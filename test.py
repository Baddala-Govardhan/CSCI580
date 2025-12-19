import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import os
from model import MLP, create_mlp
from data_loader import ProjectDataLoader
from preprocessing import preprocess_images
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def load_mnist_test(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"MNIST test samples: {len(test_dataset)}")
    return test_loader, test_dataset


def test_model(model, test_loader, device, dataset_name="Test"):
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Testing on {dataset_name}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    
    # Calculate per-class accuracy
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    per_class_acc = {}
    for digit in range(10):
        mask = all_labels == digit
        if mask.sum() > 0:
            class_correct = (all_predictions[mask] == all_labels[mask]).sum()
            class_total = mask.sum()
            per_class_acc[digit] = 100 * class_correct / class_total
        else:
            per_class_acc[digit] = 0.0
    
    return accuracy, all_predictions, all_labels, per_class_acc


def test_on_collected_images(model, device, local_path='collected_images'):
    # Load collected images
    images, labels = ProjectDataLoader(local_path=local_path)
    
    if len(images) == 0:
        print("No collected images found. Skipping collected images test.")
        return None, None, None, None
    
    print(f"\nTesting on {len(images)} collected images")
    
    # Preprocess images
    processed_images = preprocess_images(images)
    labels_tensor = torch.from_numpy(labels).long()
    
    # Create dataset and loader
    dataset = torch.utils.data.TensorDataset(processed_images, labels_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Test
    return test_model(model, loader, device, dataset_name="Collected Images")


def plot_confusion_matrix(y_true, y_pred, dataset_name, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def print_evaluation_report(accuracy, per_class_acc, predictions, labels, dataset_name):
    """Print detailed evaluation report."""
    print(f"\n{'='*60}")
    print(f"Evaluation Report: {dataset_name}")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"\nPer-Class Accuracy:")
    for digit in range(10):
        print(f"  Digit {digit}: {per_class_acc[digit]:.2f}%")
    
    print(f"\nClassification Report:")
    print(classification_report(labels, predictions, target_names=[str(i) for i in range(10)]))
    
    # Count samples per class
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nSamples per class:")
    for digit, count in zip(unique, counts):
        print(f"  Digit {digit}: {count}")


def evaluate_model(model_path='./models/best_model.pth', 
                   collected_images_path='collected_images',
                   save_results=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = create_mlp(hidden_sizes=[512, 256, 128], dropout=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded. Validation accuracy during training: {checkpoint['val_acc']:.2f}%")
    
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Test on MNIST test dataset
    print("\n" + "="*60)
    print("Testing on MNIST Test Dataset")
    print("="*60)
    mnist_loader, mnist_dataset = load_mnist_test()
    mnist_acc, mnist_pred, mnist_labels, mnist_per_class = test_model(
        model, mnist_loader, device, "MNIST Test"
    )
    
    print_evaluation_report(mnist_acc, mnist_per_class, mnist_pred, mnist_labels, "MNIST Test")
    
    if save_results:
        plot_confusion_matrix(mnist_labels, mnist_pred, "MNIST Test", 
                            os.path.join(results_dir, 'mnist_confusion_matrix.png'))
    
    # Test on collected images
    print("\n" + "="*60)
    print("Testing on Collected Team Images")
    print("="*60)
    collected_acc, collected_pred, collected_labels, collected_per_class = test_on_collected_images(
        model, device, collected_images_path
    )
    
    if collected_acc is not None:
        print_evaluation_report(collected_acc, collected_per_class, collected_pred, 
                              collected_labels, "Collected Images")
        
        if save_results:
            plot_confusion_matrix(collected_labels, collected_pred, "Collected Images",
                                os.path.join(results_dir, 'collected_confusion_matrix.png'))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"MNIST Test Accuracy: {mnist_acc:.2f}%")
    if collected_acc is not None:
        print(f"Collected Images Accuracy: {collected_acc:.2f}%")
        print(f"Accuracy Difference: {abs(mnist_acc - collected_acc):.2f}%")
    
    # Save results to file
    if save_results:
        results_file = os.path.join(results_dir, 'evaluation_results.txt')
        with open(results_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("EVALUATION RESULTS\n")
            f.write("="*60 + "\n\n")
            f.write(f"MNIST Test Accuracy: {mnist_acc:.2f}%\n\n")
            f.write("MNIST Per-Class Accuracy:\n")
            for digit in range(10):
                f.write(f"  Digit {digit}: {mnist_per_class[digit]:.2f}%\n")
            
            if collected_acc is not None:
                f.write(f"\nCollected Images Accuracy: {collected_acc:.2f}%\n\n")
                f.write("Collected Images Per-Class Accuracy:\n")
                for digit in range(10):
                    f.write(f"  Digit {digit}: {collected_per_class[digit]:.2f}%\n")
        
        print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    evaluate_model()

