import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import os
import re
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import create_mlp
from data_loader import ProjectDataLoader
from preprocessing import preprocess_images
from train import load_mnist_data, train_epoch, validate
from sklearn.metrics import classification_report, confusion_matrix


def run_experiment(hidden_sizes, learning_rate, dropout=0.3, num_epochs=20, batch_size=64, device='cpu'):
    """Run a single training experiment."""
    print(f"\nExperiment: LR={learning_rate}, Architecture={hidden_sizes}")
    
    train_loader, val_loader = load_mnist_data(batch_size=batch_size)
    model = create_mlp(hidden_sizes=hidden_sizes, dropout=dropout)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    final_train_acc = history['train_acc'][-1]
    print(f"Best Val Acc: {best_val_acc:.2f}%, Final Train Acc: {final_train_acc:.2f}%")
    
    return best_val_acc, final_train_acc, history


def hyperparameter_search(num_epochs=20):
    """Run hyperparameter search over learning rates and architectures."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    learning_rates = [0.01, 0.001, 0.0001]
    architectures = {
        '2_layers': [512, 256],
        '4_layers': [512, 256, 128, 64],
        '6_layers': [512, 256, 128, 64, 32, 16]
    }
    
    results = []
    
    print("\n" + "="*60)
    print("HYPERPARAMETER SEARCH")
    print("="*60)
    print(f"Learning Rates: {learning_rates}")
    print(f"Architectures: {list(architectures.keys())}")
    print(f"Total Experiments: {len(learning_rates) * len(architectures)}")
    
    experiment_num = 0
    total_experiments = len(learning_rates) * len(architectures)
    
    for arch_name, hidden_sizes in architectures.items():
        for lr in learning_rates:
            experiment_num += 1
            print(f"\n[{experiment_num}/{total_experiments}] Testing {arch_name} with LR={lr}")
            
            try:
                best_val_acc, final_train_acc, history = run_experiment(
                    hidden_sizes=hidden_sizes,
                    learning_rate=lr,
                    dropout=0.3,
                    num_epochs=num_epochs,
                    batch_size=64,
                    device=device
                )
                
                results.append({
                    'architecture': arch_name,
                    'hidden_sizes': hidden_sizes,
                    'learning_rate': lr,
                    'best_val_acc': best_val_acc,
                    'final_train_acc': final_train_acc,
                    'num_layers': len(hidden_sizes),
                    'total_params': sum(p.numel() for p in create_mlp(hidden_sizes=hidden_sizes).parameters())
                })
            except Exception as e:
                print(f"Error in experiment: {e}")
                continue
    
    os.makedirs('results', exist_ok=True)
    results_file = 'results/hyperparameter_search_results.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    analyze_hyperparameter_results(results)
    
    return results


def analyze_hyperparameter_results(results):
    """Analyze and visualize hyperparameter search results."""
    if not results:
        print("No results to analyze.")
        return
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    best_result = max(results, key=lambda x: x['best_val_acc'])
    
    print(f"\nBest Configuration:")
    print(f"  Architecture: {best_result['architecture']} ({best_result['num_layers']} layers)")
    print(f"  Hidden Sizes: {best_result['hidden_sizes']}")
    print(f"  Learning Rate: {best_result['learning_rate']}")
    print(f"  Best Val Accuracy: {best_result['best_val_acc']:.2f}%")
    print(f"  Total Parameters: {best_result['total_params']:,}")
    
    print(f"\nResults by Architecture:")
    arch_results = {}
    for r in results:
        arch = r['architecture']
        if arch not in arch_results:
            arch_results[arch] = []
        arch_results[arch].append(r)
    
    for arch, arch_runs in arch_results.items():
        best_lr = max(arch_runs, key=lambda x: x['best_val_acc'])
        print(f"\n{arch} ({len(arch_runs[0]['hidden_sizes'])} layers):")
        print(f"  Best LR: {best_lr['learning_rate']} -> Val Acc: {best_lr['best_val_acc']:.2f}%")
        for r in sorted(arch_runs, key=lambda x: x['learning_rate']):
            print(f"    LR={r['learning_rate']}: {r['best_val_acc']:.2f}%")
    
    plot_hyperparameter_results(results)


def plot_hyperparameter_results(results):
    """Plot hyperparameter search results with light colors."""
    os.makedirs('results', exist_ok=True)
    
    architectures = sorted(set(r['architecture'] for r in results))
    learning_rates = sorted(set(r['learning_rate'] for r in results))
    
    heatmap_data = np.zeros((len(architectures), len(learning_rates)))
    
    for r in results:
        arch_idx = architectures.index(r['architecture'])
        lr_idx = learning_rates.index(r['learning_rate'])
        heatmap_data[arch_idx, lr_idx] = r['best_val_acc']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(heatmap_data, cmap='YlGnBu', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(np.arange(len(learning_rates)))
    ax.set_yticks(np.arange(len(architectures)))
    ax.set_xticklabels([f'{lr:.4f}' for lr in learning_rates], fontsize=12)
    ax.set_yticklabels(architectures, fontsize=12)
    
    for i in range(len(architectures)):
        for j in range(len(learning_rates)):
            val = heatmap_data[i, j]
            text_color = "white" if val < 50 else "black"
            ax.text(j, i, f'{val:.1f}%',
                   ha="center", va="center", color=text_color, 
                   fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('Architecture', fontsize=12)
    ax.set_title('Hyperparameter Search: Validation Accuracy (%)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Validation Accuracy (%)')
    plt.tight_layout()
    
    save_path = 'results/hyperparameter_search_heatmap.png'
    plt.savefig(save_path, dpi=150)
    print(f"Heatmap saved to {save_path}")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for arch in architectures:
        arch_data = [r for r in results if r['architecture'] == arch]
        lrs = sorted(set(r['learning_rate'] for r in arch_data))
        accs = [max(r['best_val_acc'] for r in arch_data if r['learning_rate'] == lr) for lr in lrs]
        ax.plot(lrs, accs, marker='o', label=arch, linewidth=2)
    
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Best Validation Accuracy (%)')
    ax.set_title('Validation Accuracy vs Learning Rate by Architecture')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = 'results/hyperparameter_search_lines.png'
    plt.savefig(save_path)
    print(f"Line chart saved to {save_path}")
    plt.close()


def extract_group_info(local_path='collected_images'):
    """Extract images, labels, and group IDs from collected images."""
    images = []
    labels = []
    group_ids = []
    filenames = []
    
    if not os.path.exists(local_path):
        print(f"Error: {local_path} not found")
        return np.array([]), np.array([]), np.array([]), []
    
    image_files = [f for f in os.listdir(local_path) if f.endswith('.png')]
    image_files.sort()
    
    for filename in image_files:
        match = re.match(r'^(\d+)-(\d+)-(\d+)\.png$', filename)
        if match:
            digit = int(match.group(1))
            group_id = int(match.group(2))
            
            filepath = os.path.join(local_path, filename)
            
            try:
                from PIL import Image
                img = Image.open(filepath)
                
                if img.mode != 'L':
                    if img.mode in ('RGBA', 'LA'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'RGBA':
                            background.paste(img, mask=img.split()[3])
                        else:
                            background.paste(img)
                        img = background.convert('L')
                    else:
                        img = img.convert('L')
                
                if img.size != (28, 28):
                    img = img.resize((28, 28), Image.Resampling.LANCZOS)
                
                img_array = np.array(img)
                images.append(img_array)
                labels.append(digit)
                group_ids.append(group_id)
                filenames.append(filename)
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
    
    return np.array(images), np.array(labels), np.array(group_ids), filenames


def test_by_group(model, images, labels, group_ids, device):
    """Test model on collected images and group results by group ID."""
    processed_images = preprocess_images(images)
    labels_tensor = torch.from_numpy(labels).long()
    
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_labels = []
    all_group_ids = []
    
    dataset = torch.utils.data.TensorDataset(processed_images, labels_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    with torch.no_grad():
        batch_idx = 0
        for batch_images, batch_labels in tqdm(loader, desc="Testing by group"):
            batch_images = batch_images.to(device)
            outputs = model(batch_images)
            _, predicted = torch.max(outputs.data, 1)
            
            start_idx = batch_idx * 64
            end_idx = start_idx + len(batch_labels)
            batch_group_ids = group_ids[start_idx:end_idx]
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_group_ids.extend(batch_group_ids)
            
            batch_idx += 1
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_group_ids = np.array(all_group_ids)
    
    group_results = {}
    unique_groups = np.unique(all_group_ids)
    
    for group_id in unique_groups:
        mask = all_group_ids == group_id
        group_predictions = all_predictions[mask]
        group_labels = all_labels[mask]
        
        correct = (group_predictions == group_labels).sum()
        total = len(group_labels)
        accuracy = 100 * correct / total if total > 0 else 0
        
        per_class_acc = {}
        for digit in range(10):
            digit_mask = group_labels == digit
            if digit_mask.sum() > 0:
                digit_correct = (group_predictions[digit_mask] == group_labels[digit_mask]).sum()
                per_class_acc[digit] = 100 * digit_correct / digit_mask.sum()
            else:
                per_class_acc[digit] = 0.0
        
        group_results[int(group_id)] = {
            'accuracy': float(accuracy),
            'correct': int(correct),
            'total': int(total),
            'per_class_acc': {str(k): float(v) for k, v in per_class_acc.items()},
            'predictions': group_predictions.tolist(),
            'labels': group_labels.tolist()
        }
    
    return group_results


def analyze_groups(model_path='./models/best_model.pth', collected_images_path='collected_images'):
    """Analyze collected images by group."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print(f"\nLoading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model = create_mlp(hidden_sizes=[512, 256, 128], dropout=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\nExtracting images and group information from {collected_images_path}")
    images, labels, group_ids, filenames = extract_group_info(collected_images_path)
    
    if len(images) == 0:
        print("No images found!")
        return
    
    print(f"Loaded {len(images)} images")
    print(f"Groups found: {sorted(np.unique(group_ids))}")
    print(f"Images per group: {dict(zip(*np.unique(group_ids, return_counts=True)))}")
    
    print("\n" + "="*60)
    print("Testing by Group")
    print("="*60)
    group_results = test_by_group(model, images, labels, group_ids, device)
    
    print("\n" + "="*60)
    print("GROUP ANALYSIS RESULTS")
    print("="*60)
    
    sorted_groups = sorted(group_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print(f"\n{'Group':<8} {'Accuracy':<12} {'Correct':<10} {'Total':<10}")
    print("-" * 60)
    
    for group_id, results in sorted_groups:
        print(f"{group_id:<8} {results['accuracy']:<12.2f} {results['correct']:<10} {results['total']:<10}")
    
    print("\n" + "="*60)
    print("DETAILED PER-GROUP ANALYSIS")
    print("="*60)
    
    for group_id, results in sorted_groups:
        print(f"\nGroup {group_id}:")
        print(f"  Overall Accuracy: {results['accuracy']:.2f}% ({results['correct']}/{results['total']})")
        print(f"  Per-Class Accuracy:")
        for digit in range(10):
            acc = results['per_class_acc'].get(str(digit), 0)
            if acc > 0:
                print(f"    Digit {digit}: {acc:.2f}%")
    
    os.makedirs('results', exist_ok=True)
    results_file = 'results/group_analysis_results.json'
    
    with open(results_file, 'w') as f:
        json.dump(group_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    plot_group_results(group_results)
    
    return group_results


def plot_group_results(group_results):
    """Create visualizations for group analysis."""
    os.makedirs('results', exist_ok=True)
    
    groups = sorted(group_results.keys())
    accuracies = [group_results[g]['accuracy'] for g in groups]
    totals = [group_results[g]['total'] for g in groups]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#90EE90' if acc >= 80 else '#FFD700' if acc >= 60 else '#FFB6C1' for acc in accuracies]
    bars = ax.bar([f'Group {g}' for g in groups], accuracies, color=colors)
    
    ax.set_xlabel('Group ID')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Accuracy by Group')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, acc, total in zip(bars, accuracies, totals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%\n(n={total})',
                ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/group_accuracy_comparison.png')
    print("Group accuracy comparison saved to results/group_accuracy_comparison.png")
    plt.close()
    
    digit_acc_matrix = np.zeros((len(groups), 10))
    
    for i, group_id in enumerate(groups):
        for digit in range(10):
            digit_acc_matrix[i, digit] = group_results[group_id]['per_class_acc'].get(str(digit), 0)
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(groups) * 0.5)))
    im = ax.imshow(digit_acc_matrix, cmap='YlGnBu', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(len(groups)))
    ax.set_xticklabels([f'Digit {i}' for i in range(10)])
    ax.set_yticklabels([f'Group {g}' for g in groups])
    
    for i in range(len(groups)):
        for j in range(10):
            val = digit_acc_matrix[i, j]
            if val > 0:
                text = ax.text(j, i, f'{val:.0f}%',
                             ha="center", va="center", 
                             color="black", 
                             fontsize=8)
    
    ax.set_xlabel('Digit')
    ax.set_ylabel('Group')
    ax.set_title('Per-Group, Per-Digit Accuracy Heatmap')
    plt.colorbar(im, ax=ax, label='Accuracy (%)')
    plt.tight_layout()
    plt.savefig('results/group_digit_heatmap.png')
    print("Group-digit heatmap saved to results/group_digit_heatmap.png")
    plt.close()
    
    print("\n" + "="*60)
    print("GROUP STATISTICS")
    print("="*60)
    print(f"Best Group: Group {max(groups, key=lambda g: group_results[g]['accuracy'])} "
          f"({max(accuracies):.2f}%)")
    print(f"Worst Group: Group {min(groups, key=lambda g: group_results[g]['accuracy'])} "
          f"({min(accuracies):.2f}%)")
    print(f"Average Accuracy: {np.mean(accuracies):.2f}%")
    print(f"Std Deviation: {np.std(accuracies):.2f}%")


def main():
    """Main function to run analyses."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Analysis: Hyperparameter Search & Group Analysis')
    parser.add_argument('--hyperparameter', action='store_true', 
                       help='Run hyperparameter search')
    parser.add_argument('--group', action='store_true',
                       help='Run group-based analysis')
    parser.add_argument('--all', action='store_true',
                       help='Run both analyses')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs for hyperparameter search')
    
    args = parser.parse_args()
    
    if args.all or (not args.hyperparameter and not args.group):
        args.hyperparameter = True
        args.group = True
    
    if args.hyperparameter:
        print("\n" + "="*60)
        print("RUNNING HYPERPARAMETER SEARCH")
        print("="*60)
        print("This will test:")
        print("  - Learning rates: [0.01, 0.001, 0.0001]")
        print("  - Architectures: 2, 4, 6 layers")
        print("  - Total experiments: 9")
        print(f"  - Epochs per experiment: {args.epochs}")
        
        response = input("\nContinue? (y/n): ")
        if response.lower() == 'y':
            hyperparameter_search(num_epochs=args.epochs)
        else:
            print("Skipping hyperparameter search.")
    
    if args.group:
        print("\n" + "="*60)
        print("RUNNING GROUP ANALYSIS")
        print("="*60)
        
        if not os.path.exists('./models/best_model.pth'):
            print("\nWarning: Model not found. Please train a model first.")
            print("Run: python main.py --mode train")
            return
        
        analyze_groups()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()

