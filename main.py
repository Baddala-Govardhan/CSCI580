import argparse
import os
from train import train_model, load_mnist_data, plot_training_history
from test import evaluate_model
from model import create_mlp


def main():
    parser = argparse.ArgumentParser(description='MLP Handwritten Digit Recognition')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['train', 'test', 'all'],
                       help='Mode: train, test, or all (default: all)')
    parser.add_argument('--model_path', type=str, default='./models/best_model.pth',
                       help='Path to model checkpoint for testing')
    parser.add_argument('--collected_images', type=str, default='collected_images',
                       help='Path to collected images directory')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[512, 256, 128],
                       help='Hidden layer sizes')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout probability')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MLP Handwritten Digit Recognition Project")
    print("="*60)
    
    if args.mode in ['train', 'all']:
        print("\n[1/2] TRAINING PHASE")
        print("-"*60)
        
        # Load data
        train_loader, val_loader = load_mnist_data(batch_size=args.batch_size)
        
        # Create model
        model = create_mlp(
            hidden_sizes=args.hidden_sizes,
            dropout=args.dropout
        )
        
        # Train
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            device=device
        )
        
        # Plot history
        plot_training_history(history)
        print("\nTraining completed!")
    
    if args.mode in ['test', 'all']:
        print("\n[2/2] TESTING PHASE")
        print("-"*60)
        
        if not os.path.exists(args.model_path):
            print(f"Error: Model not found at {args.model_path}")
            print("Please train the model first or specify correct model path.")
            return
        
        evaluate_model(
            model_path=args.model_path,
            collected_images_path=args.collected_images,
            save_results=True
        )
    
    print("\n" + "="*60)
    print("Pipeline completed!")
    print("="*60)


if __name__ == "__main__":
    main()

