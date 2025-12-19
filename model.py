import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for digit classification.
    Input: 28x28 grayscale images (784 features)
    Output: 10 classes (digits 0-9)
    """
    
    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], num_classes=10, dropout=0.3):
        """
        Initialize MLP model.
        
        Args:
            input_size: Size of input features (28*28 = 784 for MNIST)
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes (10 for digits 0-9)
            dropout: Dropout probability for regularization
        """
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28) or (batch_size, 784)
        
        Returns:
            logits: Output tensor of shape (batch_size, 10)
        """
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        return self.network(x)
    
    def predict(self, x):
        """
        Make predictions.
        
        Args:
            x: Input tensor
        
        Returns:
            predictions: Predicted class indices
        """
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions


def create_mlp(hidden_sizes=[512, 256, 128], dropout=0.3):
    """
    Factory function to create MLP model.
    
    Args:
        hidden_sizes: List of hidden layer sizes
        dropout: Dropout probability
    
    Returns:
        model: MLP model instance
    """
    return MLP(hidden_sizes=hidden_sizes, dropout=dropout)


if __name__ == "__main__":
    # Test model
    model = create_mlp()
    print(model)
    
    # Test forward pass
    dummy_input = torch.randn(32, 1, 28, 28)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    predictions = model.predict(dummy_input)
    print(f"Predictions shape: {predictions.shape}")

