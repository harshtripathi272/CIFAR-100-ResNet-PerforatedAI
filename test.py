import argparse
import torch
import torch.nn as nn
from dataset import get_cifar100_loaders
from models import get_model
import time

def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    print("Loading test data...")
    _, test_loader = get_cifar100_loaders(batch_size=args.batch_size, resize=224)

    # Model
    print(f"Initializing model: {args.model}...")
    # For testing, we must use the same structure (num_classes=100)
    model = get_model(args.model, num_classes=100, pretrained=False) 
    
    # Load Weights
    if args.checkpoint:
        print(f"Loading weights from {args.checkpoint}...")
        try:
            state_dict = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(state_dict)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            return
    else:
        print("WARNING: No checkpoint provided! Testing with random/initialized weights.")

    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            
            if i % 10 == 0:
                print(f"Step [{i}/{len(test_loader)}]")

    val_acc = 100 * val_correct / val_total
    val_loss = val_loss / len(test_loader)
    total_time = time.time() - start_time

    print(f"\nTest Finished in {total_time:.2f}s")
    print(f"Test Loss: {val_loss:.4f}")
    print(f"Test Accuracy: {val_acc:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test ResNet on CIFAR-100')
    parser.add_argument('--model', type=str, required=True, choices=['resnet18', 'resnet34', 'resnet18_perforated'],
                        help='Model architecture')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to .pth checkpoint file')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

    args = parser.parse_args()
    test(args)
