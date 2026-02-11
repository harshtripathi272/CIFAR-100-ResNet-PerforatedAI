import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_cifar100_loaders
from models import get_model

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    print(f"Loading data for model: {args.model}...")
    train_loader, test_loader = get_cifar100_loaders(batch_size=args.batch_size, resize=224)

    # Set seed for reproducibility across all models
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Model
    print(f"Initializing model: {args.model}...")
    model = get_model(args.model, num_classes=100, pretrained=True)
    model = model.to(device)

    # Criterion and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # PerforatedAI Tracker Integration
    pai_enabled = (args.model == 'resnet18_perforated')
    if pai_enabled:
        from perforatedai import globals_perforatedai as GPA
        GPA.pai_tracker.set_optimizer_instance(optimizer)
        GPA.pai_tracker.set_scheduler(type(scheduler))
        GPA.pai_tracker.member_vars["scheduler_instance"] = scheduler

    # Metrics
    best_acc = 0.0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        if pai_enabled:
            GPA.pai_tracker.start_epoch()
            
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        epoch_start = time.time()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(test_loader)
        
        epoch_time = time.time() - epoch_start
        scheduler.step()
        
        if pai_enabled:
            GPA.pai_tracker.add_loss(train_loss)
            # This handles graphing and PAI logic
            # returns potentially updated model (if restructuring happens)
            model, status, _ = GPA.pai_tracker.add_validation_score(val_acc, model)
            if status == 2: # TRAINING_COMPLETE
                 print("PAI signals training is complete.")
                 break

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Time: {epoch_time:.2f}s | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"cifar100_{args.model}_best.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stop triggered at epoch {epoch+1}")
                break

    total_time = time.time() - start_time
    print(f"Training finished in {total_time/60:.2f} minutes.")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    
    # Save results to file
    with open("experiment_results.txt", "a") as f:
        f.write(f"{args.model},{args.lr},{args.batch_size},{args.momentum},{args.weight_decay},{best_acc:.2f},{total_time:.2f}\n")
    
    if pai_enabled:
        print("Saving PAI graphs...")
        GPA.pai_tracker.save_graphs()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet on CIFAR-100')
    parser.add_argument('--model', type=str, required=True, choices=['resnet18', 'resnet34', 'resnet18_perforated'],
                        help='Model name')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size') # Reduced batch size for 224x224
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')

    args = parser.parse_args()
    train(args)
