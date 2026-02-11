import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_cifar100_loaders(batch_size=128, resize=224, num_workers=2):
    """
    Returns train and test loaders for CIFAR-100 with resizing to match ImageNet.
    
    Args:
        batch_size (int): Batch size for loaders.
        resize (int): Size to resize images to (e.g., 224 for ResNet).
        num_workers (int): Number of worker threads for data loading.
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    
    # Standard ImageNet normalization since we are using pretrained models
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        normalize,
    ])

    train_set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers)

    test_set = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers)

    return train_loader, test_loader
