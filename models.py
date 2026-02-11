import torch
import torch.nn as nn
import torchvision
from perforatedai import library_perforatedai as LPA
from perforatedai import utils_perforatedai as UPA

def get_model(model_name, num_classes=100, pretrained=True):
    """
    Factory function to get the requested model.

    Args:
        model_name (str): 'resnet18', 'resnet34', or 'resnet18_perforated'
        num_classes (int): Number of output classes (e.g., 100 for CIFAR-100)
        pretrained (bool): Whether to use pretrained weights (ImageNet)

    Returns:
        nn.Module: The requested model
    """
    
    if model_name == 'resnet18':
        weights = 'IMAGENET1K_V1' if pretrained else None
        model = torchvision.models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif model_name == 'resnet34':
        weights = 'IMAGENET1K_V1' if pretrained else None
        model = torchvision.models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif model_name == 'resnet18_perforated':
        # 1. Base model
        base_model = torchvision.models.resnet18(weights=None) # We load specific weights later
        
        # 2. Perforated wrapper
        model = LPA.ResNetPAIPreFC(base_model)
        
        # 3. Load pretrained perforated weights
        if pretrained:
            print("Loading perforated-ai/resnet-18-perforated from HuggingFace...")
            try:
                model = UPA.from_hf_pretrained(model, 'perforated-ai/resnet-18-perforated')
            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
                print("Proceeding with random weights (WARNING: Performance will be poor)")

        # 4. Replace FC layer for transfer learning
        # Accessing the internal FC layer of the wrapped model might depend on implementation
        # The wrapper exposes .fc
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        return model

    else:
        raise ValueError(f"Unknown model name: {model_name}")
