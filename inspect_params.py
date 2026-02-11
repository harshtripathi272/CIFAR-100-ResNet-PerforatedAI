import sys
import torch
import torchvision
from perforatedai import library_perforatedai as LPA
from perforatedai import utils_perforatedai as UPA

# Ensure stdout is unbuffered
sys.stdout.reconfigure(line_buffering=True)

print("Starting inspection...", flush=True)

try:
    print("Loading base model...", flush=True)
    base_model = torchvision.models.resnet18(weights=None)
    
    print("Creating Perforated model...", flush=True)
    model = LPA.ResNetPAIPreFC(base_model)
    
    print("Model created successfully.", flush=True)
    
    print("\nModel Structure (Top Level):")
    for name, module in model.named_children():
        print(f"  - {name}: {type(module).__name__}")
        
    print("\nConfigurable Parameters (via __init__ or attributes):")
    # Check attributes that might be hyperparameters
    params_of_interest = []
    for attr in dir(model):
        if not attr.startswith('_') and not callable(getattr(model, attr)):
            val = getattr(model, attr)
            if isinstance(val, (int, float, bool, str, list, dict)):
                params_of_interest.append((attr, val))
                
    for name, val in params_of_interest:
        print(f"  - {name}: {val}")
        
    print("\nChecking for internal PAI modules params...")
    # PAI modules often have specific configurations
    # We can use UPA helper functions
    try:
        pai_params = UPA.get_pai_module_params(model, 0)
        print(f"  - Found {len(pai_params)} PAI specific parameters (tensors/weights).")
    except Exception as e:
        print(f"  - Could not get PAI module params: {e}")

    print("\nInspection complete.", flush=True)

except Exception as e:
    print(f"Error during inspection: {e}", flush=True)
