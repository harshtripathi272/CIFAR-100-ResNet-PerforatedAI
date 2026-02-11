import os
import sys
import subprocess
import pandas as pd
import itertools

def run_command(command):
    print(f"Running: {command}")
    # Run command and let stdout/stderr flow through to parent process (which is captured by run_command tool)
    subprocess.run(command, shell=True, check=False)

def main():
    # Clear previous results
    if os.path.exists("experiment_results.txt"):
        os.remove("experiment_results.txt")
        
    models = ['resnet18', 'resnet34', 'resnet18_perforated']
    # BENCHMARK CONFIGURATION
    lrs = [0.01, 0.001]
    batch_sizes = [64,128]
    momentums = [0.9]
    weight_decays = [1e-4, 5e-4]
    epochs = 100
    patience = 15
    
    print("Starting BENCHMARK sweep (50 epochs per run)...")
    
    # Generate all combinations
    combinations = list(itertools.product(models, lrs, batch_sizes, momentums, weight_decays))
    
    for model, lr, batch_size, momentum, weight_decay in combinations:
        print(f"\n{'='*60}")
        print(f"Sweeping {model} | LR: {lr} | Batch: {batch_size} | Mom: {momentum} | WD: {weight_decay}")
        print(f"{'='*60}")
        
        cmd = f'"{sys.executable}" train.py --model {model} --epochs {epochs} --batch_size {batch_size} --lr {lr} --momentum {momentum} --weight_decay {weight_decay} --patience {patience}'
        run_command(cmd)
        
    print("\nSweep completed.")
    
    # Generate Report
    if os.path.exists("experiment_results.txt"):
        print("\nGenerating sweep report...")
        columns = ['Model', 'LR', 'Batch_Size', 'Momentum', 'Weight_Decay', 'Best_Val_Acc', 'Total_Time_Sec']
        
        try:
            df = pd.read_csv("experiment_results.txt", header=None, names=columns)
            
            # Find best config for each model
            best_configs = df.loc[df.groupby('Model')['Best_Val_Acc'].idxmax()]
            
            print("\nBest Configurations per Model:")
            print(best_configs.to_markdown(index=False))
            
            print("\nFull Sweep Results:")
            print(df.sort_values(by=['Model', 'Best_Val_Acc'], ascending=[True, False]).to_markdown(index=False))
            
            with open("sweep_report.md", "w") as f:
                f.write("# CIFAR-100 Hyperparameter Sweep Report\n\n")
                f.write("## Best Configurations\n")
                f.write(best_configs.to_markdown(index=False))
                f.write("\n\n## Full Results\n")
                f.write(df.sort_values(by=['Model', 'Best_Val_Acc'], ascending=[True, False]).to_markdown(index=False))
                
        except Exception as e:
            print(f"Error parsing results: {e}")
            with open("experiment_results.txt", "r") as f:
                print(f.read())
    else:
        print("No results found.")

if __name__ == "__main__":
    main()
