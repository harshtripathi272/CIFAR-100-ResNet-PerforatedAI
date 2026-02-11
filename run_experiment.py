import os
import sys
import subprocess
import pandas as pd

def run_command(command):
    print(f"Running: {command}")
    process = subprocess.Popen(command, shell=True)
    process.wait()

def main():
    # Clear previous results
    if os.path.exists("experiment_results.txt"):
        os.remove("experiment_results.txt")
        
    models = ['resnet18', 'resnet34', 'resnet18_perforated']
    epochs = 5
    batch_size = 64
    lr = 0.001 # Lower LR for fine-tuning
    
    print("Starting experiments...")
    
    for model in models:
        print(f"\n{'='*40}")
        print(f"Evaluating {model}")
        print(f"{'='*40}")
        
        cmd = f"{sys.executable} train.py --model {model} --epochs {epochs} --batch_size {batch_size} --lr {lr}"
        run_command(cmd)
        
    print("\nExperiments completed.")
    
    # Generate Report
    if os.path.exists("experiment_results.txt"):
        print("\nGenerating comparison report...")
        columns = ['Model', 'Best_Val_Acc', 'Total_Time_Sec']
        
        try:
            df = pd.read_csv("experiment_results.txt", header=None, names=columns)
            
            # Calculate metrics relative to ResNet-18
            base_acc = df[df['Model'] == 'resnet18']['Best_Val_Acc'].values[0]
            base_time = df[df['Model'] == 'resnet18']['Total_Time_Sec'].values[0]
            
            df['Acc_Gain'] = df['Best_Val_Acc'] - base_acc
            df['Time_Factor'] = df['Total_Time_Sec'] / base_time
            
            print("\nComparison Results:")
            print(df.to_markdown(index=False))
            
            with open("final_report.md", "w") as f:
                f.write("# PerforatedAI CIFAR-100 Evaluation Report\n\n")
                f.write(df.to_markdown(index=False))
                f.write("\n\n")
                f.write(f"*Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}*\n")
                
        except Exception as e:
            print(f"Error parsing results: {e}")
            with open("experiment_results.txt", "r") as f:
                print(f.read())
    else:
        print("No results found.")

if __name__ == "__main__":
    main()
