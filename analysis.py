import subprocess
import os
import sys

def main():
    print("==================================================")
    print("DSQD Symbolic Regression Autoresearch Automator")
    print("==================================================\n")
    
    # 1. Run training (train.py)
    print("Step 1: Running Symbolic Regression (train.py)...")
    try:
        # Run train.py and capture output to run.log
        with open("run.log", "w") as log_file:
            subprocess.run([sys.executable, "train.py"], stdout=log_file, stderr=subprocess.STDOUT, check=True)
        print("Symbolic Regression search completed successfully.")
    except subprocess.CalledProcessError as e:
        print("Error: train.py failed during execution.")
        # Read and print the last few lines of run.log
        if os.path.exists("run.log"):
            with open("run.log", "r") as f:
                print("Tail of run.log:")
                lines = f.readlines()
                for line in lines[-30:]:
                    print(line, end="")
        sys.exit(1)
        
    # Read the summary from run.log
    if os.path.exists("run.log"):
        print("\n--- Summary from run.log ---")
        with open("run.log", "r") as f:
            content = f.read()
            # Find the summary section (indicated by ---)
            if "---" in content:
                summary = content.split("---")[-1]
                print(summary.strip())
            else:
                print(content[-500:])
                
    # 2. Run visualization (visualize.py)
    print("\nStep 2: Generating Visualizations (visualize.py)...")
    try:
        result = subprocess.run([sys.executable, "visualize.py"], capture_output=True, text=True, check=True)
        print("Visualizations generated successfully.")
        print(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print("Error: visualize.py failed during execution.")
        print(e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
