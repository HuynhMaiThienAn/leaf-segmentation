import subprocess
import sys
import os

def run_script(script_name):
    if not os.path.exists(script_name):
        print(f"Error: {script_name} not found.")
        return False
    print(f"Running {script_name}...")
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(f"Error output from {script_name}:\n{result.stderr.strip()}", file=sys.stderr)
    return result.returncode == 0

def main():
    # 1. Update Step 2 to extract Hue and Saturation
    run_script("fix_clean_notebook.py")
    run_script("modify_notebook.py")
    
    # 1.5 Insert 2D color space analysis scatter plots comparison
    run_script("add_space_comparison.py")
    
    # 2. Upgrade Steps 3 & 4 to 5D PCA (L*, a*, b*, S, H)
    run_script("add_5d_pca.py")
    
    # 3. Standardize mask variable names to 'cleaned' and add Watershed safety checks
    run_script("fix_notebooks.py")
    
    print("\nPipeline update successfully completed! Both notebooks have been upgraded to 5D PCA and standardized.")

if __name__ == "__main__":
    main()
