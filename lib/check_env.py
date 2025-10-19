import subprocess
import sys
import platform
import os
import psutil
try:
    import torch
except ImportError:
    torch = None

# --- Helper Function to Run Shell Commands ---
def run_command(command, print_output=True):
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            bufsize=1
        )
        if print_output:
            print(result.stdout)
        return result.stdout
    except FileNotFoundError:
        print(f"Error: Command '{command.split()[0]}' not found or not in PATH.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing '{command}' (Exit Code {e.returncode}):")
        if e.stderr:
            print(e.stderr)
        else:
            print("Command failed with no specific error output.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None

# --- Function to Check All Environment Details ---
def print_environment_info():
    
    print("=" * 70)
    print("                         System and Environment Report")
    print("=" * 70)

    # 1. PYTHON AND CONDA INFORMATION
    print("\n## Conda & Python Environment")
    
    # Conda Version
    print("\n--- Conda Version ---")
    run_command("conda --version")
    
    # Conda Environment Status
    conda_env = os.environ.get('CONDA_DEFAULT_ENV') or os.environ.get('CONDA_PREFIX')
    env_name = os.path.basename(conda_env) if conda_env and os.path.isabs(conda_env) else conda_env
    print(f"\n--- Active Environment ---")
    if env_name:
        print(f"Name: {env_name}")
        print(f"Path: {conda_env}")
    else:
        print("No Conda environment is currently active.")
    
    # Python and OS
    print("\n--- Python & OS Information ---")
    print(f"Python Version: {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"OS Platform: {platform.platform()}")
    print(f"Python Executable Path: {sys.executable}")
    
    # 2. CPU INFORMATION (using psutil)
    print("\n## CPU Information (via psutil and lscpu)")
    
    # Basic CPU counts
    print(f"Logical CPU Cores: {psutil.cpu_count(logical=True)}")
    print(f"Physical CPU Cores: {psutil.cpu_count(logical=False)}")
    
    # CPU Frequency
    cpu_freq = psutil.cpu_freq()
    if cpu_freq:
        print(f"Current CPU Frequency: {cpu_freq.current / 1000:.2f} GHz")
        print(f"Max CPU Frequency: {cpu_freq.max / 1000:.2f} GHz")
    
    # CPU Model Name (Linux specific)
    print("\n--- CPU Model Name ---")
    run_command("lscpu | grep 'Model name:' | sed 's/Model name:\\s*//'")
    
    # 3. PACKAGE LISTS
    print("\n## Installed Packages")
    
    # Conda packages
    print("\n--- Conda Packages (conda list) ---")
    run_command("conda list --export | grep -v '^#'")
    
    # Pip packages (newly added)
    print("\n--- Pip Packages (pip list) ---")
    run_command("pip list")

    # 4. GPU AND DRIVER INFORMATION (Requires PyTorch)
    print("\n## GPU & Driver Information")
    
    if torch:
        # PyTorch Check
        print("\n--- PyTorch CUDA Status ---")
        if torch.cuda.is_available():
            print("CUDA is available: True (PyTorch detected)")
            print(f"PyTorch built with CUDA: {torch.version.cuda}")
            
            gpu_count = torch.cuda.device_count()
            print(f"Total GPU Devices: {gpu_count}")
            
            for i in range(gpu_count):
                print(f"  - GPU {i} Name: {torch.cuda.get_device_name(i)}")
                
            # NVIDIA Driver Check (using nvidia-smi)
            print("\n--- NVIDIA Driver & System Info (nvidia-smi) ---")
            run_command("nvidia-smi")
            
        else:
            print("CUDA is available: False (Running on CPU mode)")
            print("Driver check still running 'nvidia-smi'...")
            run_command("nvidia-smi")

    else:
        print("Warning: PyTorch is not installed. Cannot check CUDA status via PyTorch.")
        print("--- NVIDIA Driver & System Info (nvidia-smi) ---")
        run_command("nvidia-smi")
        
    print("\n" + "=" * 70)
    print("Report Finished.")
    print("=" * 70)

if __name__ == "__main__":
    if 'psutil' not in sys.modules:
        try:
            import psutil
        except ImportError:
            print("Error: The 'psutil' library is required for detailed CPU info.")
            print("Please install it: pip install psutil")
            sys.exit(1)
            
    print_environment_info()