#!/bin/bash

# --- Job Setup ---
# This script is designed to be run on a login node of your cluster.
# It activates the correct environment and executes the Python script on a specific GPU.

echo "Starting PBH Simulation..."

# --- 1. Activate Conda Environment ---
# Make sure you have created this environment as per the setup instructions.
source ~/miniconda3/etc/profile.d/conda.sh # Adjust path to your conda if different
conda activate pbh_sim

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'pbh_sim'."
    echo "Please create it first using 'conda create -n pbh_sim python=3.10'"
    exit 1
fi

# --- 2. Select the GPU ---
# Your nvidia-smi output showed GPU 1 was free.
# This environment variable tells PyTorch and CUDA to ONLY use the GPU with ID 1.
# To the Python script, this GPU will appear as 'cuda:0'.
export CUDA_VISIBLE_DEVICES=1
echo "Running on GPU ID: ${CUDA_VISIBLE_DEVICES}"

# --- 3. Run the Python Simulation Script ---
# The 'nohup' command ensures the script keeps running even if you disconnect.
# '&' runs the process in the background.
# Output is redirected to a log file.
nohup python main.py > simulation.log 2>&1 &

# --- 4. Confirmation ---
# The '$!' variable holds the process ID (PID) of the last backgrounded job.
echo "✅ Simulation submitted successfully!"
echo "Process ID (PID): $!"
echo "Monitor progress with: tail -f simulation.log"
echo "Check GPU usage on the node with: nvidia-smi"
```

---

### **How to Run on the Cluster (Step-by-Step)**

**Goal:** Run your simulation on the free A100 GPU (`ID 1`) on node `gpu1`.

#### **Step 1: Connect to the GPU Node**

First, log in to the cluster, then jump to the GPU node.
```bash
# Log in to the main login node
ssh -p 22022 username@210.77.30.33

# Connect to the gpu1 node
ssh gpu1
```
You are now on the machine with the GPUs.

#### **Step 2: One-Time Environment Setup**

You only need to do this once. If you've already created the `pbh_sim` environment, you can skip this.
```bash
# Create a dedicated conda environment
conda create -n pbh_sim python=3.10 -y

# Activate the new environment
conda activate pbh_sim

# Install PyTorch for CUDA 12.1 (compatible with your driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other packages
pip install pyyaml numpy matplotlib
```

#### **Step 3: Place Your Code and Run the Script**

1.  Upload all your `.py`, `.yaml`, and the new `run_simulation.sh` files to a directory on the cluster (e.g., `~/my_pbh_project/`).
2.  Make the submission script executable:
    ```bash
    chmod +x run_simulation.sh
    ```
3.  Execute the script to start your simulation on GPU 1:
    ```bash
    ./run_simulation.sh

