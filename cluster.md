# **Guide: Setting Up and Running the PBH Simulation on the Cluster**

This guide provides step-by-step instructions for deploying your PBH simulation project from GitHub onto your university's cluster, setting up the required environment with GPU support, and running the different simulation workflows.

### **Step 1: Connect to the Cluster and a GPU Node**

First, you need to log in to the cluster's main login node, and from there, connect to one of the GPU nodes. We'll use gpu1 which contains the NVIDIA A100 GPUs suitable for scientific computing.

\# 1\. Log in to the main login node from your local machine  
ssh \-p 22022 your\_username@210.77.30.33

\# 2\. Once logged in, connect to the gpu1 node  
ssh gpu1

Your terminal prompt should now show that you are on gpu1 (e.g., \[your\_username@gpu1 \~\]$).

### **Step 2: Clone Your Project from GitHub**

Next, use the gh command-line tool to clone your repository into your home directory on the cluster.

\# 1\. Clone your repository (replace with your actual repo name)  
gh repo clone your\_github\_username/pbh-merger-simulation

\# 2\. Navigate into the project directory  
cd pbh-merger-simulation/

### **Step 3: Set Up the Conda Environment for the GPU**

This is the most critical step. We will create a new Conda environment and install the required packages, including the version of PyTorch that can use the cluster's NVIDIA GPUs.

\# 1\. Create a new Conda environment named 'pbh\_sim' with Python 3.10  
conda create \-n pbh\_sim python=3.10 \-y

\# 2\. Activate the new environment  
conda activate pbh\_sim

\# 3\. Install the GPU-enabled version of PyTorch.  
\# The cluster's nvidia-smi output showed CUDA 12.4. The 'cu121' package is compatible.  
pip install torch torchvision toraudio \--index-url \[https://download.pytorch.org/whl/cu121\](https://download.pytorch.org/whl/cu121)

\# 4\. Install the other required packages  
pip install numpy pyyaml matplotlib tqdm

Your environment on the cluster is now correctly configured for GPU computations.

### **Step 4: Running Your Simulations on the Cluster**

#### **A Note on Resource Management (Important\!)**

To be a good citizen on the shared cluster, you should always limit the resources your jobs consume.

* **To Limit GPU Usage:** Use the CUDA\_VISIBLE\_DEVICES environment variable to select a *single* GPU. Your nvidia-smi check showed GPU 1 was free, so we will select that one.  
* **To Limit CPU Usage:** Use the OMP\_NUM\_THREADS environment variable to specify the number of CPU cores your job can use. The gpu1 node has 32 cores, so limiting your job to 4 or 8 cores is a considerate choice.

You should set these variables right before you launch your job.

#### **Workflow 1: Monte Carlo Merger Rate (Primary Goal)**

This is the main workflow for reproducing the paper's results.

1. **Prepare Configuration Files:** Create several copies of input\_cluster\_gpu.yaml and modify the f\_pbh value in each one.  
2. **Run the Simulation:**  
   \# A. Set Resource Limits  
   \# Select only the free GPU (ID 1\)  
   export CUDA\_VISIBLE\_DEVICES=1  
   \# Limit the job to use a maximum of 4 CPU cores  
   export OMP\_NUM\_THREADS=4

   echo "Running on GPU ID: ${CUDA\_VISIBLE\_DEVICES} with ${OMP\_NUM\_THREADS} CPU threads."

   \# B. Run the simulation in the background  
   nohup python monte\_carlo\_mergers.py input\_cluster\_gpu.yaml \> mc\_run.log 2\>&1 &

   echo "Job submitted with PID: $\!"

3. **Monitor Progress:**  
   * Check log file: tail \-f mc\_run.log  
   * Check GPU usage: nvidia-smi  
4. **Download and View Results:** After the jobs finish, download the .json files to your local machine to plot them.  
   * On your local machine:  
     scp \-P 22022 \-r your\_username@210.77.30.33:\~/pbh-merger-simulation/results\_cluster\_gpu/ .  
     python plot\_rate\_vs\_time.py results\_cluster\_gpu/mc\_merger\_history\_...json

#### **Workflow 2: N-Body Animation**

This simulation is CPU-intensive, so limiting its core usage is especially important.

1. **Run the Data Generator on the Cluster:**  
   \# A. Set Resource Limits (no GPU needed for this script)  
   export OMP\_NUM\_THREADS=8 \# Allow a few more cores for this intensive task

   echo "Running N-body simulation with ${OMP\_NUM\_THREADS} CPU threads."

   \# B. Run in the background  
   nohup python nbody\_visualizer.py input\_local\_cpu.yaml \> nbody\_run.log 2\>&1 &

   echo "Job submitted with PID: $\!"

2. **Download Data and Create Video Locally:**  
   * On your local machine, download the position data:  
     scp \-P 22022 \-r your\_username@210.77.30.33:\~/pbh-merger-simulation/results\_local\_cpu/positions\_history.npy .

   * Then, create the video locally:  
     python animation\_creator.py positions\_history.npy  
