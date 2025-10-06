# **Primordial Black Hole (PBH) Merger Rate Simulation**

This project contains a set of Python scripts to simulate the merger rate of Primordial Black Hole (PBH) binaries, based on the physics and analytical models presented in the paper **arXiv:2404.08416v2, "Formation of primordial black hole binaries and their merger rates"** by Raidal, Vaskonen, and Veermäe.

The primary goal is to reproduce the theoretical predictions for the "early two-body" merger channel, which is the dominant channel for most PBH abundance scenarios.

There are two main simulation modes:

1. **Monte Carlo Simulation:** A fast, efficient method to calculate the merger rate evolution and compare it directly with the paper's theoretical predictions. This is the recommended approach for quantitative analysis.  
2. **N-Body Visualization:** A simplified, direct N-body simulation designed to produce a visual animation of the spatial evolution and clustering of PBHs.

## **Setup**

Before running any simulations, you must create a dedicated Conda environment with all the necessary packages.

1. **Create and Activate Conda Environment:**  
   \# Create the environment  
   conda create \-n pbh\_sim python=3.10 \-y

   \# Activate the environment (must be done in every new terminal session)  
   conda activate pbh\_sim

2. **Install Required Packages:**  
   \# For core simulation and plotting  
   pip install torch numpy PyYAML matplotlib tqdm

   \# For creating animations (optional)  
   \# You may also need to install ffmpeg via your system's package manager  
   \# e.g., 'sudo apt-get install ffmpeg' or 'conda install \-c conda-forge ffmpeg'

## **Workflow 1: Monte Carlo Simulation (Recommended)**

This workflow is the most direct way to compare your results with the paper's theoretical predictions for the merger rate.

### **Step 1: Run the Monte Carlo Simulation**

This script uses the analytical formulas from the paper to generate a large sample of merging binaries and their coalescence times. It's very fast.

\# Run the simulation using a configuration file  
python monte\_carlo\_mergers.py input\_local\_cpu.yaml

* **Input:** input\_local\_cpu.yaml (or any other config file).  
* **Output:** A JSON file containing the merger history, e.g., results\_local\_cpu/mc\_merger\_history\_fpbh\_1.0e-01.json.

### **Step 2: Plot the Merger Rate vs. Time**

This script takes the merger history data and plots the merger rate R(t) as a function of cosmic time. It overlays the theoretical t^(-34/37) scaling law for direct comparison.

\# Provide the path to the JSON file generated in the previous step  
python plot\_rate\_vs\_time.py results\_local\_cpu/mc\_merger\_history\_fpbh\_1.0e-01.json

* **Input:** The JSON results file from the Monte Carlo simulation.  
* **Output:** A PNG image showing the comparison plot, e.g., results\_local\_cpu/rate\_vs\_time\_comparison\_fpbh\_1.0e-01.png.

## **Workflow 2: N-Body Spatial Visualization**

This workflow is for generating a video to visualize how a population of PBHs clusters under gravity over time.

### **Step 1: Generate the Position Data**

This script runs a simplified N-body simulation and saves the position of every particle at each time step. **Warning: This is computationally intensive.** It's recommended to use a small number of particles (e.g., set number\_density: 10 in your config file).

\# Run the N-body visualizer  
python nbody\_visualizer.py input\_local\_cpu.yaml

* **Input:** A configuration file (preferably with a low particle count).  
* **Output:** A NumPy data file containing the position history, e.g., results\_local\_cpu/positions\_history.npy.

### **Step 2: Create the Animation Video**

This script reads the position data and uses matplotlib to render an MP4 video file.

\# Provide the path to the .npy file from the previous step  
python create\_animation.py results\_local\_cpu/positions\_history.npy

* **Input:** The .npy position data file.  
* **Output:** A video file named pbh\_evolution.mp4 in the root project directory.