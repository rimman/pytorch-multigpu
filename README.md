# Training of PyramidNet on CIFAR-10

This repository contains the scripts and setup instructions for training PyramidNet model using CIFAR-10 dataset.

## Setup

The following setup describes training this model on M60 GPUs in Azure and in LCOW container.

### 1. Install CUDA

Follow the instructions below to install CUDA runtime and required NVIDIA drivers.

#### Azure VM (NV12)

- Request Azure NV12 VM
- Install CUDA and its pre-requisites using the following commands

    ```bash

    sudo apt install gcc
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
    sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
    sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
    sudo apt-get update
    sudo apt-get -y install cuda

    ```

#### LCOW

Current setup assumes that the LCOW container has two GPUs and CUDA already installed.

### 2. Install PyTorch and other utilities

Ensure that the CUDA setup is completed as described in the previous section. Verify that the CUDA installation is successful by running the `nvidia-smi` command. This should show the available GPUs.

Install Git, VIM, Python, PyTorch and other required packages with the following commands.

```bash
# run the following two commands for LCOW only
apt-get update
apt-get install -y sudo

# run the following commands for both Azure VM and LCOW
sudo apt-get install -y git
sudo apt-get install -y vim
sudo apt-get -y install python3
sudo apt-get -y install python3-pip
python3 -m pip install torch
python3 -m pip install torchvision
python3 -m pip install tensorboardX

```

### 3. Clone the demo repository

```bash

git clone https://github.com/rimman/pytorch-multigpu.git
cd pytorch-multigpu
```

## Demo

### 0. Show GPUs

Show the GPU information through `nvidia-smi` command.

```bash
nvidia-smi

```

### 1. Sanity Check

Show that the PyTorch is installed successfully and can use see the GPUs.

```bash
python3 verify-setup.py

```

### 2. Run Training

There are 3 directories for training. The command below show how to run them.

**For the demo run the distributed data parallel training.**
**While the demo is running in another shell run `watch nvidia-smi` that shows the usage of the GPU.**

- Single GPU training

    ```bash
    cd single_gpu
    python3 train.py --batch_size 100
    # takes about 8 min
    ```

- Multi-GPU training (data parallel strategy)

    ```bash
    cd data_parallel
    python3 train.py --gpu_devices 0 1 --batch_size 100
    # takes about 4 min
    ```

- Multi-GPU training (distributed data parallel strategy)

    ```bash
    cd data_parallel
    python3 train.py --gpu_devices 0 1 --batch_size 100
    # takes about 4 min
    ```
