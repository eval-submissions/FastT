# FastT
## Abstract 
FastT is a software module that automatically produces deployment strategies for different DNNs in heterogeneous environments. To successfully launch FastT, each server used in experiments should be equipped with at least one GPU. 
The GNN in FastT is implemented using Tensorflow1.14 which is modified to support customized execution orders. 
The detailed software and hardware requirements are introduced in following sections. We will also introduce the dependencies needed and the procedures to install FastT, and the detailed steps to conduct corresponding experiments.

## Hardware dependency
We deploy FastT-boosted TensorFlow framework in 2 physical machines: each equipped with 8 NVIDIA 16GBTesla V100 GPUs, two 10-core Intel Xeon processor E5-2630v4 CPUs and one 100GbE Mellanox RDMA card; The machines are connected through a 100Gbps switch.

## Software dependency
Dependency | Version 
--- | --- 
OS  | Ubuntu-16.04   
Linux Kernel | Linux 4.4.0-131-generic x86_64 
GCC | gcc 5.4.0
CUDA-Toolkit |  cuda-10.0
CUDNN | cudnn-7.6.0
NCCL |  nccl-2.6.4 
Python |  python3.7
TensorFlow |  Modified version based on 1.14

The software dependency is listed in the table above. 
CUDA-Toolkit can be downloaded for https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604. 
CUDNN can be downloaded from https://developer.nvidia.com/cudnn-download-survey. NCCL can be downloaded from https://developer.nvidia.com/nccl/nccl2-download-survey. 
The modified Tensorflow should be downloaded from our this repository.

## Dataset
We conduct experiments based on both synthetic data and real training data sets including cifar10 and SQuAD. These data sets are already included in the repository.

## Models
We conduct experiments using 8 models including VGG-19,ResNet200, InceptionV3, MobileNetV2, NasNet, Transformer,Bert-large and XLNet-large. The implementation of all these models is already included in the github repository.

## Installation
We will detailed introduced the installation steps of FastT in this part.
### Install python environment. 
We recommand to use anaconda: https://docs.anaconda.com/anaconda/install/
After the installation, create an environment named FastT with python3.7

`conda create -n FastT python=3.7`

Then activate the environment:

`conda activate FastT`

### Install CUDA CUDNN and NCCL. 
CUDA-Toolkit can be downloaded for https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604. 

CUDNN can be downloaded from https://developer.nvidia.com/cudnn-download-survey. 

NCCL can be downloaded from https://developer.nvidia.com/nccl/nccl2-download-survey. 

### build modified tensorflow from source
First, clone the FastT project and the submodule of it from repo:

`git clone https://github.com/eval-submissions/FastT.git --recursive`

Then step into tensorflow folder:

`cd tensorflow`

Then install bazel to build tensorflow:

`conda install bazel=0.24`

Then configure the build:

`./configure`

The following shows a sample run of ./configure script (your session may differ):

  You have bazel 0.24 installed.

  Please specify the location of python. [Default is /usr/bin/python3]: 


Found possible Python library paths:

/usr/lib/python3/dist-packages

/usr/local/lib/python3.7/dist-packages

Please input the desired Python library path to use.  Default is [/usr/lib/python3/dist-packages]

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: 

No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with ROCm support? [y/N]: 

No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: Y

CUDA support will be enabled for TensorFlow.

Do you wish to build TensorFlow with TensorRT support? [y/N]: 

No TensorRT support will be enabled for TensorFlow.

Found CUDA 10.1 in:

/usr/local/cuda-10.1/targets/x86_64-linux/lib

/usr/local/cuda-10.1/targets/x86_64-linux/include

Found cuDNN 7 in:

/usr/lib/x86_64-linux-gnu

/usr/include



Please specify a list of comma-separated CUDA compute capabilities you want to build with.


You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus Each capability can be specified as "x.y" or "compute_xy" to include both virtual 
and binary GPU code, or as "sm_xy" to only include the binary code.

Please note that each additional compute capability significantly increases your build time and binary size, and that TensorFlow only supports compute capabilities >= 3.5 [Default is: 3.5,7.0]: 6.1


Do you want to use clang as CUDA compiler? [y/N]: 

nvcc will be used as CUDA compiler.

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: 


Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native -Wno-sign-compare]: 


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: 

Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.

  --config=mkl            # Build with MKL support.

  --config=monolithic     # Config for mostly static monolithic build.

  --config=ngraph         # Build with Intel nGraph support.

  --config=numa           # Build with NUMA support.

  --config=dynamic_kernels    # (Experimental) Build kernels into separate shared objects.

  --config=v2             # Build TensorFlow 2.x instead of 1.x.

Preconfigured Bazel build configs to DISABLE default on features:

  --config=noaws          # Disable AWS S3 filesystem support.

  --config=nogcp          # Disable GCP support.

  --config=nohdfs         # Disable HDFS support.

  --config=nonccl         # Disable NVIDIA NCCL support.

Configuration finished

After Configuration, run the build script in the folder. Before running it, please modify the file accordingly to specify a path to store the whl file. After that you can run:

`sh build.sh`

After the execution of the script. The modified tensorflow should be successfully installed.

