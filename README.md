# DSAN (Dynamic Switch-Attention Network)

[1] Haoxing Lin, Rufan Bai, Weijia Jia, Xinyu Yang, Yongjian You.
Preserving Dynamic Attention for Long-Term Spatial-Temporal Prediction. KDD 2020.
Arxiv link: https://arxiv.org/abs/2006.08849

**Quick install dependencies: ```pip install -r requirements.txt```**

## 1. About DSAN

Dynamic Switch-Attention Network is designed to achieve effective long-term spatial-temporal prediction
by filtering our spatial noise and alleviating long-term error propagation. It relies on attention mechanism instead of
CNN or RNN to measure the spatial-temporal correlation. You are welcomed to check the technical details and experimental 
results in our paper.

Model architecture:

<p align="center">
<img src="./figs/fig_arch.png" width="800" />
</p>

## 2. Environments

Prerequisites:

 - Tensorflow & Tensorflow-GPU: 2.2.0
 - CUDA 10.1
 - CUDNN 7.6.5


Docker is strongly recommended:

```docker pull tensorflow/tensorflow:2.2.0-gpu```

We tested our model on two different machines:

 1. A duo Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz machine with 256G RAM and 4 RTX 2080 Ti GPUs
 2. NVIDIA DGX-2

We configured the prerequisites on the first machine manually and use the aforementioned docker on NVIDIA DGX-2 without any modification. Both resulted in the same outcome, and the only difference is the training time. If you have problem configuring the environment, pulling the official docker image is recommended.

## 3. Pretrained DSAN Checkpoints

We have provided some pretrained checkpoints in the `checkpoints.zip` file together with the corresponding training 
and testing logs. Checkpoints of other DSAN variants are coming soon.

Checkpoints info (RMSE/MAPE):

|       Model    | Data set | # GPU | Batch Size | 1     |  2    | 3     |4      |  5    |  6    |  7     | 8     | 9     | 10    | 11    | 12        |
|----------      |------    |------ |------      |------ |------ |------ |------ | ------| ------| ------ |------ |------ |------ |------ |  :------: |
|     DSAN-LT    | Taxi-NYC | 1     | 64         |       |       |       |       |       |       |        |       |       |       |       |           |
