# DeepSeparator


## Introduction

Electroencephalogram (EEG) recordings are often contaminated with artifacts. Various methods have been developed to eliminate or weaken the influence of artifacts. However, most of them rely on prior experience for analysis. Here, we propose an deep learning framework to separate neural signal and artifacts in the embedding space and reconstruct the denoised signal, which is called DeepSeparator. DeepSeparator employs an encoder to extract and amplify the features in the raw EEG, a module called decomposer to extract the trend, detect and suppress artifact and a decoder to reconstruct the denoised signal. Besides, DeepSeparator can extract the artifact, which largely increases the model interpretability. The proposed method is tested with a semi-synthetic EEG dataset and a real task-related EEG dataset, suggesting that DeepSeparator outperforms the conventional models in both EOG and EMG artifact removal. DeepSeparator can be extended to multi-channel EEG and data with any arbitrary length. It may motivate future developments and application of deep learning-based EEG denoising.

Our main contributions are summarized as follows:

1. **Novel architecture**: DeepSeparator is an end-to-end deep learning framework which does not rely on manually designed prior assumptions and knowledge of artifacts. It can be considered as a nonlinear decomposition and reconstruction of the input, as an extension of linear blind source separation methods. DeepSeparator learns to decompose the clean EEG signal and artifacts in the latent space for single channel EEG, as ICA does for multi-channel EEG denoising；
2. **Strong interpretability**: Compared with other deep learning models, the network design of DeepSeparator fosters its interpretability. Specifically, the encoder is responsible for capturing and amplifying the features in the raw EEG, the decomposer for extracting the trend, detecting and suppressing the artifacts in the embedding space, and the decoder for reconstructing the EEG signal and artifact；
3. **High capacity**: DeepSeparator can deal with various artifacts, such as EOG and EMG. It reliably achieves better performance compared to traditional EEG denoising methods (e.g., adaptive filter, HHT, EEMD-ICA) across multiple SNR levels. The DeepSeparator trained with single-channel, semi-synthetic EEG data can be applied in multi-channel, real EEG data.

The goal of the repository is to provide an implementation of DeepSeparator and replicate the experiments in the paper.


 ## Getting Started

### Setup Enviroment


* [PyTorch](http://pytorch.org/) version = 1.9.0
* MNE = 0.22.1
* Python version = 3.6


### Dataset

EEGdenoiseNet: a benchmark dataset that is suited for training and testing deep learning-based EEG denoising models, as well as for comparing the performance across different models.

The paper of this dataset is publicly available on Journal of Neural Engineering (https://iopscience.iop.org/article/10.1088/1741-2552/ac2bf8).

Due to size limitations, EEG and EMG epochs with a sample rate of 512hz are temporarily placed in the G-node database (https://gin.g-node.org/NCClab/EEGdenoiseNet). 

Single-Channel-EEG-Denoise tool box could be find in Github(https://github.com/ncclabsustech/Single-Channel-EEG-Denoise)

### Model Training

1. data/generate_data.py for data generation

2. code/train.py for model training

3. code/predict.py for checking the EEG artifact removal performance



