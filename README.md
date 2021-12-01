# DeepSeparator


## Introduction

Electroencephalogram (EEG) recordings are often contaminated with artifacts. Various methods have been developed to eliminate or weaken the influence of artifacts. However, most of them rely on prior experience for analysis. Here, we propose an deep learning framework to separate neural signal and artifacts in the embedding space and reconstruct the denoised signal, which is called DeepSeparator. DeepSeparator employs an encoder to extract and amplify the features in the raw EEG, a module called decomposer to extract the trend, detect and suppress artifact and a decoder to reconstruct the denoised signal. Besides, DeepSeparator can extract the artifact, which largely increases the model interpretability. The proposed method is tested with a semi-synthetic EEG dataset and a real task-related EEG dataset, suggesting that DeepSeparator outperforms the conventional models in both EOG and EMG artifact removal. DeepSeparator can be extended to multi-channel EEG and data with any arbitrary length. It may motivate future developments and application of deep learning-based EEG denoising.


The goal of the repository is to provide an implementation of DeepSeparator and replicate the experiments in the paper.


 ## Getting Started

### Setup Enviroment


* [PyTorch](http://pytorch.org/) version = 1.9.0
* MNE = 0.22.1
* Python version = 3.6



### Model Training

1. data/generate_data.py for data generation

2. code/train.py for model training

3. code/predict.py for checking the EEG artifact removal performance



