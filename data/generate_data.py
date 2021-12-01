from data_input import *

import os

epochs = 1  # training epoch
batch_size = 1000  # training batch size
train_num = 3000  # how many trails for train
test_num = 400  # how many trails for test

# 主要为了增加数据数量，不同的EEG和噪声混合会产生更多的数据，同时也可增强模型鲁棒性，引入不同的噪声仍可还原信号
combin_num = 10  # combin EEG and noise ? times


EEG_all = np.load('EEG_all_epochs.npy')
EOG_all = np.load('EOG_all_epochs.npy')
EMG_all = np.load('EMG_all_epochs.npy')

EOGEEG_train_input, EOGEEG_train_output, EOGEEG_test_input, EOGEEG_test_output, test_std_VALUE = data_prepare(EEG_all, EOG_all, combin_num,
                                                                                  train_num, test_num)

EMGEEG_train_input, EMGEEG_train_output, EMGEEG_test_input, EMGEEG_test_output, test_std_VALUE = data_prepare(EEG_all, EMG_all, combin_num,
                                                                                  train_num, test_num)

train_input = np.vstack((EOGEEG_train_input, EMGEEG_train_input))
train_output = np.vstack((EOGEEG_train_output, EMGEEG_train_output))
test_input = np.vstack((EOGEEG_test_input, EMGEEG_test_input))
test_output = np.vstack((EOGEEG_test_output, EMGEEG_test_output))

np.save('./train_input.npy', train_input)
np.save('./train_output.npy', train_output)
np.save('./test_input.npy', test_input)
np.save('./test_output.npy', test_output)

np.save('./EOG_EEG_test_input.npy', EOGEEG_test_input)
np.save('./EOG_EEG_test_output.npy', EOGEEG_test_output)
np.save('./EMG_EEG_test_input.npy', EMGEEG_test_input)
np.save('./EMG_EEG_test_output.npy', EMGEEG_test_output)


