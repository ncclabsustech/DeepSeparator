import os
import torch
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from network import DeepSeparator


model_name = 'DeepSeparator'

# choose one sample for visualization
index = 35

test_input = np.load('../data/test_input.npy')
test_input = test_input[index]
test_output = np.load('../data/test_output.npy')
test_output = test_output[index]

test_input = torch.from_numpy(test_input)
test_output = torch.from_numpy(test_output)

test_input = torch.unsqueeze(test_input, 0)
test_output = torch.unsqueeze(test_output, 0)

test_torch_dataset = Data.TensorDataset(test_input)

print("torch.cuda.is_available() = ", torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = DeepSeparator()
model.to(device)  # 移动模型到cuda

if os.path.exists('checkpoint/' + model_name + '.pkl'):
    print('load model')
    model.load_state_dict(torch.load('checkpoint/' + model_name + '.pkl'))


test_input = test_input.float().to(device)

extracted_signal = model(test_input, 0)  # 0 for denoising, 1 for extracting artifact
extracted_artifact = model(test_input, 1)  # 0 for denoising, 1 for extracting artifact


test_input_value = test_input.cpu()
test_input_value = test_input_value.detach().numpy()
test_input_value = test_input_value[0]

test_output_value = test_output.cpu()
test_output_value = test_output_value.detach().numpy()
test_output_value = test_output_value[0]

extracted_signal_value = extracted_signal.cpu()
extracted_signal_value = extracted_signal_value.detach().numpy()
extracted_signal_value = extracted_signal_value[0]

extracted_artifact_value = extracted_artifact.cpu()
extracted_artifact_value = extracted_artifact_value.detach().numpy()
extracted_artifact_value = extracted_artifact_value[0]

l0, = plt.plot(test_input_value)
l1, = plt.plot(extracted_signal_value)
# l2, = plt.plot(extracted_artifact_value)
l3, = plt.plot(test_output_value)


# plt.legend([l0, l1, l2, l3], ['Raw EEG', 'Denoised EEG', 'Extracted Artifact', 'Clean EEG'], loc='upper right')
plt.legend([l0, l1, l3], ['Raw EEG', 'Denoised EEG', 'Clean EEG'], loc='upper right')

plt.show()





