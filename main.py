

from os.path import join
import os
import torchaudio
from torch.utils.data import DataLoader
from create_dataset import CustomAudioDataset, cal_istft
import argparse
import torch
import torch.nn as nn

from models import SimpleLSTM

device = "cuda" if torch.cuda.is_available() else "cpu"
print("current device:", device)

num_epochs, batch_size, learning_rate = 100, 32, 0.002

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_folder', type=str, required=True)
    parser.add_argument('--model_parameter', type=str, required=True)
    parser.add_argument('--test_result', type=str, required=True)
    parser.add_argument('--output_train_log', type=str, required=True)
    parser.add_argument('--output_test_log', type=str, required=True)
    args = parser.parse_args()
    return args

args = get_args()

train_dataset = CustomAudioDataset(
    root = args.data_root_folder,
    train=True,
    transform=True,
    target_transform=True
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = CustomAudioDataset(
    root = args.data_root_folder,
    train=False,
    transform=True,
    target_transform=True
)

test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

model = SimpleLSTM().to(device)
model_parameter = os.path.exists(args.model_parameter)
sampling_rate = 16000

if model_parameter:
    model.load_state_dict(torch.load(args.model_parameter))
    print("Model parameter exists in file system.")
else:
    train_log_file = open(args.output_train_log, 'w', newline='')

    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    print("There is no model parameter in file system. The model is going to be trained now.")

    for epoch in range(num_epochs):
        for i, (noisy, clean) in enumerate(train_loader):
            noisy = noisy['mag'].to(device)
            clean = clean['mag'].to(device)

            output = model(noisy).to(device)
        
            loss = criterion(output, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                message = f'Epoch: {epoch + 1}/{num_epochs}, Batch Step: {i + 1}/{train_loader.__len__()}, Loss: {loss.item():.4f}'
                print(message)
                train_log_file.write(message + '\n')

    torch.save(model.state_dict(), args.model_parameter)
    train_log_file.close()

test_creterion = nn.MSELoss()
test_loss = 0

model.eval()
with torch.no_grad():
    total, correct = 0, 0
    test_log_file = open(args.output_test_log, 'w', newline='')
    for i, (noisy, clean) in enumerate(test_loader):

        noisy_mag, noisy_angle = noisy['mag'].to(device), noisy['ang'].to(device)
        clean_mag, clean_angle = clean['mag'].to(device), clean['ang'].to(device)
        noisy_mag, clean_mag = noisy_mag.to(device), clean_mag.to(device)

        audio_length = clean['len']

        output_mag = model(noisy_mag)

        test_loss += test_creterion(output_mag, clean_mag).item()

        result_path = join(args.test_result, "result")
        noisy_path = join(args.test_result, "noisy")
        clean_path = join(args.test_result, "clean")

        if (i + 1) % 100 == 0:

            file_name = "result" + str(i + 1) + ".wav"
            result_file = join(result_path, file_name)
            noisy_file = join(noisy_path, file_name)
            clean_file = join(clean_path, file_name)

            output = cal_istft(output_mag, noisy_angle).to('cpu')
            noisy_amp = cal_istft(noisy_mag, noisy_angle).to('cpu')
            clean_amp = cal_istft(clean_mag, clean_angle).to('cpu')
            mag_factor = (0.9 / torch.max(output))

            output *= mag_factor

            torchaudio.save(
                result_file,
                torch.unsqueeze(output, 0)[:, :audio_length],
                sampling_rate
            )

            torchaudio.save(
                noisy_file,
                torch.unsqueeze(noisy_amp, 0)[:, :audio_length],
                sampling_rate
            )

            torchaudio.save(
                clean_file,
                torch.unsqueeze(clean_amp, 0)[:, :audio_length],
                sampling_rate
            )

            message = f"{file_name} is created with {test_creterion(output_mag, clean_mag).item()} loss"
            print(message)
            test_log_file.write(message + '\n')

    test_loss /= test_loader.__len__()
    message = f"Test Loss: {test_loss}"
    print(message)
    test_log_file.write(message)
    test_log_file.close()
