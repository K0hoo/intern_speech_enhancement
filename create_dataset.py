
import pandas as pd
import torch
import torchaudio
from os.path import join
from torch.utils.data import Dataset

CONST_LENGTH = (1 << 17)

def cal_stft(amp):
    stft = torch.stft(
        amp,
        n_fft=512,
        hop_length=128,
        win_length=512,
    )

    stft = torch.transpose(
        torch.complex(stft[0, :, :, 0], stft[0, :, :, 1]),
        0, 1
    )

    return stft.abs(), stft.angle()

def cal_istft(stft_mag, stft_angle):
    stft_real = stft_mag * torch.cos(stft_angle)
    stft_img = stft_mag * torch.sin(stft_angle)

    stft = torch.view_as_real(
        torch.transpose(
            torch.complex(stft_real, stft_img)[0],
            0, 1
        )
    )

    istft = torch.istft(
        stft,
        n_fft=512,
        hop_length=128,
        win_length=512
    )

    return istft

class CustomAudioDataset(Dataset):
    def __init__(self, root='./', train=True, transform=None, target_transform=None):
        super().__init__()
        self.train = train
        data_folder = join(root, "train") if self.train else join(root, "test")
        annotations_file = join(data_folder, "dataset.csv")
        self.file_name = pd.read_csv(annotations_file, names=["noisy_file", "clean_file"])
        self.noisy_dir = join(data_folder, "mix")
        self.clean_dir = join(data_folder, "clean")
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, index):
        noisy_file = join(self.noisy_dir, self.file_name.iloc[index, 0])
        clean_file = join(self.clean_dir, self.file_name.iloc[index, 1])

        noisy_amp, noisy_sample_rate = torchaudio.load(noisy_file)
        clean_amp, clean_sample_rate = torchaudio.load(clean_file)

        audio_length = noisy_amp.size()[1]
        zeros = torch.zeros((1, CONST_LENGTH - audio_length))

        noisy_amp = torch.cat((noisy_amp, zeros), dim=1)
        clean_amp = torch.cat((clean_amp, zeros), dim=1)

        noisy_mag, noisy_angle = cal_stft(noisy_amp)
        clean_mag, clean_angle = cal_stft(clean_amp)

        sample = (
            {"mag": noisy_mag, "ang": noisy_angle, "len": audio_length}, 
            {"mag": clean_mag, "ang": clean_angle, "len": audio_length}
        )
        
        return sample
