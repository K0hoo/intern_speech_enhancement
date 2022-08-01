
import pandas as pd
import torch
import torchaudio
from os.path import join
from torch.utils.data import Dataset, random_split, DataLoader
from torchaudio.transforms import Spectrogram, InverseSpectrogram

CONST_LENGTH = 64000

"""
!!!The data should be get by 'get function'.!!!
"""

"""
The STFT of time-domain function is returned.
When transform is true, it returns magnitude and angle form.
When it is not, it returns real and imaginary form.
"""
def cal_stft(amp, transform=False):
    stft_transform = Spectrogram(
        n_fft=512,
        hop_length=128,
        win_length=512,
        power=None
    )

    stft = torch.transpose(stft_transform(amp)[0], 0, 1)

    if transform:
        return stft.abs(), stft.angle()
    else:
        stft_real, stft_img = stft.real, stft.imag
        stft = torch.cat((stft_real, stft_img), dim=1)
        return stft
        

def cal_istft(stft=None, stft_mag=None, stft_angle=None, transform=False):
    if transform:
        stft_real = stft_mag * torch.cos(stft_angle)
        stft_img = stft_mag * torch.sin(stft_angle)
    else:
        stft_real, stft_img = stft[:, :, :257], stft[:, :, 257:]

    stft = torch.transpose(
        torch.complex(stft_real, stft_img),
        1, 2
    )

    istft_transform = InverseSpectrogram(
        n_fft=512,
        hop_length=128,
        win_length=512
    )

    istft = istft_transform(stft)
    return istft


"""
This class work well with the below file system structure.
When transform is true, the dataset is brought out with the mag-angle format.
If it is not, the format might be real-img format.

root_folder
   ├─train
   │   ├─noisy
   │   ├─clean
   │   └─dataset.csv
   └─test
       ├─seen
       │   ├─noisy
       │   ├─clean
       │   └─dataset.csv
       └─unseen
           ├─noisy
           ├─clean
           └─dataset.csv
"""
class CustomAudioDataset(Dataset):
    def __init__(self, root='./', train=True, seen=True, transform=None, logarithm=False):
        super().__init__()
        self.train = train
        self.seen = seen
        self.logarithm = logarithm
        if self.logarithm:
            data_folder = join(root, "data_log")
        else:
            data_folder = join(root, "data")
        if self.train:
            data_folder = join(data_folder, "train")
        else:
            data_folder = join(data_folder, "test")
            if self.seen:
                data_folder = join(data_folder, "seen")
            else:
                data_folder = join(data_folder, "unseen")
        annotations_file = join(data_folder, "dataset.csv")
        self.file_name = pd.read_csv(annotations_file, names=["noisy_file", "clean_file"])
        self.noisy_dir = join(data_folder, "noisy")
        self.clean_dir = join(data_folder, "clean")
        self.transform = transform

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, index):
        noisy_file = join(self.noisy_dir, self.file_name.iloc[index, 0])
        clean_file = join(self.clean_dir, self.file_name.iloc[index, 1])

        noisy_amp, _ = torchaudio.load(noisy_file)
        clean_amp, _ = torchaudio.load(clean_file)

        if self.train and not self.logarithm:
            audio_length = noisy_amp.shape[1]
            if audio_length < CONST_LENGTH:
                zeros = torch.zeros((1, CONST_LENGTH - audio_length))
                noisy_amp = torch.cat((noisy_amp, zeros), dim=1)
                clean_amp = torch.cat((clean_amp, zeros), dim=1)
            else:
                noisy_amp = noisy_amp[:, :CONST_LENGTH]
                clean_amp = clean_amp[:, :CONST_LENGTH]

        if self.transform:
            noisy_mag, noisy_angle = cal_stft(noisy_amp, self.transform)
            clean_mag, clean_angle = cal_stft(clean_amp, self.transform)
            sample = (
                {"mag": noisy_mag, "ang": noisy_angle}, 
                {"mag": clean_mag, "ang": clean_angle}
            )
        else:
            noisy = cal_stft(noisy_amp, self.transform)
            clean = cal_stft(clean_amp, self.transform)
            sample = (noisy, clean)
        
        return sample


def get_train_dataset(root_folder='\.', transform=False, logarithm=False, validation_ratio=5, batch_size=32, num_workers=1):
    train_dataset = CustomAudioDataset(root=root_folder, transform=transform, logarithm=logarithm)

    train_dataset_length = train_dataset.__len__()

    train_dataset, validation_dataset = random_split(
        train_dataset,
        [train_dataset_length - train_dataset_length // validation_ratio, train_dataset_length // validation_ratio],
        generator=torch.Generator().manual_seed(1)
    )

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, validation_loader


def get_test_dataset(root_folder='\.', transform=None, logarithm=False, num_workers=1):
    seen_test_dataset = CustomAudioDataset(root=root_folder, train=False, transform=transform, logarithm=logarithm)
    unseen_test_dataset = CustomAudioDataset(root=root_folder, train=False, seen=False, transform=transform, logarithm=logarithm)

    seen_test_loader = DataLoader(
        dataset=seen_test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    unseen_test_loader = DataLoader(
        dataset=unseen_test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return seen_test_loader, unseen_test_loader