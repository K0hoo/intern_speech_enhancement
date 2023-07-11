
import pandas as pd
import torch
import torchaudio
from os.path import join
from torch.utils.data import Dataset

from dataset_util import cal_stft

CONST_LENGTH = 64000

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


# TODO make more class not separate the case in one class
# magnitude - complex / log - normal

class MagnitudeDataset(Dataset):
    def __init__(self, root='./', train=True, seen=True):
        super().__init__()
        self.train = train
        self.seen = seen
        
        data_folder = join(root, 'data')
        data_folder = join(data_folder, 'train') if self.train else join(data_folder, 'test')
        self.data_folder = join(data_folder, 'seen') if self.seen else join(data_folder, 'unseen')

        annotation_file = join(self.data_folder, 'dataset.csv')
        self.file_name = pd.read_csv(annotation_file, names=['noisy_file', 'clean_file', 'noise_file'])
        self.noisy_dir = join(self.data_folder, 'noisy')
        self.clean_dir = join(self.data_folder, 'clean')
        self.noise_dir = join(self.data_folder, 'noise')

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, index):
        
        if self.train:
            noisy_file = join(self.noisy_dir, self.file_name.iloc[index, 0])
            clean_file = join(self.clean_dir, self.file_name.iloc[index, 1])
            noise_file = join(self.noise_dir, self.file_name.iloc[index, 2])
        else:
            noisy_name = self.file_name.iloc[index, 0]
            clean_name = self.file_name.iloc[index, 1]
            noise_name = self.file_name.iloc[index, 2]

            noisy_file = join(self.noisy_dir, noisy_name)
            clean_file = join(self.clean_dir, clean_name)
            noise_file = join(self.noise_dir, noise_name)

        noisy_amp, _ = torchaudio.load(noisy_file)
        clean_amp, _ = torchaudio.load(clean_file)
        noise_amp, _ = torchaudio.load(noise_file)

        if self.train:
            audio_length = noisy_amp.shape[1]
            if audio_length < CONST_LENGTH:
                zeros = torch.zeros((1, CONST_LENGTH - audio_length))
                noisy_amp = torch.cat((noisy_amp, zeros), dim=1)
                clean_amp = torch.cat((clean_amp, zeros), dim=1)
                noise_amp = torch.cat((noise_amp, zeros), dim=1)
            else:
                noisy_amp = noisy_amp[:, :CONST_LENGTH]
                clean_amp = clean_amp[:, :CONST_LENGTH]
                noise_amp = noise_amp[:, :CONST_LENGTH]

        mag_angle = True
        noisy_mag, noisy_angle = cal_stft(noisy_amp, mag_angle=mag_angle)
        clean_mag, clean_angle = cal_stft(clean_amp, mag_angle=mag_angle)
        noise_mag, noise_angle = cal_stft(noise_amp, mag_angle=mag_angle)

        if self.train:
            sample = (
                {"mag": noisy_mag, "angle": noisy_angle}, 
                {"mag": clean_mag, "angle": clean_angle},
                {"mag": noise_mag, "angle": noise_angle}
            )
        else:
            sample = (
                {"mag": noisy_mag, "angle": noisy_angle, "name": noisy_name}, 
                {"mag": clean_mag, "angle": clean_angle, "name": clean_name},
                {"mag": noise_mag, "angle": noise_angle, "name": noise_name}
            )

        return sample


class MagnitudeLogDataset(Dataset):
    def __init__(self, root='./', train=True, seen=True):
        super().__init__()
        self.train = train
        self.seen = seen
        
        self.data_folder = join(root, 'data_log')
        if self.train:
            self.data_folder = join(self.data_folder, 'train')
        else:
            self.data_folder = join(self.data_folder, 'test')
            self.data_folder = join(self.data_folder, 'seen') if self.seen else join(self.data_folder, 'unseen')

        annotation_file = join(self.data_folder, 'dataset.csv')
        self.file_name = pd.read_csv(annotation_file, names=['noisy_file', 'clean_file', 'noise_file'])
        self.noisy_dir = join(self.data_folder, 'noisy')
        self.clean_dir = join(self.data_folder, 'clean')
        self.noise_dir = join(self.data_folder, 'noise')

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, index):
        
        if self.train:
            noisy_file = join(self.noisy_dir, self.file_name.iloc[index, 0])
            clean_file = join(self.clean_dir, self.file_name.iloc[index, 1])
            noise_file = join(self.noise_dir, self.file_name.iloc[index, 2])
        else:
            noisy_name = self.file_name.iloc[index, 0]
            clean_name = self.file_name.iloc[index, 1]
            noise_name = self.file_name.iloc[index, 2]

            noisy_file = join(self.noisy_dir, noisy_name)
            clean_file = join(self.clean_dir, clean_name)
            noise_file = join(self.noise_dir, noise_name)

        noisy_amp, _ = torchaudio.load(noisy_file)
        clean_amp, _ = torchaudio.load(clean_file)
        noise_amp, _ = torchaudio.load(noise_file)

        mag_angle = True
        noisy_mag, noisy_angle = cal_stft(noisy_amp, mag_angle=mag_angle)
        clean_mag, clean_angle = cal_stft(clean_amp, mag_angle=mag_angle)
        noise_mag, noise_angle = cal_stft(noise_amp, mag_angle=mag_angle)

        if self.train:
            sample = (
                {"mag": noisy_mag, "angle": noisy_angle}, 
                {"mag": clean_mag, "angle": clean_angle},
                {"mag": noise_mag, "angle": noise_angle}
            )
        else:
            sample = (
                {"mag": noisy_mag, "angle": noisy_angle, "name": noisy_name}, 
                {"mag": clean_mag, "angle": clean_angle, "name": clean_name},
                {"mag": noise_mag, "angle": noise_angle, "name": noise_name}
            )
        
        return sample


class ComplexDataset(Dataset):
    def __init__(self, root='./', train=True, seen=True):
        self.train = train
        self.seen = seen
        
        self.data_folder = join(root, 'data_log')
        if self.train:
            self.data_folder = join(self.data_folder, 'train')
        else:
            self.data_folder = join(self.data_folder, 'test')
            self.data_folder = join(self.data_folder, 'seen') if self.seen else join(self.data_folder, 'unseen')

        annotation_file = join(self.data_folder, 'dataset.csv')
        self.file_name = pd.read_csv(annotation_file, names=['noisy_file', 'clean_file', 'noise_file'])
        self.noisy_dir = join(self.data_folder, 'noisy')
        self.clean_dir = join(self.data_folder, 'clean')
        self.noise_dir = join(self.data_folder, 'noise')

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, index):
        
        if self.train:
            noisy_file = join(self.noisy_dir, self.file_name.iloc[index, 0])
            clean_file = join(self.clean_dir, self.file_name.iloc[index, 1])
            noise_file = join(self.noise_dir, self.file_name.iloc[index, 2])
        else:
            noisy_name = self.file_name.iloc[index, 0]
            clean_name = self.file_name.iloc[index, 1]
            noise_name = self.file_name.iloc[index, 2]

            noisy_file = join(self.noisy_dir, noisy_name)
            clean_file = join(self.clean_dir, clean_name)
            noise_file = join(self.noise_dir, noise_name)

        noisy_amp, _ = torchaudio.load(noisy_file)
        clean_amp, _ = torchaudio.load(clean_file)
        noise_amp, _ = torchaudio.load(noise_file)        

        if self.train:
            audio_length = noisy_amp.shape[1]
            if audio_length < CONST_LENGTH:
                zeros = torch.zeros((1, CONST_LENGTH - audio_length))
                noisy_amp = torch.cat((noisy_amp, zeros), dim=1)
                clean_amp = torch.cat((clean_amp, zeros), dim=1)
                noise_amp = torch.cat((noise_amp, zeros), dim=1)
            else:
                noisy_amp = noisy_amp[:, :CONST_LENGTH]
                clean_amp = clean_amp[:, :CONST_LENGTH]
                noise_amp = noise_amp[:, :CONST_LENGTH]

        mag_angle = False
        noisy = cal_stft(noisy_amp, mag_angle=mag_angle)
        clean = cal_stft(clean_amp, mag_angle=mag_angle)
        noise = cal_stft(noise_amp, mag_angle=mag_angle)

        if self.train:
            sample = (noisy, clean, noise)
        else:
            sample = (
                {"value": noisy, "name": noisy_name},
                {"value": clean, "name": clean_name},
                {"value": noise, "name": noise_name}
            )
        
        return sample


class SoundDataset(Dataset):
    def __init__(self, root='./', train=True, seen=True):
        self.train = train
        self.seen = seen
        
        self.data_folder = join(root, 'data_log')
        if self.train:
            self.data_folder = join(self.data_folder, 'train')
        else:
            self.data_folder = join(self.data_folder, 'test')
            self.data_folder = join(self.data_folder, 'seen') if self.seen else join(self.data_folder, 'unseen')

        annotation_file = join(self.data_folder, 'dataset.csv')
        self.file_name = pd.read_csv(annotation_file, names=['noisy_file', 'clean_file', 'noise_file'])
        self.noisy_dir = join(self.data_folder, 'noisy')
        self.clean_dir = join(self.data_folder, 'clean')
        self.noise_dir = join(self.data_folder, 'noise')

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, index):
        
        if self.train:
            noisy_file = join(self.noisy_dir, self.file_name.iloc[index, 0])
            clean_file = join(self.clean_dir, self.file_name.iloc[index, 1])
            noise_file = join(self.noise_dir, self.file_name.iloc[index, 2])
        else:
            noisy_name = self.file_name.iloc[index, 0]
            clean_name = self.file_name.iloc[index, 1]
            noise_name = self.file_name.iloc[index, 2]

            noisy_file = join(self.noisy_dir, noisy_name)
            clean_file = join(self.clean_dir, clean_name)
            noise_file = join(self.noise_dir, noise_name)

        noisy_amp, _ = torchaudio.load(noisy_file)
        clean_amp, _ = torchaudio.load(clean_file)
        noise_amp, _ = torchaudio.load(noise_file)        

        if self.train:
            audio_length = noisy_amp.shape[1]
            if audio_length < CONST_LENGTH:
                zeros = torch.zeros((1, CONST_LENGTH - audio_length))
                noisy_amp = torch.cat((noisy_amp, zeros), dim=1)
                clean_amp = torch.cat((clean_amp, zeros), dim=1)
                noise_amp = torch.cat((noise_amp, zeros), dim=1)
            else:
                noisy_amp = noisy_amp[:, :CONST_LENGTH]
                clean_amp = clean_amp[:, :CONST_LENGTH]
                noise_amp = noise_amp[:, :CONST_LENGTH]

        if self.train:
            sample = (noisy_amp, clean_amp, noise_amp)
        else:
            sample = (
                {"value": noisy_amp, "name": noisy_name},
                {"value": clean_amp, "name": clean_name},
                {"value": noise_amp, "name": noise_name}
            )
        
        return sample
        