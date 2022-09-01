
import pandas as pd
import torch
import torchaudio
from os.path import join
from torch.utils.data import Dataset, random_split, DataLoader
from torchaudio.transforms import Spectrogram, InverseSpectrogram

CONST_LENGTH = 64000
sampling_rate = 16000

"""
!!!The data should be get by 'get function'.!!!
"""

"""
The STFT of time-domain function is returned.
When transform is true, it returns magnitude and angle form.
When it is not, it returns real and imaginary form.
"""

"""
mag_angle == True return (magnitude, angle)
mag_angle == False return (complex)
"""
def cal_stft(amp, mag_angle=False):

    stft_transform = Spectrogram(
        n_fft=512,
        hop_length=128,
        win_length=512,
        power=None
    )

    stft = torch.transpose(stft_transform(amp)[0], 0, 1)

    if mag_angle:
        return stft.abs(), stft.angle()
    else:
        return stft
        

def cal_istft(stft=None, stft_mag=None, stft_angle=None, mag_angle=False):
    
    if mag_angle:
        stft_real = stft_mag * torch.cos(stft_angle)
        stft_img = stft_mag * torch.sin(stft_angle)

    stft = torch.transpose(
        stft,
        1, 2
    )

    istft_transform = InverseSpectrogram(
        n_fft=512,
        hop_length=128,
        win_length=512
    )

    istft = istft_transform(stft)
    return istft


def save_result(log_file, root_path, file_name, esti_amp, noisy_amp, clean_amp):
    
    try:
        result_file = join(join(root_path, "result"), file_name['result_name'])
        noisy_file = join(join(root_path, "noisy"), file_name['noisy_name'])
        clean_file = join(join(root_path, "clean"), file_name['clean_name'])

        mag_factor = (0.9 / torch.max(esti_amp))
        esti_amp *= mag_factor

        torchaudio.save(
            result_file,
            esti_amp,
            sampling_rate
        )

        torchaudio.save(
            noisy_file,
            noisy_amp,
            sampling_rate
        )

        torchaudio.save(
            clean_file,
            clean_amp,
            sampling_rate
        )

        return True
    except Exception as e:
        test_log_file = open(log_file, 'a', newline='')
        test_log_file.write(f"Exception when creating reuslt file. {e}\n")
        test_log_file.close()
        return False



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


dataset_dict = {
    'mag': MagnitudeDataset,
    'mag_log': MagnitudeLogDataset,
    'complex': ComplexDataset
}


def get_train_dataset(root_folder='\.', transform=None, validation_ratio=5, batch_size=32, num_workers=1):
    
    mag_angle, logarithm = transform['mag_angle'], transform['logarithm']

    assert(mag_angle or not logarithm)

    if mag_angle and not logarithm:
        dataset = dataset_dict['mag']
    elif mag_angle and logarithm:
        dataset = dataset_dict['mag_log']
    elif not mag_angle and not logarithm:
        dataset = dataset_dict['complex']

    dataset = dataset(root=root_folder)

    train_dataset_length = dataset.__len__()

    train_dataset, validation_dataset = random_split(
        dataset,
        [train_dataset_length - train_dataset_length // validation_ratio, train_dataset_length // validation_ratio],
        generator=torch.Generator().manual_seed(1)
    )

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, validation_loader


def get_test_dataset(root_folder='\.', transform=None, num_workers=1):
    
    mag_angle, logarithm = transform['mag_angle'], transform['logarithm']

    assert(mag_angle or not logarithm)

    if mag_angle and not logarithm:
        dataset = dataset_dict['mag']
    elif mag_angle and logarithm:
        dataset = dataset_dict['mag_log']
    elif not mag_angle and not logarithm:
        dataset = dataset_dict['complex']

    seen_test_dataset = dataset(root=root_folder, train=False)
    unseen_test_dataset = dataset(root=root_folder, train=False, seen=False)

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