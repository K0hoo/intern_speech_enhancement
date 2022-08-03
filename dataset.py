
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
        stft_real, stft_img = stft.real, stft.imag
        stft = torch.cat((stft_real, stft_img), dim=1)
        return stft
        

def cal_istft(stft=None, stft_mag=None, stft_angle=None, mag_angle=False):
    
    if mag_angle:
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


def save_result(log_file, root_path, file_name, esti_amp, noisy_amp, clean_amp):
    
    try:
        result_file = join(join(root_path, "result"), file_name)
        noisy_file = join(join(root_path, "noisy"), file_name)
        clean_file = join(join(root_path, "clean"), file_name)

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
class CustomAudioDataset(Dataset):
    def __init__(self, root='./', train=True, seen=True, transform=None):
        super().__init__()
        self.train = train
        self.seen = seen
        self.logarithm = transform['logarithm']
        self.mag_angle = transform['mag_angle']
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
        self.file_name = pd.read_csv(annotations_file, names=["noisy_file", "clean_file", "noise_file"])
        self.noisy_dir = join(data_folder, "noisy")
        self.clean_dir = join(data_folder, "clean")
        self.noise_dir = join(data_folder, "noise")

    def __len__(self):
        return len(self.file_name)

    # The return data is consisted of three data; noisy, target, clean
    def __getitem__(self, index):
        
        noisy_file = join(self.noisy_dir, self.file_name.iloc[index, 0])
        clean_file = join(self.clean_dir, self.file_name.iloc[index, 1])
        noise_file = join(self.noise_dir, self.file_name.iloc[index, 2])

        noisy_amp, _ = torchaudio.load(noisy_file)
        clean_amp, _ = torchaudio.load(clean_file)
        noise_amp, _ = torchaudio.load(noise_file)

        # When logarithm is true, there is no need to adjust the length of data. They are already 2 seconds.
        # When logarithm is false, the data is adjusted to 4 seconds. Zero-padding is used if it is needed.
        if self.train and not self.logarithm:
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

        # The return data is consisted of three data; noisy, clean, and noise.
        # when mag_angle is true, each of them is also consisted of magitude and angle.
        # when mag_angle is false, real and img of STFT form is concatenated.
        if self.mag_angle:
            noisy_mag, noisy_angle = cal_stft(noisy_amp, mag_angle=self.mag_angle)
            clean_mag, clean_angle = cal_stft(clean_amp, mag_angle=self.mag_angle)
            noise_mag, noise_angle = cal_stft(noise_amp, mag_angle=self.mag_angle)
            sample = (
                {"mag": noisy_mag, "angle": noisy_angle}, 
                {"mag": clean_mag, "angle": clean_angle},
                {"mag": noise_mag, "angle": noise_angle}
            )
        else:
            noisy = cal_stft(noisy_amp, mag_angle=self.mag_angle)
            clean = cal_stft(clean_amp, mag_angle=self.mag_angle)
            noise = cal_stft(noise_amp, mag_angle=self.mag_angle)
            sample = (noisy, clean, noise)
        
        return sample


def get_train_dataset(root_folder='\.', transform=None, validation_ratio=5, batch_size=32, num_workers=1):
    
    train_dataset = CustomAudioDataset(root=root_folder, transform=transform)

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


def get_test_dataset(root_folder='\.', transform=None, num_workers=1):
    
    seen_test_dataset = CustomAudioDataset(root=root_folder, train=False, transform=transform)
    unseen_test_dataset = CustomAudioDataset(root=root_folder, train=False, seen=False, transform=transform)

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