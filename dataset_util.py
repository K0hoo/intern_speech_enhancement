
from os.path import join
from torch.utils.data import random_split, DataLoader
from torchaudio.transforms import Spectrogram, InverseSpectrogram

from dataset import *

"""
!!!The data should be get by 'get function'.!!!
"""

sampling_rate = 16000
dataset_dict = {
    'sound': SoundDataset,
    'mag': MagnitudeDataset,
    'mag_log': MagnitudeLogDataset,
    'complex': ComplexDataset
}

def get_train_dataset(root_folder='\.', transform=None, validation_ratio=5, batch_size=32, num_workers=1):
    
    mag_angle, logarithm, stft = transform['mag_angle'], transform['logarithm'], transform['stft']

    assert(mag_angle or not logarithm)

    if not stft:
        dataset = dataset_dict['stft']
    elif mag_angle and not logarithm:
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
    
    mag_angle, logarithm, stft = transform['mag_angle'], transform['logarithm'], transform['stft']

    assert(mag_angle or not logarithm)

    if not stft:
        dataset = dataset_dict['sound']
    elif mag_angle and not logarithm:
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
        stft = torch.complex(stft_real, stft_img)

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
