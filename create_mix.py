
from distutils.command.clean import clean
import os, glob, csv
from os.path import join, exists
import numpy as np
import argparse
import random
import torch
import torchaudio
from vis_sound import *

"""
To create noisy data, file system should be set like below.

dataset
  ├─data (data for experience)
  │   ├─train
  │   │   ├─clean
  │   │   ├─noisy
  │   │   └─dataset.csv
  │   └─test
  │       ├─seen
  │       │   ├─clean
  │       │   ├─noisy
  │       │   └─dataset.csv
  │       └─unseen
  │           ├─clean
  │           ├─noisy
  │           └─dataset.csv
  ├─TIMIT (speech)
  │   ├─TRAIN
  │   │   ├─DR1
  │   │   │   ├─FCJF0
  │   │   │   │ ...
  │   │   │   └─MWAR0
  │   │   │ ...
  │   │   │
  │   │   └─DR8
  │   │       ├─FBCG1
  │   │       │ ...
  │   │       └─MTCS0
  │   └─TEST
  │       ├─DR1
  │       │   ├─FAKS0
  │       │   │ ...
  │       │   └─MWBT0
  │       │ ...
  │       │
  │       └─DR8
  │           ├─FCHM1
  │           │ ...
  │           └─MSLB0
  └─NoiseX (noise)
      ├─seen
      └─unseen
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--vis_sound', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--root_dataset_folder', type=str, required=True)
    parser.add_argument('--output_seen_clean_folder', type=str, default='')
    parser.add_argument('--output_unseen_clean_folder', type=str, default='')
    parser.add_argument('--output_seen_noisy_folder', type=str, default='')
    parser.add_argument('--output_unseen_noisy_folder', type=str, default='')
    parser.add_argument('--output_seen_noise_folder', type=str, default='')
    parser.add_argument('--output_unseen_noise_folder', type=str, default='')
    parser.add_argument('--output_seen_dataset_csv', type=str, default='')
    parser.add_argument('--output_unseen_dataset_csv', type=str, default='')
    parser.add_argument('--output_log_txt', type=str, default='')
    parser.add_argument('--snr', type=float, default=10)
    args = parser.parse_args()
    return args


def cal_rms(amp):
    return torch.sqrt(torch.mean(torch.square(amp), dim=-1))


def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a)
    return noise_rms

"""
The table which shows the length of clean sound is printed.
The row means second and col means deci-second.
"""
def vis_clean_sound_length(clean_folder):
    second_lengths = np.zeros(5)
    deci_second_lengths = np.zeros((50))

    clean_sub_folders = os.listdir(clean_folder)
    for clean_sub_folder in clean_sub_folders:
        clean_sub_sub_folders = os.listdir(join(clean_folder, clean_sub_folder))
        for clean_sub_sub_folder in clean_sub_sub_folders:
            clean_cur_folder = join(clean_folder, clean_sub_folder)
            clean_cur_folder = join(clean_cur_folder, clean_sub_sub_folder)
            os.chdir(clean_cur_folder)
            clean_files = glob.glob("*.WAV")
            for clean_file in clean_files:
                clean_amp, sample_rate = torchaudio.load(join(clean_cur_folder, clean_file))
                clean_amp_length = clean_amp.size(1)
                amp_second = clean_amp_length // sample_rate
                amp_deci_second = (clean_amp_length % sample_rate) // (sample_rate // 10)
                try:
                    deci_second_lengths[amp_second * 10 + amp_deci_second] += 1
                    second_lengths[amp_second] += 1
                except IndexError:
                    expand_factor = amp_second - (second_lengths.shape[0] - 1)
                    second_lengths = np.concatenate((second_lengths, np.zeros(expand_factor)), axis=0)
                    deci_second_lengths = np.concatenate((deci_second_lengths, np.zeros(expand_factor * 10)), axis=0)
                    deci_second_lengths[amp_second * 10 + amp_deci_second] += 1
                    second_lengths[amp_second] += 1
    
    deci_second_lengths = np.reshape(deci_second_lengths, (-1, 10))

    print("   deci\t|", end='')
    for i in range(10):
        print(f"{i}\t|", end='')
    print("\nsecond\t|", end='')
    for i in range(10):
        print("\t|", end='')
    print()
    for i in range(12):
        print("--------", end='')
    print()
    max_second_length = second_lengths.shape[0]
    for sl in range(max_second_length):
        print(f"{sl}\t|", end='')
        for dsl in range(10):
            print(f"{deci_second_lengths[sl][dsl]}\t|", end='')
        print(f"{second_lengths[sl]}")


def save_data(name, amp, length, csv_writer, args, sample_rate=16000, b_seen=True):

    clean_name, noisy_name, noise_name = name['clean'], name['noisy'], name['noise']
    clean_amp, noise_amp = amp['clean'], amp['noise']
    clean_len, noise_len = length['clean'], length['noise']

    start = random.randint(0, noise_len - clean_len)
    mag_factor = 0.9 / torch.max(clean_amp)
    clean_amp *= mag_factor
    clean_rms = cal_rms(clean_amp)

    split_noise_amp = noise_amp[:, start: start + clean_len]
    noise_rms = cal_rms(split_noise_amp)
    adjusted_noise_rms = cal_adjusted_rms(clean_rms, args.snr)

    adjusted_noise_amp = split_noise_amp * (adjusted_noise_rms / noise_rms)
    noisy_amp = clean_amp + adjusted_noise_amp

    if b_seen:
        output_noisy_folder = args.output_seen_noisy_folder
        output_clean_folder = args.output_seen_clean_folder
        output_noise_folder = args.output_seen_noise_folder
    else:
        output_noisy_folder = args.output_unseen_noisy_folder
        output_clean_folder = args.output_unseen_clean_folder
        output_noise_folder = args.output_unseen_noise_folder
    
    output_noisy_file = join(output_noisy_folder, noisy_name)
    output_clean_file = join(output_clean_folder, clean_name)
    output_noise_file = join(output_noise_folder, noise_name)

    torchaudio.save(
        output_noisy_file,
        noisy_amp,
        sample_rate
    )
    
    clean_exist = exists(output_clean_file)
    if not clean_exist:
        torchaudio.save(
            output_clean_file,
            clean_amp,
            sample_rate
        )

    noise_exist = exists(output_noise_file)
    if not noise_exist:
        torchaudio.save(
            output_noise_file,
            adjusted_noise_amp,
            sample_rate
        )

    csv_writer.writerow([noisy_name, clean_name, noise_name])


def create_train_data(args):

    root_dataset_folder = args.root_dataset_folder
    clean_folder = join(join(root_dataset_folder, "TIMIT"), "TRAIN")
    noise_folder = join(join(root_dataset_folder, "NoiseX"), "seen")

    log_file = open(args.output_log_txt, 'w', newline='')
    csv_file = open(args.output_seen_dataset_csv, 'w', newline='')
    csv_writer = csv.writer(csv_file)

    os.chdir(noise_folder)
    noise_files = glob.glob("*.wav")
    for noise_file in noise_files:
        noise_amp, _ = torchaudio.load(join(noise_folder, noise_file))
        noise_amp = torch.repeat_interleave(noise_amp, 2, dim=1)
        noise_len = noise_amp.size(1)
        clean_sub_folders = os.listdir(clean_folder)
        noise_idx = 0
        for clean_sub_folder in clean_sub_folders:
            clean_sub_sub_folders = os.listdir(join(clean_folder, clean_sub_folder))
            for clean_sub_sub_folder in clean_sub_sub_folders:
                clean_cur_folder = join(clean_folder, clean_sub_folder)
                clean_cur_folder = join(clean_cur_folder, clean_sub_sub_folder)
                os.chdir(clean_cur_folder)
                clean_files = glob.glob("*.WAV")
                for clean_file in clean_files:
                    clean_amp, clean_sample_rate = torchaudio.load(join(clean_cur_folder, clean_file))
                    clean_len = clean_amp.size(1)
                    noisy_name = '_'.join(
                        (
                            clean_sub_folder, 
                            clean_sub_sub_folder, 
                            clean_file.split('.')[0], 
                            noise_file.split('.')[0]
                        )
                    ) + '.WAV'
                    clean_name = '_'.join(
                        (
                            clean_sub_folder,
                            clean_sub_sub_folder,
                            clean_file
                        )
                    )
                    noise_name = '_'.join(
                        (
                            noise_file.split('.')[0],
                            str(noise_idx)
                        )
                    ) + '.WAV'
                    noise_idx += 1
                    save_data(
                        name={'clean': clean_name, 'noisy': noisy_name, 'noise': noise_name},
                        amp={'clean': clean_amp, 'noise': noise_amp},
                        length={'clean': clean_len, 'noise': noise_len},
                        csv_writer=csv_writer,
                        args=args,
                        sample_rate=clean_sample_rate,
                        b_seen=True
                    )

                print(f"Done. noise: {noise_file}, clean: {clean_sub_folder}/{clean_sub_sub_folder}")
                log_file.write(f"Done. noise: {noise_file}, clean: {clean_sub_folder}/{clean_sub_sub_folder}\n")

    csv_file.close()
    log_file.close()


def create_test_data(args):

    root_dataset_folder = args.root_dataset_folder
    clean_folder = join(join(root_dataset_folder, "TIMIT"), "TRAIN")
    seen_noise_folder = join(join(root_dataset_folder, "NoiseX"), "seen")
    unseen_noise_folder = join(join(root_dataset_folder, "NoiseX"), "unseen")

    log_file = open(args.output_log_txt, 'w', newline='')
    seen_csv_file = open(args.output_seen_dataset_csv, 'w', newline='')
    unseen_csv_file = open(args.output_unseen_dataset_csv, 'w', newline='')

    seen_csv_writer = csv.writer(seen_csv_file)
    unseen_csv_writer = csv.writer(unseen_csv_file)

    lb_seen = [True, False]
    for b_seen in lb_seen:
        noise_folder = seen_noise_folder if b_seen else unseen_noise_folder
        os.chdir(noise_folder)
        noise_files = glob.glob("*.wav")
        for i, noise_file in enumerate(noise_files):
            noise_amp, _ = torchaudio.load(join(noise_folder, noise_file))
            noise_amp =  torch.repeat_interleave(noise_amp, 2, dim=1)
            noise_len = noise_amp.size(1)
            clean_sub_folders = os.listdir(clean_folder)
            noise_idx = 0
            for clean_sub_folder in clean_sub_folders:
                clean_sub_sub_folders = os.listdir(join(clean_folder, clean_sub_folder))
                for clean_sub_sub_folder in clean_sub_sub_folders:
                    clean_cur_folder = join(clean_folder, clean_sub_folder)
                    clean_cur_folder = join(clean_cur_folder, clean_sub_sub_folder)
                    os.chdir(clean_cur_folder)
                    clean_files = glob.glob("*.WAV")
                    for clean_file in clean_files:
                        clean_amp, clean_sample_rate = torchaudio.load(join(clean_cur_folder, clean_file))
                        clean_len = clean_amp.size(1)
                        noisy_name = '_'.join(
                            (
                                clean_sub_folder,
                                clean_sub_sub_folder,
                                clean_file.split('.')[0],
                                noise_file.split('.')[0]
                            )
                        ) + ".WAV"
                        clean_name = '_'.join(
                            (
                                clean_sub_folder,
                                clean_sub_sub_folder,
                                clean_file
                            )
                        )
                        noise_name = '_'.join(
                            (
                                noise_file.split('.')[0],
                                str(noise_idx)
                            )
                        ) + '.WAV'

                        noise_idx += 1

                        save_data(
                            name={'clean': clean_name, 'noisy': noisy_name, 'noise': noise_name},
                            amp={'clean': clean_amp, 'noise': noise_amp},
                            length={'clean': clean_len, 'noise': noise_len},
                            csv_writer= seen_csv_writer if b_seen else unseen_csv_writer,
                            args=args,
                            sample_rate=clean_sample_rate,
                            b_seen=b_seen
                        )
                    
                    if b_seen:
                        print(f"Done. seen noise: {noise_file}, clean: {clean_sub_folder}/{clean_sub_sub_folder}")
                        log_file.write(f"Done. seen noise: {noise_file}, clean: {clean_sub_folder}/{clean_sub_sub_folder}\n")
                    else:
                        print(f"Done. unseen noise: {noise_file}, clean: {clean_sub_folder}/{clean_sub_sub_folder}")
                        log_file.write(f"Done. unseen noise: {noise_file}, clean: {clean_sub_folder}/{clean_sub_sub_folder}\n")

    seen_csv_file.close()
    unseen_csv_file.close()
    log_file.close()


if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("current device:", device)

    args = get_args()

    if args.vis_sound:
        root_dataset_folder = args.root_dataset_folder
        clean_folder = join(join(root_dataset_folder, "TIMIT"), "TRAIN")
        print("TRAIN")
        vis_clean_sound_length(clean_folder)
        clean_folder = join(join(root_dataset_folder, "TIMIT"), "TEST")
        print("TEST")
        vis_clean_sound_length(clean_folder)
    else:
        if args.train:
            create_train_data(args)
        else:
            create_test_data(args)

