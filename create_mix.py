
import os, glob, csv
from os.path import join, exists
import argparse
import random
import torch
import torchaudio
from vis_sound import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_folder', type=str, required=True)
    parser.add_argument('--noise_folder', type=str, required=True)
    parser.add_argument('--output_clean_folder', type=str, default='')
    parser.add_argument('--output_noise_folder', type=str, default='')
    parser.add_argument('--output_noisy_folder', type=str, default='')
    parser.add_argument('--output_dataset_csv', type=str, default='')
    parser.add_argument('--output_log_txt', type=str, default='')
    parser.add_argument('--snr', type=float, default='', required=True)
    args = parser.parse_args()
    return args

def cal_rms(amp):
    return torch.sqrt(torch.mean(torch.square(amp), dim=-1))

def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a)
    return noise_rms

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("current device:", device)

    args = get_args()

    csv_file = open(args.output_dataset_csv, 'w', newline='')
    csv_writer = csv.writer(csv_file)

    log_file = open(args.output_log_txt, 'w', newline='')

    noise_folder = args.noise_folder
    os.chdir(noise_folder)
    noise_files = glob.glob("*.WAV")

    clean_folder = args.clean_folder
    clean_sub_folders = os.listdir(clean_folder)
    total_file_count = 0

    log_file.write(f"Creating files is started.\n")
    log_file.write(f"clean sub-folder: {clean_sub_folders}\n")
    print(f"Creating files is started.")
    print(f"clean sub-folder: {clean_sub_folders}")

    for noise_file in noise_files:
        
        noise_amp, noise_sample_rate = torchaudio.load(join(noise_folder, noise_file))
        noise_amp = torch.repeat_interleave(noise_amp, 2, dim=1)
        noise_len = noise_amp.size(1)
        noise_file_count = 0

        log_file.write(f"   noise {noise_file} file is opened.\n")
        print(f"    noise {noise_file} file is opened.")

        for clean_sub_folder in clean_sub_folders:
            
            log_file.write(f"       clean sub-folder {clean_sub_folder} is opened.\n")
            print(f"        clean sub-folder {clean_sub_folder} is opened.")

            clean_sub_sub_folders = os.listdir(join(clean_folder, clean_sub_folder))
            clean_file_count = 0

            for clean_sub_sub_folder in clean_sub_sub_folders:
                
                clean_cur_folder = join(clean_folder, clean_sub_folder)
                clean_cur_folder = join(clean_cur_folder, clean_sub_sub_folder)
                os.chdir(clean_cur_folder)

                clean_files = glob.glob("*.WAV")
            
                for clean_file in clean_files:
                
                    clean_amp, clean_sample_rate = torchaudio.load(join(clean_cur_folder, clean_file))
                    clean_len = clean_amp.size()[1]

                    start = random.randint(0, noise_len-clean_len)
                    mag_factor = (0.9 / torch.max(clean_amp))
                    clean_amp *= mag_factor
                    clean_rms = cal_rms(clean_amp)

                    split_noise_amp = noise_amp[:, start: start + clean_len]
                    noise_rms = cal_rms(split_noise_amp)
                    adjusted_noise_rms = cal_adjusted_rms(clean_rms, args.snr)
                    
                    adjusted_noise_amp = split_noise_amp * (adjusted_noise_rms / noise_rms)
                    noisy_amp = (clean_amp + adjusted_noise_amp)

                    noisy_name = clean_sub_folder + "_" + clean_sub_sub_folder + "_" + clean_file.split(".")[0] + "_" + noise_file.split(".")[0] + ".WAV"
                    clean_name = clean_sub_folder + "_" + clean_sub_sub_folder + "_" + clean_file
                    output_noisy_file = join(args.output_noisy_folder , noisy_name)
                    output_clean_file = join(args.output_clean_folder, clean_name)

                    torchaudio.save(
                        output_noisy_file,
                        noisy_amp,
                        clean_sample_rate
                    )

                    clean_exist = exists(output_clean_file)
                    if (not clean_exist):
                        torchaudio.save(
                            output_clean_file,
                            clean_amp,
                            clean_sample_rate
                        )

                    csv_writer.writerow([noisy_name, clean_name])

                    clean_file_count += 1
                    noise_file_count += 1
                    total_file_count += 1

            print(f"        clean {clean_sub_folder} folders are done. {str(clean_file_count)} files are created.")
            log_file.write(f"       clean {clean_sub_folder} folders are done. {str(clean_file_count)} files are created.\n")

        print(f"    noise {noise_file} folders are done. {str(noise_file_count)} files are created.")
        log_file.write(f"   noise {noise_file} folders are done. {str(noise_file_count)} files are created.\n")

    print(f"Creating files is done. {str(total_file_count)} files are created.")
    log_file.write(f"Creating files is done. {str(total_file_count)} files are created.\n")

    csv_file.close()
    log_file.close()

    """
    print_stats(clean_amp, sample_rate=clean_sample_rate)
    plot_waveform(clean_amp, clean_sample_rate)
    plot_specgram(clean_amp, clean_sample_rate)
    play_audio(clean_amp, clean_sample_rate)

    print_stats(adjusted_noise_amp, sample_rate=clean_sample_rate)
    plot_waveform(adjusted_noise_amp, clean_sample_rate)
    plot_specgram(adjusted_noise_amp, clean_sample_rate)
    play_audio(adjusted_noise_amp, clean_sample_rate)

    print_stats(noisy_amp, sample_rate=clean_sample_rate)
    plot_waveform(noisy_amp, clean_sample_rate)
    plot_specgram(noisy_amp, clean_sample_rate)
    play_audio(noisy_amp, clean_sample_rate)

    print("hi")
    """
