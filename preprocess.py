#import librosa
import numpy as np
import os, sys
import argparse
#import pyworld
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from utils import world_encode_wav, logf0_statistics, coded_sp_statistics, normalize_coded_sp
from tqdm import tqdm
#from collections import defaultdict
#from collections import namedtuple
from sklearn.model_selection import train_test_split
import glob
from os.path import join, basename
import subprocess

# python preprocess.py --sample_rate 16000 --origin_wavpath ../datasets/vctk/VCTK-Corpus/wav48/ --target_wavpath ../datasets/vctk/VCTK-Corpus/wav16/ --mc_dir_train data/mc/train --mc_dir_test data/mc/test

def resample(spk, origin_wavpath, target_wavpath):
    wavfiles = [i for i in os.listdir(join(origin_wavpath, spk)) if i.endswith(".wav")]
    for wav in wavfiles:
        folder_to = join(target_wavpath, spk)
        os.makedirs(folder_to, exist_ok=True)
        wav_to = join(folder_to, wav)
        wav_from = join(origin_wavpath, spk, wav)
        subprocess.call(['sox', wav_from, "-r", "16000", wav_to])
        #subprocess.call(['sox', wav_from, "-r", "24000", wav_to])
    return 0

def resample_to_16k(origin_wavpath, target_wavpath, num_workers=1):
    os.makedirs(target_wavpath, exist_ok=True)
    spk_folders = os.listdir(origin_wavpath)
    spk_folders = [spk_folder for spk_folder in spk_folders if os.path.isdir(os.path.join(origin_wavpath, spk_folder)) and spk_folder!='Transcriptions']
    print(f"> Using {num_workers} workers!")
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    for spk in spk_folders:
        futures.append(executor.submit(partial(resample, spk, origin_wavpath, target_wavpath)))
    result_list = [future.result() for future in tqdm(futures)]
    print(result_list)

def split_data(paths):
    indices = np.arange(len(paths))
    test_size = 0.1 # make sure every speaker have utts in train data
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=1234)
    train_paths = list(np.array(paths)[train_indices])
    test_paths = list(np.array(paths)[test_indices])
    return train_paths, test_paths

def get_spk_world_feats(spk_fold_path, mc_dir_train, mc_dir_test,
        sample_rate=16000,
        spk_fold_path_test=None):
    paths = glob.glob(join(spk_fold_path, '*.wav'))
    spk_name = basename(spk_fold_path)
    output_dir_train=os.path.join(mc_dir_train, spk_name)
    output_dir_test=os.path.join(mc_dir_test, spk_name)
    os.makedirs(output_dir_train, exist_ok=True)
    os.makedirs(output_dir_test, exist_ok=True)
    if spk_fold_path_test is not None:
        train_paths=paths
        test_paths=glob.glob(join(spk_fold_path_test, '*.wav'))
    else:
        train_paths, test_paths = split_data(paths)
    f0s = []
    coded_sps = []
    for wav_file in train_paths:
        f0, _, _, _, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
        f0s.append(f0)
        coded_sps.append(coded_sp)
    log_f0s_mean, log_f0s_std = logf0_statistics(f0s)
    coded_sps_mean, coded_sps_std = coded_sp_statistics(coded_sps)
    np.savez(join(output_dir_train, spk_name+'_stats.npz'),
            log_f0s_mean=log_f0s_mean,
            log_f0s_std=log_f0s_std,
            coded_sps_mean=coded_sps_mean,
            coded_sps_std=coded_sps_std)

    for wav_file in tqdm(train_paths):
        wav_nam = basename(wav_file)
        f0, timeaxis, sp, ap, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
        normed_coded_sp = normalize_coded_sp(coded_sp, coded_sps_mean, coded_sps_std)
        np.save(join(output_dir_train, wav_nam.replace('.wav', '.npy')), normed_coded_sp, allow_pickle=False)

    for wav_file in tqdm(test_paths):
        wav_nam = basename(wav_file)
        f0, timeaxis, sp, ap, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
        normed_coded_sp = normalize_coded_sp(coded_sp, coded_sps_mean, coded_sps_std)
        np.save(join(output_dir_test, wav_nam.replace('.wav', '.npy')), normed_coded_sp, allow_pickle=False)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    sample_rate_default = 16000

    parser.add_argument("--sample_rate", type = int, default = sample_rate_default, help = "Sample rate.")
    parser.add_argument('--dataset', type=str, default='vcc2018', choices=['vctk', 'vcc2018'])
    #parser.add_argument("--origin_wavpath", type = str, default = origin_wavpath_default, help = "The original wav path to resample.")
    #parser.add_argument("--target_wavpath", type = str, default = target_wavpath_default, help = "The original wav path to resample.")
    #parser.add_argument("--mc_dir_train", type = str, default = mc_dir_train_default, help = "The directory to store the training features.")
    #parser.add_argument("--mc_dir_test", type = str, default = mc_dir_test_default, help = "The directory to store the testing features.")
    parser.add_argument("--num_workers", type = int, default = None, help = "The number of cpus to use.")

    argv = parser.parse_args()

    if argv.dataset=='vctk':
        # WE only use 10 speakers listed below for this experiment.
        speaker_used = ['262', '272', '229', '232', '292', '293', '360', '361', '248', '251']
        speaker_used = ['p'+i for i in speaker_used]
    else:
        speaker_used = ['VCC2SF1', 'VCC2SF2', 'VCC2SM1', 'VCC2SM2']

    num_workers = argv.num_workers if argv.num_workers is not None else cpu_count()
    num_workers = min(len(speaker_used), num_workers)

    sample_rate = argv.sample_rate

    if argv.dataset=='vctk':
        origin_wavpath= "./data/VCTK-Corpus/wav48"
        target_wavpath= "./data/VCTK-Corpus/wav16"
        mc_dir_train= './data/mc/train'
        mc_dir_test= './data/mc/test'
        # The original wav in VCTK is 48K, first we want to resample to 16K
        resample_to_16k(origin_wavpath, target_wavpath, num_workers=num_workers)
        #sys.exit(0)

    else:
        origin_wavpath= "./vcc2018/vcc2018_training"
        target_wavpath= "./vcc2018/vcc2018_training_16k"
        origin_wavpath_test = "./vcc2018/vcc2018_evaluation"
        target_wavpath_test = "./vcc2018/vcc2018_evaluation_16k"
        mc_dir_train= './vcc2018/mc_vcc2018/train'
        mc_dir_test= './vcc2018/mc_vcc2018/test'
        # The original wav in VCC2018 is 22.05K, first we want to resample to 16K
        resample_to_16k(origin_wavpath, target_wavpath, num_workers=num_workers)
        resample_to_16k(origin_wavpath_test, target_wavpath_test, num_workers=num_workers)
        #sys.exit(0)



    ## Next we are to extract the acoustic features (MCEPs, lf0) and compute the corresponding stats (means, stds). 
    # Make dirs to contain the MCEPs
    os.makedirs(mc_dir_train, exist_ok=True)
    os.makedirs(mc_dir_test, exist_ok=True)

    print("number of workers: ", num_workers)
    executor = ProcessPoolExecutor(max_workers=num_workers)

    work_dir = target_wavpath
    if argv.dataset=='vcc2018':
        test_dir = target_wavpath_test

    futures = []
    for spk in speaker_used:
        spk_path = os.path.join(work_dir, spk)
        if argv.dataset=='vctk':
            futures.append(executor.submit(partial(get_spk_world_feats, spk_path, mc_dir_train, mc_dir_test, sample_rate)))
        else:
            spk_path_test = os.path.join(test_dir, spk)
            futures.append(executor.submit(partial(get_spk_world_feats, spk_path, mc_dir_train, mc_dir_test, sample_rate, spk_path_test)))
    result_list = [future.result() for future in tqdm(futures)]
    print(result_list)
    sys.exit(0)

