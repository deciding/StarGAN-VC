from torch.utils import data
import torch
#import os
import random
import glob
from os.path import join, basename, dirname, exists
#from os.path import dirname, split
import numpy as np
import pickle

# Below is the accent info for the used 10 speakers.
min_length = 256   # Since we slice 256 frames from each utterance when training. also is the sample len
#min_length = 128   # Since we slice 256 frames from each utterance when training. also is the sample len
# Build a dict useful when we want to get one-hot representation of speakers.
vctk_speakers = ['p262', 'p272', 'p229', 'p232', 'p292', 'p293', 'p360', 'p361', 'p248', 'p251']
vctk_spk2idx = dict(zip(vctk_speakers, range(len(vctk_speakers))))
vcc2018_speakers = ['VCC2SF1', 'VCC2SF2', 'VCC2SM1', 'VCC2SM2']
vcc2018_spk2idx = dict(zip(vcc2018_speakers, range(len(vcc2018_speakers))))

# [B] -> [B, C]
def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    From Keras np_utils
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class MyDataset(data.Dataset):
    """Dataset for MCEP features and speaker labels."""
    def __init__(self, data_dir, version='v2'):
        self.version=version
        if 'vcc2018' in data_dir:
            self.speakers=vcc2018_speakers
            self.spk2idx=vcc2018_spk2idx
        else:
            self.speakers=vctk_speakers
            self.spk2idx=vctk_spk2idx

        mc_pickle_filename='mc_files.pickle'
        mc_files_pickle=join(data_dir, mc_pickle_filename)
        #if False:
        if exists(mc_files_pickle):
            with open(mc_files_pickle, 'rb') as f:
                self.mc_files=pickle.load(f)
        else:
            mc_files = glob.glob(join(data_dir, '*', '*.npy'))
            mc_files = [i for i in mc_files if basename(dirname(i)) in self.speakers]
            self.mc_files = self.rm_too_short_utt(mc_files)
            #import pdb;pdb.set_trace()
            with open(mc_files_pickle, 'wb') as f:
                pickle.dump(self.mc_files, f)
        #random.shuffle(self.mc_files)
        self.num_files = len(self.mc_files)
        print("\t Number of training samples: ", self.num_files)
        #for f in self.mc_files:
        #    mc = np.load(f)
        #    if mc.shape[0] <= min_length:
        #        print(f)
        #        raise RuntimeError(f"The data may be corrupted! We need all MCEP features having more than {min_length} frames!")

    def rm_too_short_utt(self, mc_files, min_length=min_length):
        new_mc_files = []
        for mcfile in mc_files:
            mc = np.load(mcfile)
            if mc.shape[0] > min_length:
                new_mc_files.append(mcfile)
            else:
                print("%s is eliminated, since it is too short" % mcfile)
        return new_mc_files

    def sample_seg(self, feat, sample_len=min_length):
        assert feat.shape[0] - sample_len >= 0
        s = np.random.randint(0, feat.shape[0] - sample_len + 1)
        return feat[s:s+sample_len, :]

    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        filename = self.mc_files[index]
        spk = basename(dirname(filename))
        spk_idx = self.spk2idx[spk]
        mc = np.load(filename)
        mc = self.sample_seg(mc)
        mc = np.transpose(mc, (1, 0))  # (T, D) -> (D, T), since pytorch need feature having shape
        # to one-hot [batch]
        spk_cat = np.squeeze(to_categorical([spk_idx], num_classes=len(self.speakers)))

        if self.version=='v2':
            ## Below is the version where no need mcep of target
            #label_trg=random.randint(0, len(self.speakers)-1-1)
            #if label_trg>=spk_idx:
            #    label_trg+=1
            # Below is the version where trg mcep is needed
            label_trg=spk_idx
            while label_trg==spk_idx:
                trg_idx=random.randint(0, len(self.mc_files)-1)
                trg_filename = self.mc_files[trg_idx]
                trg_spk = basename(dirname(trg_filename))
                label_trg = self.spk2idx[trg_spk]
            trg_mc = np.load(trg_filename)
            trg_mc = self.sample_seg(trg_mc)
            trg_mc = np.transpose(trg_mc, (1, 0))  # (T, D) -> (D, T)

            trg_cat = np.squeeze(to_categorical([label_trg], num_classes=len(self.speakers)))
            duo_domain=spk_idx*4+label_trg
            duo_cat=np.squeeze(to_categorical([duo_domain], len(self.speakers)*len(self.speakers)))
            duo_domain_rev=label_trg*4+spk_idx
            duo_cat_rev=np.squeeze(to_categorical([duo_domain_rev], len(self.speakers)*len(self.speakers)))
            duo_domain_id=spk_idx*4+spk_idx
            duo_cat_id=np.squeeze(to_categorical([duo_domain_id], len(self.speakers)*len(self.speakers)))

            # source speech, speaker id, speaker id one hot
            return torch.FloatTensor(mc), \
                    torch.LongTensor([spk_idx]).squeeze_(), torch.FloatTensor(spk_cat), \
                    torch.LongTensor([label_trg]).squeeze_(), torch.FloatTensor(trg_cat), \
                    torch.LongTensor([duo_domain]).squeeze_(), torch.FloatTensor(duo_cat), \
                    torch.LongTensor([duo_domain_rev]).squeeze_(), torch.FloatTensor(duo_cat_rev), \
                    torch.LongTensor([duo_domain_id]).squeeze_(), torch.FloatTensor(duo_cat_id), \
                    torch.FloatTensor(trg_mc)
        else:
            # source speech, speaker id, speaker id one hot
            return torch.FloatTensor(mc), \
                    torch.LongTensor([spk_idx]).squeeze_(), torch.FloatTensor(spk_cat)

class TestDataset(object):
    """Dataset for testing."""
    def __init__(self, data_dir, wav_dir, src_spk='VCC2SM1', trg_spk='VCC2SF1', version='v2'):
        self.version=version
        if 'vcc2018' in data_dir:
            self.speakers=vcc2018_speakers
            self.spk2idx=vcc2018_spk2idx
        else:
            self.speakers=vctk_speakers
            self.spk2idx=vctk_spk2idx

        self.src_spk = src_spk
        self.trg_spk = trg_spk
        self.mc_files = sorted(glob.glob(join(data_dir, self.src_spk, '*.npy')))

        self.src_spk_stats = np.load(join(data_dir.replace('test', 'train'), self.src_spk, '{}_stats.npz'.format(src_spk)))
        self.trg_spk_stats = np.load(join(data_dir.replace('test', 'train'), self.trg_spk, '{}_stats.npz'.format(trg_spk)))

        #stats, src_wav_dir, spk_c_trg
        self.logf0s_mean_src = self.src_spk_stats['log_f0s_mean']
        self.logf0s_std_src = self.src_spk_stats['log_f0s_std']
        self.logf0s_mean_trg = self.trg_spk_stats['log_f0s_mean']
        self.logf0s_std_trg = self.trg_spk_stats['log_f0s_std']
        self.mcep_mean_src = self.src_spk_stats['coded_sps_mean']
        self.mcep_std_src = self.src_spk_stats['coded_sps_std']
        self.mcep_mean_trg = self.trg_spk_stats['coded_sps_mean']
        self.mcep_std_trg = self.trg_spk_stats['coded_sps_std']
        self.src_wav_dir = f'{wav_dir}/{src_spk}'
        self.spk_idx = self.spk2idx[trg_spk]
        if self.version=='v2':
            num_spk=len(self.speakers)
            src_spk_idx = self.spk2idx[src_spk]
            spk_cat = to_categorical([num_spk*src_spk_idx+self.spk_idx], num_classes=num_spk*num_spk)
        else:
            spk_cat = to_categorical([self.spk_idx], num_classes=len(self.speakers))
        self.spk_c_trg = spk_cat

    def get_batch_test_data(self, batch_size=8):
        batch_data = []
        for i in range(batch_size):
            mcfile = self.mc_files[i]
            #filename = basename(mcfile).split('-')[-1]# why split
            filename=basename(mcfile)
            wavfile_path = join(self.src_wav_dir, filename.replace('npy', 'wav'))
            batch_data.append(wavfile_path)
        # return wavs
        return batch_data

def get_loader(data_dir, batch_size=32, mode='train', num_workers=1, version='v2'):
    dataset = MyDataset(data_dir, version)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers,
                                  drop_last=True)
    return data_loader


if __name__ == '__main__':
    loader = get_loader('./data/mc/train')
    data_iter = iter(loader)
    for i in range(10):
        mc, spk_idx, spk_acc_cat = next(data_iter)
        print('-'*50)
        print(mc.size())
        print(spk_idx.size())
        print(spk_acc_cat.size())
        print(spk_idx.squeeze_())
        print(spk_acc_cat)
        print('-'*50)

