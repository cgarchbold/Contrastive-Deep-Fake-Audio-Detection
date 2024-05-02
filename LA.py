import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import librosa
import torch



class ASVSpoof2019LA_Dataset(Dataset):
    """ASVSpoof2019 LA dataset."""

    def __init__(self, root_dir, mode = 'train', transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            mode (string): 'train','val','test'
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if mode == 'train':
            self.train = True
            self.flac_dir = os.path.join(root_dir, 'LA','ASVspoof2019_LA_train','flac')
            csv_file = os.path.join(root_dir, 'LA','ASVspoof2019_LA_cm_protocols','ASVspoof2019.LA.cm.train.trn.txt')
        elif mode =='test':
            self.train = False
            self.flac_dir = os.path.join(root_dir, 'LA','ASVspoof2019_LA_eval','flac')
            csv_file = os.path.join(root_dir, 'LA','ASVspoof2019_LA_cm_protocols','ASVspoof2019.LA.cm.eval.trl.txt')
        elif mode == 'val':
            self.train = False
            self.flac_dir = os.path.join(root_dir, 'LA','ASVspoof2019_LA_dev','flac')
            csv_file = os.path.join(root_dir, 'LA','ASVspoof2019_LA_cm_protocols','ASVspoof2019.LA.cm.dev.trl.txt')

        self.annot_frame = pd.read_csv(csv_file,header=None, sep='\\s+')
        if mode == 'train':
            minority_samples = self.annot_frame.iloc[:2580]
            remaining_samples = self.annot_frame.iloc[2580:]
            sampled_remaining_samples = remaining_samples.sample(n=2580)
            self.annot_frame = pd.concat([minority_samples, sampled_remaining_samples])

        if mode == 'val':
            minority_samples = self.annot_frame.iloc[:250]
            remaining_samples = self.annot_frame.iloc[2548:]
            sampled_remaining_samples = remaining_samples.sample(n=250)
            self.annot_frame = pd.concat([minority_samples, sampled_remaining_samples])

        self.root_dir = root_dir
        
        self.transform = transform

    def __len__(self):
        return len(self.annot_frame)

    def __getitem__(self, idx):
        annot = self.annot_frame.iloc[idx]

        file_name = annot[1]
        file_path = os.path.join(self.flac_dir, file_name+'.flac')
        y, sr = librosa.load(file_path)

        y = crop_audio(y)

        #if self.transform:
        #    y = self.transform(y)

        if annot[4] == 'bonafide':
            label = 0.0
        else:
            label = 1.0

        label = torch.tensor(label).float()

        return y, sr, label
    

def crop_audio(audio_signal, crop_length=40000):
    # Calculate the amount of padding needed
    padding = max(0, crop_length - len(audio_signal))

    # Pad the audio signal if necessary
    padded_signal = np.pad(audio_signal, (0, padding), mode='constant', constant_values=0)

    # Calculate the maximum valid starting index after padding
    max_start_index = len(padded_signal) - crop_length

    # Generate a random starting index within the valid range
    start_index = np.random.randint(0, max_start_index + 1)

    # Crop the padded audio signal
    cropped_signal = padded_signal[start_index:start_index + crop_length]

    return cropped_signal