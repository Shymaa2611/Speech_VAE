import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from glob import glob

def extract_mfcc(audio_path, max_pad_len=100):
    audio, sr = librosa.load(audio_path, mono=True)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs

def load_data(data_dir):
    mfccs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".wav"):
            audio_path = os.path.join(data_dir, filename)
            mfcc = extract_mfcc(audio_path)
            mfccs.append(mfcc)
    return np.array(mfccs)


class UnsupervisedSpeechDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.audio_paths = glob(os.path.join(root_dir, '*.wav'))

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        mfcc = extract_mfcc(audio_path)
        return mfcc

if __name__ == "__main__":
    dataset_dir = 'data'
    dataset = UnsupervisedSpeechDataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in dataloader:
        print(batch.shape)  