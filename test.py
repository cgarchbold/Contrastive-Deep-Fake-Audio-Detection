from config import config
from LA import ASVSpoof2019LA_Dataset
from torch.utils.data import DataLoader
from audio_cnn import AudioModel
import torch
from sklearn import metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(test_loader, model):
    true_labels = []

    for sample in test_loader:
        inputs, sr, labels = sample
        inputs = inputs.to(device)
        labels = labels.to(device)

        #gather outputs
        outputs = model(inputs)

    # compute errors

    # write results to txt

if __name__ == '__main__':
    test()