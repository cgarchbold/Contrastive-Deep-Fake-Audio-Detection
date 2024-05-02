import torch
import torchvision
from LA import ASVSpoof2019LA_Dataset
from config import config
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from audio_cnn import AudioModel
from torchinfo import summary
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pdist = nn.PairwiseDistance(p=2)

def train_one_epoch(epoch_index, model, train_loader, optimizer, loss_fn):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, sr, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        #contrastive loss
        if labels[0] == labels[1]:
            loss = pdist(outputs[0], outputs[1])
        else:
            loss = torch.max(torch.tensor(0),1 - pdist(outputs[0],outputs[1]))

        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10 # loss per batch
            print('EPOCH: {} Batch [{}/{}]  loss: {}'.format(epoch_index+1, i + 1,len(train_loader), last_loss))
            running_loss = 0.

    return last_loss


def train():
    print("Initiating Training...")

    # Load train data loaders
    train_dataset = ASVSpoof2019LA_Dataset(root_dir=config['root_dir'], mode = 'train')
    val_dataset = ASVSpoof2019LA_Dataset(root_dir=config['root_dir'], mode = 'val')
    
    # For contrastive, batch size must be 2
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)

    # Load Model
    model = AudioModel(2048,1, contrastive=True).to(device)
    print(summary(model, input_size=(2,40000)))
    # Define Loss
    loss_fn =nn.BCELoss().to(device) #BCELoss_class_weighted([5,0.5]).to(device)

    # Define Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Training Loop
    best_vloss = 1_000_000.
    epoch_number = 0
    for epoch in range(config['epochs']):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, model, train_loader, optimizer, loss_fn)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vsr, vlabels = vdata
                vlabels = vlabels.to(device)
                vinputs = vinputs.to(device)

                voutputs = model(vinputs)

                if vlabels[0] == vlabels[1]:
                    vloss = pdist(voutputs[0],voutputs[1])
                else:
                    vloss = torch.max(torch.tensor(0),1 - pdist(voutputs[0],voutputs[1]))

                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            print('Found best, save it!')
            #model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            model_path = './best_contr.pth'
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

if __name__ == '__main__':
    train()