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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        # Compute the loss and its gradients
        loss = loss_fn(outputs.squeeze(1), labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        #if i % 10 == 9:
        last_loss = running_loss #/ 10 # loss per batch
        print('EPOCH: {} Batch [{}/{}]  loss: {}'.format(epoch_index+1, i + 1,len(train_loader), last_loss))
        running_loss = 0.

    return last_loss


def train():
    print("Initiating Training...")

    # Load train data loaders
    train_dataset = ASVSpoof2019LA_Dataset(root_dir=config['root_dir'], mode = 'train')
    val_dataset = ASVSpoof2019LA_Dataset(root_dir=config['root_dir'], mode = 'val')

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)

    # Load Model
    model = AudioModel(2048,1).to(device)
    print(summary(model, input_size=(config['batch_size'],40000)))
    if config['contrastive_pretrained']:
        model.load_state_dict(torch.load('./best_contr.pth'))


    # Define Loss
    loss_fn =nn.BCELoss().to(device)

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
                vloss = loss_fn(voutputs.squeeze(1), vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            print('Found best, save it!')
            #model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            model_path = './best_.pth'
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

if __name__ == '__main__':
    train()