from nnAudio import features
import torch
import torch.nn as nn

class AudioModel(torch.nn.Module):
    def __init__(self, n_fft, output_dim, contrastive=False):
        super().__init__()
        self.epsilon=1e-10
        # Getting Mel Spectrogram on the fly
        self.spec_layer = features.STFT(n_fft=n_fft, freq_bins=None,
                                           window='hann',
                                           freq_scale='no', center=True,
                                           pad_mode='reflect', fmin=50,
                                           fmax=6000, sr=22050, trainable=True,
                                           output_format='Magnitude')
        self.n_bins = n_fft//2

        # Creating CNN Layers
        self.CNN_freq_kernel_size=(128,1)
        self.CNN_freq_kernel_stride=(2,1)
        k_out = 128
        k2_out = 256
        self.CNN_freq = nn.Conv2d(1,k_out,
                                kernel_size=self.CNN_freq_kernel_size,stride=self.CNN_freq_kernel_stride)
        self.CNN_time = nn.Conv2d(k_out,k2_out,
                                kernel_size=(1,3),stride=(1,1))

        self.region_v = 1 + (self.n_bins-self.CNN_freq_kernel_size[0])//self.CNN_freq_kernel_stride[0]
        self.linear_1 = torch.nn.Linear(k2_out*self.region_v, 512, bias=True)

        self.linear_2 = torch.nn.Linear(512, 256, bias=True)
        self.linear_3 = torch.nn.Linear(256, 128, bias=True)
        self.linear_4 = torch.nn.Linear(128, 64, bias=True)
        self.linear_5 = torch.nn.Linear(64, 1, bias=False)

        self.contrastive = contrastive


    def forward(self,x):
        z = self.spec_layer(x)
        z = torch.log(z+self.epsilon)
        z2 = torch.relu(self.CNN_freq(z.unsqueeze(1)))
        z3 = torch.relu(self.CNN_time(z2)).mean(-1)
        y = self.linear_1(torch.relu(torch.flatten(z3,1)))
        if self.contrastive == False:
            y = self.linear_2(torch.relu(y))
            y = self.linear_3(torch.relu(y))
            y = self.linear_4(torch.relu(y))
            y = self.linear_5(torch.relu(y))
        return torch.sigmoid(y)