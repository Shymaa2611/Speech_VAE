import os
import torch
import torch.nn.functional as F
from PIL import Image
import librosa
from model import VAES

import torch.nn.functional as F

import torch.nn.functional as F

import torch.nn.functional as F

def loss_criterion(recon_x, x, logvar, mu):
    recon_x = F.interpolate(recon_x, size=x.size()[2])  # Interpolate to match input size
    recon_x = recon_x.view(x.size())
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def save_checkpoint(model, optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = VAES(input_size=checkpoint['model_state_dict']['fc1.weight'].size(1),
                 hidden_size=checkpoint['model_state_dict']['fc1.weight'].size(0),
                 latent_size=checkpoint['model_state_dict']['mu.weight'].size(0))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

def save_audio(output_tensor, sample_rate, file_path):
    output_tensor = output_tensor.detach().cpu().numpy()
    waveform = librosa.feature.inverse.mfcc_to_audio(output_tensor)
    librosa.output.write_wav(file_path, waveform, sample_rate)

