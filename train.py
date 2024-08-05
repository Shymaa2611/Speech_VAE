import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import save_checkpoint, loss_criterion
from dataset import UnsupervisedSpeechDataset
from model import VAES
import librosa
import soundfile as sf

def normalize(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())

def save_audio(output_tensor, sample_rate, file_path):
    output_tensor = output_tensor.detach().cpu().numpy()
    waveform = librosa.feature.inverse.mfcc_to_audio(output_tensor)
    sf.write(file_path, waveform, sample_rate)

def train(dataloader, model, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs in dataloader:
            optimizer.zero_grad()
            inputs = inputs.to(next(model.parameters()).device) 
            inputs = normalize(inputs) 
            recon_x, logvar, mu, z = model(inputs)
            recon_x = normalize(recon_x)  
            loss = loss_criterion(recon_x, inputs, logvar, mu)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            save_audio(recon_x[0], 22050, f'speech/output_audio_epoch_{epoch + 1}.wav')

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}')

def main():
    dataset = UnsupervisedSpeechDataset(root_dir='data')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    input_channels = 13  
    input_length = 100  
    hidden_size = 512
    latent_size = 30
    model = VAES(input_channels, input_length, hidden_size, latent_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(dataloader, model, optimizer, epochs=10)
    save_checkpoint(model, optimizer, 'checkpoint/checkpoint.pt')

if __name__ == "__main__":
    main()
