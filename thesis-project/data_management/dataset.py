import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio

class AudioWatermarkDataset(Dataset):
    def __init__(self, audio_dir, watermark_dir, transform=None):
        self.audio_dir = audio_dir
        self.watermark_dir = watermark_dir
        self.audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        self.transform = transform

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        watermark_path = os.path.join(self.watermark_dir, self.audio_files[idx].replace('.wav', '.pt'))

        waveform, _ = torchaudio.load(audio_path)
        watermark = torch.load(watermark_path)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform.squeeze(0), watermark

def load_data(audio_dir, watermark_dir, batch_size=32, split_ratios=(0.7, 0.2, 0.1)):
    dataset = AudioWatermarkDataset(audio_dir, watermark_dir)

    train_size = int(split_ratios[0] * len(dataset))
    valid_size = int(split_ratios[1] * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
