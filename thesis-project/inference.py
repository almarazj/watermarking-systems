import torch
from model.invertible_network import InvertibleNN
from data_management.dataset import AudioWatermarkDataset
import torchaudio

def infer(model, audio_path, watermark_path):
    waveform, _ = torchaudio.load(audio_path)
    watermark = torch.load(watermark_path)

    model.eval()
    with torch.no_grad():
        x_pred, m_pred = model(waveform.squeeze(0), watermark)
    
    return x_pred, m_pred

if __name__ == "__main__":
    model = InvertibleNN(input_dim=16000, num_blocks=3)
    model.load_state_dict(torch.load('trained_model.pth'))

    x_pred, m_pred = infer(model, 'dataset/wav_files/example.wav', 'dataset/watermarks/example.pt')
    print(f'Predicted Audio Shape: {x_pred.shape}, Predicted Watermark Shape: {m_pred.shape}')
