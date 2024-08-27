import os
import torch
import torchaudio

def preprocess_waveforms(wav_dir, output_dir):
    window_size = 1000
    hop_length = 400
    window_fn = torch.hamming_window

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for wav_file in os.listdir(wav_dir):
        if wav_file.endswith('.wav'):
            waveform, sample_rate = torchaudio.load(os.path.join(wav_dir, wav_file))
            spectrogram = torchaudio.transforms.Spectrogram(
                n_fft=window_size,
                win_length=window_size,
                hop_length=hop_length,
                window_fn=window_fn
            )(waveform)
            
            # Save spectrogram
            output_file = os.path.join(output_dir, f"{os.path.splitext(wav_file)[0]}.pt")
            torch.save(spectrogram, output_file)

def preprocess_watermark_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.txt'):
            with open(os.path.join(input_dir, file_name), 'r') as f:
                bits = f.read().strip()
                watermark = torch.tensor([int(bit) for bit in bits], dtype=torch.float32)
                torch.save(watermark, os.path.join(output_dir, file_name.replace('.txt', '.pt')))
