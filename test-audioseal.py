import numpy as np
import soundfile as sf
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from audioseal import AudioSeal
from pesq import pesq

def load_wav_files(folder_path):
    """
    Loads all WAV files from a folder and returns them as a list.

    Args:
        folder_path (str or Path): The path to the folder containing WAV files.

    Returns:
        list: A list of tuples (file_name, waveform, sample_rate).
    """
    wav_files = list(folder_path.glob('*.wav'))
    audio_data = []
    
    for wav_file in wav_files:
        data, sr = sf.read(wav_file)
        audio_data.append((wav_file.stem, data, sr))
    
    return audio_data

def to_equal_length(original, signal_watermarked):
    """
    Makes two signals of equal length by truncating the longer signal.

    Args:
        original (np.array): Original audio signal.
        signal_watermarked (np.array): Watermarked audio signal.

    Returns:
        tuple: A tuple containing two arrays of equal length.
    """
    if original.shape != signal_watermarked.shape:
        min_length = min(len(original), len(signal_watermarked))
        original = original[:min_length]
        signal_watermarked = signal_watermarked[:min_length]
    return original, signal_watermarked

def signal_noise_ratio(original, signal_watermarked):
    """
    Calculates the signal-to-noise ratio between the original and watermarked signals.

    Args:
        original (np.array): Original audio signal.
        signal_watermarked (np.array): Watermarked audio signal.

    Returns:
        float: The signal-to-noise ratio in dB.
    """
    original, signal_watermarked = to_equal_length(original, signal_watermarked)
    noise_strength = np.sum((original - signal_watermarked) ** 2)
    if noise_strength == 0:
        return np.inf
    signal_strength = np.sum(original ** 2)
    ratio = signal_strength / noise_strength
    return 10 * np.log10(ratio)

def save_results(file_name, snr, ber, pesq_score, output_file):
    """
    Saves the results to a text file.

    Args:
        file_name (str): The name of the original audio file.
        snr (float): Signal-to-noise ratio.
        ber (float): Bit Error Rate.
        output_file (Path): Path to the output file.
    """
    with open(output_file, 'a') as f:
        f.write(f"{file_name} {snr:.2f} {pesq_score} {ber:.2f}\n")

def plot_results(original, watermark_signal, watermarked_signal, sr, filename, results_folder):
    """
    Plots the spectrograms of the original audio, watermark signal, and watermarked audio.

    Args:
        original (np.array): Original audio signal.
        watermark_signal (np.array): The watermark signal.
        watermarked_signal (np.array): Watermarked audio signal.
        sr (int): Sample rate of the audio signals.
        filename (str): Name of the audio file (for the title).
    """
    original_stft = np.abs(librosa.stft(original))
    watermark_stft = np.abs(librosa.stft(watermark_signal))
    watermarked_stft = np.abs(librosa.stft(watermarked_signal))
    
    max_value = max(original_stft.max(), watermark_stft.max(), watermarked_stft.max())
    
    original_spectrogram = librosa.amplitude_to_db(original_stft, ref=max_value)
    watermark_spectrogram = librosa.amplitude_to_db(watermark_stft, ref=max_value)
    watermarked_spectrogram = librosa.amplitude_to_db(watermarked_stft, ref=max_value)
    
    original_spectrogram = np.clip(original_spectrogram, -80, 0)
    watermark_spectrogram = np.clip(watermark_spectrogram, -80, 0)
    watermarked_spectrogram = np.clip(watermarked_spectrogram, -80, 0)
    
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    librosa.display.specshow(original_spectrogram, sr=sr, x_axis='time', y_axis='log', vmin=-80, vmax=0)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Audio original - {filename}')

    plt.subplot(1, 3, 2)
    librosa.display.specshow(watermark_spectrogram, sr=sr, x_axis='time', y_axis='log', vmin=-80, vmax=0)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Marca de agua')

    plt.subplot(1, 3, 3)
    librosa.display.specshow(watermarked_spectrogram, sr=sr, x_axis='time', y_axis='log', vmin=-80, vmax=0)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Audio + marca de agua')

    plt.tight_layout()
    # Save the figure as a PNG file
    output_file_path = results_folder / f"spectrogram_{filename}.png"
    plt.savefig(output_file_path, format='png')
    plt.close()  # Close the figure to free memory

def main(signal_path, wm_path, wmd_signal_path, results_folder):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load AudioSeal generator and detector models
    generator_model = AudioSeal.load_generator("audioseal_wm_16bits").to(device)
    detector_model = AudioSeal.load_detector("audioseal_detector_16bits").to(device)
    
     # Ensure output directories exist
    wmd_signal_path.mkdir(parents=True, exist_ok=True)
    wm_path.mkdir(parents=True, exist_ok=True)
    
    audio_data = load_wav_files(signal_path)
    
    for file_name, audio, sr in audio_data:
        # Convertir el audio a tensor y asegurar que sea del tipo correcto
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # Añadir dimensión de batch
        audio_tensor = audio_tensor.unsqueeze(0)  # Añadir dimensión de canales (para audio mono)
        audio_tensor = audio_tensor.to(device)  # Mover el tensor al dispositivo correcto (CPU o GPU)

        # Generar un arreglo de NumPy con valores aleatorios 0 o 1
        msg = np.random.choice([0, 1], size=16)
        # Convertir el arreglo de NumPy a un tensor de PyTorch
        msg_tensor = torch.from_numpy(msg)
        msg_tensor = msg_tensor.to(device)
        print(f'{file_name} message: {msg_tensor}')

        # Generar la marca de agua
        watermark = generator_model.get_watermark(audio_tensor, sr, message=msg_tensor).cpu().detach().numpy()[0, 0]
        watermark_amplified = 5 * watermark
        
        # Apply watermark
        watermarked_audio = audio + watermark
        
        # Save watermarked audio
        watermarked_file = wmd_signal_path / f"wmd_{file_name}.wav"
        sf.write(watermarked_file, watermarked_audio, sr)
        
        # Save watermark signal (amplified)
        watermark_file = wm_path / f"wm_{file_name}.wav"
        sf.write(watermark_file, watermark_amplified, sr)
        
        watermarked_audio_tensor = torch.tensor(watermarked_audio, dtype=torch.float32).unsqueeze(0)  # Añadir dimensión de batch
        watermarked_audio_tensor = watermarked_audio_tensor.unsqueeze(0)  # Añadir dimensión de canales (para audio mono)
        watermarked_audio_tensor = watermarked_audio_tensor.to(device)
        # Verificar la forma del tensor antes de pasarlo al modelo
        print("Shape of watermarked_audio_tensor:", watermarked_audio_tensor.shape)

        # Detectar la marca de agua
        result, detected_message = detector_model.detect_watermark(watermarked_audio_tensor, sr)
        detected_message = detected_message.cpu().detach().numpy()
        print(f'{file_name} detected message: {detected_message}')
        snr = signal_noise_ratio(audio, watermarked_audio)
        
        pesq_score = pesq(16000, audio, watermarked_audio)
        
        print(f'Original message: {msg}')
        print(f'Decoded message:  {detected_message}')
        ber = (msg != detected_message).mean() * 100
        
        
        
        # Save results
        results_file = results_folder / 'results.txt'
        save_results(file_name, snr, ber, pesq_score, results_file)
        
        # Plot results
        plot_results(audio, watermark_amplified, watermarked_audio, sr, file_name, results_folder)

if __name__ == '__main__':
    signal_path = Path('audio-files/audioseal/signal')
    wm_path = Path('audio-files/audioseal/watermark')
    wmd_signal_path = Path('audio-files/audioseal/wmd-signal')
    results_folder = Path('audio-files/audioseal/results')
    results_folder.mkdir(parents=True, exist_ok=True)
    
    main(signal_path, wm_path, wmd_signal_path, results_folder)
