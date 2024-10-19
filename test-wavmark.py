import numpy as np
import soundfile as sf
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import wavmark
from pesq import pesq
import time


def load_files(folder_path):
    """
    Loads all WAV files from a folder and returns them as a list.

    Args:
        folder_path (str or Path): The path to the folder containing WAV files.

    Returns:
        list: A list where each element is the waveform data of a WAV file.
              Each item is a NumPy array representing the audio waveform.
    """
    wav_files = list(folder_path.glob('*.wav'))
    
    # List to store audio data
    audio_data = []
    filenames = []
    
    for wav_file in wav_files:
        # Load every wav file
        data, sr = sf.read(wav_file)
        audio_data.append(data)
        filenames.append(wav_file.stem)  # Save the filename for future use
    
    return audio_data, filenames

def to_equal_length(original, signal_watermarked):
    """
    Adjusts the length of the original and watermarked signals to be the same.
    """
    if original.shape != signal_watermarked.shape:
        print("Warning: length not equal:", len(original), len(signal_watermarked))
        min_length = min(len(original), len(signal_watermarked))
        original = original[:min_length]
        signal_watermarked = signal_watermarked[:min_length]
    assert original.shape == signal_watermarked.shape
    return original, signal_watermarked

def signal_noise_ratio(original, signal_watermarked):
    """
    Calculates the Signal-to-Noise Ratio (SNR) between the original and watermarked signals.
    """
    original, signal_watermarked = to_equal_length(original, signal_watermarked)
    noise_strength = np.sum((original - signal_watermarked) ** 2)
    if noise_strength == 0:
        return np.inf
    signal_strength = np.sum(original ** 2)
    ratio = signal_strength / noise_strength
    ratio = max(1e-10, ratio)
    return 10 * np.log10(ratio)

def save_results(file_name, snr, ber, pesq_score, time_elapsed, output_file):
    """
    Saves the results to a text file.

    Args:
        file_name (str): The name of the original audio file.
        snr (float): Signal-to-noise ratio.
        ber (float): Bit Error Rate.
        pesq_score (float): PESQ score.
        time_elapsed (float): Time taken for processing the file.
        output_file (Path): Path to the output file.
    """
    with open(output_file, 'a') as f:
        f.write(f"{file_name} {snr:.2f} {pesq_score:.2f} {ber:.2f} {time_elapsed:.2f}\n")

def plot_results(original, watermark_signal, watermarked_signal, sr, filename, results_folder):
    """
    Plots the spectrograms of the original audio, the watermark signal, and the watermarked audio.
    All spectrograms are normalized to the same reference decibel level (0 dB) for comparison.
    The dB range is limited between 0 dB and -80 dB for all plots.

    Args:
        original (np.array): Original audio signal.
        watermark_signal (np.array): The watermark signal.
        watermarked_signal (np.array): The audio signal with watermark applied.
        sr (int): Sampling rate of the audio signals.
        filename (str): The name of the audio file (for the title).
    """
    # Compute the Short-Time Fourier Transform (STFT) for each signal
    original_stft = np.abs(librosa.stft(original))
    watermark_stft = np.abs(librosa.stft(watermark_signal))
    watermarked_stft = np.abs(librosa.stft(watermarked_signal))
    
    # Find the maximum value across all STFTs to use as the reference for dB conversion
    max_value = max(original_stft.max(), watermark_stft.max(), watermarked_stft.max())
    
    # Convert the STFTs to decibel scale using the same reference
    original_spectrogram = librosa.amplitude_to_db(original_stft, ref=max_value)
    watermark_spectrogram = librosa.amplitude_to_db(watermark_stft, ref=max_value)
    watermarked_spectrogram = librosa.amplitude_to_db(watermarked_stft, ref=max_value)

    # Clip the dB range between 0 dB and -80 dB
    original_spectrogram = np.clip(original_spectrogram, -80, 0)
    watermark_spectrogram = np.clip(watermark_spectrogram, -80, 0)
    watermarked_spectrogram = np.clip(watermarked_spectrogram, -80, 0)

    # Create a figure with three subplots
    plt.figure(figsize=(18, 6))

    # Plot the original audio spectrogram
    plt.subplot(1, 3, 1)
    librosa.display.specshow(original_spectrogram, sr=sr, x_axis='time', y_axis='log', vmin=-80, vmax=0)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Audio original - {filename}')

    # Plot the watermark signal spectrogram
    plt.subplot(1, 3, 2)
    librosa.display.specshow(watermark_spectrogram, sr=sr, x_axis='time', y_axis='log', vmin=-80, vmax=0)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Marca de agua')

    # Plot the watermarked audio spectrogram
    plt.subplot(1, 3, 3)
    librosa.display.specshow(watermarked_spectrogram, sr=sr, x_axis='time', y_axis='log', vmin=-80, vmax=0)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Audio + marca de agua')

    # Show plot
    plt.tight_layout()
    output_file_path = results_folder / f"spectrogram_{filename}.png"
    plt.savefig(output_file_path, format='png')
    plt.close()  # Close the figure to free memory

def main(signal_path, wm_path, wmd_signal_path, results_folder):
    # 1. Load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = wavmark.load_model().to(device)

    # 2. Create 16-bit payload
    payload = np.random.choice([0, 1], size=16)
    print("Payload:", payload)

    # 3. Read host audio files
    audio_data, filenames = load_files(signal_path)

    # Ensure output directories exist
    wmd_signal_path.mkdir(parents=True, exist_ok=True)
    wm_path.mkdir(parents=True, exist_ok=True)
        
    # 4. Encode watermark and save the watermarked signals, watermark, and calculate SNR and BER
    for audio, filename in zip(audio_data, filenames):
        
        start_time = time.time()
        # Encode watermark
        watermarked_signal, _ = wavmark.encode_watermark(model, audio, payload, show_progress=True)
        
        # Save the watermarked signal as a new wav file in wmd_signal_path
        filename_modified = filename.replace('_or', '') + '_wm'
        watermarked_filename = f"{filename_modified}.wav"
        watermarked_filepath = wmd_signal_path / watermarked_filename
        sf.write(watermarked_filepath, watermarked_signal, 16000)
        
        # Calculate the watermark (difference between original and watermarked signal)
        watermark_signal = watermarked_signal - audio

        # Amplify watermark signal by 5 times
        amplified_watermark_signal = watermark_signal * 5
        
        # Save the amplified watermark signal in wm_path
        watermark_filename = f"wm_{filename}.wav"
        watermark_filepath = wm_path / watermark_filename
        sf.write(watermark_filepath, amplified_watermark_signal, 16000)

        # 5. Decode watermark
        payload_decoded, _ = wavmark.decode_watermark(model, watermarked_signal, show_progress=True)
        
        print(f'Original message: {payload}')
        print(f'Decoded message:  {payload_decoded}')
        ber = (payload != payload_decoded).mean() * 100
        print(f"Decode BER for {filename}: {ber:.1f}%")

        # 6. Calculate SNR
        snr = signal_noise_ratio(audio, watermarked_signal)
        print(f"SNR for {filename}: {snr:.2f} dB")
        
        print(f'Original: {audio.shape}, wmd: {watermarked_signal.shape}')
        pesq_score = pesq(16000, audio, watermarked_signal)
        print(f'pesq score for {filename}: {pesq_score:.2f}')
        
        end_time = time.time()  # <-- Captura el tiempo al final del procesamiento
        time_elapsed = end_time - start_time
        
        # Save results
        results_file = results_folder / 'results.txt'
        save_results(filename, snr, ber, pesq_score, time_elapsed, results_file)
        
        # 8. Plot the spectrograms of original, watermark, and watermarked signals
        #plot_results(audio, watermark_signal, watermarked_signal, 16000, filename, results_folder)

if __name__ == '__main__':
    # Define the paths
    signal_path = Path('D:/Music/datasets/Dataset/HABLA-spoofed')
    wm_path = Path('audio-files/wavmark/watermark')
    wmd_signal_path = Path('audio-files/wavmark/wmd-signal')
    results_folder = Path('audio-files/wavmark/results')
    results_folder.mkdir(parents=True, exist_ok=True)
    # Run the main function
    main(signal_path, wm_path, wmd_signal_path, results_folder)