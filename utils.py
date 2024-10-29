import csv
import numpy as np
import soundfile as sf
from pathlib import Path
import librosa

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
    
    for wav_file in wav_files:
        # Load every wav file
        data, sr = librosa.load(wav_file, sr=16000, mono=True)

        filename = wav_file.stem  # Save the filename for future use
        audio_data.append((filename, data, sr))
        
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

def save_results(file_name, snr, pesq_score, ber, encode_time, decode_time, msg, output_file):
    """
    Saves the results to a CSV file.

    Args:
        file_name (str): The name of the original audio file.
        snr (float): Signal-to-noise ratio.
        ber (float): Bit Error Rate.
        pesq_score (float): PESQ score.
        time_elapsed (float): Time taken for processing the file.
        output_file (Path): Path to the output file.
    """
    # Convert msg to the desired string format
    msg_str = '[' + ' '.join(map(str, msg)) + ']'
    
    # Check if the file already exists to write the header only once
    file_exists = Path(output_file).exists()

    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        # Write the header if the file doesn't exist
        if not file_exists:
            writer.writerow(["file_name", "snr", "pesq_score", "ber", "encode_time", "decode_time", "msg"])
        # Write the data
        writer.writerow([file_name, f"{snr:.2f}", f"{pesq_score:.2f}", f"{ber:.2f}", f"{encode_time:.2f}", f"{decode_time:.2f}", msg_str])

def save_ber(file_name, current_ber, new_ber, output_file):
    # Check if the file already exists to write the header only once
    file_exists = Path(output_file).exists()
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["file_name", "ber", "ber_opus_12k"])
        writer.writerow([file_name, f"{current_ber:.2f}", f"{new_ber:.2f}"])