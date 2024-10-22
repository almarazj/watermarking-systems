import numpy as np
import soundfile as sf
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import silentcipher
from pesq import pesq
import time

def load_files(folder_path, ext):
    """Loads all WAV files from a folder and returns them as a list."""
    
    SR = 44100
    
    audio_files = list(folder_path.glob(f'*.{ext}'))
    
    audio_data = []
    filenames = []
    
    for audio_file in audio_files:
        data, sr = librosa.load(audio_file, sr=SR)
        audio_data.append(data)
        filenames.append(audio_file.stem)
    
    return audio_data, filenames

def signal_noise_ratio(original, signal_watermarked):
    """Calculates the Signal-to-Noise Ratio (SNR) between the original and watermarked signals."""
        
    noise_strength = np.sum((original - signal_watermarked) ** 2)
    if noise_strength == 0:
        return np.inf
    signal_strength = np.sum(original ** 2)
    ratio = signal_strength / noise_strength
    ratio = max(1e-10, ratio)
    return 10 * np.log10(ratio)

def save_results(file_name, snr, ber, pesq_score, time_elapsed, payload, output_file):
    """Saves the results to a text file."""
    with open(output_file, 'a') as f:
        if ber == '-':
            f.write(f"{file_name} {snr:.2f} {pesq_score:.2f} {ber} {time_elapsed:.2f} {payload}\n")
        else:
            f.write(f"{file_name} {snr:.2f} {pesq_score:.2f} {ber:.2f} {time_elapsed:.2f} {payload}\n")

def encode_decode(signal_path, wm_path, wmd_signal_path, results_folder, ext='wav'):
    
    SR = 44100
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = silentcipher.get_model(model_type='44.1k', device=device)

    audio_data, filenames = load_files(signal_path, ext)

    wmd_signal_path.mkdir(parents=True, exist_ok=True)
    wm_path.mkdir(parents=True, exist_ok=True)
        
    for audio, filename in zip(audio_data, filenames):
        
        start_time = time.time()
        
        payload = np.random.randint(0, 256, size=5).tolist()
        print("Payload:", payload)
        
        watermarked_signal, _ = model.encode_wav(audio, SR, payload, message_sdr=None, calc_sdr=True, disable_checks=False)
        
        resample_wm_signal = librosa.resample(watermarked_signal, orig_sr=SR, target_sr=16000)
        resample_audio = librosa.resample(audio, orig_sr=SR, target_sr=16000)
        
        filename_modified = filename.replace('_or', '') + '_sc'
        watermarked_filepath = wmd_signal_path / f"{filename_modified}.wav"
        sf.write(watermarked_filepath, resample_wm_signal, 16000)
        
        watermark_signal = watermarked_signal - audio
        watermark_filepath = wm_path / f"wm_{filename}.wav"
        sf.write(watermark_filepath, watermark_signal, SR)

        result = model.decode_wav(watermarked_signal, SR, phase_shift_decoding=False)
        
        snr = signal_noise_ratio(resample_audio, resample_wm_signal)
        pesq_score = pesq(16000, resample_audio, resample_wm_signal)
        
        print(f"SNR for {filename}: {snr:.2f} dB")
        print(f'pesq score for {filename}: {pesq_score:.2f}')
        
        end_time = time.time()
        time_elapsed = end_time - start_time
        results_file = results_folder / 'results.txt'
        
        if result['status']:
            print(f'Original message: {payload}')
            print(f"Reconstructed message: {result['messages'][0]}")
            print(result['confidences'][0])

            payload_40bit = [int(bit) for num in payload for bit in np.binary_repr(num, width=8)]
            result_40bit = [int(bit) for num in result['messages'][0] for bit in np.binary_repr(num, width=8)]
            ber = (np.array(payload_40bit) != np.array(result_40bit)).mean() * 100
            
            save_results(filename, snr, ber, pesq_score, time_elapsed, payload, results_file)
        else:
            print(result['error'])
            ber = '-'
            save_results(filename, snr, ber, pesq_score, time_elapsed, payload, results_file)

def decode(signal_path, results_file, output_file):
    
    SR = 44100
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = silentcipher.get_model(model_type='44.1k', device=device)

    with open(results_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            filename = parts[0] + '_sc'
            snr = float(parts[1])
            pesq = float(parts[2])
            
            payload = list([int(parts[5]), int(parts[6]), int(parts[7]), int(parts[8]), int(parts[9])])
            print(payload)
            wav_file = signal_path / f"{filename}.wav"

            audio, _ = librosa.load(wav_file, sr=SR)

            start_time = time.time()
            result_decoded = model.decode_wav(audio, SR, phase_shift_decoding=False)
            end_time = time.time()

            if result_decoded['status']:
                decoded_payload = result_decoded['messages'][0]

                payload_40bit = [int(bit) for num in payload for bit in np.binary_repr(num, width=8)]
                decoded_40bit = [int(bit) for num in decoded_payload for bit in np.binary_repr(num, width=8)]
                ber = (np.array(payload_40bit) != np.array(decoded_40bit)).mean() * 100

                print(f"BER for {filename}: {ber:.2f}%")
                time_elapsed = end_time - start_time

                save_results(filename, snr, ber, pesq, time_elapsed, payload, output_file)
            else:
                print(f"Decoding error for {filename}: {result_decoded['error']}")
                ber = '-'
                save_results(filename, snr, ber, pesq, time_elapsed, payload, output_file)
    

if __name__ == '__main__':

    signal_path = Path('D:/Music/datasets/Dataset/HABLA-spoofed')
    wm_path = Path('audio-files/silentcipher/watermark')
    wmd_signal_path = Path('audio-files/silentcipher/wmd-signal')
    results_folder = Path('audio-files/silentcipher/results')
    results_folder.mkdir(parents=True, exist_ok=True)
    #encode_decode(signal_path, wm_path, wmd_signal_path, results_folder)
    
    opus_wav = Path('audio-files/silentcipher/wmd-signal/opus_wav')
    decode(opus_wav, results_folder / 'results.txt', results_folder / 'new_results.txt')
    
    # python3 .\test-sc.py > audio-files/silentcipher/results/console_output.txt 2>&1