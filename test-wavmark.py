import time
import torch
import wavmark
import numpy as np
import soundfile as sf
from pathlib import Path
from pesq import pesq
from utils import load_files, to_equal_length, signal_noise_ratio, save_results

def test_wavmark(input_path, output_path, results_folder):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = wavmark.load_model().to(device)

    audio_data = load_files(input_path)
        
    for file_name, audio, sr in audio_data:
        
        msg = np.random.choice([0, 1], size=16)
        
        start_time = time.time()
        audio_watermarked, _ = wavmark.encode_watermark(model, audio, msg, show_progress=True)
        end_time = time.time() 
        time_elapsed = end_time - start_time

        watermarked_filepath = output_path / f"{file_name}.wav"
        sf.write(watermarked_filepath, audio_watermarked, 16000)

        decoded_msg, _ = wavmark.decode_watermark(model, audio_watermarked, show_progress=True)
        
        # Métricas
        snr = signal_noise_ratio(audio, audio_watermarked)
        pesq_score = pesq(16000, audio, audio_watermarked)
        ber = (msg != decoded_msg).mean() * 100

        # Save results
        results_file = results_folder / 'results.csv'
        save_results(file_name, snr, pesq_score, ber, time_elapsed, msg, results_file)

        
if __name__ == '__main__':
    input_path = Path('D:/Music/datasets/Dataset/HABLA-spoofed')
    output_path = Path('audio-files/wavmark/wmd-signal')
    results_folder = Path('audio-files/wavmark/results')
    
    test_wavmark(input_path, output_path, results_folder)
    # python3 .\test-wavmark.py > audio-files/wavmark/results/console_output.txt 2>&1