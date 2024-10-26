import torch
import time
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import silentcipher
from pathlib import Path
from pesq import pesq
from tqdm import tqdm
from utils import load_files, to_equal_length, signal_noise_ratio, save_results

def test_silentcipher(input_path, output_path, results_folder):
    
    target_sr = 44100
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = silentcipher.get_model(model_type='44.1k', device=device)

    audio_data = load_files(input_path)
    
    with tqdm(total=len(audio_data), desc="Procesando archivos de audio", unit="archivo") as pbar:
        for file_name, audio_16, sr in audio_data:
            
            msg = np.random.randint(0, 256, size=5).tolist()
            
            audio_44 = librosa.resample(audio_16, orig_sr=sr, target_sr=target_sr)
            
            start_time = time.time()
            audio_watermarked_44, _ = model.encode_wav(audio_44, target_sr, msg, message_sdr=None, calc_sdr=True, disable_checks=False)
            end_time = time.time()
            time_elapsed = end_time - start_time
            
            audio_watermarked_16 = librosa.resample(audio_watermarked_44, orig_sr=target_sr, target_sr=sr)
            
            watermarked_filepath = output_path / f"{file_name}.wav"
            sf.write(watermarked_filepath, audio_watermarked_16, 16000)

            result = model.decode_wav(audio_watermarked_44, target_sr, phase_shift_decoding=False)
            
            snr = signal_noise_ratio(audio_16, audio_watermarked_16)
            pesq_score = pesq(16000, audio_16, audio_watermarked_16)
            ber = 100
            
            if result['status']:
                payload_40bit = [int(bit) for num in msg for bit in np.binary_repr(num, width=8)]
                result_40bit = [int(bit) for num in result['messages'][0] for bit in np.binary_repr(num, width=8)]
                ber = (np.array(payload_40bit) != np.array(result_40bit)).mean() * 100
                
            results_file = results_folder / 'results.csv'
            save_results(file_name, snr, pesq_score, ber, time_elapsed, msg, results_file)
            
            pbar.update(1)


if __name__ == '__main__':
    input_path = Path('/home/jalma/Music/datasets/HABLA-spoofed/audios')
    
    output_path = Path('audio-files/silentcipher/watermarked')
    output_path.mkdir(parents=True, exist_ok=True)

    results_folder = Path('audio-files/silentcipher/results')
    results_folder.mkdir(parents=True, exist_ok=True)

    test_silentcipher(input_path, output_path, results_folder)
    
    # python3 .\test-sc.py > audio_16-files/silentcipher/results/console_output.txt 2>&1