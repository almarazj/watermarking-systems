import torch
import time
import numpy as np
import soundfile as sf
from pesq import pesq
from pathlib import Path
from audioseal import AudioSeal
from utils import load_files, to_equal_length, signal_noise_ratio, save_results

def test_audioseal(input_path, output_path, results_folder):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    generator_model = AudioSeal.load_generator("audioseal_wm_16bits").to(device)
    detector_model = AudioSeal.load_detector("audioseal_detector_16bits").to(device)
    audio_data = load_files(input_path)
    
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

        # Generar la marca de agua
        start_time = time.time()
        watermark = generator_model.get_watermark(audio_tensor, sr, message=msg_tensor).cpu().detach().numpy()[0, 0]
        end_time = time.time()
        time_elapsed = end_time - start_time

        # Apply watermark
        audio_watermarked = audio + watermark
        
        # Save watermarked audio
        watermarked_file = output_path / f"{file_name}.wav"
        sf.write(watermarked_file, audio_watermarked, sr)
        
        watermarked_audio_tensor = torch.tensor(audio_watermarked, dtype=torch.float32).unsqueeze(0)  # Añadir dimensión de batch
        watermarked_audio_tensor = watermarked_audio_tensor.unsqueeze(0)  # Añadir dimensión de canales (para audio mono)
        watermarked_audio_tensor = watermarked_audio_tensor.to(device)

        # Detectar la marca de agua
        result, decoded_msg = detector_model.detect_watermark(watermarked_audio_tensor, sr)
        decoded_msg = decoded_msg.cpu().detach().numpy()

        # Métricas
        snr = signal_noise_ratio(audio, audio_watermarked)
        pesq_score = pesq(16000, audio, audio_watermarked)    
        ber = (msg != decoded_msg).mean() * 100
        
        # Save results
        results_file = results_folder / 'results.csv'
        save_results(file_name, snr, pesq_score, ber, time_elapsed, msg, results_file)

        
if __name__ == '__main__':
    input_path = Path('D:/Music/datasets/Dataset/HABLA-spoofed')
    output_path = Path('audio-files/audioseal/wmd-signal')
    results_folder = Path('audio-files/audioseal/results')
    
    test_audioseal(input_path, output_path, results_folder)
