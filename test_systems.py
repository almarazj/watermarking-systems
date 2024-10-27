import torch
import time
import numpy as np
import soundfile as sf
import librosa
import librosa.display
from pathlib import Path
from pesq import pesq
from tqdm import tqdm
from utils import load_files, to_equal_length, signal_noise_ratio, save_results
from audioseal import AudioSeal
import silentcipher
import wavmark

def test_audioseal(input_path, output_path, results_folder):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    generator_model = AudioSeal.load_generator("audioseal_wm_16bits").to(device)
    detector_model = AudioSeal.load_detector("audioseal_detector_16bits").to(device)
    audio_data = load_files(input_path)

    with tqdm(total=len(audio_data), desc="Procesando archivos de audio", unit="archivo") as pbar:    
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
            t1 = time.time()
            watermark = generator_model.get_watermark(audio_tensor, sr, message=msg_tensor).cpu().detach().numpy()[0, 0]
            t2 = time.time()
            encode_time = t2 - t1

            # Apply watermark
            audio_watermarked = audio + watermark
            
            # Save watermarked audio
            watermarked_file = output_path / f"{file_name}.wav"
            sf.write(watermarked_file, audio_watermarked, sr)
            
            watermarked_audio_tensor = torch.tensor(audio_watermarked, dtype=torch.float32).unsqueeze(0)  # Añadir dimensión de batch
            watermarked_audio_tensor = watermarked_audio_tensor.unsqueeze(0)  # Añadir dimensión de canales (para audio mono)
            watermarked_audio_tensor = watermarked_audio_tensor.to(device)

            # Detectar la marca de agua
            t1 = time.time()
            result, decoded_msg = detector_model.detect_watermark(watermarked_audio_tensor, sr)
            decoded_msg = decoded_msg.cpu().detach().numpy()
            t2 = time.time()
            decode_time = t2 - t1
            
            # Métricas  
            snr = signal_noise_ratio(audio, audio_watermarked)
            pesq_score = pesq(16000, audio, audio_watermarked)    
            ber = (msg != decoded_msg).mean() * 100
            
            # Save results
            results_file = results_folder / 'results.csv'
            save_results(file_name, snr, pesq_score, ber, encode_time, decode_time, msg, results_file)
            
            pbar.update(1)

        
def test_silentcipher(input_path, output_path, results_folder):
    
    target_sr = 44100
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = silentcipher.get_model(model_type='44.1k', device=device)

    audio_data = load_files(input_path)
    
    with tqdm(total=len(audio_data), desc="Procesando archivos de audio", unit="archivo") as pbar:
        for file_name, audio_16, sr in audio_data:
            
            msg = np.random.randint(0, 256, size=5).tolist()
            
            audio_44 = librosa.resample(audio_16, orig_sr=sr, target_sr=target_sr)
            
            t1 = time.time()
            audio_watermarked_44, _ = model.encode_wav(audio_44, target_sr, msg, message_sdr=47, calc_sdr=True, disable_checks=False)
            t2 = time.time()
            encode_time = t2 - t1
            
            audio_watermarked_16 = librosa.resample(audio_watermarked_44, orig_sr=target_sr, target_sr=sr)
            
            watermarked_filepath = output_path / f"{file_name}.wav"
            sf.write(watermarked_filepath, audio_watermarked_16, 16000)

            t1 = time.time()
            result = model.decode_wav(audio_watermarked_44, target_sr, phase_shift_decoding=False)
            t2 = time.time()
            decode_time = t2 - t1
            
            snr = signal_noise_ratio(audio_16, audio_watermarked_16)
            pesq_score = pesq(16000, audio_16, audio_watermarked_16)
            ber = 100
            
            if result['status']:
                payload_40bit = [int(bit) for num in msg for bit in np.binary_repr(num, width=8)]
                result_40bit = [int(bit) for num in result['messages'][0] for bit in np.binary_repr(num, width=8)]
                ber = (np.array(payload_40bit) != np.array(result_40bit)).mean() * 100
                
            results_file = results_folder / 'results.csv'
            save_results(file_name, snr, pesq_score, ber, encode_time, decode_time, msg, results_file)
            
            pbar.update(1)
        
def test_wavmark(input_path, output_path, results_folder):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = wavmark.load_model().to(device)

    audio_data = load_files(input_path)

    with tqdm(total=len(audio_data), desc="Procesando archivos de audio", unit="archivo") as pbar:        
        for file_name, audio, sr in audio_data:
            
            msg = np.random.choice([0, 1], size=16)
            
            t1 = time.time()
            audio_watermarked, _ = wavmark.encode_watermark(model, audio, msg, show_progress=True)
            t2 = time.time() 
            encode_time = t2 - t1

            watermarked_filepath = output_path / f"{file_name}.wav"
            sf.write(watermarked_filepath, audio_watermarked, 16000)

            t1 = time.time()
            decoded_msg, _ = wavmark.decode_watermark(model, audio_watermarked, show_progress=True)
            t2 = time.time()
            decode_time = t2 - t1
            
            # Métricas
            snr = signal_noise_ratio(audio, audio_watermarked)
            pesq_score = pesq(16000, audio, audio_watermarked)
            ber = (msg != decoded_msg).mean() * 100

            # Save results
            results_file = results_folder / 'results.csv'
            save_results(file_name, snr, pesq_score, ber, encode_time, decode_time, msg, results_file)

            pbar.update(1)

        
if __name__ == '__main__':
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}.')
    
    # input_path = Path('D:/Music/datasets/Dataset/HABLA-spoofed')
    # output_path = Path('audio-files/silentcipher/watermarked')
    # results_folder = Path('audio-files/silentcipher/results')

    # test_silentcipher(input_path, output_path, results_folder)
        
    input_path = Path('D:/Music/datasets/Dataset/HABLA-spoofed')
    output_path = Path('audio-files/audioseal/watermarked')
    results_folder = Path('audio-files/audioseal/results')
    
    test_audioseal(input_path, output_path, results_folder)
    
    input_path = Path('D:/Music/datasets/Dataset/HABLA-spoofed')
    output_path = Path('audio-files/wavmark/watermarked')
    results_folder = Path('audio-files/wavmark/results')
    
    test_wavmark(input_path, output_path, results_folder)