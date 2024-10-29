import torch
import numpy as np
import librosa
import librosa.display
from pathlib import Path
from tqdm import tqdm
from utils import load_files, to_equal_length, signal_noise_ratio, save_results, save_ber
from audioseal import AudioSeal
import silentcipher
import wavmark
import pandas as pd
import ast


def test_ber_sc():
    
    input_path = Path('audio-files/silentcipher/opus_12k_to_wav')
    results_folder = Path('audio-files/silentcipher/results') 
       
    target_sr = 44100
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = silentcipher.get_model(model_type='44.1k', device=device)

    audio_data = load_files(input_path)
    csv_path = results_folder / 'results.csv'
    df = pd.read_csv(csv_path)
    
    results_file = results_folder / 'ber_results.csv'
    
    # Descomentar la segunda vez
    ber_df = pd.read_csv(results_file)   
    if 'ber_opus_12k' not in ber_df.columns:
        ber_df['ber_opus_12k'] = np.nan
    
    with tqdm(total=len(audio_data), desc="Procesando archivos de audio", unit="archivo") as pbar:
        for file_name, audio_watermarked_16, sr in audio_data:
            
            msg = df.loc[df['file_name'] == file_name, 'msg'].values
            msg_str = msg[0].replace(' ', ',')
            msg = ast.literal_eval(msg_str) 
            
            current_ber = df.loc[df['file_name'] == file_name, 'ber'].values
            
            audio_watermarked_44 = librosa.resample(audio_watermarked_16, orig_sr=sr, target_sr=target_sr)
            
            result = model.decode_wav(audio_watermarked_44, target_sr, phase_shift_decoding=False)
            
            new_ber = 100
            
            if result['status']:
                payload_40bit = [int(bit) for num in msg for bit in np.binary_repr(num, width=8)]
                result_40bit = [int(bit) for num in result['messages'][0] for bit in np.binary_repr(num, width=8)]
                new_ber = (np.array(payload_40bit) != np.array(result_40bit)).mean() * 100
             
            # Comentar a la segunda vez   
            #save_ber(file_name, current_ber.item(), new_ber, results_file)
            
            # Descomentar a la segunda vez
            ber_df.loc[ber_df['file_name'] == file_name, 'ber_opus_12k'] = new_ber
            
            pbar.update(1)
        
    # Descomentar a la segunda vez     
    ber_df.to_csv(results_file, index=False)

def test_ber_as():
    
    input_path = Path('audio-files/audioseal/opus_24k_to_wav')
    results_folder = Path('audio-files/audioseal/results')
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    detector_model = AudioSeal.load_detector("audioseal_detector_16bits").to(device)
    audio_data = load_files(input_path)
    
    csv_path = results_folder / 'results.csv'
    df = pd.read_csv(csv_path)
    
    results_file = results_folder / 'ber_results.csv'
    
    # Descomentar a la segunda vez
    ber_df = pd.read_csv(results_file)   
    if 'ber_opus_24k' not in ber_df.columns:
        ber_df['ber_opus_24k'] = np.nan

    with tqdm(total=len(audio_data), desc="Procesando archivos de audio", unit="archivo") as pbar:    
        for file_name, audio_watermarked, sr in audio_data:
            # Convertir el audio a tensor y asegurar que sea del tipo correcto
            audio_tensor = torch.tensor(audio_watermarked, dtype=torch.float32).unsqueeze(0)  # Añadir dimensión de batch
            audio_tensor = audio_tensor.unsqueeze(0)  # Añadir dimensión de canales (para audio mono)
            audio_tensor = audio_tensor.to(device)  # Mover el tensor al dispositivo correcto (CPU o GPU)

            msg = df.loc[df['file_name'] == file_name, 'msg'].values
            msg_str = msg[0].replace(' ', ',')
            msg = ast.literal_eval(msg_str)
            
            watermarked_audio_tensor = torch.tensor(audio_watermarked, dtype=torch.float32).unsqueeze(0)  # Añadir dimensión de batch
            watermarked_audio_tensor = watermarked_audio_tensor.unsqueeze(0)  # Añadir dimensión de canales (para audio mono)
            watermarked_audio_tensor = watermarked_audio_tensor.to(device)

            # Detectar la marca de agua
            result, decoded_msg = detector_model.detect_watermark(watermarked_audio_tensor, sr)
            decoded_msg = decoded_msg.cpu().detach().numpy()
            
            # Métricas  
            current_ber = df.loc[df['file_name'] == file_name, 'ber'].values
            new_ber = (msg != decoded_msg).mean() * 100
            
            # Comentar a la segunda vez
            #save_ber(file_name, current_ber.item(), new_ber, results_file)

            # Descomentar a la segunda vez
            ber_df.loc[ber_df['file_name'] == file_name, 'ber_opus_24k'] = new_ber            
            
            pbar.update(1)
            
    # Descomentar a la segunda vez     
    ber_df.to_csv(results_file, index=False)

def test_ber_wm():
    
    input_path = Path('audio-files/wavmark/opus_12k_to_wav')
    results_folder = Path('audio-files/wavmark/results')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = wavmark.load_model().to(device)
    audio_data = load_files(input_path)
    
    csv_path = results_folder / 'results.csv'
    df = pd.read_csv(csv_path)
    
    results_file = results_folder / 'ber_results.csv'
    
    # Descomentar a la segunda vez
    # ber_df = pd.read_csv(results_file)   
    # if 'ber_opus_24k' not in ber_df.columns:
    #     ber_df['ber_opus_24k'] = np.nan
    
    with tqdm(total=len(audio_data), desc="Procesando archivos de audio", unit="archivo") as pbar:        
        for file_name, audio_watermarked, sr in audio_data:
            
            msg = df.loc[df['file_name'] == file_name, 'msg'].values
            msg_str = msg[0].replace(' ', ',')
            msg = ast.literal_eval(msg_str)
            
            decoded_msg, _ = wavmark.decode_watermark(model, audio_watermarked, show_progress=True)

            current_ber = df.loc[df['file_name'] == file_name, 'ber'].values
            
            print(f'original: {msg} (type {type(msg)}). Decoded: {decoded_msg} (type: {type(decoded_msg)}). SR: {sr}')

            new_ber = 100
            if decoded_msg is not None:
                new_ber = (msg != decoded_msg).mean() * 100

            # Comentar la segunda vez
            save_ber(file_name, current_ber.item(), new_ber, results_file)
            
            # Descomentar a la segunda vez
            #ber_df.loc[ber_df['file_name'] == file_name, 'ber_opus_24k'] = new_ber   

            pbar.update(1)
            
    # Descomentar a la segunda vez     
    #ber_df.to_csv(results_file, index=False)
            
if __name__ == '__main__':
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}.')
    
    # Elegir una
    
    # test_ber_sc()
    # test_ber_as()
    test_ber_wm()