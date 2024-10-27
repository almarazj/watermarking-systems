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


def test_ber_sc(input_path, results_folder):
        
    target_sr = 44100
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = silentcipher.get_model(model_type='44.1k', device=device)

    audio_data = load_files(input_path)
    csv_path = results_folder / 'results.csv'
    df = pd.read_csv(csv_path)
    
    results_file = results_folder / 'ber_results.csv'
    ber_df = pd.read_csv(results_file)
    
    if 'ber_opus_24k' not in ber_df.columns:
        ber_df['ber_opus_24k'] = np.nan
    
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
                
            #save_ber(file_name, current_ber.item(), new_ber, results_file)
            ber_df.loc[ber_df['file_name'] == file_name, 'ber_opus_24k'] = new_ber
            
            pbar.update(1)
        
    ber_df.to_csv(results_file, index=False)
            
if __name__ == '__main__':
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}.')
    
    input_path = Path('audio-files/silentcipher/opus_24k_to_wav')
    results_folder = Path('audio-files/silentcipher/results')

    test_ber_sc(input_path, results_folder)