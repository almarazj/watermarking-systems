# Watermark Generator
from audioseal import AudioSeal
import torchaudio
import torch 

# Cargar el archivo de audio
waveform, original_sample_rate = torchaudio.load('audio-files\\test.wav')

# Definir la nueva tasa de muestreo
new_sample_rate = 16000  # Ejemplo: bajar a 16 kHz

# Downsampling
resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=new_sample_rate)
downsampled_waveform = resampler(waveform)

model = AudioSeal.load_generator("audioseal_wm_16bits")

# Cargar el archivo de audio y ajustar la forma
wav, sr = downsampled_waveform, new_sample_rate

# Asegúrate de que el tensor tenga la forma correcta
if wav.dim() == 2:  
    wav = wav.unsqueeze(0)  # Añade una dimensión para el lote si es necesario

# Convertir a mono si es estéreo
if wav.shape[1] == 2:  
    wav = torch.mean(wav, dim=1, keepdim=True)  # Promedia los canales

watermark = model.get_watermark(wav, sr)
watermarked_audio = wav + watermark

# Watermark Detector
detector = AudioSeal.load_detector("audioseal_detector_16bits")
result, message = detector.detect_watermark(watermarked_audio, sr)

print(result)  # Imprime la probabilidad de que el audio esté marcado
print(message)  # Imprime el vector binario de 16 bits

# Eliminar la dimensión del lote para guardar el archivo
watermarked_audio = watermarked_audio.squeeze(0)
watermarked_audio = watermarked_audio.detach()
torchaudio.save('audio-files\\audioseal_wm_audio.wav', watermarked_audio, sr)

watermark = watermark.squeeze(0)
watermark = watermark.detach()
torchaudio.save('audio-files\\audioseal_wm.wav', watermark, sr)