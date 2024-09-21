import torch.nn as nn
import numpy as np
import torchaudio
import torchaudio.transforms as T
import torch
import wavmark
import soundfile as sf
from wavmark.utils import file_reader

# 1.load model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
model = wavmark.load_model().to(device)
nb_params = sum(param.numel() for param in model.parameters())
print(f"Number of model parameters: {nb_params}")

# 2.create 16-bit payload
payload = np.random.choice([0, 1], size=16)
print("Payload:", payload)

signal = file_reader.read_as_single_channel("audio-files\\test.wav", aim_sr=16000)

watermarked_signal, _ = wavmark.encode_watermark(model, signal, payload, show_progress=True)
payload_decoded, _ = wavmark.decode_watermark(model, watermarked_signal, show_progress=True)
BER = (payload != payload_decoded).mean() * 100
print("Decode BER:%.1f" % BER)

sf.write('audio-files\\wavmark_wm_audio.wav', watermarked_signal, 16000)


# Ejemplo de uso:
# Suponiendo que tienes implementados `encoder`, `decoder` y `discriminator`
encoder = ...  # Encoder model
decoder = ...  # Decoder model
discriminator = ...  # Discriminator model

# Datos de ejemplo
mvec = torch.randn(1, 32)               # Vector del mensaje
x_wave = torch.randn(1, 1, 16000)       # Audio original
x_wave_prime = torch.randn(1, 1, 16000) # Audio con marca de agua
z = torch.randn(1, 10)                  # Vector auxiliar

# Crear la función de pérdida
loss_fn = WatermarkLoss(lambda_a=0.1, lambda_g=0.1)

# Calcular las pérdidas
Ltotal, Ld = loss_fn(mvec, x_wave, x_wave_prime, z, discriminator, encoder, decoder)

print(f"Ltotal: {Ltotal.item()}, Ld: {Ld.item()}")
