import numpy as np
import torchaudio
import torchaudio.transforms as T
import torch
import wavmark
from wavmark.utils import file_reader
import soundfile as sf

# 1.load model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = wavmark.load_model().to(device)

# 2.create 16-bit payload
payload = np.random.choice([0, 1], size=16)
print("Payload:", payload)

signal = file_reader.read_as_single_channel("audio-files\\test.wav", aim_sr=16000)

watermarked_signal, _ = wavmark.encode_watermark(model, signal, payload, show_progress=True)
#payload_decoded, _ = wavmark.decode_watermark(model, watermarked_signal, show_progress=True)
#BER = (payload != payload_decoded).mean() * 100
#print("Decode BER:%.1f" % BER)

sf.write('audio-files\\wavmark_wm_audio.wav', watermarked_signal, 16000)