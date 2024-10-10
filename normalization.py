import os
import soundfile as sf
import pyloudnorm as pyln

# Define the folder path with the wav files
folder_path = 'audio-files/original-old'  # Cambia esto a la ruta de tu carpeta
output_folder = 'audio-files/original'  # Carpeta donde guardarás los archivos normalizados

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all the files in the folder
for file_name in os.listdir(folder_path):
    # Process only .wav files
    if file_name.endswith('.wav'):
        file_path = os.path.join(folder_path, file_name)

        # Load audio file
        data, fs = sf.read(file_path)

        # Create meter
        meter = pyln.Meter(fs)

        # Measure loudness
        loudness = meter.integrated_loudness(data)

        # Normalize loudness to -18 LUFS
        loudness_normalized_audio = pyln.normalize.loudness(data, loudness, -18.0)

        # Define output file path
        output_file_path = os.path.join(output_folder, file_name)

        # Save the normalized audio
        sf.write(output_file_path, loudness_normalized_audio, fs)

        print(f'Processed and saved: {output_file_path}')
