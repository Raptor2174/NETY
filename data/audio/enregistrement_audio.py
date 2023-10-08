import pyaudio
import wave

# Paramètres d'enregistrement audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Taux d'échantillonnage (en Hz)
CHUNK = 1024  # Taille du tampon audio
RECORD_SECONDS = 10  # Durée d'enregistrement (en secondes)
OUTPUT_FILENAME = "data/audio/audio_data.wav"

# Initialiser PyAudio
audio = pyaudio.PyAudio()

# Configuration de l'interface audio
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
                    input_device_index=1)  # Remplacez 1 par l'index du microphone désiré

print("Enregistrement audio en cours...")

frames = []

# Enregistrement audio
for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Enregistrement terminé.")

# Arrêt et fermeture du flux audio
stream.stop_stream()
stream.close()
audio.terminate()

# Sauvegarde des données audio dans un fichier WAV
with wave.open(OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"Audio enregistré dans {OUTPUT_FILENAME}")
