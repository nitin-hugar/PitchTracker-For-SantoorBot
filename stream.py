# # stream audio

import pyaudio
import numpy as np
from scipy.io.wavfile import write 
 
FORMAT = pyaudio.paFloat32
CHANNELS = 2
RATE = 44100
CHUNK = 2048
RECORD_SECONDS = 5
# WAVE_OUTPUT_FILENAME = "file.wav"
 
audio = pyaudio.PyAudio()
 
# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

print ("recording...")
frames = np.array([])
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = np.frombuffer(stream.read(CHUNK), dtype=np.float32)
    frames = np.append(frames, data)

print ("finished recording")
  
write("output.wav", RATE, frames) 
# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()
 
print("Done!")