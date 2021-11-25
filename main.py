# Calls all the functions and generates music on the santoor

from numpy.core.shape_base import block
# import stream
import features
import utilities
import makenotes
import utilities
import matplotlib.pyplot as plt


cAudioFilePath = "./voice.wav"
blockSize = 4096
hopSize = 2048

# import audio and find chromagram
fs, x = utilities.ToolReadAudio(cAudioFilePath)
xb, timeInSec = utilities.block_audio(x, blockSize, hopSize, fs)
X = utilities.compute_spectrogram(xb)
pitchChroma = features.extract_pitch_chroma(X, fs, 440.)


#plot pitch chroma
fig = plt.figure(figsize=(15, 7))
plt.pcolormesh(pitchChroma)
plt.title("PitchChroma")
plt.xlabel("Blocks")
plt.ylabel("Chroma")
plt.savefig("pitchChroma.png")

fig2 = plt.figure(figsize=(15, 7))
plt.pcolormesh(X)
plt.title("Spectrogram")
plt.xlabel("Blocks")
plt.ylabel("FrequencyBins")
plt.savefig("Spectrogram")



