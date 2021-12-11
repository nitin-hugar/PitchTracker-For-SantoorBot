# Calls all the functions and generates music on the santoor

# from numpy.core.shape_base import block
# import stream
import features
import utilities
import makenotes
import utilities
import makenotes
import matplotlib.pyplot as plt
import numpy as np

# Define variables

cAudioFilePath = "./flute.wav"
blockSize = 4096
hopSize = 2048
rec_duration = 1.0 # Block size to analyse at one go
onsets_thres = 0.8 # Peak picking threshold for onsets
t = -199 # Spectral peak detection threshold in dB for f0 estimation    t changed from -100 to -150
thres_dB = -40 # Voicing mask threshold in dB
f0min = 80
f0max = 2000
w = np.hanning(blockSize)


# import audio and find chromagram
fs, x = utilities.ToolReadAudio(cAudioFilePath)
xb, timeInSec = utilities.block_audio(x, blockSize, hopSize, fs)
X = features.compute_spectrogram(xb)


# Detect onset from STFT

# onsets = features.onset_detect(X, onsets_thres)     # Is onset in blocks ? 
# print(onsets)
# Detect f0 using TWM

f0c, f0err = features.f0_detection_TWM(xb, w, blockSize, t, f0min, f0max, fs)
f0 = features.detect_silence(xb, f0c, thres_dB)


# pitchChroma = features.extract_pitch_chroma(X, fs, 440.)
pitchChroma = features.extract_pitch_chroma(f0)


#makeNotes
# notes, durations = makenotes.makeNotes(pitchChroma, onsets, init=48, hopSize=hopSize, fs=fs)
# print(notes, durations)


# fig = plt.figure(figsize=(15, 7))
# plt.pcolormesh(pitchChroma)
# plt.savefig("pitchChromaNew.png")

# #plot pitch chroma
# fig = plt.figure(figsize=(15, 7))
# plt.pcolormesh(pitchChroma)
# plt.title("PitchChroma")
# plt.xlabel("Blocks")
# plt.ylabel("Chroma")
# plt.savefig("pitchChroma.png")

# fig2 = plt.figure(figsize=(15, 7))
# plt.pcolormesh(X)
# plt.title("Spectrogram")
# plt.xlabel("Blocks")
# plt.ylabel("FrequencyBins")
# plt.savefig("Spectrogram")



