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
import pyaudio
import numpy as np
from scipy.io.wavfile import write 


 
def santoor_playback(x, fs):

    # Define variables
    # cAudioFilePath = "./flute.wav"
    blockSize = 4096
    hopSize = 2048
    # rec_duration = 1.0 # Block size to analyse at one go
    onsets_thres = 0.6 # Peak picking threshold for onsets
    t = -199 # Spectral peak detection threshold in dB for f0 estimation    t changed from -100 to -150
    thres_dB = -40 # Voicing mask threshold in dB
    f0min = 80
    f0max = 2000
    w = np.hanning(blockSize)

    # import audio and find chromagram
    # fs, x = utilities.ToolReadAudio(cAudioFilePath)
    xb, timeInSec = utilities.block_audio(x, blockSize, hopSize, fs)
    X = features.compute_spectrogram(xb)

    # Detect onset from STFT
    onsets = features.onset_detect(X, onsets_thres, n=5)[0]    
    

    # Detect f0 using TWM
    f0c, f0err = features.f0_detection_TWM(xb, w, blockSize, t, f0min, f0max, fs)
    smoothened_f0 = utilities.smoothing_filter(f0c, 3)
    f0 = features.detect_silence(xb, smoothened_f0, thres_dB)

    # Plot f0
    # plt.figure(figsize=(15, 7))
    # plt.subplot(2, 1, 1)
    # plt.plot(f0c)
    # plt.subplot(2, 1, 2)
    # plt.plot(smoothened_f0)
    # plt.show()

    # Get Pitch Chroma
    pitchChroma = features.extract_pitch_chroma(f0)

    #makeNotes
    notes, durations = makenotes.makeNotes(pitchChroma, onsets, init=48, hopSize=hopSize, fs=fs)

    # midi playback
    makenotes.midiPlayBack(0x90 ,notes, durations)


def stream_audio(recDuration):
    FORMAT = pyaudio.paFloat32
    CHANNELS = 2
    fs = 44100
    hop_s = 2048
    p = pyaudio.PyAudio()
    # open stream   
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=fs,
                    input=True,
                    frames_per_buffer=hop_s)

    print("*** starting recording\n")
    x = np.array([], dtype=np.float32)

    while True:
       
        print("Record\n")
        # record a short phrase
        for i in range(0, int(fs / hop_s * recDuration)):
            audiobuffer = stream.read(hop_s, exception_on_overflow=False)
            signal = np.frombuffer(audiobuffer, dtype=np.float32)
            x = np.append(x, signal)

        print("Playback\n")
        

        # Main function that runs the analysis and sends midi messages to the santoor bot
        santoor_playback(x, fs)
        
        # This is where midi playback happens        
        x = np.array([])

    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == '__main__':
    stream_audio(5)









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