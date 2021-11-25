
import librosa
import scipy
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.fft


def  block_audio(x,blockSize,hopSize,fs):    
    # allocate memory    
    numBlocks = math.ceil(x.size / hopSize)    
    xb = np.zeros([numBlocks, blockSize])    
    # compute time stamps    
    t = (np.arange(0, numBlocks) * hopSize) / fs    
    x = np.concatenate((x, np.zeros(blockSize)),axis=0)    
    for n in range(0, numBlocks):        
        i_start = n * hopSize        
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])        
        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]    
    return (xb,t)


def compute_hann(iWindowLength):
    return 0.5 - (0.5 * np.cos(2 * np.pi / iWindowLength * 
np.arange(iWindowLength)))


def compute_spectrogram(xb):
    numBlocks = xb.shape[0]
    afWindow = compute_hann(xb.shape[1])
    X = np.zeros([math.ceil(xb.shape[1]/2+1), numBlocks])
    
    for n in range(0, numBlocks):
        # apply window
        tmp = abs(scipy.fft.fft(xb[n,:] * afWindow))*2/xb.shape[1]
    
        # compute magnitude spectrum
        X[:,n] = tmp[range(math.ceil(tmp.size/2+1))] 
        X[[0,math.ceil(tmp.size/2)],n]= X[[0,math.ceil(tmp.size/2)],n]/np.sqrt(2) 
    
    return X.T


def plot_spectrogram(spectrogram, fs, hopSize):
    
    t = hopSize*np.arange(spectrogram.shape[0])/fs
    f = np.arange(0,fs/2, fs/2/spectrogram.shape[1])

    plt.figure(figsize = (15, 7))
    plt.xlabel('Time (s)')
    plt.ylabel('Freq (Hz)')
    plt.pcolormesh(t, f, spectrogram.T)
    plt.show()


def extract_spectral_flux(X):    
    
    # X = compute_spectrogram(xb)
    # Compute spectral flux
    # Initialise blockNum and freqIndex
    n = 0
    k = 0

    spectral_flux = np.zeros(X.shape[0])

    for n in np.arange(X.shape[0]-1):
        flux_frame = 0
        for k in np.arange(X.shape[1]):
            flux = (abs(X[n+1, k]) - abs(X[n, k]))**2
            flux_frame += flux
        flux_frame = np.sqrt(flux_frame)/(X.shape[1]//2+1)
        spectral_flux[n] = flux_frame

    return spectral_flux

def half_wave_rectification(spectral_flux):
    envelope = np.max([spectral_flux, np.zeros_like(spectral_flux)], axis = 0)
    envelope = envelope/max(envelope)
    return envelope

def pick_onsets(envelope, thres):
    peaks = envelope[envelope>thres] # pick peaks ? 
    return peaks

def onset_detect(X, thres):
    spectral_flux = extract_spectral_flux(X)
    envelope = half_wave_rectification(spectral_flux)
    peaks = pick_onsets(envelope, thres)
    return peaks


# get silences from audio

def extract_rmsDb(xb, DB_TRUNCATION_THRESHOLD=-100):
    rmsDb = np.maximum(20 * np.log10(np.sqrt(np.mean(xb ** 2, axis=-1))), DB_TRUNCATION_THRESHOLD)
    return rmsDb


def create_voicing_mask(rmsDb, thresholdDb): 
    return 0 + (rmsDb >= thresholdDb)


def apply_voicing_mask(f0, mask):
    return f0 * mask

# get pitch chromagram 

# utility functions
def lowerBound(f_mid, Xsize, fs, notesPerOctave):
    return 2 ** (-1 / (2 * notesPerOctave)) * f_mid * 2 * (Xsize - 1) / fs


def upperBound(f_mid, Xsize, fs, notesPerOctave):
    return 2 ** (1 / (2 * notesPerOctave)) * f_mid * 2 * (Xsize - 1) / fs


## Mask generation function
def generate_mask(Xsize, fs, tfInHz):
    p = 48  # C3
    f_mid = tfInHz * 2 ** ((p - 69) / 12)
    numberOfOctaves = 3
    notesPerOctave = 12

    mask = np.zeros([notesPerOctave, Xsize])
    for i in range(0, notesPerOctave):
        bounds = np.array([lowerBound(f_mid, Xsize, fs, notesPerOctave), upperBound(f_mid, Xsize, fs, notesPerOctave)])
        for j in range(0, numberOfOctaves):
            noteLowerBound = np.ceil(2 ** j * bounds[0])
            noteUpperBound = np.ceil(2 ** j * bounds[1])
            diff = noteUpperBound - noteLowerBound
            # avoid division by zero
            if diff == 0:
                diff = 1
            mask[i, range(int(noteLowerBound), int(noteUpperBound))] = 1 / diff
        f_mid *= 2 ** (1 / notesPerOctave)
    return mask


def extract_pitch_chroma(X, fs, tfInHz):
    pitchChroma = np.zeros([12, X.shape[1]])
    mask = generate_mask(X.shape[0], fs, tfInHz)
    pitchChroma = np.dot(mask, X ** 2)

    # Normalize to length of 1
    norm = np.sqrt(np.square(pitchChroma).sum(axis=0, keepdims=True))
    pitchChroma = pitchChroma / norm

    return pitchChroma

