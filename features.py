import librosa
import numpy as np
import matplotlib.pyplot as plt


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
        tmp = abs(sp.fft(xb[n,:] * afWindow))*2/xb.shape[1]
    
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
        flux_frame = np.sqrt(flux_frame)/(xb.shape[1]//2+1)
        spectral_flux[n] = flux_frame

    return spectral_flux

def half_wave_rectification(spectral_flux):
    envelope = np.max([spectral_flux, np.zeros_like(spectral_flux)], axis = 0)
    envelope = envelope/max(envelope)
    return envelope

def pick_onsets(envelope, thres):
    peaks = envelope[envelope>thres]
    return peaks

def onset_detect(X, thres):
    spectral_flux = extract_spectral_flux(X)
    envelope = half_wave_rectification(spectral_flux)
    peaks = pick_onsets(envelope, thres)
    return (envelope)