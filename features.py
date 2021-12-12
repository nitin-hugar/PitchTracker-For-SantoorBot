import scipy
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.fft import fft
import utilities


def compute_spectrogram(xb):
    numBlocks = xb.shape[0]
    afWindow = utilities.compute_hann(xb.shape[1])
    X = np.zeros([math.ceil(xb.shape[1] / 2 + 1), numBlocks])

    for n in range(0, numBlocks):
        # apply window
        tmp = abs(scipy.fftpack.fft(xb[n, :] * afWindow)) * 2 / xb.shape[1]

        # compute magnitude spectrum
        X[:, n] = tmp[range(math.ceil(tmp.size / 2 + 1))]
        X[[0, math.ceil(tmp.size / 2)], n] = X[[0, math.ceil(tmp.size / 2)], n] / np.sqrt(2)
        # let's be pedantic about normalization
    return X.T


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

def onset_smoothening(envelope, n):
    # n: filter length - odd
    fltr = np.ones(n)
    envelope = np.append(np.zeros(n//2), envelope)
    envelope = np.append(envelope, np.zeros(n//2))
    filtered_envelope = np.zeros(len(envelope))
    
    for i in np.arange(n//2, len(envelope)-n//2):
        block = envelope[i-n//2:i+n//2+1]
        avg = np.dot(block, fltr)/n
        filtered_envelope[i-n//2] = envelope[i]-avg
        
    return filtered_envelope

def half_wave_rectification(spectral_flux):
    envelope = np.max([spectral_flux, np.zeros_like(spectral_flux)], axis = 0)
    envelope = envelope/max(envelope)
    return envelope

def pick_onsets(envelope, thres):
    peaks = np.where(envelope>thres)
    return peaks

def onset_detect(X, thres, n=5):
    # n = moving average filter for smoothening the envelope
    spectral_flux = extract_spectral_flux(X)
    smoothened_envelope = onset_smoothening(spectral_flux, n)
    hwr_envelope = half_wave_rectification(smoothened_envelope)
    print (max(hwr_envelope))
    norm_envelope = hwr_envelope/max(hwr_envelope)
    print (max(norm_envelope))
    peaks = pick_onsets(norm_envelope, thres)
    return peaks

# get silences from audio
def extract_rmsDb(xb, DB_TRUNCATION_THRESHOLD=-100):
    print(np.all((xb==0)))
    rmsDb = np.maximum(20 * np.log10(np.sqrt(np.mean(xb ** 2, axis=-1))), DB_TRUNCATION_THRESHOLD)  #need to handle zero blocks
    return rmsDb


def create_voicing_mask(rmsDb, thresholdDb): 
    return 0 + (rmsDb >= thresholdDb)


def apply_voicing_mask(f0, mask):
    return f0 * mask

def detect_silence(xb, f0, thres_dB):
    rmsDb = extract_rmsDb(xb, DB_TRUNCATION_THRESHOLD=-100)
    mask = create_voicing_mask(rmsDb, thres_dB)
    f0f = apply_voicing_mask(f0, mask)
    return f0f

# get pitch chromagram 

# utility functions
def lowerBound(f_mid, Xsize, fs, notesPerOctave):
    return 2 ** (-1 / (2 * notesPerOctave)) * f_mid * 2 * (Xsize - 1) / fs


def upperBound(f_mid, Xsize, fs, notesPerOctave):
    return 2 ** (1 / (2 * notesPerOctave)) * f_mid * 2 * (Xsize - 1) / fs

# Pitch Chroma
def extract_pitch_chroma(f0c):
    init = 48 #C3
    pitch_classes = np.arange(init, init+12)

    # scale = np.array([48, 50, 51, 53, 55, 56, 58, 60]) # C_MINOR
    pitchChroma = np.zeros([12, f0c.shape[0]])
    midi = utilities.hz2midi(f0c)
    
    tmp = midi-init
    
    for i in range(tmp.shape[0]):
        if tmp[i] == -48:
            pitchChroma[:, i] == 0
        if (tmp[i] >= 0) and (tmp[i] < 12):
            pitchChroma[int(tmp[i])-1, i] = 1      # Velocity value #tmp-1
        elif tmp[i] >= 12:
            val = tmp[i] - (int(tmp[i]/12)*12)      #changed tmp to val since there was a def error
            pitchChroma[int(val)-1, i] = 1         # Velocity value 
        elif tmp[i] < 0:
            val = tmp[i] + ((1+int(np.abs(tmp[i]/12)))*12)
            pitchChroma[int(val)-1, i] = 1         # Velocity value
        
    return pitchChroma


def dft(x, w, N):
    """
    Analysis of a signal using the discrete Fourier transform
    x: input signal, w: analysis window, N: FFT size
    returns mX, pX: magnitude and phase spectrum
    """

    hN = (N // 2) + 1                                           # size of positive spectrum, it includes sample 0
    hM1 = (w.size + 1) // 2                                     # half analysis window size by rounding
    hM2 = w.size // 2                                         # half analysis window size by floor
    fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
    w = w / sum(w)                                          # normalize analysis window
    xw = x * w                                                # window the input sound
    fftbuffer[:hM1] = xw[hM2:]                              # zero-phase window in fftbuffer
    fftbuffer[-hM2:] = xw[:hM2]
    X = fft(fftbuffer)                                      # compute FFT
    absX = abs(X[:hN])                                      # compute ansolute value of positive side
    absX[absX < np.finfo(float).eps] = np.finfo(float).eps    # if zeros add epsilon to handle log
    mX = 20 * np.log10(absX)                                # magnitude spectrum of positive frequencies in dB
    X[:hN].real[np.abs(X[:hN].real) < 1e-14] = 0.0            # for phase calculation set to 0 the small values
    X[:hN].imag[np.abs(X[:hN].imag) < 1e-14] = 0.0            # for phase calculation set to 0 the small values
    pX = np.unwrap(np.angle(X[:hN]))                        # unwrapped phase spectrum of positive frequencies
    return mX, pX

def peakDetection(mX, t):
    """
    Detect spectral peak locations
    mX: magnitude spectrum, t: threshold
    returns ploc: peak locations
    """

    thresh = np.where(np.greater(mX[1:-1], t), mX[1:-1], 0);  # locations above threshold
    next_minor = np.where(mX[1:-1] > mX[2:], mX[1:-1], 0)     # locations higher than the next one
    prev_minor = np.where(mX[1:-1] > mX[:-2], mX[1:-1], 0)    # locations higher than the previous one
    ploc = thresh * next_minor * prev_minor                 # locations fulfilling the three criteria
    ploc = ploc.nonzero()[0] + 1                            # add 1 to compensate for previous steps
    return ploc

def peakInterp(mX, pX, ploc):
    """
    Interpolate peak values using parabolic interpolation
    mX, pX: magnitude and phase spectrum, ploc: locations of peaks
    returns iploc, ipmag, ipphase: interpolated peak location, magnitude and phase values
    """

    val = mX[ploc]                                          # magnitude of peak bin
    lval = mX[ploc - 1]                                       # magnitude of bin at left
    rval = mX[ploc + 1]                                       # magnitude of bin at right
    iploc = ploc + 0.5 * (lval - rval) / (lval - 2 * val + rval)        # center of parabola
    ipmag = val - 0.25 * (lval - rval) * (iploc - ploc)             # magnitude of peaks
    ipphase = np.interp(iploc, np.arange(0, pX.size), pX)   # phase of peaks by linear interpolation
    return iploc, ipmag, ipphase

def TWM(pfreq, pmag, f0c):
    """

    pfreq, pmag: peak frequencies in Hz and magnitudes,
    f0c: frequencies of f0 candidates
    returns f0, f0Error: fundamental frequency detected and its error
    """

    p = 0.5                                          # weighting by frequency value
    q = 1.4                                          # weighting related to magnitude of peaks
    r = 0.5                                          # scaling related to magnitude of peaks
    rho = 0.33                                       # weighting of MP error
    Amax = max(pmag)                                 # maximum peak magnitude
    maxnpeaks = 10                                   # maximum number of peaks used
    harmonic = np.matrix(f0c)
    ErrorPM = np.zeros(harmonic.size)                # initialize PM errors
    MaxNPM = min(maxnpeaks, pfreq.size)
    for i in range(0, MaxNPM):                      # predicted to measured mismatch error
        difmatrixPM = harmonic.T * np.ones(pfreq.size)
        difmatrixPM = abs(difmatrixPM - np.ones((harmonic.size, 1)) * pfreq)
        FreqDistance = np.amin(difmatrixPM, axis=1)    # minimum along rows
        peakloc = np.argmin(difmatrixPM, axis=1)
        Ponddif = np.array(FreqDistance) * (np.array(harmonic.T)**(-p))
        PeakMag = pmag[peakloc]
        MagFactor = 10**((PeakMag - Amax) / 20)
        ErrorPM = ErrorPM + (Ponddif + MagFactor * (q * Ponddif - r)).T
        harmonic = harmonic + f0c

    ErrorMP = np.zeros(harmonic.size)                # initialize MP errors
    MaxNMP = min(maxnpeaks, pfreq.size)
    for i in range(0, f0c.size):                    # measured to predicted mismatch error
        nharm = np.round(pfreq[:MaxNMP] / f0c[i])
        nharm = (nharm >= 1) * nharm + (nharm < 1)
        FreqDistance = abs(pfreq[:MaxNMP] - nharm * f0c[i])
        Ponddif = FreqDistance * (pfreq[:MaxNMP]**(-p))
        PeakMag = pmag[:MaxNMP]
        MagFactor = 10**((PeakMag - Amax) / 20)
        ErrorMP[i] = sum(MagFactor * (Ponddif + MagFactor * (q * Ponddif - r)))
    Error = (ErrorPM[0] / MaxNPM) + (rho * ErrorMP / MaxNMP)  # total error
    f0c_shortlist = np.array([])
    Error_shortlist = np.array([])    
    if len(Error) == 1:
#         print (Error)
        f0c_shortlist = np.append(f0c_shortlist, f0c)
        Error_shortlist = np.append(Error_shortlist, Error)
    if len(Error) >= 2: 
#         print (Error)
#         for i in range(10):                                   # get candidates with smallest error
        f0index = np.argmin(Error)                       # get the smallest error
#         print (f0index)
        Error1 = Error[f0index]
#         print (Error1)
        f01 = f0c[f0index]                                # f0 with the smallest error
#         print (f01)
        f0c_shortlist = np.append(f0c_shortlist, f01)
        Error_shortlist = np.append(Error_shortlist, Error1)
        f0c = np.delete(f0c, f0index)
        Error = np.delete(Error, f0index)
    
    return f0c_shortlist, Error_shortlist

def TWM_errors(pfreq, pmag, f0c):
    """
    Two-way mismatch algorithm for f0 detection (by Beauchamp&Maher)
    pfreq, pmag: peak frequencies in Hz and magnitudes,
    f0c: frequencies of f0 candidates
    returns f0, f0Error: fundamental frequency detected and its error
    """

    p = 0.5                                          # weighting by frequency value
    q = 1.4                                          # weighting related to magnitude of peaks
    r = 0.5                                          # scaling related to magnitude of peaks
    rho = 0.33                                       # weighting of MP error
    Amax = max(pmag)                                 # maximum peak magnitude
    maxnpeaks = 10                                   # maximum number of peaks used
    harmonic = np.matrix(f0c)
    ErrorPM = np.zeros(harmonic.size)                # initialize PM errors
    MaxNPM = min(maxnpeaks, pfreq.size)
    for i in range(0, MaxNPM):                      # predicted to measured mismatch error
        difmatrixPM = harmonic.T * np.ones(pfreq.size)
        difmatrixPM = abs(difmatrixPM - np.ones((harmonic.size, 1)) * pfreq)
        FreqDistance = np.amin(difmatrixPM, axis=1)    # minimum along rows
        peakloc = np.argmin(difmatrixPM, axis=1)
        Ponddif = np.array(FreqDistance) * (np.array(harmonic.T)**(-p))
        PeakMag = pmag[peakloc]
        MagFactor = 10**((PeakMag - Amax) / 20)
        ErrorPM = ErrorPM + (Ponddif + MagFactor * (q * Ponddif - r)).T
        harmonic = harmonic + f0c

    ErrorMP = np.zeros(harmonic.size)                # initialize MP errors
    MaxNMP = min(maxnpeaks, pfreq.size)
    for i in range(0, f0c.size):                    # measured to predicted mismatch error
        nharm = np.round(pfreq[:MaxNMP] / f0c[i])
        nharm = (nharm >= 1) * nharm + (nharm < 1)
        FreqDistance = abs(pfreq[:MaxNMP] - nharm * f0c[i])
        Ponddif = FreqDistance * (pfreq[:MaxNMP]**(-p))
        PeakMag = pmag[:MaxNMP]
        MagFactor = 10**((PeakMag - Amax) / 20)
        ErrorMP[i] = sum(MagFactor * (Ponddif + MagFactor * (q * Ponddif - r)))

    Error = (ErrorPM[0] / MaxNPM) + (rho * ErrorMP / MaxNMP)  # total error
    
    return Error

def f0_detection_TWM(xb, w, blockSize, t, f0min, f0max, fs):
    
    num_blocks = xb.shape[0]
    f0cf = np.zeros(num_blocks)
    f0Errors = np.zeros(num_blocks)
    for i, block in enumerate(xb):
#     print (i)
        mX, pX = dft(block, w, blockSize)
        ploc = peakDetection(mX, t)                            # detect peak locations
        iploc, ipmag, ipphase = peakInterp(mX, pX, ploc)
    
        if len(ipmag) != 0:
            ipfreq = fs*iploc/blockSize
            f0c = np.argwhere((ipfreq>f0min) & (ipfreq<f0max))[:, 0]
            f0cf_block = ipfreq[f0c]
            f0c, Error = TWM(ipfreq, ipmag, f0cf_block)
            f0cf[i] = f0c
            f0Errors[i] = Error
    
        if len(ipmag) == 0:
            f0c = 0     # set to zero to account for definition error
            Error = 0
            f0cf[i] = f0c
            f0Errors[i] = Error
            
    return f0cf, f0Errors

