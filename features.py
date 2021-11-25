import scipy
import numpy as np
import matplotlib.pyplot as plt



# get onsets from audio 

# get f0 from audio 

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

