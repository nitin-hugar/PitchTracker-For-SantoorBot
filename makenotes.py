# convert onsets and f0 into notes 

# import libraries
import numpy as np
import scipy.stats


#Generate notes: 
def makeNotes(pitchChroma, onsetBlocks, init, hopSize, fs):
    # pitchChroma --> (12 X no.of blocks)
    # find the pitch class from the chromagram:
    
    pitches = np.zeros(pitchChroma.shape[1])
    for block in range(pitchChroma.shape[1]):
        pitches[block] = init + np.argmax(pitchChroma[:, block])
    
    notes = np.array([])
    for onset in range(onsetBlocks.shape[0] - 1):
        notes = np.append(notes, scipy.stats.mode(pitches[onsetBlocks[onset]: onsetBlocks[onset+1]][0][0]))
    
    durations = np.diff(onsetBlocks) * (hopSize / fs)
    
    return notes, durations
    

# convert notes into midi
