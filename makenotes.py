# convert onsets and f0 into notes 

# import libraries
import numpy as np


#Generate notes: 
def makeNotes(pitchChroma, onsetTimes, silenceMask):
    
    # find the pitch class from the chromagram:
    pitch = np.zeros(pitchChroma.shape[1])
    for block in pitchChroma.shape[1]:
        pitch[block] = np.argmax(pitchChroma[:, block])
    
    

    
    

        

    



# convert notes into midi
