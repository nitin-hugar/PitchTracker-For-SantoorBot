# convert onsets and f0 into notes 

# import libraries
import numpy as np
import scipy.stats
import rtmidi
import time

#Generate notes: 
def makeNotes(pitchChroma, onsetBlocks, init, hopSize, fs):
    # pitchChroma --> (12 X no.of blocks)
    # find the pitch class from the chromagram:
    
    pitches = np.zeros(pitchChroma.shape[1], dtype= np.int32)
    for block in range(pitchChroma.shape[1]):
        pitches[block] = init + np.argmax(pitchChroma[:, block])
    
    notes = np.array([], dtype=np.int32)
    for onset in range(onsetBlocks.shape[0] - 1):
        notes = np.append(notes, scipy.stats.mode(pitches[onsetBlocks[onset]: onsetBlocks[onset+1]])[0][0])
    
    durations = np.diff(onsetBlocks) * (fs / hopSize)
    
    return np.array([notes, durations])
    

# function to send midi out through port
def send_midi(channel, note, velocity=80):
    midiout = rtmidi.MidiOut()
    available_ports = midiout.get_ports()
    print(available_ports)
    if available_ports:
        midiout.open_port(1)
    else:
        midiout.open_virtual_port("My virtual output")
    
    with midiout:
        note_on = [channel, note, velocity]
        midiout.send_message(note_on)
    del midiout


# function to send notes as midi messages 
def midiPlayBack(channel, notes, durations):
    for note, duration in zip(notes, durations):
        print(int(note))
        send_midi(channel, int(note), velocity=100) # Omit when no note or make velocity = 0 for sending any note.
        time.sleep(duration)
