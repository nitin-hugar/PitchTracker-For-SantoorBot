# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import mido
import random
import time
import matrix
print(mido.get_output_names())
outport = mido.open_output('Network Forest')

#### initialze
note = [76, 77, 79, 81, 83, 84, 86, 88, 89]

# def CreateMidiNote(NoteIndex, TimeIndex, velocity):
#     tick=192
#     midMsg = mido.Message('note_on', channel=5, note=note[NoteIndex], velocity=velocity, time=int(mido.second2tick(NoteTime[TimeIndex],tick,500000)))
#     # track.append(midMsg)
#     # track.append(mido.Message('note_off',channel=5, note=note[NoteIndex], velocity=velocity, time=int(mido.second2tick(NoteTime[TimeIndex],tick,500000))))
#     print("*********",mido.second2tick(NoteTime[TimeIndex],tick,500000),'**********')
#     return (midMsg)

def CreateMidiNote(NoteIndex, sec, velocity):
    tick = 192
    midMsg = mido.Message('note_on', channel=5, note=note[NoteIndex], velocity=velocity, time=int(mido.second2tick(sec,tick,500000)))
    # track.append(midMsg)
    # track.append(mido.Message('note_off', channel=5, note=note[NoteIndex], velocity=velocity, time=0))
    # print("*********",mido.second2tick(sec,192,500000),'**********')
    return (midMsg)
#scale = np.array([48, 50, 51, 53, 55, 56, 58, 60])
def Riz(note, dur):
    rizNum = 10
    timRiz = dur / rizNum
    for i in range(rizNum):
        velocity = 50 + i * 6
        outport.send(CreateMidiNote(note, dur, velocity))
        print('Riz')
        time.sleep(timRiz)
    return

def motif1(note, dur):

    dur = dur/3
    if note != 8:
        outport.send(CreateMidiNote(note, dur, random.randrange(65,110,4)))
        time.sleep(dur)
        outport.send(CreateMidiNote(note+1, dur, random.randrange(65, 110, 4)))
        time.sleep(dur)
        outport.send(CreateMidiNote(note, dur, random.randrange(65, 110, 4)))
        time.sleep(dur)
    else:
        outport.send(CreateMidiNote(note, dur, random.randrange(65, 110, 4)))

    return

def motif2(note, dur):

    dur = dur/3
    if note != 0:
        outport.send(CreateMidiNote(note, dur, random.randrange(65,110,4)))
        time.sleep(dur)
        outport.send(CreateMidiNote(note-1, dur, random.randrange(65, 110, 4)))
        time.sleep(dur)
        outport.send(CreateMidiNote(note, dur, random.randrange(65, 110, 4)))
        time.sleep(dur)
    else:
        outport.send(CreateMidiNote(note, dur, random.randrange(65, 110, 4)))

    return

def SantoorBot(notes,duration):
    """

    :param notes:  Index 0-8
    :param duration: in Sec
    :return:
    """
    for i in range(notes):
        dur=duration[i]
        note=notes[i]
        if duration > 1:
            Riz(note, dur)

        else:

            outport.send(CreateMidiNote(note, dur, random.randrange(65,110,4)))
            time.sleep(dur)

    return

def SantoorBotPlusMotifs(notes,duration):
    """

    :param notes:  Index 0-8
    :param duration: in Sec
    :return:
    """
    for i in range(notes):
        dur=duration[i]
        note=notes[i]
        if duration > 1:
            n = np.random.choice(3, 1, p=[0.5, 0.25, 0.25])
            if n == 1:
                Riz(note, dur)
            elif n == 2:
                motif1(note, dur)
            else:
                motif2(note, dur)
        else:

            outport.send(CreateMidiNote(note, dur, random.randrange(65,110,4)))
            time.sleep(dur/3)


    return
outport.close()