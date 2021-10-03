import numpy


def stream_data(blockSize, hopSize, fs):
  return data

def f0_estimation(data, fs):
  return f0, timestamps

def onset_detection(data, fs):
  return onsets

def offset_detection(data, fs):
  return offsets

def detect_silence(data, fs):   ## Need this to see if music is actually playing. 
  return silence_flag

def energy_estimate(data, fs):
  return energies

def quantize_pitches(f0, timestamps, scale):
  return quantized_pitches

def get_note_durations(onsets,offsets):
  return note_durations

def make_notes():
  return notes, durations, velocities, timestamps

def convert_to_midi(notes, durations, velocities, timestamps): # timestamps may or may not be needed depending on the audio buffer
  return midi_message


def make_awesome(midi_message):
  """
  Looks for long notes and sends articulation messages to the santoor bot. 
  Adds phrases when long notes are detected.
  """
  return awesome_santoor_messages
