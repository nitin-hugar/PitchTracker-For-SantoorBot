import pyaudio
import numpy as np
import rtmidi
from numpy import interp
from scipy.fftpack import fft
from scipy.signal import windows
import math
import aubio
import soundfile
import matplotlib.pyplot as plt
import librosa
from scipy.signal import medfilt, butter, lfilter, freqz
from midiutil.MidiFile import MIDIFile

# Initializations
silence_flag = False
pyaudio_format = pyaudio.paFloat32
n_channels = 1
samplerate = 44100
silenceThresholdindB = -40
win_s = 2048  # fft size
hop_s = 1024  # hop size
fs = 44100


def block_audio(x, blockSize, hopSize, fs):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])
    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs

    x = np.concatenate((x, np.zeros(blockSize)), axis=0)

    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])

        xb[n][np.arange(0, blockSize)] = x[np.arange(i_start, i_stop + 1)]

    return (xb, t)


def comp_acf(inputVector):
    r = np.correlate(inputVector, inputVector, "full")
    return r


def get_f0_from_acf(r, fs):
    first_index = np.argmax(r)
    threshold = 40
    second_index = np.argmax(r[first_index + threshold:])
    period_samples = second_index + threshold
    f0 = fs / period_samples
    return f0


def track_pitch_acf(x, blockSize, hopSize, fs, smoothing):
    xbs, timeInSec = block_audio(x, blockSize, hopSize, fs)
    all_f0 = np.array([])
    for block in xbs:
        r = comp_acf(block)
        freq = get_f0_from_acf(r, fs)
        all_f0 = np.append(all_f0, freq)
    return all_f0


def smoothing_filter(data, filter_duration, hop_s, fs):
    filter_size = int(filter_duration * fs / float(hop_s))
    if filter_size % 2 == 0:
        filter_size += 1
    smooth_pitches = medfilt(data, filter_size)
    return smooth_pitches


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def fourier(x):
    w = windows.hann(np.size(x))
    windowed = x * w
    w1 = int((x.size + 1) // 2)
    w2 = int(x.size / 2)
    fftans = np.zeros(x.size)
    fftans[0:w1] = windowed[w2:]  # Centre to make even function
    fftans[w2:] = windowed[0:w1]
    X = fft(fftans)
    magX = abs(X[0:int(x.size // 2 + 1)])
    return magX


def extract_spectral_flux(xb):
    magX = np.zeros((xb.shape[0], int(xb.shape[1] / 2 + 1)))
    flux = np.zeros(xb.shape[0])
    magX[0] = fourier(xb[0])
    for block in np.arange(1, xb.shape[0]):
        magX[block] = fourier(xb[block])
        den = magX[block].shape[0]
        flux[block] = np.sqrt(np.sum(np.square(magX[block] - magX[block - 1])))
    return flux


def get_onsets(x, threshold):
    xb, t = block_audio(x, win_s, hop_s, samplerate)
    flux = extract_spectral_flux(xb)
    # half_wave_rectification
    flux = np.max([flux, np.zeros_like(flux)], axis=0)
    flux = flux / max(flux)
    flux = np.where(flux < threshold, 0, flux)  # setting values less than threshold to zero

    return flux


def extract_spectral_crest(xb):
    magX = np.zeros((xb.shape[0], int(xb.shape[1] / 2 + 1)))
    spc = np.zeros(xb.shape[0])
    for block in range(xb.shape[0]):
        magX[block] = fourier(xb[block])
        summa = np.sum(magX[block], axis=0)
        if not summa:
            summa = 1
        spc[block] = np.max(magX[block]) / summa
    return spc


def extract_spectral_centroid(xb, fs):
    magX = np.zeros((xb.shape[0], int(xb.shape[1] / 2 + 1)))
    centroids = np.zeros(xb.shape[0])
    for block in range(xb.shape[0]):
        magX[block] = fourier(xb[block])
        N = magX[block].size
        den = np.sum(magX[block])
        if den == 0:
            den = 1
        centroid = 0
        for n in range(N):
            num = magX[block][n] * n
            centroid += num / den
        centroid = (centroid / (N - 1)) * fs / 2
        centroids[block] = centroid
    return centroids


def get_offset(x, threshold):
    xb, _ = block_audio(x, win_s, hop_s, samplerate)
    centroid = extract_spectral_centroid(xb, samplerate)
    #     crest = np.max([crest, np.zeros_like(crest)], axis = 0)
    centroid = centroid / max(centroid)
    centroid = np.where(centroid > threshold, 0, centroid)  # setting values greater than threshold to zero
    return centroid


def detect_sound_activity(audio_block, silenceThresholdindB):
    global silence_flag
    rms = np.sqrt(np.mean(np.square(audio_block)))
    dB = 20 * np.log10(rms)
    if dB < silenceThresholdindB:
        silence_flag = True
    else:
        silence_flag = False
    return silence_flag


def send_midi(channel, note, velocity):
    midiout = rtmidi.MidiOut()
    available_ports = midiout.get_ports()
    if available_ports:
        midiout.open_port(0)
    else:
        midiout.open_virtual_port("My virtual output")
    with midiout:
        note_on = [channel, note, velocity]
        midiout.send_message(note_on)
    del midiout


def freq2midi(freq):
    midi = 69 + 12 * np.log2(freq / 440)
    return midi


# melodia implementation
def hz2midi(hz):
    # convert from Hz to midi note
    hz_nonneg = hz.copy()
    idx = hz_nonneg <= 0
    hz_nonneg[idx] = 1
    midi = 69 + 12 * np.log2(hz_nonneg / 440.)
    midi[idx] = 0

    # round
    midi = np.round(midi)

    return midi


def scale_values(source_low, source_high, dest_low, dest_high, data):
    m = interp(data, [source_low, source_high], [dest_low, dest_high])
    return int(m)


# melodia implementation
def save_midi(outfile, notes, tempo):
    track = 0
    time = 0
    midifile = MIDIFile(1)

    # Add track name and tempo.
    midifile.addTrackName(track, time, "MIDI TRACK")
    midifile.addTempo(track, time, tempo)

    channel = 0
    volume = 100

    for note in notes:
        onset = note[0] * (tempo / 60.)
        duration = note[1] * (tempo / 60.)
        # duration = 1
        pitch = int(note[2])
        midifile.addNote(track, channel, pitch, onset, duration, volume)

    # And write it to disk.
    binfile = open(outfile, 'wb')
    midifile.writeFile(binfile)
    binfile.close()


def make_notes(midi, samplerate, hop_s, smooth, minduration):
    smoothed_midi = smoothing_filter(midi, smooth, hop_s, samplerate)
    notes = []
    p_prev = 0
    duration = 0
    onset = 0
    for n, p in enumerate(smoothed_midi):
        if p == p_prev:
            duration += 1
        else:
            # treat 0 as silence
            if p_prev > 0:
                # add note
                duration_sec = duration * hop_s / float(samplerate)
                # only add notes that are long enough
                if duration_sec >= minduration:
                    onset_sec = onset * hop_s / float(samplerate)
                    notes.append((onset_sec, duration_sec, p_prev))

            # start new note
            onset = n
            duration = 1
            p_prev = p

    # add last note
    if p_prev > 0:
        # add note
        duration_sec = duration * hop_s / float(samplerate)
        onset_sec = onset * hop_s / float(samplerate)
        notes.append((onset_sec, duration_sec, p_prev))

    print("Saving MIDI to disk...")
    save_midi("outfile.mid", notes, int(120))

    return notes


"""
Main Part Of Code
"""

# initialise pyaudio
p = pyaudio.PyAudio()

pitch_o = aubio.pitch("default", win_s, hop_s, samplerate)
pitch_o.set_unit("freq")
pitch_o.set_tolerance(0.4)

onset_o = aubio.onset("mkl", win_s, hop_s, samplerate)
onset_o.set_silence(-30.0)
onset_o.set_threshold(0.4)

# open stream
stream = p.open(format=pyaudio_format,
                channels=n_channels,
                rate=samplerate,
                input=True,
                frames_per_buffer=hop_s)

print("*** starting recording")
audio_block = np.array([], dtype=np.float32)
section_pitches = np.array([])
section_onsets = np.array([])
record_time = 5

# low pass filter initializations:
order = 3
cutOff = 1200  # Hz
notes = []
while True:
    try:
        # record a short phrase
        for i in range(0, int(samplerate / hop_s * record_time)):
            audiobuffer = stream.read(hop_s, exception_on_overflow=False)
            signal = np.frombuffer(audiobuffer, dtype=np.float32)
            audio_block = np.append(audio_block, signal)

        audio_block = butter_lowpass_filter(audio_block, cutOff, samplerate, order)

        pitches = track_pitch_acf(audio_block, win_s, hop_s, samplerate, smoothing=0.25)
        pitches[pitches > 400] = 0

        # get_onsets
        onsets = get_onsets(audio_block, 0.3)

        # # convert f0 to midi notes
        # midi_pitch = hz2midi(pitches)
        # notes.append(make_notes(midi_pitch, samplerate, hop_s, smooth=0.25 , minduration= 0.25))
        # print(notes)

        # plot audio, onsets
        pitches_plot = pitches / np.max(pitches)  # just to plot on the same scale as the audio.
        pitches_t = librosa.frames_to_time(range(len(pitches_plot)), sr=samplerate, hop_length=hop_s)
        onsets_t = librosa.frames_to_time(range(len(onsets)), sr=samplerate, hop_length=hop_s)
        frames_total = range(len(audio_block))
        time = librosa.frames_to_time(frames_total, sr=samplerate, hop_length=1)
        plt.plot(pitches_t, pitches_plot, 'g--', time, audio_block, onsets_t, onsets, 'r--')
        plt.show()

        # reinitiate audio_block
        audio_block = np.array([])

    except KeyboardInterrupt:
        print("***Ending Stream")
        break

stream.stop_stream()
stream.close()
p.terminate()

print("exporting audio...")
# soundfile.write(record, audio_block, samplerate)
print("done exporting!")
