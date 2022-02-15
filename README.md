# RT_Pitchtracker

In this project, we have implemented a call and response based pitch tracker for a robotic santoor player. 
The input audio stream is read in chunks of a specified time and then the onsets, and pitch are tracked from the audiofile. 

## Block Diagram

<img width="939" alt="image" src="https://user-images.githubusercontent.com/77855667/153995975-6602d7f9-0fef-4cb0-897c-a92f52d92bcc.png">

## Stream Audio 

1. Library used : Pyaudio
2. Record 2-3 seconds of audio
3. Send recorded audio for processing and santoor playback
4. Continue recording the next chunk while santoor plays back the notes
5. System works as a call and response accompanist

## Onset Detection 

<img width="941" alt="image" src="https://user-images.githubusercontent.com/77855667/153995210-67cff2c2-df10-40c0-b6fd-acef82a75c9d.png">

## Fundamental Frequency Detection 

<img width="938" alt="image" src="https://user-images.githubusercontent.com/77855667/153995282-8ea69ac7-0a90-4b04-afc0-a264f06c4abb.png">

## Silence Detection 

Steps: 
1. Extract energy per block 
2. Create voicing mask with threshold -40dB
3. Apply voicing mask to eliminate blocks with energy less than threshold

## Pitch Chroma

<img width="326" alt="image" src="https://user-images.githubusercontent.com/77855667/153995425-5c155475-3602-4143-9b47-bcef24f174b7.png">

Steps: 

1. Create Chroma template with midi values 48-60 owing to the physical constraints of the Santoor bot
2. Convert detected F0s to MIDI 
3. Create Pitch Chroma with respect to the given template


## Make Notes: 

1. Find Pitch class per block by picking the pitch with the maximum amplitude in the pitch chromagram
2. Create Note: Note is defined as the mode of pitch values between subsequent ‘onset’ blocks
3. Note Duration: Note duration is the difference in time between two subsequent onsets

## Santoor Playback

Two types of improvisations: 

1. Riz : triggered when the duration of the midi note is longer than 1 second with a 50% probability
2. Tekiyeh : triggered when the duration of the midi note is longer than 1 second with 25% probability of playing the higher note and 25% probability of playing the  lower note than the detected pitch.


