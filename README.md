 # Audio Modem: Frequency-Based ASCII Communication in Python
 This project implements a software-based acoustic modem using PyAudio, NumPy, and SciPy. 
 It encodes ASCII characters as audio tones using multiple simultaneous sine waves (representing bits) and decodes them using FFT-based frequency detection.

## Features
Encodes ASCII characters as multi-tone sine waves.

Each bit of the character is mapped to a distinct frequency.

Plays encoded audio directly via speakers.

Saves encoded messages to .wav files.

Decodes real-time audio from microphone input using FFT.

Supports redundancy and signal stability for improved decoding.
