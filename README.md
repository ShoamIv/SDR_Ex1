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

## How It Works
### Encoding:

Each ASCII character (8 bits) is mapped to a set of frequencies. A bit value of 1 activates the corresponding frequency.

The encoder supports:

Character repetition (--repetitions) for redundancy.

Pause between characters for separation.

Fade-in/fade-out to avoid audio clicks.

### Decoding:

Continuously listens to microphone input.

Applies a Hanning window + FFT to detect present frequencies.

Converts detected frequencies back to binary states.

Stabilizes detection by requiring the same state to persist over several windows.

## Dependencies

pip install numpy pyaudio matplotlib scipy

sudo apt-get install portaudio19-dev python3-pyaudio

