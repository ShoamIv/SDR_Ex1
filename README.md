 # Audio Modem: Frequency-Based ASCII Communication in Python
 This project implements a software-based acoustic modem using PyAudio, NumPy, and SciPy. 
 It encodes ASCII characters as audio tones using multiple simultaneous sine waves (representing bits) and decodes them using FFT-based frequency detection.

## Features
Encodes ASCII characters as multi-tone sine waves.

Each bit of the character is mapped to a distinct frequency.

Plays encoded audio directly via speakers.

Saves encoded messages to .wav files.

Decodes real-time audio from microphone input using FFT.

Decodes from saved .wav files.

Supports redundancy and signal stability for improved decoding.

## How It Works
### Encoding:

Each ASCII character (8 bits) is mapped to a set of frequencies. A bit value of 1 activates the corresponding frequency.

The encoder supports:

Character repetition (--repetitions) for redundancy.

Pause between characters for separation.

### Decoding:

Continuously listens to microphone input.

Applies a Hanning window + FFT to detect present frequencies.

Converts detected frequencies back to binary states.

Stabilizes detection by requiring the same state to persist over several windows.

## Dependencies

pip install numpy pyaudio matplotlib scipy

sudo apt-get install portaudio19-dev python3-pyaudio

## Encode and Play Text

### Encode live audio

 python audio_modem.py encode "example"

### Encode and save to .wav file

python audio_modem.py encode "example" --save example.wav

## Decode

### Deco Live Audio (Microphone Input)

python audio_modem.py decode

### Decode from .wav file

python audio_modem.py decode --file example.wav

## Optinal Flags 

### Encoding Flags

#### --active-duration (float):

Duration (seconds) of each character's audio tone (default: 0.1s).

Increase for better decoding at the cost of slower transmission.

#### --pause-duration (float):

Silence duration (seconds) between characters (default: 0.02s).

Increase if decoding misses characters.

#### --repetitions (int):

How many times each character is repeated (default: 5).

Higher values improve reliability but slow transmission.

#### --save (filename):

Save to a WAV file instead of playing live (e.g., --save output.wav).

### Decoding Flags:

#### --file (filename):

Decode from a WAV file instead of microphone (e.g., --file output.wav).

#### --duration (int):

Seconds to listen via microphone (default: 10). Ignored if --file is used.

#### --threshold (float):

Frequency detection sensitivity (0.0–1.0, default: 0.15).

Lower values detect weaker signals but may catch noise.

#### --duplicates (int):

Required consecutive detections before accepting a character (default: 3).

Increase to reduce errors (recommended: 3–5).

#### --show-spectrogram:

Visualize frequencies (requires matplotlib).

#### --debug:

Show detailed detection logs.
