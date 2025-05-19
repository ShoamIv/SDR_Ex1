import numpy as np
import sounddevice as sd
import sys

samplerate = 48000
blocksize = 480  # ~10ms at 48kHz
device = 5  # Replace with your actual monitor input index

time = 1 / float(sys.argv[1])
freq0 = float(sys.argv[2])
freq1 = float(sys.argv[3])

def dominatfreq(audio_chunk):
    fft = np.fft.fft(audio_chunk)
    freqs = np.fft.fftfreq(len(audio_chunk), d=1/samplerate)
    magnitude = np.abs(fft)
    positive_freqs = freqs[:len(freqs)//2]
    magnitude = magnitude[:len(magnitude)//2]
    return positive_freqs[np.argmax(magnitude)]

def record_and_decode():
    print("Listening...")

    code = ""
    try:
        with sd.InputStream(samplerate=samplerate, channels=1, device=device, dtype='float32') as stream:
            while True:
                audio_chunk, _ = stream.read(blocksize)
                audio_chunk = audio_chunk[:, 0]  # Mono
                freq = dominatfreq(audio_chunk)

                if abs(freq - freq0) < 100:
                    code += '0'
                elif abs(freq - freq1) < 100:
                    code += '1'
                else:
                    if code:
                        print("Raw code:", code)
                        try:
                            # Call your decode function here
                            print("Decoded:", decode(code))
                        except Exception as e:
                            print("Decoding error:", e)
                        code = ""
    except KeyboardInterrupt:
        print("Stopped.")
