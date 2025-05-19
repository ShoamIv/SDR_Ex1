#!/usr/bin/env python3

import sys
import numpy as np
import sounddevice as sd
import wave
import os

nchannels = 1
samplerate = 44100
time = float(1 / float(sys.argv[1]))
freq0 = float(sys.argv[2])
freq1 = float(sys.argv[3])

todecode = {
    '11110': '0000', '01001': '0001', '10100': '0010', '10101': '0011',
    '01010': '0100', '01011': '0101', '01110': '0110', '01111': '0111',
    '10010': '1000', '10011': '1001', '10110': '1010', '10111': '1011',
    '11010': '1100', '11011': '1101', '11100': '1110', '11101': '1111'
}

def dominatfreq(buffer):
    if len(buffer) == 0:
        return -1
    fft_result = np.fft.fft(buffer)
    half_fft = fft_result[:len(fft_result)//2]
    freqs = np.fft.fftfreq(len(buffer), 1/samplerate)[:len(half_fft)]
    idx = np.argmax(np.abs(half_fft))
    return abs(freqs[idx])

def decodenrz(code):
    prev = 1
    output = ""
    for c in code:
        cur = int(c)
        if cur != prev:
            output += "1"
        else:
            output += "0"
        prev = cur
    return output

def decode4b5b(code):
    output = ""
    for i in range(0, len(code), 5):
        five_bit = code[i:i+5]
        if five_bit in todecode:
            output += todecode[five_bit]
        else:
            print(f"Warning: invalid 5-bit code '{five_bit}'")
            output += "0000"
    return output

def decode(code):
    out = decodenrz(code)
    output = decode4b5b(out)
    output1 = str(int(output[48:96], 2)) + " "
    output1 += str(int(output[:48], 2)) + " "
    endofmsg = 112 + 8 * int(output[96:112], 2)
    msg_bin = output[112:endofmsg]
    msg_bytes = int(msg_bin, 2).to_bytes(len(msg_bin)//8, byteorder='big')
    output1 += msg_bytes.decode('utf-8', errors='replace')
    return output1

def read_blocks_from_wav(wavfile):
    wf = wave.open(wavfile, 'rb')
    block_size = int(time * wf.getframerate())
    while True:
        frames = wf.readframes(block_size)
        if len(frames) < block_size * wf.getsampwidth():
            break
        buffer = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        yield buffer

def main():
    todecodein = ""
    wavfile = None

    if len(sys.argv) >= 5:
        wavfile = sys.argv[4]
        if not os.path.isfile(wavfile):
            print(f"Error: WAV file '{wavfile}' not found.")
            sys.exit(1)

    if wavfile:
        print(f"Reading from WAV file: {wavfile}")

        for buffer in read_blocks_from_wav(wavfile):
            freq = dominatfreq(buffer)
            if freq == -1:
                break
            if abs(freq - freq0) < 100:
                todecodein += '0'
            elif abs(freq - freq1) < 100:
                todecodein += '1'
        if todecodein:
            try:
                print("Decoded message:", decode(todecodein))
            except Exception as e:
                print(f"Decoding error: {e}")
        else:
            print("No data decoded.")
    else:
        block_size = int(time * samplerate)
        with sd.InputStream(samplerate=samplerate, channels=nchannels, dtype='float32', blocksize=block_size) as stream:
            print("Listening on microphone... Press Ctrl+C to stop.")
            try:
                while True:
                    buffer, overflowed = stream.read(block_size)
                    buffer = buffer.flatten()
                    freq = dominatfreq(buffer)
                    if freq == -1:
                        continue
                    if abs(freq - freq0) < 100:
                        todecodein += '0'
                    elif abs(freq - freq1) < 100:
                        todecodein += '1'
            except KeyboardInterrupt:
                print("\nStopped.")
            if todecodein:
                try:
                    print("Decoded message:", decode(todecodein))
                except Exception as e:
                    print(f"Decoding error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python receiver.py <time_divisor> <freq0> <freq1> [wavfile]")
        sys.exit(1)
    main()
