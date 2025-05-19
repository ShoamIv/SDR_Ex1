#!/usr/bin/env python3

import sys
import math
import sounddevice as sd
import numpy as np
from bitarray import bitarray
import zlib
import wave
import os

# Sampling parameters for audio generation
samplerate = 44100    # Number of audio samples per second
amplitude = 2000      # Amplitude (volume) of the generated sine waves

# Read command-line arguments for bit rate and frequencies
if len(sys.argv) < 4:
    print("Usage: python transmitter.py <rate (Hz)> <freq0> <freq1>")
    sys.exit(1)

time = float(1 / float(sys.argv[1]))  # Duration of each bit in seconds (bit period)
freq0 = float(sys.argv[2])            # Frequency to represent binary '0'
freq1 = float(sys.argv[3])            # Frequency to represent binary '1'

# 4b5b encoding table: maps 4-bit nibbles to 5-bit codes for DC balance & error detection
tocode = {
    '0000': '11110', '0001': '01001', '0010': '10100', '0011': '10101',
    '0100': '01010', '0101': '01011', '0110': '01110', '0111': '01111',
    '1000': '10010', '1001': '10011', '1010': '10110', '1011': '10111',
    '1100': '11010', '1101': '11011', '1110': '11100', '1111': '11101'
}

def code4b5b(code):
    """
    Encode a binary string using 4b5b encoding.
    Converts each 4-bit chunk into a 5-bit code.
    """
    output = ""
    while len(code):
        output += tocode[code[:4]]  # Map first 4 bits to 5 bits
        code = code[4:]
    return output

def codenrz(code):
    """
    Encode the given binary string using NRZ-I (Non-Return-to-Zero Inverted).
    This flips the signal state when the bit is 1, otherwise keeps it the same.
    This ensures signal transitions for better clock recovery.
    """
    output = ""
    prev = 1
    for cur in code:
        if cur == "1":
            prev = (prev + 1) % 2  # Toggle between 0 and 1 on '1' bits
        output += str(prev)
    return output

def encoding(to, fromwho, msg):
    """
    Encode the message with addressing, length, CRC, 4b5b, and NRZ-I encoding.

    - to: destination address (integer, converted to 48 bits)
    - fromwho: source address (integer, converted to 48 bits)
    - msg: ASCII message string

    Output is a binary string ready for modulation.
    """
    output = bin(int(fromwho))[2:].zfill(48)  # Source address: 48 bits zero-padded
    output += bin(int(to))[2:].zfill(48)      # Destination address: 48 bits zero-padded
    output += bin(len(msg))[2:].zfill(16)     # Length of message in bytes: 16 bits

    # Convert message characters to their 8-bit ASCII binary form
    for ch in msg:
        output += bin(ord(ch))[2:].zfill(8)

    # Compute CRC32 checksum on the bitarray of output for error detection
    crc = zlib.crc32(bitarray(output).tobytes()) & 0xffffffff
    output += bin(crc)[2:].zfill(32)  # Append 32-bit CRC checksum

    # Encode the entire bitstream with 4b5b and NRZ-I
    output = code4b5b(output)
    output = codenrz(output)

    # Add a 64-bit preamble (alternating bits ending with '11') for receiver sync
    output = "1010101010101010101010101010101010101010101010101010101010101011" + output
    return output

def generplayandwav(code, filename="output.wav"):
    """
    Generate the modulated audio wave for the given binary code,
    play it via the sound card, and save it as a WAV file.

    - code: string of '0' and '1' representing the bitstream
    - filename: output WAV filename
    """
    duration_samples = int(time * samplerate)  # Number of samples per bit
    t = np.linspace(0, time, duration_samples, endpoint=False)  # Time vector for one bit

    # Generate sine waves for 0 and 1 bits
    wave0 = (np.sin(2 * math.pi * freq0 * t) * amplitude).astype(np.int16)
    wave1 = (np.sin(2 * math.pi * freq1 * t) * amplitude).astype(np.int16)

    # Create the full audio by concatenating wave0 or wave1 based on bits
    samples = [wave0 if bit == '0' else wave1 for bit in code]
    full_wave = np.concatenate(samples)

    # Play the generated sound
    sd.play(full_wave, samplerate=samplerate)
    sd.wait()  # Wait until playback is done

    # Save the audio to a WAV file for later playback or processing
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)          # Mono audio
        wf.setsampwidth(2)          # 2 bytes per sample (16-bit audio)
        wf.setframerate(samplerate) # Sample rate (Hz)
        wf.writeframes(full_wave.tobytes())  # Write audio data

    print(f"Played and saved WAV to: {filename}")

def main():
    """
    Main program: reads input line with 'TO FROM MESSAGE', encodes and plays message.
    """
    print("Enter message as: <TO> <FROM> <MESSAGE>")
    try:
        line = input().strip()
        if not line:
            return
        to, fromwho, msg = line.split(" ", 2)  # Split input line into parts
        bitstream = encoding(to, fromwho, msg)  # Encode to bitstream
        print(bitstream)
        generplayandwav(bitstream)               # Generate and play WAV
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
