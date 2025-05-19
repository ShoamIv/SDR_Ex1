# !/usr/bin/env python


import sys
import math
import sounddevice as sd
import numpy as np
from bitarray import bitarray
import zlib
import wave

amplitude = 2000
time = float(1 / float(sys.argv[1]))
freq0 = float(sys.argv[2])
freq1 = float(sys.argv[3])

tocode = {
    '0000': '11110',
    '0001': '01001',
    '0010': '10100',
    '0011': '10101',
    '0100': '01010',
    '0101': '01011',
    '0110': '01110',
    '0111': '01111',
    '1000': '10010',
    '1001': '10011',
    '1010': '10110',
    '1011': '10111',
    '1100': '11010',
    '1101': '11011',
    '1110': '11100',
    '1111': '11101'
}


def code4b5b(code):
    output = ""
    while (len(code)):
        output += tocode[code[:4]]
        code = code[4:]
    return output


def codenrz(code):
    output = ""
    prev = 1
    while (len(code)):
        cur = code[:1]
        code = code[1:]
        if (cur == "1"):
            prev = (prev + 1) % 2
        output += str(prev)
    return output


def encoding(to, fromwho, msg):
    output = bin(int(fromwho))[2:].zfill(48)
    output += bin(int(to))[2:].zfill(48)
    output += bin(len(msg))[2:].zfill(16)
    for i in range(0, len(msg)):
        output += bin(ord(msg[i]))[2:].zfill(8)
    output += bin(zlib.crc32(bitarray(output).tobytes()) & 0xffffffff)[2:].zfill(32)
    output = code4b5b(output)
    output = codenrz(output)
    output = "1010101010101010101010101010101010101010101010101010101010101011" + output
    return output


def generplay(code):
    frames = [0, 0]
    duration_samples = int(time * 44100)

    t = np.linspace(0, time, duration_samples, endpoint=False)
    frames[0] = (np.sin(2 * math.pi * freq0 * t) * amplitude).astype(np.int16)
    frames[1] = (np.sin(2 * math.pi * freq1 * t) * amplitude).astype(np.int16)

    for i in code:
        sd.play(frames[int(i)], samplerate=44100)
        sd.wait()

while True:
    try:
        input = input()
    except EOFError:
        break

    if (input == ""):
        break
    else:
        inputspl = input.split(" ", 2)
        output = encoding(inputspl[0], inputspl[1], inputspl[2])
        generplay(output)
    print("Generating completed.")
    break
