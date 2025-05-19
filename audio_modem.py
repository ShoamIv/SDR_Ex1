import numpy as np
import pyaudio
import time
import argparse
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.io import wavfile
import wave

# Constants - same frequencies as the original web implementation
FREQUENCIES = [392, 784, 1046.5, 1318.5, 1568, 1864.7, 2093, 2637]
SAMPLE_RATE = 44100  # Standard audio sample rate


class AudioModem:
    def __init__(self):
        self.audio = pyaudio.PyAudio()

    def __del__(self):
        self.audio.terminate()


class Encoder(AudioModem):
    def __init__(self, active_duration=0.1, pause_duration=0.02):
        super().__init__()
        self.active_duration = active_duration  # Duration in seconds for each character
        self.pause_duration = pause_duration  # Pause between characters

    def char_to_frequencies(self, char):
        """Convert a character to a list of active frequencies based on its binary representation."""
        char_code = ord(char)
        active_freqs = []
        for i, freq in enumerate(FREQUENCIES):
            if char_code & (1 << i):
                active_freqs.append(freq)
        return active_freqs

    def generate_audio_for_char(self, char):
        """Generate audio data for a single character."""
        active_freqs = self.char_to_frequencies(char)

        # Create time array
        t = np.linspace(0, self.active_duration, int(SAMPLE_RATE * self.active_duration), False)

        # Initialize with silence if no frequencies are active
        if not active_freqs:
            return np.zeros_like(t)

        # Generate sum of sine waves for active frequencies
        signal = np.zeros_like(t)
        for freq in active_freqs:
            # Use a small fade-in and fade-out to avoid clicks
            fade_samples = int(0.01 * SAMPLE_RATE)  # 10ms fade
            fade_in = np.linspace(0, 1, min(fade_samples, len(t)))
            fade_out = np.linspace(1, 0, min(fade_samples, len(t)))

            # Apply fades only if the signal is long enough
            envelope = np.ones_like(t)
            if len(t) > 2 * fade_samples:
                envelope[:fade_samples] = fade_in
                envelope[-fade_samples:] = fade_out

            # Add the sine wave with envelope
            signal += 0.8 * (1.0 / len(active_freqs)) * envelope * np.sin(2 * np.pi * freq * t)

        return signal

    def encode_text(self, text):
        """Encode the entire text as audio."""
        encoded_signal = np.array([])

        # Add initial silence to help decoding
        initial_pause = np.zeros(int(SAMPLE_RATE * 0.1))
        encoded_signal = np.append(encoded_signal, initial_pause)

        for char in text:
            # Generate audio for character
            char_signal = self.generate_audio_for_char(char)
            encoded_signal = np.append(encoded_signal, char_signal)

            # Add pause between characters
            pause = np.zeros(int(SAMPLE_RATE * self.pause_duration))
            encoded_signal = np.append(encoded_signal, pause)

        # Add final silence
        final_pause = np.zeros(int(SAMPLE_RATE * 0.1))
        encoded_signal = np.append(encoded_signal, final_pause)

        return encoded_signal

    def play_audio(self, audio_data):
        """Play the audio data."""
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = 0.9 * audio_data / max_val

        stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=SAMPLE_RATE,
            output=True
        )

        # Convert to bytes and play in chunks to avoid buffer issues
        CHUNK = 1024
        audio_bytes = (audio_data.astype(np.float32)).tobytes()

        for i in range(0, len(audio_bytes), CHUNK * 4):  # 4 bytes per float32
            chunk = audio_bytes[i:i + CHUNK * 4]
            stream.write(chunk)

        stream.stop_stream()
        stream.close()

    def encode_and_play(self, text):
        """Encode and play the text."""
        print(f"Encoding and playing: {text}")
        audio_data = self.encode_text(text)
        self.play_audio(audio_data)

    def save_to_file(self, text, filename):
        """Encode text and save to WAV file."""
        import wave

        audio_data = self.encode_text(text)

        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = 0.9 * audio_data / max_val

        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for 16-bit audio
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())

        print(f"Saved encoded audio to {filename}")


class Decoder(AudioModem):
    def __init__(self, bin_value_threshold=0.2, duplicate_state_threshold=3):
        super().__init__()
        self.bin_value_threshold = bin_value_threshold  # Threshold for frequency detection (relative)
        self.duplicate_state_threshold = duplicate_state_threshold  # How many duplicates before outputting
        self.debug = False

    def detect_frequencies(self, fft_data, fft_freqs):
        """Detect which frequencies are present in the FFT data."""
        active_freqs = []
        relative_strengths = []

        # Find the overall peak for relative threshold
        peak_magnitude = np.max(fft_data)
        noise_floor = np.median(fft_data)

        if peak_magnitude < 1e-6:  # Almost silence
            return [], []

        # Check each target frequency
        for freq in FREQUENCIES:
            # Find the closest frequency bin and a small range around it
            center_idx = np.argmin(np.abs(fft_freqs - freq))

            # Look at a small window around the expected frequency to account for slight tuning issues
            window_size = 3  # Number of bins to look at on each side
            start_idx = max(0, center_idx - window_size)
            end_idx = min(len(fft_data) - 1, center_idx + window_size)

            # Find the peak within this window
            window_peak_idx = start_idx + np.argmax(fft_data[start_idx:end_idx + 1])
            peak_value = fft_data[window_peak_idx]

            # Calculate signal-to-noise ratio
            snr = peak_value / (noise_floor + 1e-10)  # Avoid division by zero

            # If peak is above absolute and relative thresholds
            adaptive_threshold = self.bin_value_threshold * peak_magnitude

            if peak_value > adaptive_threshold and snr > 3.0:
                active_freqs.append(freq)
                relative_strengths.append(peak_value / peak_magnitude)

        return active_freqs, relative_strengths

    def get_state_from_frequencies(self, active_freqs):
        """Convert active frequencies to a state byte."""
        state = 0
        for i, freq in enumerate(FREQUENCIES):
            if freq in active_freqs:
                state |= (1 << i)

        return state

    def decode_audio_stream(self, duration=10):
        """Decode audio from microphone for the specified duration."""
        CHUNK = 4096  # Larger chunk for better frequency resolution

        stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        print(f"Listening for {duration} seconds...")

        decoded_text = ""
        prev_state = 0
        duplicates = 0
        silent_chunks = 0

        start_time = time.time()

        while time.time() - start_time < duration:
            # Read audio chunk
            audio_bytes = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(audio_bytes, dtype=np.float32)

            # Compute FFT
            fft_data = np.abs(rfft(audio_data))
            fft_freqs = rfftfreq(len(audio_data), 1.0 / SAMPLE_RATE)

            # Check if the chunk is mostly silence
            if np.max(np.abs(audio_data)) < 0.01:
                silent_chunks += 1
                if silent_chunks > 5:  # Reset state after continuous silence
                    prev_state = 0
                    duplicates = 0
                continue
            else:
                silent_chunks = 0

            # Detect active frequencies
            active_freqs, strengths = self.detect_frequencies(fft_data, fft_freqs)

            # Get state from active frequencies
            state = self.get_state_from_frequencies(active_freqs)

            # Print current binary state
            binary_str = format(state, '08b')
            active_str = ', '.join([f"{f}Hz" for f in active_freqs])
            print(f"State: 0b{binary_str} | Active: {active_str}", end='\r')

            # Check for duplicate states (this helps filter out noise)
            if state == prev_state and state > 0:  # Only count duplicates for non-zero states
                duplicates += 1
            else:
                prev_state = state
                duplicates = 0

            # Output character when enough duplicates are detected
            if duplicates >= self.duplicate_state_threshold and state > 0:
                if state < 256:  # Valid ASCII range
                    char = chr(state)
                    decoded_text += char
                    print(f"\nDetected: '{char}' (ASCII: {state})")
                duplicates = 0

        stream.stop_stream()
        stream.close()

        return decoded_text

    def show_spectrogram(self, audio_data, sample_rate):
        """Display a spectrogram of the audio data with frequency markers."""
        # Calculate the spectrogram
        plt.figure(figsize=(12, 8))

        # Plot main spectrogram
        plt.specgram(audio_data, NFFT=2048, Fs=sample_rate, noverlap=512,
                     cmap='viridis', vmin=-120, vmax=0)

        # Add markers for the encoding frequencies
        for freq in FREQUENCIES:
            plt.axhline(y=freq, color='r', linestyle='--', alpha=0.5)
            plt.text(0, freq, f'{freq}Hz', color='white',
                     bbox=dict(facecolor='red', alpha=0.5))

        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title('Audio Spectrogram with Encoding Frequencies')
        plt.ylim(0, 3000)  # Limit y-axis to relevant frequencies
        plt.colorbar(label='Intensity (dB)')
        plt.tight_layout()
        plt.show()

    def decode_from_file(self, filename, show_spectrogram=False):
        """Decode audio from a WAV file."""
        try:
            # Try using scipy.io.wavfile for robust reading
            sample_rate, audio_data = wavfile.read(filename)

            # Convert to float between -1 and 1
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32767.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483647.0
            elif audio_data.dtype == np.uint8:
                audio_data = (audio_data.astype(np.float32) - 128) / 128.0
        except:
            # Fallback to wave module if scipy fails
            with wave.open(filename, 'rb') as wf:
                # Get sample rate and frames
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()

                # Read all frames
                frames = wf.readframes(n_frames)

                # Convert to numpy array
                if sample_width == 1:  # 8-bit audio
                    dtype = np.uint8
                    audio_data = np.frombuffer(frames, dtype=dtype)
                    audio_data = (audio_data.astype(np.float32) - 128) / 128.0
                elif sample_width == 2:  # 16-bit audio
                    dtype = np.int16
                    audio_data = np.frombuffer(frames, dtype=dtype)
                    audio_data = audio_data.astype(np.float32) / 32767.0
                elif sample_width == 4:  # 32-bit audio
                    dtype = np.int32
                    audio_data = np.frombuffer(frames, dtype=dtype)
                    audio_data = audio_data.astype(np.float32) / 2147483647.0
                else:
                    raise ValueError(f"Unsupported sample width: {sample_width}")

                # Handle multi-channel audio
                if n_channels > 1:
                    audio_data = audio_data[::n_channels]  # Take first channel

        print(f"Decoding file: {filename} ({len(audio_data) / sample_rate:.2f} seconds)")

        if show_spectrogram:
            self.show_spectrogram(audio_data, sample_rate)

        # Process audio in overlapping chunks
        chunk_size = 4096
        hop_size = chunk_size // 4  # 75% overlap

        decoded_text = ""
        prev_state = 0
        duplicates = 0
        last_char_time = -1  # To prevent duplicate characters

        # Create a list to store states and their timestamps
        states = []

        for i in range(0, len(audio_data) - chunk_size, hop_size):
            chunk = audio_data[i:i + chunk_size]
            time_pos = i / sample_rate

            # Apply window to reduce spectral leakage
            chunk = chunk * np.hanning(len(chunk))

            # Compute FFT
            fft_data = np.abs(rfft(chunk))
            fft_freqs = rfftfreq(len(chunk), 1.0 / sample_rate)

            # Detect active frequencies
            active_freqs, strengths = self.detect_frequencies(fft_data, fft_freqs)

            # Get state from active frequencies
            state = self.get_state_from_frequencies(active_freqs)

            # Store state with timestamp
            if state > 0:  # Only store non-zero states
                states.append((time_pos, state))

            # Print current binary state
            binary_str = format(state, '08b')
            print(f"Time: {time_pos:.2f}s | State: 0b{binary_str}", end='\r')

            # Check for duplicate states (this helps filter out noise)
            if state == prev_state and state > 0:
                duplicates += 1
            else:
                prev_state = state
                duplicates = 0

            # Output character when enough duplicates are detected
            if duplicates >= self.duplicate_state_threshold and state > 0:
                # Only add character if we haven't added one recently
                if time_pos - last_char_time > 0.05:  # Minimum time between characters
                    if state < 256:  # Valid ASCII range
                        char = chr(state)
                        decoded_text += char
                        print(f"\nDetected at {time_pos:.2f}s: '{char}' (ASCII: {state})")
                        last_char_time = time_pos
                duplicates = 0

        # Post-processing: If we didn't get any text, try with a lower threshold
        if not decoded_text and states:
            print("\nTrying again with more aggressive detection...")

            # Group states by similarity
            grouped_states = []
            current_group = [states[0]]

            for i in range(1, len(states)):
                if states[i][1] == current_group[0][1]:  # Same state
                    current_group.append(states[i])
                else:
                    if len(current_group) >= 2:  # Only keep groups with at least 2 instances
                        grouped_states.append(current_group)
                    current_group = [states[i]]

            if len(current_group) >= 2:
                grouped_states.append(current_group)

            # Extract characters from consistent state groups
            for group in grouped_states:
                state = group[0][1]
                if 0 < state < 256:
                    char = chr(state)
                    decoded_text += char
                    time_pos = group[0][0]
                    print(f"Detected at {time_pos:.2f}s: '{char}' (ASCII: {state})")

        return decoded_text


def main():
    parser = argparse.ArgumentParser(description='Web Audio Modem in Python')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Encoder arguments
    encoder_parser = subparsers.add_parser('encode', help='Encode text to audio')
    encoder_parser.add_argument('text', help='Text to encode')
    encoder_parser.add_argument('--active-duration', type=float, default=0.1,
                                help='Duration of active bit in seconds (default: 0.1)')
    encoder_parser.add_argument('--pause-duration', type=float, default=0.02,
                                help='Pause between characters in seconds (default: 0.02)')
    encoder_parser.add_argument('--save', type=str, default=None,
                                help='Save to WAV file instead of playing')

    # Decoder arguments
    decoder_parser = subparsers.add_parser('decode', help='Decode audio to text')
    decoder_parser.add_argument('--duration', type=int, default=10,
                                help='Duration to listen in seconds (default: 10)')
    decoder_parser.add_argument('--threshold', type=float, default=0.2,
                                help='Relative threshold for frequency detection (default: 0.2)')
    decoder_parser.add_argument('--duplicates', type=int, default=3,
                                help='Duplicate states before output (default: 3)')
    decoder_parser.add_argument('--file', type=str, default=None,
                                help='Decode from WAV file instead of microphone')
    decoder_parser.add_argument('--show-spectrogram', action='store_true',
                                help='Show spectrogram (requires matplotlib)')

    args = parser.parse_args()

    if args.command == 'encode':
        encoder = Encoder(
            active_duration=args.active_duration,
            pause_duration=args.pause_duration
        )

        if args.save:
            encoder.save_to_file(args.text, args.save)
        else:
            encoder.encode_and_play(args.text)

    elif args.command == 'decode':
        decoder = Decoder(
            bin_value_threshold=args.threshold,
            duplicate_state_threshold=args.duplicates
        )

        if args.file:
            decoded_text = decoder.decode_from_file(args.file, args.show_spectrogram)
        else:
            decoded_text = decoder.decode_audio_stream(duration=args.duration)

        print(f"\nDecoded text: {decoded_text}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()