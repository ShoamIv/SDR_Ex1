import numpy as np
import pyaudio
import time
import argparse
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.io import wavfile
import wave

FREQUENCIES = [392, 784, 1046.5, 1318.5, 1568, 1864.7, 2093, 2637]
SAMPLE_RATE = 44100  # Standard audio sample rate


class AudioModem:
    def __init__(self):
        self.audio = pyaudio.PyAudio()

    def __del__(self):
        self.audio.terminate()


class Encoder(AudioModem):
    def __init__(self, active_duration=0.1, pause_duration=0.02, repetitions=5):
        super().__init__()
        self.active_duration = active_duration  # Duration in seconds for each character
        self.pause_duration = pause_duration  # Pause between characters
        self.repetitions = repetitions  # Number of times to repeat each character

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
        """Encode the entire text as audio with repetitions for stability."""
        encoded_signal = np.array([])

        # Add initial silence to help decoding
        initial_pause = np.zeros(int(SAMPLE_RATE * 0.2))  # Longer initial pause
        encoded_signal = np.append(encoded_signal, initial_pause)

        for char in text:
            print(f"Encoding '{char}' ({self.repetitions} times)")

            # Repeat each character multiple times for better detection
            for rep in range(self.repetitions):
                # Generate audio for character
                char_signal = self.generate_audio_for_char(char)
                encoded_signal = np.append(encoded_signal, char_signal)

                # Add small pause between repetitions (except for the last one)
                if rep < self.repetitions - 1:
                    small_pause = np.zeros(int(SAMPLE_RATE * (self.pause_duration / 2)))
                    encoded_signal = np.append(encoded_signal, small_pause)

            # Add longer pause between different characters
            char_pause = np.zeros(int(SAMPLE_RATE * self.pause_duration * 5))
            encoded_signal = np.append(encoded_signal, char_pause)

        # Add final silence
        final_pause = np.zeros(int(SAMPLE_RATE * 0.2))
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
        print(f"Encoding and playing: '{text}' (each character repeated {self.repetitions} times)")
        audio_data = self.encode_text(text)
        print(f"Total audio duration: {len(audio_data) / SAMPLE_RATE:.2f} seconds")
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
        print(f"Audio duration: {len(audio_data) / SAMPLE_RATE:.2f} seconds")


class Decoder(AudioModem):
    def __init__(self, bin_value_threshold=0.2, duplicate_state_threshold=3):
        super().__init__()
        self.bin_value_threshold = bin_value_threshold  # Threshold for frequency detection (relative)
        self.duplicate_state_threshold = duplicate_state_threshold  # How many duplicates before outputting
        self.debug = False

    def detect_frequencies(self, fft_data, fft_freqs):
        """Detect which frequencies are present in the FFT data with improved accuracy."""
        active_freqs = []
        relative_strengths = []

        # Find the overall peak for relative threshold
        peak_magnitude = np.max(fft_data)
        noise_floor = np.percentile(fft_data, 25)  # Use 25th percentile instead of median for better noise estimation

        if peak_magnitude < 1e-5:  # Almost silence
            return [], []

        # More conservative frequency detection
        detected_peaks = []
        for freq in FREQUENCIES:
            # Find the closest frequency bin
            center_idx = np.argmin(np.abs(fft_freqs - freq))

            # Look at a small window around the expected frequency
            window_size = 2  # Smaller window for more precision
            start_idx = max(0, center_idx - window_size)
            end_idx = min(len(fft_data) - 1, center_idx + window_size)

            # Find the peak within this window
            window_peak_idx = start_idx + np.argmax(fft_data[start_idx:end_idx + 1])
            peak_value = fft_data[window_peak_idx]

            # Calculate various metrics
            snr = peak_value / (noise_floor + 1e-10)
            relative_strength = peak_value / peak_magnitude

            # More stringent detection criteria
            adaptive_threshold = max(
                self.bin_value_threshold * peak_magnitude,  # Relative to peak
                noise_floor * 5  # Absolute noise floor multiple
            )

            if (peak_value > adaptive_threshold and
                    snr > 4.0 and  # Stricter SNR requirement
                    relative_strength > 0.1):  # Must be at least 10% of peak

                detected_peaks.append((freq, peak_value, relative_strength))

        # Sort by strength and only take the strongest peaks if we have too many
        detected_peaks.sort(key=lambda x: x[1], reverse=True)

        # Limit to maximum reasonable number of frequencies (ASCII has max 8 bits)
        max_freqs = 6  # Conservative limit
        for freq, peak_value, rel_strength in detected_peaks[:max_freqs]:
            active_freqs.append(freq)
            relative_strengths.append(rel_strength)

        return active_freqs, relative_strengths

    def get_state_from_frequencies(self, active_freqs):
        """Convert active frequencies to a state byte."""
        state = 0
        for i, freq in enumerate(FREQUENCIES):
            if freq in active_freqs:
                state |= (1 << i)

        return state

    def decode_audio_stream(self, duration=10):
        """Decode audio from microphone using sliding buffer and tone stability detection."""
        window_size = int(SAMPLE_RATE * 0.50)  # Larger window for better frequency resolution
        step_size = window_size // 8  # More overlap for stability
        min_persistent_windows = max(5, self.duplicate_state_threshold * 2)  # More conservative
        silence_threshold = 0.005  # Lower silence threshold
        min_signal_strength = 0.01  # Minimum signal strength to consider

        stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=step_size
        )

        print(f"Listening for {duration} seconds...")
        print(f"Using duplicate threshold: {min_persistent_windows}")

        decoded_text = ""
        current_state = None
        persistent_count = 0
        sliding_buffer = np.zeros(window_size, dtype=np.float32)
        last_output_time = 0
        state_history = []  # Track recent states

        start_time = time.time()

        while time.time() - start_time < duration:
            current_time = time.time() - start_time

            # Read new audio chunk
            try:
                audio_bytes = stream.read(step_size, exception_on_overflow=False)
                new_data = np.frombuffer(audio_bytes, dtype=np.float32)
            except:
                continue

            # Slide buffer and append new data
            sliding_buffer = np.roll(sliding_buffer, -step_size)
            sliding_buffer[-step_size:] = new_data

            # Check signal strength
            signal_rms = np.sqrt(np.mean(sliding_buffer ** 2))
            if signal_rms < min_signal_strength:
                persistent_count = 0
                current_state = None
                state_history = []
                continue

            # Apply window function for better FFT
            windowed_buffer = sliding_buffer * np.hanning(len(sliding_buffer))

            # FFT and frequency detection
            fft_data = np.abs(rfft(windowed_buffer))
            fft_freqs = rfftfreq(len(windowed_buffer), 1.0 / SAMPLE_RATE)
            active_freqs, strengths = self.detect_frequencies(fft_data, fft_freqs)

            # Map to binary state
            state = self.get_state_from_frequencies(active_freqs)

            # Keep track of recent states for stability analysis
            state_history.append(state)
            if len(state_history) > 10:
                state_history.pop(0)

            # Print debug info
            binary_str = format(state, '08b')
            active_str = ', '.join([f"{f:.1f}Hz" for f in sorted(active_freqs)])
            print(
                f"Time: {current_time:.1f}s | State: 0b{binary_str} | Active: {active_str} | RMS: {signal_rms:.4f} | Count: {persistent_count}",
                end='\r')

            # Enhanced state stability check
            if state == current_state and state > 0:
                persistent_count += 1
            else:
                # Only change state if the new state appears multiple times recently
                if len(state_history) >= 3 and state_history[-3:].count(state) >= 2:
                    current_state = state
                    persistent_count = 1
                else:
                    persistent_count = 0

            # Emit character only for valid printable ASCII and with strong stability
            if (persistent_count >= min_persistent_windows and
                    32 <= state <= 126 and  # Only printable ASCII characters
                    current_time - last_output_time > 0.5):  # Longer minimum time between characters

                # Double-check by looking at recent state consistency
                recent_consistent = len(state_history) >= 5 and state_history[-5:].count(state) >= 4

                if recent_consistent:
                    char = chr(state)
                    decoded_text += char
                    print(f"\nDetected: '{char}' (ASCII: {state}) at {current_time:.1f}s")
                    last_output_time = current_time
                    persistent_count = 0
                    current_state = None
                    state_history = []  # Clear history after successful detection

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

            # Track stable sequences of the same state
            min_repeats = 3  # You can increase this if needed
            if state == prev_state and state > 0:
                duplicates += 1
            else:
                if prev_state > 0 and duplicates >= min_repeats:
                    if prev_state < 256:
                        char = chr(prev_state)
                        decoded_text += char
                        print(f"\nDetected stable group at {time_pos:.2f}s: '{char}' (ASCII: {prev_state})")
                prev_state = state
                duplicates = 1

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
    encoder_parser.add_argument('--repetitions', type=int, default=5,
                                help='Number of times to repeat each character (default: 5)')
    encoder_parser.add_argument('--save', type=str, default=None,
                                help='Save to WAV file instead of playing')

    # Decoder arguments
    decoder_parser = subparsers.add_parser('decode', help='Decode audio to text')
    decoder_parser.add_argument('--duration', type=int, default=10,
                                help='Duration to listen in seconds (default: 10)')
    decoder_parser.add_argument('--threshold', type=float, default=0.15,
                                help='Relative threshold for frequency detection (default: 0.15)')
    decoder_parser.add_argument('--duplicates', type=int, default=3,
                                help='Duplicate states before output (default: 3)')
    decoder_parser.add_argument('--file', type=str, default=None,
                                help='Decode from WAV file instead of microphone')
    decoder_parser.add_argument('--show-spectrogram', action='store_true',
                                help='Show spectrogram (requires matplotlib)')
    decoder_parser.add_argument('--debug', action='store_true',
                                help='Enable debug output for troubleshooting')

    args = parser.parse_args()

    if args.command == 'encode':
        encoder = Encoder(
            active_duration=args.active_duration,
            pause_duration=args.pause_duration,
            repetitions=args.repetitions
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

        if hasattr(args, 'debug') and args.debug:
            decoder.debug = True

        if args.file:
            decoded_text = decoder.decode_from_file(args.file, args.show_spectrogram)
        else:
            decoded_text = decoder.decode_audio_stream(duration=args.duration)

        print(f"\nDecoded text: '{decoded_text}'")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()