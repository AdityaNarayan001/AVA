"""Quick diagnostic: test mic capture + VAD processing."""
import sounddevice as sd
import numpy as np
import time
from vad import VADProcessor

frames_received = []

def callback(indata, frames, time_info, status):
    if status:
        print(f"Mic status warning: {status}")
    chunk = indata[:, 0].copy()
    frames_received.append(chunk)

print("Opening mic stream at 16kHz, blocksize=512...")
s = sd.InputStream(samplerate=16000, channels=1, dtype="int16", blocksize=512, callback=callback)
s.start()
print("Recording 2 seconds of audio...")
time.sleep(2)
s.stop()
s.close()

print(f"\nFrames received: {len(frames_received)}")
if not frames_received:
    print("ERROR: No frames received from mic!")
    exit(1)

all_audio = np.concatenate(frames_received)
print(f"Total samples: {len(all_audio)} ({len(all_audio)/16000:.2f}s)")
print(f"Range: [{all_audio.min()}, {all_audio.max()}]")
print(f"Mean: {all_audio.mean():.2f}, Std: {all_audio.std():.2f}")
non_zero = np.count_nonzero(all_audio)
print(f"Non-zero: {non_zero}/{len(all_audio)} ({100*non_zero/len(all_audio):.1f}%)")

if all_audio.std() < 5:
    print("\nWARNING: Audio is nearly silent — check macOS microphone permissions!")
    print("Go to System Settings > Privacy & Security > Microphone and grant access to Terminal.")

print("\n--- VAD Test ---")
vad = VADProcessor(hop_size=512, threshold=0.5)
print(f"Backend: {vad.backend_name}")

speech_count = 0
total_count = 0
max_conf = 0.0
for i in range(0, len(all_audio), 512):
    frame = all_audio[i:i+512]
    if len(frame) < 512:
        break
    r = vad.process(frame)
    total_count += 1
    if r.confidence > max_conf:
        max_conf = r.confidence
    if r.is_speech:
        speech_count += 1

print(f"Frames: {total_count}, Speech: {speech_count}, Max confidence: {max_conf:.4f}")

print("\nFirst 10 frame confidences:")
vad2 = VADProcessor(hop_size=512, threshold=0.5)
for i in range(min(10, len(frames_received))):
    frame = frames_received[i][:512]
    if len(frame) < 512:
        frame = np.pad(frame, (0, 512 - len(frame)))
    r = vad2.process(frame)
    print(f"  Frame {i}: conf={r.confidence:.4f} speech={r.is_speech}")

print("\nDone.")
