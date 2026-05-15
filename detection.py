from ultralytics import YOLO
import cv2
import threading
import winsound
import time
import numpy as np
import wave
import struct
import os

model = YOLO("best (3).pt")
cap = cv2.VideoCapture("video.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

out = cv2.VideoWriter('output_silent.mp4', cv2.VideoWriter_fourcc(*'avc1'), 20, (640, 480))
if not out.isOpened():
    out = cv2.VideoWriter('output_silent.avi', cv2.VideoWriter_fourcc(*'XVID'), 20, (640, 480))

weapon_counter = 0
last_beep_time = 0

# ✅ Track when beeps happen for audio generation
beep_timestamps = []

def beep():
    winsound.Beep(1000, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    results = model.predict(frame, conf=0.20, iou=0.4, imgsz=640, verbose=False)

    humans = []
    weapons = []

    for r in results:
        for box in r.boxes:
            name = model.names[int(box.cls[0])].lower()
            conf = float(box.conf[0])
            coords = list(map(int, box.xyxy[0]))
            if name == "human":
                humans.append((coords, conf))
            elif name == "weapon":
                weapons.append((coords, conf))

    if weapons:
        weapon_counter = min(weapon_counter + 1, 5)
    else:
        weapon_counter = max(weapon_counter - 1, 0)

    armed = weapon_counter >= 5

    if armed:
        now = time.time()
        if now - last_beep_time > 2.0:
            last_beep_time = now
            beep_timestamps.append(now)  # ✅ save beep time
            threading.Thread(target=beep, daemon=True).start()

    for (x1, y1, x2, y2), conf in weapons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"Weapon {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for (x1, y1, x2, y2), conf in humans:
        color = (0, 0, 255) if armed else (0, 255, 0)
        label = f"Armed {conf:.2f}" if armed else f"Unarmed {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if armed:
        cv2.putText(frame, "ARMED MAN DETECTED", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "UNARMED", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

out.release()
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

# ✅ Generate WAV audio with beeps at correct timestamps
print("Generating audio...")
sample_rate = 44100
duration = time.time() - (beep_timestamps[0] if beep_timestamps else time.time())
total_samples = int(sample_rate * max(duration, 1))
audio = np.zeros(total_samples)

start_time = beep_timestamps[0] if beep_timestamps else 0

for t in beep_timestamps:
    offset = int((t - start_time) * sample_rate)
    beep_samples = int(0.6 * sample_rate)  # 600ms beep
    for i in range(beep_samples):
        if offset + i < total_samples:
            audio[offset + i] = 0.5 * np.sin(2 * np.pi * 1000 * i / sample_rate)

# save as wav
with wave.open("beeps.wav", 'w') as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(sample_rate)
    for sample in audio:
        wav.writeframes(struct.pack('<h', int(sample * 32767)))

# ✅ Merge video + audio using ffmpeg
print("Merging video and audio...")
os.system("ffmpeg -i output_silent.mp4 -i beeps.wav -c:v copy -c:a aac -shortest final_output.mp4 -y")

# cleanup
os.remove("beeps.wav")
os.remove("output_silent.mp4")
print("Done! Saved as final_output.mp4")