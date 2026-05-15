
# 🔥 Real-Time Weapon Detection & Armed Person Identification (CCP - Computer Vision)

## 🚀 Project Overview

This project is a **Complex Computing Problem (CCP) in Computer Vision** focused on building a real-time surveillance system that detects humans and weapons from live CCTV/video feeds.

We used a custom-trained **YOLOv8 model** to detect weapons and determine whether a person is armed or unarmed in real time. The system also includes smoothing logic to reduce false detections and flickering.


## 🎯 Objectives

- Detect weapons in real-time video streams  
- Detect human presence in CCTV footage  
- Identify armed vs unarmed persons  
- Reduce false positives using frame smoothing  
- Achieve fast and efficient real-time performance  


## 🧠 Key Features

- Real-time object detection (weapon & human)
- Armed / Unarmed classification logic
- Confidence-based predictions
- Frame smoothing using detection counter
- Live webcam / CCTV video support
- Output video recording support


## 🛠️ Technology Stack

- Python  
- OpenCV  
- Ultralytics YOLOv8  
- PyTorch  
- Google Colab  
- Matplotlib  


## 📁 Project Structure

weapon-detection-ccp/
│
├── dataset/              # Custom dataset (images + labels)
├── runs/                 # Training outputs
├── best.pt               # Trained YOLO model
├── detection.py          # Real-time detection script
├── train.py              # Training script
└── README.md


## ⚙️ How It Works

### 1. Dataset Training
- Custom dataset trained using YOLOv8  
- Classes: Human, Weapon  

### 2. Real-Time Detection
- Webcam/video feed captured using OpenCV  
- Each frame processed by YOLO model  
- Bounding boxes drawn around detections  

### 3. Decision Logic
- Weapon detected consistently → ARMED  
- No weapon detected → UNARMED  

### 4. Smoothing Logic
Reduces flickering using frame-based counter:

```python
if weapons:
    weapon_counter = min(weapon_counter + 1, 5)
else:
    weapon_counter = max(weapon_counter - 1, 0)

armed = weapon_counter >= 3
````



## 📦 Installation

```bash
pip install ultralytics opencv-python matplotlib
```


## ▶️ Run Project

```bash
python detection.py
```

OR use Google Colab for training and testing.


## 📌 Applications

* Smart CCTV surveillance
* Security monitoring systems
* Public safety detection
* AI-based threat detection

## ⚠️ Limitations

* Depends on dataset quality
* May fail in low-light conditions
* Small weapons may not always detect
* Requires GPU for best performance


## 🔮 Future Improvements

* Weapon type classification (gun/knife etc.)
* Real-time alert system (SMS/email)
* Multi-camera support
* Edge deployment (Raspberry Pi / Jetson)
* Object tracking integration


## 👨‍💻 Team Members

* Waleed Kamal
* Tehamee Raheel
* Owais Ahmed


## 📜 License

Academic CCP (Computer Vision) Project

