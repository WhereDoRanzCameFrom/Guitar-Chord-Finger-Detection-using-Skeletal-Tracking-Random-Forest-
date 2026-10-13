# 🎸 Guitar Chord & Finger Detection using Skeletal Tracking (Random Forest)

A machine learning-based application for automatic guitar chord recognition and finger placement tracking using **skeletal hand data** (MediaPipe/OpenPose) combined with the **Random Forest** algorithm, rather than raw audio signals.

# *The main source code of this project is stored in a private repository for security and copyright reasons.*

## 🚀 Features

- **Skeletal Hand Tracking:** Utilizes computer vision (e.g., MediaPipe Hands) to extract 3D/2D hand joint coordinates (landmarks) from camera frames.
- **Geometric Feature Extraction:** Computes relative distances, angles between finger joints, and fretboard positioning vectors.
- **Robust Classification:** Employs an ensemble of decision trees via **Random Forest** to accurately classify guitar chords based on finger configurations.
- **Real-Time Inference:** Fast processing pipeline suitable for live webcam feeds, practice assistants, or interactive guitar learning tools.

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **MediaPipe / OpenCV** (for hand tracking and video frame capture)
- **Scikit-Learn** (for Random Forest Classifier)
- **NumPy & Pandas** (for geometric data manipulation and coordinate processing)
- **Joblib** (for saving and loading trained models)

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/chord-skeleton-rf.git](https://github.com/your-username/chord-skeleton-rf.git)
   cd chord-skeleton-rf
