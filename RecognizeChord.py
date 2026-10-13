import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import deque

# Load model
with open("chord_model.pkl", "rb") as f:
    model = pickle.load(f)

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
                       

# Buffer untuk stabilisasi prediksi
prediction_buffer = deque(maxlen=10)  # simpan 10 frame terakhir
stable_prediction = None

# Fungsi preprocessing: relatif terhadap wrist (landmark 0)
def get_relative_landmarks(hand_landmarks):
    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    wrist_x, wrist_y = landmarks[0]
    rel_landmarks = []
    for (x, y) in landmarks:
        rel_landmarks.append(x - wrist_x)
        rel_landmarks.append(y - wrist_y)
    return rel_landmarks

# Jalankan kamera
cap = cv2.VideoCapture(0)

print("🎸 Tekan 'q' untuk keluar")

# Tambahan: penghitung ketika tangan tidak terdeteksi
no_hand_counter = 0
MAX_NO_HAND_FRAMES = 15  # hilangkan teks setelah ~15 frame tanpa tangan

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Kamera tidak terbaca")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        no_hand_counter = 0  # reset counter kalau tangan terdeteksi
        # Ambil hanya 1 tangan (yang pertama)
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks,
                               mp_hands.HAND_CONNECTIONS)

        # Ambil fitur relatif (harus 42 dimensi)
        rel_landmarks = get_relative_landmarks(hand_landmarks)

        if len(rel_landmarks) == 42:
            features = np.array(rel_landmarks).reshape(1, -1)

            try:
                prediction = model.predict(features)[0]
                prediction_buffer.append(prediction)

                # Stabilisasi: pilih prediksi terbanyak dari buffer
                if len(prediction_buffer) == prediction_buffer.maxlen:
                    stable_prediction = max(set(prediction_buffer),
                                            key=prediction_buffer.count)

            except Exception as e:
                print("❌ Error prediksi:", e)

    else:
        # Tidak ada tangan terdeteksi
        no_hand_counter += 1
        if no_hand_counter > MAX_NO_HAND_FRAMES:
            stable_prediction = None  # hapus teks chord

    # Tampilkan prediksi stabil di layar
    if stable_prediction:
        cv2.putText(frame, f"Chord: {stable_prediction}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Chord Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
