import cv2
import numpy as np
import pickle
import h5py

# === Load ANN model từ h5 ===
with h5py.File("face_recognition_ann.h5", "r") as f:
    W1 = f["W1"][:]
    b1 = f["b1"][:]
    W2 = f["W2"][:]
    b2 = f["b2"][:]

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# === Hàm ANN ===
def relu(x): 
    return np.maximum(0, x)

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def predict_face(img_gray):
    try:
        # Resize và chuẩn hóa ảnh
        img = cv2.resize(img_gray, (100, 100)) / 255.0
        x = img.flatten().reshape(1, -1)

        # Forward ANN
        Z1 = x @ W1 + b1
        A1 = relu(Z1)
        Z2 = A1 @ W2 + b2
        A2 = softmax(Z2)

        # Lấy nhãn dự đoán
        pred_class = np.argmax(A2)
        prob = np.max(A2)

        return encoder.inverse_transform([pred_class])[0], prob
    except Exception as e:
        return "Lỗi", 0.0

# === Load Haar cascade ===
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# === Mở camera ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không mở được camera!")
    exit()

print("📷 Camera đang chạy... Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không nhận được khung hình từ camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]

        # Dự đoán khuôn mặt
        label, prob = predict_face(face_roi)

        # Vẽ khung + label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame, f"{label} ({prob*100:.1f}%)", 
            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, (0, 255, 0), 2
        )

    cv2.imshow("Face Recognition - ANN", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
