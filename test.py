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

# Hàm ANN
def relu(x): return np.maximum(0, x)
def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def predict_face(img_gray):
    img = cv2.resize(img_gray, (100, 100)).flatten()/255.0
    x = img.reshape(1, -1)
    Z1 = x @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    pred_class = np.argmax(A2)
    return encoder.inverse_transform([pred_class])[0]

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Mở camera
cap = cv2.VideoCapture(0)
print("📷 Camera đang chạy... nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        label = predict_face(face_roi)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    cv2.imshow("Face Recognition - ANN", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
