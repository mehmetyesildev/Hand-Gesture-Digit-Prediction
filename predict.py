import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Modeli yükle
model = load_model('model/hand_gesture_model.h5')

# Kameradan görüntü almak için OpenCV kullanıyoruz
cap = cv2.VideoCapture(0)  # 0, varsayılan kamerayı temsil eder

while True:
    # Kameradan bir kare al
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntü boyutlarını al
    height, width, _ = frame.shape

    # ROI için çerçeve (örneğin, ekranın ortasında 300x300 boyutunda bir bölge)
    x_start = width // 2 - 150
    y_start = height // 2 - 150
    x_end = width // 2 + 150
    y_end = height // 2 + 150

    # ROI'yi çiz (ekrana gösterilecek çerçeve)
    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)

    # Sadece ROI bölgesini al
    roi = frame[y_start:y_end, x_start:x_end]

    # ROI'yi yeniden boyutlandır ve işleme hazırlığı yap
    img = cv2.resize(roi, (64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Tahmin yap
    prediction = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Tahmin edilen sınıfı ekrana yazdır
    cv2.putText(frame, f'Tahmin: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Görüntüyü ekranda göster
    cv2.imshow('Hand Gesture Recognition', frame)

    # 'q' tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı serbest bırak ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()
