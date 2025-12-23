import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('model/hand_gesture_model.h5')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Define ROI (Region of Interest)
    x_start = width // 2 - 150
    y_start = height // 2 - 150
    x_end = width // 2 + 150
    y_end = height // 2 + 150

    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)

    # Process ROI
    roi = frame[y_start:y_end, x_start:x_end]
    img = cv2.resize(roi, (64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]

    cv2.putText(frame, f'Prediction: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()