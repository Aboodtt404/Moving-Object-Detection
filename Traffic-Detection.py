import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

cap = cv2.VideoCapture("traffic.mp4")

gmm = GaussianMixture(n_components=2)
alpha = 0.4
learning_rate = 0.01
prev_mask = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    features = gray.reshape(-1, 1)

    gmm.fit(features)

    predictions = gmm.predict(features)
    car_mask = predictions.reshape(gray.shape)

    car_mask_uint8 = car_mask.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    car_mask_processed = cv2.morphologyEx(car_mask_uint8, cv2.MORPH_CLOSE, kernel)

    if prev_mask is None:
        prev_mask = car_mask_processed.copy()
    else:
        prev_mask = cv2.addWeighted(prev_mask, 1 - learning_rate, car_mask_processed, learning_rate, 0)

    result = cv2.bitwise_and(frame, frame, mask=car_mask_processed.astype(np.uint8) * 255)
    cv2.imshow('Car Detection', result)
    cv2.waitKey(1)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()