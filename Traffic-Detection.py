import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

def process_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gmm = GaussianMixture(n_components=2)
    gmm.fit(gray_frame.reshape(-1, 1))

    predictions = gmm.predict(gray_frame.reshape(-1, 1))
    car_mask = predictions.reshape(gray_frame.shape)

    car_mask_uint8 = car_mask.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    car_mask_processed = cv2.morphologyEx(car_mask_uint8, cv2.MORPH_CLOSE, kernel)

    foreground = cv2.bitwise_and(frame, frame, mask=car_mask_processed.astype(np.uint8) * 255)
    background = cv2.bitwise_and(frame, frame, mask=(1 - car_mask_processed).astype(np.uint8) * 255)
    
    return foreground, background

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    foreground_frames = []
    background_frames = []

    if not cap.isOpened():
        print("Error opening video file")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        foreground, background = process_frame(frame)
        
        foreground_frames.append(foreground)
        background_frames.append(background)

        cv2.imshow('Original Frame', frame)
        cv2.imshow('Foreground', foreground)
        cv2.imshow('Background', background)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main('people.mp4')
