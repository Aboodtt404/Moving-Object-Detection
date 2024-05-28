# Moving Object Detection in Videos (Traffic Monitoring)

This Python script implements a real-time moving object detection system suitable for basic traffic monitoring applications. It leverages OpenCV and the Gaussian Mixture Model (GMM) for background subtraction

## Features

- **Real-time Processing:** Analyzes video frames on-the-fly for real-time object detection
- **Background Subtraction with GMM:** Adapts to gradual lighting changes and handles static objects in the scene
- **Morphological Closing:** Refines the foreground mask by removing noise and holes
- **Learning Rate Update:** Continuously updates the background model to account for slow scene changes
- **Output:** Generates a video output where only moving objects are highlighted within the original video frame (black background with detected objects shown in original colors)

## Applications

- **Basic traffic monitoring:** Detecting moving vehicles in a scene.
- **People counting:** Identifying and counting people entering or leaving a specific area

## Limitations

This approach might not be ideal for complex scenarios with:
- Significant background variations (e.g., flickering lights, shadows)
- Slow-moving objects
- Multiple object types requiring individual classification

## Getting Started

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/moving-object-detection.git
    ```
    Use code with caution.

2. **Install dependencies:**

    This script requires OpenCV and scikit-learn. You can install them using pip:
    ```bash
    pip install opencv-python scikit-learn
    ```
    Use code with caution.

