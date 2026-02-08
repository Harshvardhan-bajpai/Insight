import cv2
import numpy as np

# Hyperparameters (TUNE LATER)
COLOR_SAMPLE_HEIGHT_RATIO = 0.6   # how much of bbox height to sample


def dominant_color(image):
    """
    Estimate dominant color using k-means clustering.
    """
    if image.size == 0:
        return "unknown"

    # Resize for speed
    img = cv2.resize(image, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img.reshape((-1, 3)).astype(np.float32)

    # K-means to find dominant color
    K = 3
    _, labels, centers = cv2.kmeans(
        pixels,
        K,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    counts = np.bincount(labels.flatten())
    dominant = centers[np.argmax(counts)]

    return rgb_to_color_name(dominant)


def rgb_to_color_name(rgb):
    r, g, b = rgb

    if r > 200 and g < 80 and b < 80:
        return "red"
    if r < 80 and g > 200 and b < 80:
        return "green"
    if r < 80 and g < 80 and b > 200:
        return "blue"
    if r > 200 and g > 200 and b < 80:
        return "yellow"
    if r > 200 and g > 200 and b > 200:
        return "white"
    if r < 60 and g < 60 and b < 60:
        return "black"
    if r > 150 and g > 100 and b < 80:
        return "orange"
    if r > 150 and g < 100 and b > 150:
        return "pink"

    return "unknownknn"


def get_attributes(frame, person_bbox, face_bbox=None):
    """
    Demo-safe attributes:
    - Always returns gender = "male"
    - Estimates clothing color from upper body
    """

    x1, y1, x2, y2 = person_bbox
    h = y2 - y1

    # Sample upper body region
    sample_y2 = int(y1 + h * COLOR_SAMPLE_HEIGHT_RATIO)
    upper_body = frame[y1:sample_y2, x1:x2]

    color = dominant_color(upper_body)

    # Demo mode: fixed gender
    gender = "male"

    return {
        "color": color,
        "gender": gender
    }
