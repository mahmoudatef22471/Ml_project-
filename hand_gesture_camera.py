"""
Hand Gesture Recognition - Real-time Camera Processing
Save this as: hand_gesture_camera.py
Make sure svm_hand_model.pkl is in the same folder
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import joblib
from datetime import datetime
import os
import urllib.request

# ──────────────────────────────────────────────────────────────────────────────
# 1. Check model file
# ──────────────────────────────────────────────────────────────────────────────
if not os.path.exists('svm_hand_model.pkl'):
    print("ERROR: svm_hand_model.pkl not found in current directory!")
    exit()

# ──────────────────────────────────────────────────────────────────────────────
# 2. Download MediaPipe hand landmarker model if needed
# ──────────────────────────────────────────────────────────────────────────────
model_path = 'hand_landmarker.task'
MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/'
             'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task')

def download_model(path, url):
    print("Downloading MediaPipe hand landmarker model...")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"✓ Download complete! File size: {os.path.getsize(path)} bytes")
    except Exception as e:
        print(f"ERROR downloading model: {e}")
        print(f"\nPlease download manually from:\n{url}")
        print(f"Save it as '{path}' in the current directory")
        exit()

if not os.path.exists(model_path):
    download_model(model_path, MODEL_URL)
elif os.path.getsize(model_path) < 1_000_000:   # should be ~3-4 MB
    print(f"WARNING: Model file too small ({os.path.getsize(model_path)} bytes). Re-downloading...")
    os.remove(model_path)
    download_model(model_path, MODEL_URL)
else:
    print(f"✓ Hand landmarker model found ({os.path.getsize(model_path)} bytes)")

# ──────────────────────────────────────────────────────────────────────────────
# 3. MediaPipe setup
# ──────────────────────────────────────────────────────────────────────────────
BaseOptions          = mp.tasks.BaseOptions
HandLandmarker       = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode    = mp.tasks.vision.RunningMode



# ──────────────────────────────────────────────────────────────────────────────
def preprocess_landmarks(landmarks):
    """
    Replicate normalize_landmarks() from main.ipynb exactly.

    MediaPipe gives 21 landmarks (0-based). Because the training CSV used
    1-based column names (x1…x21), reshaping produces the same mapping:
      array index 0  → wrist
      array index 13 → middle finger tip  (CSV column set 14)
    """
    # Build (21, 3) array from MediaPipe landmark objects
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])  # (21, 3)

    # Step 1 – recenter x and y to wrist (index 0)
    wrist = coords[0].copy()
    coords[:, 0] -= wrist[0]
    coords[:, 1] -= wrist[1]
    # z is NOT recentered – matches the notebook

    # Step 2 – scale using middle finger tip (index 13 in 0-based array)
    middle_finger_tip = coords[13].copy()
    scale = np.sqrt(middle_finger_tip[0] ** 2 + middle_finger_tip[1] ** 2)

    # Step 3 – normalise x and y only (z stays raw – matches the notebook)
    if scale > 1e-6:
        coords[:, 0] /= scale
        coords[:, 1] /= scale

    return coords.flatten()   # 63 features


# ──────────────────────────────────────────────────────────────────────────────
# 5. Drawing helper
# ──────────────────────────────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
]

def draw_hand_landmarks(frame, hand_landmarks, width, height):
    for lm in hand_landmarks:
        cx, cy = int(lm.x * width), int(lm.y * height)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
    for start_idx, end_idx in HAND_CONNECTIONS:
        s = hand_landmarks[start_idx]
        e = hand_landmarks[end_idx]
        cv2.line(frame,
                 (int(s.x * width), int(s.y * height)),
                 (int(e.x * width), int(e.y * height)),
                 (255, 0, 0), 2)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Main camera loop
# ──────────────────────────────────────────────────────────────────────────────
def process_camera(model, label_encoder,
                   output_video_path='output_gesture_video.mp4',
                   window_size=10):
    """
    Process real-time camera input and save the output.

    The model was trained on raw string labels (no integer encoding), so
    model.predict() already returns the gesture name directly (e.g. 'fist').
    label_encoder is only used here to display the known classes on startup.
    """
    print(f"Creating hand landmarker with model: {os.path.abspath(model_path)}")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = 20
    print(f"Camera resolution: {width}x{height}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print("ERROR: Could not create video writer!")
        cap.release()
        return

    # Sliding window for prediction stabilisation (stores raw string labels)
    prediction_window = deque(maxlen=window_size)

    try:
        with HandLandmarker.create_from_options(options) as landmarker:

            frame_count = 0
            start_time  = datetime.now()
            elapsed     = 0

            print("\n" + "=" * 60)
            print("HAND GESTURE RECOGNITION - CAMERA MODE")
            print("=" * 60)
            print("Controls:")
            print("  - Press 'q' to QUIT and save video")
            print("  - Press 's' to take a SCREENSHOT")
            print("=" * 60 + "\n")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Mirror effect
                frame     = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                detection_result = landmarker.detect(mp_image)

                gesture         = "No Hand Detected"
                confidence_text = ""

                if detection_result.hand_landmarks:
                    for hand_landmarks in detection_result.hand_landmarks:
                        draw_hand_landmarks(frame, hand_landmarks, width, height)

                        try:
                            # --- Preprocess (matches notebook exactly) ---
                            features = preprocess_landmarks(hand_landmarks)

                            # --- Predict ---
                            # The model was trained with raw string labels, so
                            # predict() returns a string directly (e.g. 'fist').
                            # DO NOT call label_encoder.inverse_transform() here.
                            raw_prediction = model.predict([features])[0]

                            # --- Confidence (SVC with probability=True) ---
                            if hasattr(model, 'predict_proba'):
                                proba     = model.predict_proba([features])[0]
                                max_proba = max(proba)
                                confidence_text = f"Confidence: {max_proba:.2%}"
                            elif hasattr(model, 'decision_function'):
                                # SVC without probability – use decision margin as proxy
                                margins   = model.decision_function([features])[0]
                                # For multi-class, margins is a 1-D array of scores
                                if hasattr(margins, '__len__'):
                                    confidence_text = f"Score: {max(margins):.2f}"
                                else:
                                    confidence_text = f"Score: {margins:.2f}"

                            # --- Stabilisation window (string labels) ---
                            prediction_window.append(raw_prediction)
                            gesture = max(set(prediction_window),
                                         key=prediction_window.count)

                        except Exception as e:
                            print(f"Prediction error: {e}")
                            gesture = "Error"

                # ── Overlay ──────────────────────────────────────────────────
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

                display_gesture = gesture.replace('_', ' ').title()
                cv2.putText(frame, f"Gesture: {display_gesture}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 255, 0), 3)

                if confidence_text:
                    cv2.putText(frame, confidence_text,
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 0), 2)

                # Recording indicator
                cv2.circle(frame, (width - 30, 30), 10, (0, 0, 255), -1)
                cv2.putText(frame, "REC",
                            (width - 80, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)

                elapsed   = (datetime.now() - start_time).total_seconds()
                time_text = f"Time: {int(elapsed)}s | Frames: {frame_count}"
                cv2.putText(frame, time_text,
                            (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)

                out.write(frame)
                cv2.imshow('Hand Gesture Recognition (Press q to quit)', frame)
                frame_count += 1

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nRecording stopped by user")
                    break
                elif key == ord('s'):
                    screenshot_name = (f"screenshot_"
                                       f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(screenshot_name, frame)
                    print(f"Screenshot saved: {screenshot_name}")

    except Exception as e:
        print(f"ERROR creating hand landmarker: {e}")
        print(f"Model file path:   {os.path.abspath(model_path)}")
        print(f"Model file exists: {os.path.exists(model_path)}")
        if os.path.exists(model_path):
            print(f"Model file size:   {os.path.getsize(model_path)}")
        cap.release()
        out.release()
        return

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("RECORDING COMPLETE!")
    print("=" * 60)
    print(f"✓ Video saved:            {output_video_path}")
    print(f"✓ Total frames recorded:  {frame_count}")
    print(f"✓ Duration:               {int(elapsed)} seconds")
    print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading SVM model...")
    svm_model = joblib.load('svm_hand_model.pkl')
    print("✓ Model loaded successfully!")

    # The model carries its own class list; label_encoder is optional here.
    # We load it only to display the known gesture classes on startup.
    label_encoder = None
    if os.path.exists('label_encoder.pkl'):
        label_encoder = joblib.load('label_encoder.pkl')
        print(f"✓ Label encoder loaded.")
        print(f"  Classes: {', '.join(label_encoder.classes_)}")
    else:
        print("label_encoder.pkl not found – that's OK, it is not needed for prediction.")
        print(f"  Model classes: {', '.join(svm_model.classes_)}")

    output_video = 'output_gesture_recognition.mp4'
    process_camera(svm_model, label_encoder, output_video, window_size=10)

    print(f"\nSaved your video to '{output_video}'")
