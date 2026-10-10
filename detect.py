"""Gender and age detection with OpenCV's DNN module.

Based on the Gender-and-Age-Detection project by Mahesh Sawant (MIT, see LICENSE).
Restructured into reusable functions with a headless mode, an output option and
model paths resolved relative to this file.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

MODEL_DIR = Path(__file__).resolve().parent

FACE_PROTO = "opencv_face_detector.pbtxt"
FACE_MODEL = "opencv_face_detector_uint8.pb"
AGE_PROTO = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GENDERS = ["Male", "Female"]
PADDING = 20
WINDOW_NAME = "Detecting age and gender"


@dataclass
class Prediction:
    box: tuple  # (x1, y1, x2, y2)
    gender: str
    gender_confidence: float
    age: str
    age_confidence: float


@dataclass
class Models:
    face: cv2.dnn.Net
    age: cv2.dnn.Net
    gender: cv2.dnn.Net


def load_models(model_dir=MODEL_DIR):
    """Load the face, age and gender networks, failing clearly if a file is missing."""
    model_dir = Path(model_dir)
    required = [FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL]
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing model files in {model_dir}: {', '.join(missing)}. "
            "Run `python download_models.py` to fetch them."
        )

    def read(model, proto):
        return cv2.dnn.readNet(str(model_dir / model), str(model_dir / proto))

    return Models(
        face=read(FACE_MODEL, FACE_PROTO),
        age=read(AGE_MODEL, AGE_PROTO),
        gender=read(GENDER_MODEL, GENDER_PROTO),
    )


def detect_faces(face_net, frame, conf_threshold=0.7):
    """Return face boxes as (x1, y1, x2, y2), clipped to the frame."""
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    face_net.setInput(blob)
    detections = face_net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        if detections[0, 0, i, 2] <= conf_threshold:
            continue
        x1 = max(0, int(detections[0, 0, i, 3] * width))
        y1 = max(0, int(detections[0, 0, i, 4] * height))
        x2 = min(width - 1, int(detections[0, 0, i, 5] * width))
        y2 = min(height - 1, int(detections[0, 0, i, 6] * height))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def predict(models, frame, conf_threshold=0.7):
    """Detect every face in the frame and classify its gender and age bucket."""
    height, width = frame.shape[:2]
    predictions = []
    for x1, y1, x2, y2 in detect_faces(models.face, frame, conf_threshold):
        face = frame[max(0, y1 - PADDING):min(y2 + PADDING, height - 1),
                     max(0, x1 - PADDING):min(x2 + PADDING, width - 1)]
        if face.size == 0:
            continue
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        models.gender.setInput(blob)
        gender_preds = models.gender.forward()[0]
        models.age.setInput(blob)
        age_preds = models.age.forward()[0]

        g, a = int(gender_preds.argmax()), int(age_preds.argmax())
        predictions.append(Prediction(
            box=(x1, y1, x2, y2),
            gender=GENDERS[g],
            gender_confidence=float(gender_preds[g]),
            age=AGE_BUCKETS[a],
            age_confidence=float(age_preds[a]),
        ))
    return predictions


def annotate(frame, predictions):
    """Return a copy of the frame with a box and label drawn for each prediction."""
    out = frame.copy()
    thickness = max(1, int(round(frame.shape[0] / 150)))
    for p in predictions:
        x1, y1, x2, y2 = p.box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), thickness, 8)
        cv2.putText(out, f"{p.gender}, {p.age}", (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def print_predictions(predictions):
    if not predictions:
        print("No face detected")
    for p in predictions:
        print(f"Gender: {p.gender} ({p.gender_confidence:.0%})")
        print(f"Age: {p.age[1:-1]} years ({p.age_confidence:.0%})")


def run_image(models, path, output=None, display=True, conf_threshold=0.7):
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"Could not read image: {path}", file=sys.stderr)
        return 1
    predictions = predict(models, frame, conf_threshold)
    print_predictions(predictions)
    result = annotate(frame, predictions)
    if output:
        cv2.imwrite(str(output), result)
        print(f"Saved annotated image to {output}")
    if display:
        cv2.imshow(WINDOW_NAME, result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def run_webcam(models, camera=0, conf_threshold=0.7):
    video = cv2.VideoCapture(camera)
    if not video.isOpened():
        print(f"Could not open camera {camera}", file=sys.stderr)
        return 1
    print("Press q or Esc to quit.")
    try:
        while True:
            has_frame, frame = video.read()
            if not has_frame:
                break
            cv2.imshow(WINDOW_NAME, annotate(frame, predict(models, frame, conf_threshold)))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        video.release()
        cv2.destroyAllWindows()
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Detect faces and estimate gender and age.")
    parser.add_argument("--image", help="path to an image; omit to use the webcam")
    parser.add_argument("--output", help="save the annotated image to this path")
    parser.add_argument("--no-display", action="store_true", help="do not open a window (image mode)")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument("--confidence", type=float, default=0.7,
                        help="minimum face detection confidence (default: 0.7)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        models = load_models()
    except (FileNotFoundError, cv2.error) as exc:
        print(f"Failed to load models: {exc}", file=sys.stderr)
        return 1
    if args.image:
        return run_image(models, args.image, args.output, not args.no_display, args.confidence)
    return run_webcam(models, args.camera, args.confidence)


if __name__ == "__main__":
    sys.exit(main())
