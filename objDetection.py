import cv2
from ultralytics import YOLO

# Load model once
model = YOLO("yolo11n.pt")

def detect_objects(image):

    results = model(image)

    for box in results[0].boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        print(f"{class_name}: {confidence:.2f}")
        print(f"Coordinates: ({x1}, {y1}) ({x2}, {y2})")

        # Draw rectangle
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Draw label
        label = f"{class_name} {confidence:.2f}"

        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    return image