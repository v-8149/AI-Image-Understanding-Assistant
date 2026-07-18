"""
main.py
3-phase pipeline:
  Phase 1: Input image (OpenCV)
  Phase 2: Object detection (YOLOv11)
  Phase 3: OCR (PaddleOCR)
"""

import os
import cv2

from objDetection import detect_objects
from ocr import run_ocr

# ------------------------------------------------------------
# Edit these values to point at your own image/model, then run:
#   python main.py
# ------------------------------------------------------------
IMAGE_PATH = "img2.png"  # input image path

DETECTION_MODEL_PATH = "yolo11n.pt"
DETECTION_OUTPUT_PATH = "detected_object.jpg"
CONFIDENCE_THRESHOLD = 0.25

OCR_LANG = "en"
OCR_ENABLE_MKLDNN = False  # set True if your CPU supports it, for a speed-up
OCR_OUTPUT_IMAGE_PATH = "ocr_result.jpg"
OCR_OUTPUT_JSON_PATH = "ocr_result.json"

VERBOSE = True


# --------------------------------------------------------------------------
# Phase 1: Input image
# --------------------------------------------------------------------------
def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    return image


# --------------------------------------------------------------------------
# Reading / summarizing the results from both phases
# --------------------------------------------------------------------------
def read_results(detections, ocr_detections):
    print("\n--- Object Detection Results ---")
    if detections:
        for det in detections:
            print(f"  {det['class_name']} ({det['confidence']:.2f}) at {det['box']}")
    else:
        print("  No objects detected.")

    print("\n--- OCR Results ---")
    if ocr_detections:
        for det in ocr_detections:
            print(f"  \"{det['text']}\" ({det['confidence']:.2f}) at {det['bbox']}")
    else:
        print("  No text detected.")


def main():
    # Phase 1: input image
    image = load_image(IMAGE_PATH)
    print(f"Image loaded: {IMAGE_PATH}")

    # Phase 2: object detection (input image, output stored)
    _, detections = detect_objects(
        image,
        model_path=DETECTION_MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        output_path=DETECTION_OUTPUT_PATH,
        verbose=VERBOSE,
    )
    print(f"Object detection completed. Saved to: {DETECTION_OUTPUT_PATH}")

    # Phase 3: OCR (input image, output stored)
    _, ocr_detections = run_ocr(
        image,
        lang=OCR_LANG,
        enable_mkldnn=OCR_ENABLE_MKLDNN,
        output_image_path=OCR_OUTPUT_IMAGE_PATH,
        output_json_path=OCR_OUTPUT_JSON_PATH,
    )
    print(f"OCR completed. Saved to: {OCR_OUTPUT_IMAGE_PATH} and {OCR_OUTPUT_JSON_PATH}")

    # Reading the results (object detection results, OCR results)
    read_results(detections, ocr_detections)


if __name__ == "__main__":
    main()
