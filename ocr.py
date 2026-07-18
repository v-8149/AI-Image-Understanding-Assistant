"""
Phase 3: OCR (PaddleOCR)

Structured in explicit stages: preprocessing -> model loading ->
inference (text boxes + orientation) -> post-processing -> storing
(texts and bounding boxes).
"""

import json

import cv2
from paddleocr import PaddleOCR

# Cache the loaded PaddleOCR engine so it isn't reloaded on every call
_model_cache = {}


# --------------------------------------------------------------------------
# 1. Preprocessing
# --------------------------------------------------------------------------
def preprocess(image):
    """
    Prepare the raw BGR image for PaddleOCR.
    PaddleOCR expects RGB images, so convert from OpenCV's default BGR.
    """
    if image is None:
        raise ValueError("preprocess() received an empty image")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------
# 2. Model loading
# --------------------------------------------------------------------------
def load_model(lang="en", enable_mkldnn=False):
    """
    Load (and cache) a PaddleOCR engine.
    use_textline_orientation=True enables the text-orientation classifier so
    rotated text boxes are corrected before recognition.
    enable_mkldnn defaults to False since the oneDNN/PIR CPU backend can
    crash on some CPUs; set True for a speed-up if your machine supports it.
    """
    cache_key = (lang, enable_mkldnn)
    if cache_key not in _model_cache:
        _model_cache[cache_key] = PaddleOCR(
            use_textline_orientation=True,
            lang=lang,
            enable_mkldnn=enable_mkldnn,
        )
    return _model_cache[cache_key]


# --------------------------------------------------------------------------
# 3. Inference
# --------------------------------------------------------------------------
def run_inference(model, image):
    """
    Run PaddleOCR on the image.

    Returns the raw OCRResult for the image (dict-like, with rec_texts,
    rec_scores and rec_polys/rec_boxes fields), or None if nothing came back.
    """
    results = model.predict(image)
    return results[0] if results else None


# --------------------------------------------------------------------------
# 4. Post-processing
# --------------------------------------------------------------------------
def postprocess(raw_result):
    """
    Convert a raw PaddleOCR OCRResult into a clean list of dicts:
    [{"text": str, "confidence": float, "bbox": [(x, y), ...4 points]}, ...]
    """
    detections = []

    if raw_result is None:
        return detections

    texts = raw_result["rec_texts"]
    scores = raw_result["rec_scores"]
    polys = raw_result["rec_polys"]

    for text, confidence, poly in zip(texts, scores, polys):
        bbox = [(int(x), int(y)) for x, y in poly]

        detections.append({
            "text": text,
            "confidence": float(confidence),
            "bbox": bbox,
        })

    return detections


# --------------------------------------------------------------------------
# 5. Storing (draw text boxes + save results)
# --------------------------------------------------------------------------
def draw_detections(image, detections):
    """Draw the 4-point text boxes and recognized text onto the image."""
    for detection in detections:
        pts = detection["bbox"]
        for i in range(4):
            cv2.line(image, pts[i], pts[(i + 1) % 4], (255, 0, 0), 2)

        label = f"{detection['text']} ({detection['confidence']:.2f})"
        text_x, text_y = pts[0]

        cv2.putText(
            image,
            label,
            (text_x, max(text_y - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

    return image


def save_results(image, detections, output_image_path="ocr_result.jpg",
                  output_json_path="ocr_result.json"):
    """Draw text boxes onto the image and save both image and JSON results."""
    annotated_image = draw_detections(image, detections)
    cv2.imwrite(output_image_path, annotated_image)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(detections, f, ensure_ascii=False, indent=2)

    return annotated_image


# --------------------------------------------------------------------------
# End-to-end pipeline (used by main.py)
# --------------------------------------------------------------------------
def run_ocr(image, lang="en", enable_mkldnn=False, output_image_path="ocr_result.jpg",
            output_json_path="ocr_result.json"):
    """
    Full OCR pipeline: preprocess -> load model -> infer -> postprocess ->
    save annotated image + JSON.

    Returns:
        (annotated_image, detections)
    """
    preprocessed = preprocess(image)
    model = load_model(lang, enable_mkldnn=enable_mkldnn)
    raw_result = run_inference(model, preprocessed)
    detections = postprocess(raw_result)

    # Draw on a BGR copy of the original image (not the RGB preprocessed one)
    annotated_image = save_results(image.copy(), detections, output_image_path, output_json_path)

    return annotated_image, detections
