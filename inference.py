"""
inference.py — Face detection + prediction for EfficientNet-B4
Works entirely on CPU (Render free tier has no GPU).
"""

import io
import gc
import urllib.request

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import DEFAULT_CONFIG

_PROTO_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
_WEIGHTS_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/"
    "dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)

_TRANSFORM = transforms.Compose([
    transforms.Resize((DEFAULT_CONFIG["image_size"], DEFAULT_CONFIG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=DEFAULT_CONFIG["mean"], std=DEFAULT_CONFIG["std"]),
])


def download_face_detector(detector_dir: str) -> tuple[str, str]:
    import os
    os.makedirs(detector_dir, exist_ok=True)
    proto   = os.path.join(detector_dir, "deploy.prototxt")
    weights = os.path.join(detector_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    if not os.path.exists(proto):
        print("[inference] Downloading face-detector config …")
        urllib.request.urlretrieve(_PROTO_URL, proto)
    if not os.path.exists(weights):
        print("[inference] Downloading face-detector weights (~10 MB) …")
        urllib.request.urlretrieve(_WEIGHTS_URL, weights)
    return proto, weights


def load_face_net(detector_dir: str):
    proto, weights = download_face_detector(detector_dir)
    return cv2.dnn.readNetFromCaffe(proto, weights)


def _decode_image(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _detect_face(img_bgr: np.ndarray, face_net, margin=0.2) -> tuple[Image.Image, bool]:
    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(img_bgr, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), swapRB=False)
    face_net.setInput(blob)
    dets = face_net.forward()

    best_conf, best_box = 0.0, None
    for i in range(dets.shape[2]):
        c = float(dets[0, 0, i, 2])
        if c > 0.5 and c > best_conf:
            best_conf = c
            x1 = int(dets[0, 0, i, 3] * w)
            y1 = int(dets[0, 0, i, 4] * h)
            x2 = int(dets[0, 0, i, 5] * w)
            y2 = int(dets[0, 0, i, 6] * h)
            best_box = (x1, y1, x2, y2)

    if best_box is None:
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        s   = min(w, h)
        return pil.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2)), False

    x1, y1, x2, y2 = best_box
    bw, bh = x2-x1, y2-y1
    x1 = max(0, int(x1 - margin*bw))
    y1 = max(0, int(y1 - margin*bh))
    x2 = min(w, int(x2 + margin*bw))
    y2 = min(h, int(y2 + margin*bh))
    return Image.fromarray(cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)), True


def predict(image_bytes: bytes, model, face_net, threshold: float) -> dict:
    img_bgr = _decode_image(image_bytes)
    if img_bgr is None:
        raise ValueError("Could not decode image.")

    face_pil, face_found = _detect_face(img_bgr, face_net)
    tensor = _TRANSFORM(face_pil).unsqueeze(0)  # CPU

    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()

    gc.collect()

    label = "FAKE" if prob >= threshold else "REAL"
    conf  = prob if prob >= threshold else 1.0 - prob
    return {
        "label"        : label,
        "confidence"   : round(conf * 100, 2),
        "fake_prob"    : round(prob * 100, 2),
        "real_prob"    : round((1 - prob) * 100, 2),
        "face_detected": face_found,
        "threshold"    : round(threshold, 2),
    }
