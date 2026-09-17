import base64
import cv2
import numpy as np
import pytest
from PIL import Image
from facenn.utils.image import (
    clip_box,
    load_image,
    align_face_5point,
    crop_and_align_face,
    ARCFACE_DST_112,
)


def test_clip_box():
    # Box exceeding boundaries
    box = [-10, -5, 200, 300]
    clipped = clip_box(box, img_shape=(100, 100))
    x, y, w, h = clipped
    assert x >= 0 and y >= 0
    assert x + w <= 100
    assert y + h <= 100
    assert w > 0 and h > 0


def test_load_image_numpy():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    loaded = load_image(img)
    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == (100, 100, 3)


def test_load_image_pil():
    pil_img = Image.new("RGB", (60, 40), color="red")
    loaded = load_image(pil_img)
    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == (40, 60, 3)
    # PIL red (255, 0, 0) should be BGR (0, 0, 255)
    assert loaded[0, 0, 2] == 255
    assert loaded[0, 0, 0] == 0


def test_load_image_base64():
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    _, buffer = cv2.imencode(".png", img)
    b64_str = "data:image/png;base64," + base64.b64encode(buffer).decode("utf-8")
    loaded = load_image(b64_str)
    assert loaded.shape == (32, 32, 3)


def test_align_face_5point():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Put sample landmarks in order: [left_eye, right_eye, nose, mouth_l, mouth_r]
    landmarks = np.array([
        [70.0, 80.0],
        [130.0, 80.0],
        [100.0, 110.0],
        [80.0, 140.0],
        [120.0, 140.0],
    ], dtype=np.float32)

    aligned = align_face_5point(img, landmarks, image_size=(112, 112))
    assert aligned.shape == (112, 112, 3)


def test_crop_and_align_face_fallback():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    box = [10, 10, 80, 80]
    # No landmarks -> fallback to bbox crop
    chip = crop_and_align_face(img, box=box, landmarks=None, target_size=(112, 112))
    assert chip.shape == (112, 112, 3)
