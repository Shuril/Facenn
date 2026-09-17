import numpy as np
import pytest
import torch

from facenn import Facenn
from facenn.models.sface import SFace
from facenn.models.mobilefacenet import MobileFaceNet


def test_sface_inference():
    sface = SFace()
    dummy_input = torch.randn(1, 3, 112, 112)
    emb = sface.predict(dummy_input)
    assert emb.shape == (1, 128)
    norm = torch.linalg.norm(emb, dim=-1).item()
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_mobilefacenet_inference():
    mbf = MobileFaceNet()
    dummy_input = torch.randn(1, 3, 112, 112)
    emb = mbf.predict(dummy_input)
    assert emb.shape == (1, 512)
    norm = torch.linalg.norm(emb, dim=-1).item()
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_core_model_zoo_dispatch():
    app_sface = Facenn(recognition_model_name="SFace")
    assert isinstance(app_sface.recognition_model, SFace)
    assert app_sface.recognition_model.input_shape == (112, 112)

    app_mbf = Facenn(recognition_model_name="MobileFaceNet")
    assert isinstance(app_mbf.recognition_model, MobileFaceNet)
    assert app_mbf.recognition_model.input_shape == (112, 112)
