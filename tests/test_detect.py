from pathlib import Path

import cv2
import numpy as np
import pytest

import detect

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def models():
    return detect.load_models()


def test_load_models_reports_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="age_net.caffemodel"):
        detect.load_models(tmp_path)


def test_blank_image_has_no_faces(models):
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    assert detect.predict(models, blank) == []


SAMPLE = ROOT / "samples" / "astronaut.jpg"


def test_sample_image_returns_valid_prediction(models):
    frame = cv2.imread(str(SAMPLE))
    predictions = detect.predict(models, frame)
    assert len(predictions) == 1
    height, width = frame.shape[:2]
    p = predictions[0]
    x1, y1, x2, y2 = p.box
    assert 0 <= x1 < x2 < width and 0 <= y1 < y2 < height
    assert p.gender in detect.GENDERS
    assert p.age in detect.AGE_BUCKETS
    assert 0.0 <= p.gender_confidence <= 1.0
    assert 0.0 <= p.age_confidence <= 1.0


def test_annotate_does_not_modify_input(models):
    frame = cv2.imread(str(SAMPLE))
    original = frame.copy()
    annotated = detect.annotate(frame, detect.predict(models, frame))
    assert np.array_equal(frame, original)
    assert not np.array_equal(annotated, original)


def test_cli_writes_annotated_image(tmp_path, capsys):
    out = tmp_path / "result.jpg"
    code = detect.main(["--image", str(SAMPLE), "--output", str(out), "--no-display"])
    assert code == 0
    assert out.is_file()
    assert "Gender:" in capsys.readouterr().out


def test_cli_rejects_unreadable_image(tmp_path):
    assert detect.main(["--image", str(tmp_path / "missing.jpg"), "--no-display"]) == 1
