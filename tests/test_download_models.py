import hashlib

import pytest

import download_models

PAYLOAD = b"model weights"
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def fake_fetch(monkeypatch):
    calls = []

    def fetch(url, target):
        calls.append(url)
        target.write_bytes(PAYLOAD)

    monkeypatch.setattr(download_models, "fetch", fetch)
    return calls


def test_downloads_and_verifies(tmp_path, fake_fetch):
    status = download_models.download_model("m.bin", "https://x/m.bin", PAYLOAD_SHA, tmp_path)
    assert status == "downloaded"
    assert (tmp_path / "m.bin").read_bytes() == PAYLOAD
    assert fake_fetch == ["https://x/m.bin"]


def test_skips_valid_existing_file(tmp_path, fake_fetch):
    (tmp_path / "m.bin").write_bytes(PAYLOAD)
    assert download_models.download_model("m.bin", "https://x/m.bin", PAYLOAD_SHA, tmp_path) == "skipped"
    assert fake_fetch == []


def test_replaces_corrupted_existing_file(tmp_path, fake_fetch):
    (tmp_path / "m.bin").write_bytes(b"truncated")
    assert download_models.download_model("m.bin", "https://x/m.bin", PAYLOAD_SHA, tmp_path) == "downloaded"
    assert (tmp_path / "m.bin").read_bytes() == PAYLOAD


def test_checksum_mismatch_keeps_existing_file_and_cleans_up(tmp_path, fake_fetch):
    (tmp_path / "m.bin").write_bytes(b"old")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_models.download_model("m.bin", "https://x/m.bin", "0" * 64, tmp_path, force=True)
    assert len(fake_fetch) == 3
    assert (tmp_path / "m.bin").read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["m.bin"]


def test_main_reports_failure(tmp_path, monkeypatch, capsys):
    def broken_fetch(url, target):
        raise OSError("network down")

    monkeypatch.setattr(download_models, "fetch", broken_fetch)
    assert download_models.main(["--dest", str(tmp_path)]) == 1
    assert "network down" in capsys.readouterr().err


def test_checksums_match_bundled_models():
    for name, (_, sha256) in download_models.MODELS.items():
        path = download_models.DEFAULT_DEST / name
        if not path.is_file():
            pytest.skip(f"{name} not downloaded")
        assert download_models.sha256_of(path) == sha256


def test_retries_after_truncated_download(tmp_path, monkeypatch):
    responses = [PAYLOAD[:5], PAYLOAD]

    def flaky_fetch(url, target):
        target.write_bytes(responses.pop(0))

    monkeypatch.setattr(download_models, "fetch", flaky_fetch)
    assert download_models.download_model("m.bin", "https://x/m.bin", PAYLOAD_SHA, tmp_path) == "downloaded"
    assert (tmp_path / "m.bin").read_bytes() == PAYLOAD
