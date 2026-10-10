"""Download the pretrained model weights from their original sources.

The large weight files are not meant to be committed. This script fetches each
one, verifies its SHA-256 checksum and writes it next to detect.py. Files that
are already present and valid are skipped.
"""

import argparse
import hashlib
import http.client
import sys
import tempfile
import urllib.request
from pathlib import Path

DEFAULT_DEST = Path(__file__).resolve().parent
CHUNK_SIZE = 1 << 20

_LEVI_HASSNER = "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models"
_OPENCV_3RDPARTY = ("https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
                    "dnn_samples_face_detector_20180220_uint8")

MODELS = {
    "age_net.caffemodel": (
        f"{_LEVI_HASSNER}/age_net.caffemodel",
        "6dde5d07df5ca1d66ff39e525693f05ccfb9d2c437e188fdd1a10d42e57fabd6",
    ),
    "gender_net.caffemodel": (
        f"{_LEVI_HASSNER}/gender_net.caffemodel",
        "ac7571b281ae078817764b645a20541bd6aa1babeac20a45e6d8de7d61ba0e50",
    ),
    "opencv_face_detector_uint8.pb": (
        f"{_OPENCV_3RDPARTY}/opencv_face_detector_uint8.pb",
        "5c71d752ef2cbf2f457ac82fdd580fcb2522fd04c5efdaed18eb6d9e2843fbed",
    ),
}


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url, target):
    """Stream url into target. Separate so tests can replace it."""
    with urllib.request.urlopen(url, timeout=60) as response, open(target, "wb") as out:
        for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
            out.write(chunk)


def download_model(name, url, expected_sha256, dest, force=False, attempts=3):
    """Ensure dest/name exists with the expected checksum. Returns "skipped" or "downloaded".

    A dropped connection can leave a truncated file, so failed downloads and
    checksum mismatches are retried up to `attempts` times.
    """
    target = Path(dest) / name
    if not force and target.is_file() and sha256_of(target) == expected_sha256:
        return "skipped"

    error = None
    for _ in range(attempts):
        # Download to a temporary file in the same directory so a failed or
        # corrupted download never replaces a good file.
        with tempfile.NamedTemporaryFile(dir=dest, prefix=f".{name}.", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            fetch(url, tmp_path)
            actual = sha256_of(tmp_path)
            if actual == expected_sha256:
                tmp_path.replace(target)
                return "downloaded"
            error = ValueError(f"Checksum mismatch for {name}: expected {expected_sha256}, got {actual}")
        except (OSError, http.client.HTTPException) as exc:
            error = exc
        finally:
            tmp_path.unlink(missing_ok=True)
    raise error


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download and verify the pretrained model weights.")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help="directory to save the models in (default: next to this script)")
    parser.add_argument("--force", action="store_true", help="re-download even if a valid file exists")
    args = parser.parse_args(argv)

    args.dest.mkdir(parents=True, exist_ok=True)
    failed = False
    for name, (url, sha256) in MODELS.items():
        try:
            status = download_model(name, url, sha256, args.dest, args.force)
            print(f"{status:>10}  {name}")
        except Exception as exc:  # report every file, then fail
            failed = True
            print(f"{'failed':>10}  {name}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
