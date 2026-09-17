from typing import Optional
import hashlib
import logging
import os
import requests
from tqdm import tqdm

logger = logging.getLogger("facenn")


def download_file_from_url(
    url: str,
    dest_path: str,
    expected_sha256: Optional[str] = None,
    chunk_size: int = 1024 * 64,
) -> str:
    """
    Downloads a file atomically to dest_path.
    If the file already exists and passes size/hash checks, it is returned immediately.
    """
    dest_dir = os.path.dirname(os.path.abspath(dest_path))
    os.makedirs(dest_dir, exist_ok=True)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        if expected_sha256 is None or _verify_sha256(dest_path, expected_sha256):
            return dest_path

    tmp_path = f"{dest_path}.tmp"
    logger.info("Downloading %s to %s", url, dest_path)

    if "drive.google.com" in url:
        try:
            import gdown

            gdown.download(url, tmp_path, quiet=False)
        except ImportError as exc:
            raise ImportError(
                "Google Drive download requires 'gdown'. Install it via `pip install gdown`."
            ) from exc
    else:
        response = requests.get(url, stream=True, allow_redirects=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        hasher = hashlib.sha256() if expected_sha256 else None

        with open(tmp_path, "wb") as file, tqdm(
            total=total_size if total_size > 0 else None,
            unit="iB",
            unit_scale=True,
            desc=os.path.basename(dest_path),
        ) as bar:
            for chunk in response.iter_content(chunk_size):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))
                    if hasher:
                        hasher.update(chunk)

        if expected_sha256 and hasher and hasher.hexdigest() != expected_sha256:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise ValueError(
                f"SHA256 mismatch for {dest_path}: expected {expected_sha256}, got {hasher.hexdigest()}"
            )

    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Download failed or empty file received from {url}")

    os.replace(tmp_path, dest_path)
    return dest_path


def _verify_sha256(path: str, expected_sha256: str) -> bool:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 64), b""):
            hasher.update(chunk)
    return hasher.hexdigest() == expected_sha256
