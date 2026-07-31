from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


MODEL_URL = os.environ.get(
    "IMAGE_MODEL_URL",
    "https://huggingface.co/laeroah/realisticvision5.1.safetensors/resolve/main/realisticvision5.1.safetensors?download=true",
)
MODEL_FILENAME = "realisticvision5.1.safetensors"
MODEL_PATH = Path(os.environ.get("IMAGE_MODEL_PATH", Path(__file__).resolve().parent / MODEL_FILENAME))
MODEL_SHA256 = os.environ.get(
    "IMAGE_MODEL_SHA256",
    "15012c538f503ce2ebfc2c8547b268c75ccdaff7a281db55399940ff1d70e21d",
)
MODEL_SIZE = int(os.environ.get("IMAGE_MODEL_SIZE", "2132625894"))
CHUNK_SIZE = 1024 * 1024


def format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_existing_model() -> bool:
    if not MODEL_PATH.exists():
        return False
    actual_size = MODEL_PATH.stat().st_size
    if actual_size != MODEL_SIZE:
        print(f"Existing {MODEL_PATH.name} has size {format_bytes(actual_size)}, expected {format_bytes(MODEL_SIZE)}.")
        return False
    print(f"Verifying existing {MODEL_PATH.name}...")
    actual_hash = sha256_file(MODEL_PATH)
    if actual_hash != MODEL_SHA256:
        raise RuntimeError(
            f"Existing {MODEL_PATH.name} failed SHA256 verification.\n"
            f"Expected: {MODEL_SHA256}\n"
            f"Actual:   {actual_hash}\n"
            "Delete the file and run again to redownload it."
        )
    print(f"{MODEL_PATH.name} is ready.")
    return True


def download_model() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial_path = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".part")
    downloaded = partial_path.stat().st_size if partial_path.exists() else 0

    headers = {"User-Agent": "General-AI-model-downloader/1.0"}
    mode = "wb"
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"
        mode = "ab"
        print(f"Resuming {MODEL_PATH.name} at {format_bytes(downloaded)}.")
    else:
        print(f"Downloading {MODEL_PATH.name} from Hugging Face.")

    request = urllib.request.Request(MODEL_URL, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial_path.open(mode) as output:
            start_time = time.monotonic()
            last_print = start_time
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_print >= 1:
                    percent = min(100, downloaded * 100 / MODEL_SIZE)
                    rate = downloaded / max(1, now - start_time)
                    remaining = max(0, MODEL_SIZE - downloaded)
                    eta = format_duration(remaining / rate) if rate > 0 else "unknown"
                    print(
                        f"\r{percent:5.1f}%  {format_bytes(downloaded)} / {format_bytes(MODEL_SIZE)}  "
                        f"{format_bytes(int(rate))}/s  ETA {eta}",
                        end="",
                        flush=True,
                    )
                    last_print = now
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and partial_path.exists():
            pass
        else:
            raise
    finally:
        print()

    if partial_path.stat().st_size != MODEL_SIZE:
        raise RuntimeError(
            f"Download incomplete: got {format_bytes(partial_path.stat().st_size)}, "
            f"expected {format_bytes(MODEL_SIZE)}."
        )

    print("Verifying downloaded model...")
    actual_hash = sha256_file(partial_path)
    if actual_hash != MODEL_SHA256:
        raise RuntimeError(
            f"Downloaded model failed SHA256 verification.\n"
            f"Expected: {MODEL_SHA256}\n"
            f"Actual:   {actual_hash}"
        )
    partial_path.replace(MODEL_PATH)
    print(f"{MODEL_PATH.name} downloaded and verified.")


def ensure_model() -> None:
    if verify_existing_model():
        return
    download_model()
    if not verify_existing_model():
        raise RuntimeError(f"{MODEL_PATH.name} is still missing after download.")


def main() -> int:
    try:
        ensure_model()
    except Exception as exc:
        print(f"Model setup failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
