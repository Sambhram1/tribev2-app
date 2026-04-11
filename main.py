import os
import uuid
import shutil
import subprocess
import tempfile
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Global model handle
# ---------------------------------------------------------------------------
model = None
DOWNLOAD_DIR = Path("./downloads")

# Cap input videos at 5 minutes to keep CPU inference time reasonable
MAX_DURATION_SECONDS = 300


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    print("Loading TRIBE v2 model …")
    from tribev2 import TribeModel  # type: ignore
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
    print("Model ready.")
    yield
    # cleanup on shutdown (nothing to do for the model)


app = FastAPI(title="TRIBE v2 Inference", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    # Validate extension
    allowed = {".mp4", ".avi", ".mov"}
    ext = Path(video.filename or "video").suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Save upload to a temp file
    tmp_path = Path(tempfile.gettempdir()) / f"tribe_{uuid.uuid4().hex}{ext}"
    trimmed_path = Path(tempfile.gettempdir()) / f"tribe_{uuid.uuid4().hex}_trimmed.mp4"
    try:
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(video.file, f)

        # Trim to MAX_DURATION_SECONDS so long videos don't stall CPU inference
        trimmed_ok = _ffmpeg_trim(str(tmp_path), str(trimmed_path), MAX_DURATION_SECONDS)
        input_path = str(trimmed_path) if trimmed_ok else str(tmp_path)

        # Run inference in a thread so we don't block the event loop
        loop = asyncio.get_event_loop()
        preds, segments = await loop.run_in_executor(None, _run_inference, input_path)
    finally:
        tmp_path.unlink(missing_ok=True)
        trimmed_path.unlink(missing_ok=True)

    # Persist predictions
    out_path = DOWNLOAD_DIR / "brain_predictions.npy"
    np.save(str(out_path), preds)

    # Build a JSON-serialisable preview (first 5 rows)
    arr = np.array(preds)
    preview = arr.flat[:5].tolist() if arr.size > 0 else []

    return JSONResponse({
        "shape": list(arr.shape),
        "preview": preview,
        "download_url": "/download/brain_predictions.npy",
    })


def _ffmpeg_trim(src: str, dst: str, max_seconds: int) -> bool:
    """Trim video to max_seconds using ffmpeg (no re-encode, fast).
    Returns True on success, False if ffmpeg is unavailable or fails."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-t", str(max_seconds), "-c", "copy", dst],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _run_inference(video_path: str):
    """Blocking inference call – executed in a thread pool."""
    torch.set_num_threads(os.cpu_count() or 2)
    df = model.get_events_dataframe(video_path=video_path)
    # inference_mode can conflict with some model internals — fall back to no_grad
    try:
        with torch.inference_mode():
            preds, segments = model.predict(events=df)
    except Exception:
        with torch.no_grad():
            preds, segments = model.predict(events=df)
    return preds, segments


@app.get("/download/brain_predictions.npy")
async def download():
    out_path = DOWNLOAD_DIR / "brain_predictions.npy"
    if not out_path.exists():
        raise HTTPException(status_code=404, detail="No predictions available yet.")
    return FileResponse(
        path=str(out_path),
        media_type="application/octet-stream",
        filename="brain_predictions.npy",
    )


# Serve frontend last so API routes take priority
app.mount("/", StaticFiles(directory="static", html=True), name="static")
