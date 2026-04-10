import os
import uuid
import tempfile
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Global model handle
# ---------------------------------------------------------------------------
model = None
DOWNLOAD_DIR = Path("./downloads")


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
    try:
        contents = await video.read()
        tmp_path.write_bytes(contents)

        # Run inference in a thread so we don't block the event loop
        loop = asyncio.get_event_loop()
        preds, segments = await loop.run_in_executor(None, _run_inference, str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

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


def _run_inference(video_path: str):
    """Blocking inference call – executed in a thread pool."""
    df = model.get_events_dataframe(video_path=video_path)
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
