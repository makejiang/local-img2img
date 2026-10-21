# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Single-process inference runtime for local-img2img."""

from __future__ import annotations

import argparse
import os
import random
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Optional

try:
    from ai_generated_watermark import (
        save_ai_generated_png,
        validate_image_dimensions,
    )
except ModuleNotFoundError:
    common_scripts = Path(__file__).resolve().parents[4] / "common" / "scripts"
    if common_scripts.is_dir() and str(common_scripts) not in sys.path:
        sys.path.insert(0, str(common_scripts))
    from ai_generated_watermark import (  # type: ignore[no-redef]
        save_ai_generated_png,
        validate_image_dimensions,
    )

from model_download import (
    ensure_models,
    load_model_infos,
    validate_model_dir,
)

STATE_STARTING = "starting"
STATE_DOWNLOADING = "downloading"
STATE_LOADING = "loading"
STATE_RUNNING = "running"
STATE_ERROR = "error"

OPENVINO_ROOT = Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".openvino"
MODELS_ROOT = OPENVINO_ROOT / "models"
OUTPUTS_ROOT = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Pictures"

_HERE = Path(__file__).resolve().parent
SKILL_ROOT = _HERE.parent
INFO_JSON = _HERE / "info.json" if (_HERE / "info.json").exists() else _HERE.parent / "info.json"
MODEL_INFOS = load_model_infos(INFO_JSON)
MODEL_INFO = MODEL_INFOS[0]
MODEL_ID = MODEL_INFO.model_id
MODEL_DIR = MODELS_ROOT / MODEL_INFO.dir_name
REQUIRED_MODEL_FILES = MODEL_INFO.required_files

DEFAULT_STEPS = 4
DEFAULT_GUIDANCE = 1.0
_INTEL_VENDOR_ID = "0x8086"

os.environ.setdefault("OPENVINO_TELEMETRY_OPT_OUT", "1")

for stream_name in ("stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    if stream is not None and hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

LOG_LOCK = threading.Lock()

def _normalize_log_path(log_path: str | None) -> Path | None:
    if not log_path:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = f"~/.openvino/log/img2img-runtime-py-{timestamp}.log"

    path = Path(log_path).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _log_message(log_path: Path | None, message: str) -> None:
    if log_path is None:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with LOG_LOCK:
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(f"[{timestamp}] [runtime pid={os.getpid()}] {message}\n")
    except OSError:
        pass


def _report_download_message(log_path: Path | None, message: str) -> None:
    print(f"[runtime] {message}", flush=True)
    _log_message(log_path, message)


def _ensure_required_model(log_path: Path | None = None) -> None:
    """Download the model to MODEL_DIR if not already valid."""
    validation = validate_model_dir(MODEL_DIR, REQUIRED_MODEL_FILES)
    if validation.ok:
        return

    def _logger(message: str):
        _report_download_message(log_path, message)

    ensure_models(MODEL_INFOS, MODELS_ROOT, logger=_logger)

    validation = validate_model_dir(MODEL_DIR, REQUIRED_MODEL_FILES)
    if not validation.ok:
        raise RuntimeError(
            f"required model download did not complete: {MODEL_DIR} ({validation.reason})"
        )


# ---------------------------------------------------------------------------
# OpenVINO GenAI pipeline helpers (FLUX.2 klein via openvino_genai)
# ---------------------------------------------------------------------------
def _load_pipeline(model_dir: Path, device: str) -> Any:
    """Build an ``openvino_genai.Image2ImagePipeline`` on the given device."""
    try:
        import openvino_genai as ov_genai  # type: ignore
    except Exception as exc:
        raise ImportError(
            "openvino_genai could not be loaded from the managed environment. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return ov_genai.Image2ImagePipeline(str(model_dir), device)


def _resolve_device() -> str:
    return "GPU"


def _tensor_to_image(image_tensor: Any) -> Optional[Any]:
    """Convert an ``openvino_genai`` image tensor into a PIL image."""
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore

    arr = np.array(image_tensor.data, copy=True)
    get_shape = getattr(image_tensor, "get_shape", None)
    if callable(get_shape):
        arr = arr.reshape(get_shape())
    if arr.ndim == 4:
        arr = arr[0]
    return Image.fromarray(arr.astype(np.uint8))


def _load_reference_tensor(reference_image_path: str) -> tuple[Any, int, int, int, int]:
    """Load the source image, resize to multiples of 16, and return ``(ov.Tensor, original_width, original_height, new_width, new_height)``."""
    from PIL import Image  # type: ignore

    with Image.open(reference_image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        validate_image_dimensions(width, height)
        import numpy as np  # type: ignore
        import openvino as ov  # type: ignore

        new_width = max(16, int(round(width / 16.0)) * 16)
        new_height = max(16, int(round(height / 16.0)) * 16)
        if width != new_width or height != new_height:
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS if hasattr(Image, "LANCZOS") else 1
            rgb = rgb.resize((new_width, new_height), resample)
        ref_arr = np.array(rgb)[None]
    return ov.Tensor(ref_arr), width, height, new_width, new_height


def _safe_seed() -> int:
    return random.SystemRandom().randint(0, 2**31 - 1)


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
class Runtime:
    def __init__(self, log_path: Path | None = None) -> None:
        self.runtime_lock = threading.Lock()
        self.pipeline: Any = None
        self.loaded_model_id: str | None = None
        self.loaded_device: str | None = None
        self.log_path = log_path
        self.state: str = STATE_STARTING
        self.init_error: str = ""

    def log(self, message: str) -> None:
        _log_message(self.log_path, message)

    def initialize(self) -> None:
        try:
            with self.runtime_lock:
                self.state = STATE_DOWNLOADING
            self.log("initialization: downloading required model")
            _ensure_required_model(self.log_path)

            with self.runtime_lock:
                self.state = STATE_LOADING
            self.log("initialization: loading OpenVINO FLUX.2 klein image-to-image pipeline")

            device = _resolve_device()
            self.log(f"initialization: resolved device={device}")

            pipeline = _load_pipeline(MODEL_DIR, device)

            with self.runtime_lock:
                self.pipeline = pipeline
                self.loaded_model_id = MODEL_ID
                self.loaded_device = device
                self.state = STATE_RUNNING
            self.log(f"initialization: pipeline ready on device={device}")
        except Exception:
            error_text = traceback.format_exc()
            with self.runtime_lock:
                self.init_error = error_text
                self.state = STATE_ERROR
            self.log(f"initialization failed:\n{error_text}")

    def _do_generate(self, reference_image_path: str, prompt: str, steps: int, seed: int) -> dict:
        t0 = time.time()
        ref_path = Path(reference_image_path)
        parent_dir = ref_path.parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{ref_path.stem}_edited_{timestamp}.png"
        output_path = parent_dir / filename

        load_start = time.time()
        try:
            ref_tensor, width, height, new_width, new_height = _load_reference_tensor(reference_image_path)
        except Exception as exc:
            return {
                "ok": True,
                "success": False,
                "error_code": "BAD_IMAGE",
                "error": f"{type(exc).__name__}: {exc}",
                "reference_image_path": reference_image_path,
                "prompt": prompt,
                "seed": seed,
                "timing": {"load_s": time.time() - load_start, "infer_s": 0.0, "save_s": 0.0, "total_s": time.time() - t0},
            }
        load_s = time.time() - load_start

        self.log(f"starting generation: original_size=({width}x{height}), resized_size=({new_width}x{new_height}), steps={steps}, seed={seed}")
        infer_start = time.time()
        try:
            result = self.pipeline.generate(
                prompt,
                ref_tensor,
                width=int(new_width),
                height=int(new_height),
                num_inference_steps=int(steps),
                guidance_scale=float(DEFAULT_GUIDANCE),
                rng_seed=int(seed),
            )
        except Exception as exc:
            return {
                "ok": True,
                "success": False,
                "error_code": "GENERATION_FAILED",
                "error": f"{type(exc).__name__}: {exc}",
                "reference_image_path": reference_image_path,
                "prompt": prompt,
                "seed": seed,
                "timing": {"load_s": load_s, "infer_s": time.time() - infer_start, "save_s": 0.0, "total_s": time.time() - t0},
            }
        infer_s = time.time() - infer_start

        image = _tensor_to_image(result)
        if image is None:
            return {
                "ok": True,
                "success": False,
                "error_code": "GENERATION_FAILED",
                "error": "pipeline returned no images",
                "reference_image_path": reference_image_path,
                "prompt": prompt,
                "seed": seed,
                "timing": {"load_s": load_s, "infer_s": infer_s, "save_s": 0.0, "total_s": time.time() - t0},
            }

        # Restore size of the generated image to original width and height!
        if image.width != width or image.height != height:
            try:
                from PIL import Image  # type: ignore
                try:
                    resample = Image.Resampling.LANCZOS
                except AttributeError:
                    resample = Image.LANCZOS if hasattr(Image, "LANCZOS") else 1
                image = image.resize((width, height), resample)
            except Exception as exc:
                return {
                    "ok": True,
                    "success": False,
                    "error_code": "RESIZE_FAILED",
                    "error": f"failed to restore image size: {type(exc).__name__}: {exc}",
                    "reference_image_path": reference_image_path,
                    "prompt": prompt,
                    "seed": seed,
                    "timing": {"load_s": load_s, "infer_s": infer_s, "save_s": 0.0, "total_s": time.time() - t0},
                }

        save_start = time.time()
        try:
            save_ai_generated_png(image, output_path)
        except Exception as exc:
            return {
                "ok": True,
                "success": False,
                "error_code": "SAVE_FAILED",
                "error": f"{type(exc).__name__}: {exc}",
                "reference_image_path": reference_image_path,
                "prompt": prompt,
                "seed": seed,
                "timing": {"load_s": load_s, "infer_s": infer_s, "save_s": time.time() - save_start, "total_s": time.time() - t0},
            }
        save_s = time.time() - save_start

        return {
            "ok": True,
            "success": True,
            "image_path": str(output_path),
            "reference_image_path": reference_image_path,
            "prompt": prompt,
            "seed": int(seed),
            "steps": int(steps),
            "width": int(width),
            "height": int(height),
            "device": self.loaded_device,
            "model_id": self.loaded_model_id,
            "timing": {"load_s": load_s, "infer_s": infer_s, "save_s": save_s, "total_s": time.time() - t0},
        }

    def dispatch(self, msg: dict) -> dict:
        op = msg.get("op")
        self.log(f"dispatching op={op!r}")
        if op == "generate":
            prompt = msg.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                return {
                    "ok": False,
                    "success": False,
                    "error_code": "BAD_PROMPT",
                    "error": "prompt must be a non-empty string",
                }
            reference_image_path = msg.get("reference_image_path")
            if not isinstance(reference_image_path, str) or not reference_image_path.strip():
                return {
                    "ok": False,
                    "success": False,
                    "error_code": "BAD_IMAGE",
                    "error": "reference_image_path must be a non-empty string",
                }
            reference_image = Path(reference_image_path).expanduser()
            if not reference_image.exists() or not reference_image.is_file():
                return {
                    "ok": False,
                    "success": False,
                    "error_code": "BAD_IMAGE",
                    "error": f"reference image does not exist: {reference_image}",
                }
            with self.runtime_lock:
                state = self.state
                init_error = self.init_error
                pipeline = self.pipeline
            if state != STATE_RUNNING or pipeline is None:
                self.log(f"generate rejected: state={state}")
                reply = {
                    "ok": False,
                    "success": False,
                    "state": state,
                    "prompt": prompt,
                    "error": f"runtime not ready: {state}",
                }
                if state == STATE_ERROR and init_error:
                    reply["error"] = init_error
                return reply
            steps = msg.get("steps", DEFAULT_STEPS)
            seed = msg.get("seed")
            if not isinstance(seed, int) or isinstance(seed, bool):
                seed = _safe_seed()
            try:
                reply = self._do_generate(str(reference_image), prompt, steps, seed)
                reply["state"] = STATE_RUNNING
                self.log(
                    f"generate completed: success={reply.get('success')} "
                    f"image_path={reply.get('image_path')}"
                )
                return reply
            except Exception:
                error_text = traceback.format_exc()
                self.log(f"generate failed:\n{error_text}")
                return {
                    "ok": False,
                    "success": False,
                    "state": STATE_RUNNING,
                    "prompt": prompt,
                    "error": error_text,
                }
        self.log(f"unknown operation received: {op!r}")
        return {"ok": False, "error": f"unknown op: {op!r}"}
