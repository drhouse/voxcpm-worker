"""Runpod Serverless worker: VoxCPM2 cross-lingual voice cloning, batch in, WAVs out.

Game-agnostic. The caller renders the spoken text (pronunciation tables,
stutter respelling) and picks the reference clip; the worker only synthesises.
Settings default to what the WitchSpring R v1 pack was generated with, so a
regenerated line matches the rest of that pack:

  ultimate cloning (prompt wav + its exact transcript, same clip as reference),
  cfg 2.0, 30 diffusion steps, retry_badcase up to 5 times with a per-line
  frame-ratio threshold of 2.0 + 20 / len(text).

Input:  {"input": {"items": [{"id", "text", "ref_b64", "ref_text"}, ...],
                   "cfg": 2.0, "steps": 30}}
Output: {"results": [{"id", "wav_b64", "seconds"} | {"id", "error"}], "device": ...}
"""
import base64
import glob
import io
import os
import tempfile
import time

import runpod
import soundfile as sf
import torch
from voxcpm import VoxCPM

DEVICE = "cuda" if torch.cuda.is_available() else "CPU-FALLBACK"
# optimize=False: the default runs torch.compile plus a warm-up at load, which
# needs a C compiler the slim runtime image lacks - the worker died before
# taking a job. Compilation changes speed, not output.
MODEL_ID = os.environ.get("VOXCPM_MODEL", "openbmb/VoxCPM2")


def model_source(model_id):
    """Runpod's cached copy if the endpoint has one, else the id (baked or downloaded).

    Cached models live at /runpod-volume/huggingface-cache/hub/models--org--name/
    snapshots/<hash>/ (docs.runpod.io/serverless/endpoints/model-caching).
    """
    pattern = "/runpod-volume/huggingface-cache/hub/models--{}/snapshots/*/".format(model_id.replace("/", "--"))
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else model_id


SOURCE = model_source(MODEL_ID)
# VOXCPM_OPTIMIZE=1 turns on torch.compile (needs the compiler the image now
# has). Off by default so existing endpoints behave exactly as before.
OPTIMIZE = os.environ.get("VOXCPM_OPTIMIZE", "0") == "1"
print(f"loading {MODEL_ID} from {SOURCE}", flush=True)
try:
    MODEL = VoxCPM.from_pretrained(SOURCE,
                                   load_denoiser=False, optimize=OPTIMIZE)
    LOAD_ERROR = None
except Exception as exc:  # report through every job instead of crash-looping silently
    MODEL, LOAD_ERROR = None, f"model load failed: {type(exc).__name__}: {exc}"
    print(LOAD_ERROR, flush=True)


def ratio_threshold(text, base=2.0, allowance=20.0):
    # Frames per character; a flat ratio punishes short lines (see wsrvoice.voxcpm_backend).
    return base + allowance / max(len(text or ""), 1)


def synth(item, cfg, steps, retries=5):
    handle, ref = tempfile.mkstemp(suffix=".wav")
    os.close(handle)
    t0 = time.time()
    try:
        with open(ref, "wb") as fh:
            fh.write(base64.b64decode(item["ref_b64"]))
        # "mode": "ref" = timbre from the clip only, no transcript. Hi-Fi (the
        # default) ignores parenthesized style instructions; this mode obeys them.
        prompt = {} if item.get("mode") == "ref" else {"prompt_wav_path": ref, "prompt_text": item["ref_text"]}
        wav = MODEL.generate(
            text=item["text"], **prompt,
            reference_wav_path=ref, cfg_value=cfg, inference_timesteps=steps,
            retry_badcase=retries > 0, retry_badcase_max_times=max(retries, 1),
            retry_badcase_ratio_threshold=ratio_threshold(item["text"]))
        buf = io.BytesIO()
        rate = MODEL.tts_model.sample_rate
        sf.write(buf, wav, rate, format="WAV", subtype="PCM_16")
        return {"id": item["id"], "wav_b64": base64.b64encode(buf.getvalue()).decode(),
                "seconds": round(len(wav) / rate, 3), "gen_s": round(time.time() - t0, 2)}
    finally:
        os.unlink(ref)


def handler(job):
    if LOAD_ERROR:
        return {"error": LOAD_ERROR}
    inp = job["input"]
    cfg, steps = float(inp.get("cfg", 2.0)), int(inp.get("steps", 30))
    retries = int(inp.get("retries", 5))
    started, out = time.time(), []
    for item in inp["items"]:
        try:
            out.append(synth(item, cfg, steps, retries))
        except Exception as exc:  # one bad line must not cost the batch
            out.append({"id": item.get("id"), "error": f"{type(exc).__name__}: {exc}"})
    return {"results": out, "elapsed": round(time.time() - started, 2), "device": DEVICE,
            "model_source": SOURCE, "optimize": OPTIMIZE}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
