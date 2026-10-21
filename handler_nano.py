"""Runpod Serverless worker: VoxCPM2 on Nano-vLLM-VoxCPM (continuous batching).

Same job contract as handler.py, so the existing clients work unchanged:

Input:  {"input": {"items": [{"id", "text", "ref_b64", "ref_text", "mode"}, ...],
                   "cfg": 2.0}}
Output: {"results": [{"id", "wav_b64", "seconds", "gen_s"}], "elapsed", "device",
         "engine": "nanovllm", "steps"}

handler.py runs one generate() per line, so the GPU idles between the small
kernels of a single sequence. Here every line of a job is submitted at once and
the engine schedules them together, which is where the throughput comes from.

Differences from handler.py:
  * The diffusion step count is fixed when the engine loads (VOXCPM_STEPS,
    default 30, the v1 pack setting); a per-job "steps" is ignored.
  * "mode": "ref" = reference-only cloning (ref_audio_latents only), which obeys
    parenthesized style instructions. The default, Hi-Fi, also passes the
    reference as a decode prefix with its transcript (prompt_latents +
    prompt_text), matching handler.py.
  * No retry_badcase: length checks happen client-side.
"""
import asyncio
import base64
import io
import os
import time

import numpy as np
import runpod
import soundfile as sf

MODEL_ID = os.environ.get("VOXCPM_MODEL", "openbmb/VoxCPM2")
STEPS = int(os.environ.get("VOXCPM_STEPS", "30"))
MAX_SEQS = int(os.environ.get("VOXCPM_MAX_SEQS", "32"))

_server, _rate, _load_error = None, None, None


async def server():
    """The engine pool must be built inside the running event loop: from_pretrained
    returns the async pool only when it finds one."""
    global _server, _rate, _load_error
    if _server is None and _load_error is None:
        try:
            from huggingface_hub import snapshot_download
            from nanovllm_voxcpm import VoxCPM
            path = snapshot_download(MODEL_ID)
            _server = VoxCPM.from_pretrained(path, inference_timesteps=STEPS, max_num_seqs=MAX_SEQS,
                                             gpu_memory_utilization=0.9, devices=[0])
            await _server.wait_for_ready()
            _rate = int((await _server.get_model_info())["output_sample_rate"])
        except Exception as exc:  # report through every job instead of crash-looping
            import traceback
            _load_error = f"engine load failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}"
            print(_load_error, flush=True)
    return _server


async def synth(srv, item, cfg):
    t0 = time.time()
    latents = await srv.encode_latents(base64.b64decode(item["ref_b64"]), "wav")
    kwargs = {"ref_audio_latents": latents}
    if item.get("mode") != "ref":
        kwargs.update(prompt_latents=latents, prompt_text=item["ref_text"])
    chunks = [c async for c in srv.generate(target_text=item["text"], cfg_value=cfg, **kwargs)]
    wav = np.concatenate(chunks, axis=0)
    buf = io.BytesIO()
    sf.write(buf, wav, _rate, format="WAV", subtype="PCM_16")
    return {"id": item["id"], "wav_b64": base64.b64encode(buf.getvalue()).decode(),
            "seconds": round(len(wav) / _rate, 3), "gen_s": round(time.time() - t0, 2)}


async def handler(job):
    srv = await server()
    if _load_error:
        return {"error": _load_error}
    inp = job["input"]
    cfg = float(inp.get("cfg", 2.0))
    started = time.time()

    async def one(item):
        try:
            return await synth(srv, item, cfg)
        except Exception as exc:  # one bad line must not cost the batch
            return {"id": item.get("id"), "error": f"{type(exc).__name__}: {exc}"}

    out = await asyncio.gather(*(one(i) for i in inp["items"]))
    return {"results": out, "elapsed": round(time.time() - started, 2), "device": "cuda",
            "engine": "nanovllm", "steps": STEPS}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
