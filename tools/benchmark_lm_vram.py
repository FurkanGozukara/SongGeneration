import argparse
import json
import os
import subprocess
import sys
import threading
import time
from typing import Dict, Optional

import numpy as np
import soundfile as sf
import torch

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

sys.path.append(os.path.join(APP_DIR, "tools", "gradio"))
from levo_inference_lowmem import LeVoInference

from logic.generation import compose_description_from_params, format_lyrics_for_model


def _query_gpu_used_mb(gpu_index: int) -> Optional[float]:
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if not result:
            return None
        return float(result.splitlines()[0].strip())
    except Exception:
        return None


def _build_gen_params(preset: Dict) -> Dict:
    duration_from_steps = float(preset["max_gen_length"]) / 25.0
    try:
        top_k_value = int(preset.get("top_k", -1))
    except (TypeError, ValueError):
        top_k_value = -1
    if top_k_value < 0:
        top_k_value = -1
    try:
        top_p_value = float(preset.get("top_p", 0.0))
    except (TypeError, ValueError):
        top_p_value = 0.0
    return {
        "duration": duration_from_steps,
        "num_steps": int(preset["diffusion_steps"]),
        "temperature": float(preset["temperature"]),
        "top_k": top_k_value,
        "top_p": top_p_value,
        "cfg_coef": float(preset["cfg_coef"]),
        "guidance_scale": float(preset["guidance_scale"]),
        "use_sampling": bool(preset["use_sampling"]),
        "extend_stride": float(preset["extend_stride"]),
        "chunked": bool(preset["chunked"]),
        "chunk_size": int(preset["chunk_size"]),
        "record_tokens": bool(preset["record_tokens"]),
        "record_window": int(preset["record_window"]),
    }


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def main():
    parser = argparse.ArgumentParser(description="Benchmark SongGeneration LM VRAM usage")
    parser.add_argument("--ckpt_path", default=os.path.join(APP_DIR, "ckpt", "songgeneration_v2_large"))
    parser.add_argument("--preset", default=os.path.join(APP_DIR, "presets", "Default.json"))
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--sample-rate", type=float, default=10.0, help="VRAM sample frequency (Hz)")
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--audio-out", default=None)
    args = parser.parse_args()

    if args.log_path is None:
        args.log_path = os.path.join(APP_DIR, "output", f"vram_benchmark_{args.label}.json")
    if args.audio_out is None:
        args.audio_out = os.path.join(APP_DIR, "output", f"benchmark_{args.label}.wav")

    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.audio_out), exist_ok=True)

    with open(args.preset, "r", encoding="utf-8") as f:
        preset = json.load(f)

    requested_gen_type = preset.get("gen_type", "mixed")
    require_vocal = requested_gen_type != "bgm"
    lyrics = format_lyrics_for_model(preset["lyrics"], require_vocal=require_vocal)
    description = compose_description_from_params(preset)
    gen_type = "mixed" if requested_gen_type == "separate" else requested_gen_type
    gen_params = _build_gen_params(preset)

    stage_state = {"stage": None, "message": None}
    stage_lock = threading.Lock()
    progress_events = []

    poll_interval = 1.0 / max(1e-3, float(args.sample_rate))
    monitor_stop = threading.Event()
    vram_samples = []

    def monitor_loop():
        while not monitor_stop.is_set():
            used_mb = _query_gpu_used_mb(args.gpu_index)
            ts = time.time()
            with stage_lock:
                stage = stage_state["stage"]
                msg = stage_state["message"]
            if used_mb is not None:
                vram_samples.append(
                    {
                        "timestamp": ts,
                        "used_mb": float(used_mb),
                        "stage": stage,
                        "message": msg,
                    }
                )
            time.sleep(poll_interval)

    def progress_callback(info: Dict):
        event = {
            "timestamp": time.time(),
            "stage": info.get("stage"),
            "phase": info.get("phase"),
            "progress": info.get("progress"),
            "message": info.get("message"),
            "current_step": info.get("current_step"),
            "total_steps": info.get("total_steps"),
            "speed": info.get("speed"),
            "eta_seconds": info.get("eta_seconds"),
            "stage_progress": info.get("stage_progress"),
        }
        progress_events.append(event)
        with stage_lock:
            stage_state["stage"] = info.get("stage")
            stage_state["message"] = info.get("message")
        message = info.get("message")
        if message:
            print(message, flush=True)

    start_time = time.time()
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
    error = None
    audio = None

    try:
        model = LeVoInference(args.ckpt_path)
        audio = model(
            lyrics,
            description,
            preset.get("audio_path"),
            None,
            preset.get("auto_prompt_path"),
            gen_type,
            gen_params,
            disable_offload=bool(preset.get("disable_offload", False)),
            disable_cache_clear=bool(preset.get("disable_cache_clear", False)),
            disable_fp16=bool(preset.get("disable_fp16", False)),
            disable_sequential=bool(preset.get("disable_sequential", False)),
            enable_lm_block_swap=bool(preset.get("enable_lm_block_swap", False)),
            enable_lm_mlp_int8=bool(preset.get("enable_lm_mlp_int8", False)),
            enable_lm_mlp_int4=bool(preset.get("enable_lm_mlp_int4", False)),
            lm_blocks_to_swap=_safe_int(preset.get("lm_blocks_to_swap", 1), 1),
            lm_sub_blocks_to_swap=_safe_int(preset.get("lm_sub_blocks_to_swap", 0), 0),
            lm_block_swap_use_pinned=bool(preset.get("lm_block_swap_use_pinned", True)),
            seed=_safe_int(preset.get("seed", -1), -1),
            progress_callback=progress_callback,
            cancellation_token=None,
        )
    except Exception as exc:
        error = str(exc)
    finally:
        monitor_stop.set()
        monitor_thread.join(timeout=5)

    elapsed = time.time() - start_time

    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().cpu().permute(1, 0).float().numpy()
        sample_rate = 48000
        sf.write(args.audio_out, audio_np, sample_rate)
        audio_saved = True
    else:
        audio_saved = False

    used_values = [s["used_mb"] for s in vram_samples if s.get("used_mb") is not None]
    lm_values = [s["used_mb"] for s in vram_samples if s.get("stage") == "lm" and s.get("used_mb") is not None]
    peak_overall_mb = max(used_values) if used_values else None
    peak_lm_mb = max(lm_values) if lm_values else None

    report = {
        "label": args.label,
        "timestamp": time.time(),
        "elapsed_seconds": elapsed,
        "gpu_index": args.gpu_index,
        "ckpt_path": args.ckpt_path,
        "preset": args.preset,
        "audio_out": args.audio_out,
        "audio_saved": audio_saved,
        "error": error,
        "peak_vram_mb_overall": peak_overall_mb,
        "peak_vram_mb_lm_stage": peak_lm_mb,
        "sample_count": len(vram_samples),
        "progress_event_count": len(progress_events),
        "vram_samples": vram_samples,
        "progress_events": progress_events,
    }

    with open(args.log_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)

    print(json.dumps(
        {
            "label": args.label,
            "elapsed_seconds": round(elapsed, 2),
            "peak_vram_mb_overall": peak_overall_mb,
            "peak_vram_mb_lm_stage": peak_lm_mb,
            "audio_saved": audio_saved,
            "error": error,
            "log_path": args.log_path,
            "audio_out": args.audio_out,
        },
        ensure_ascii=True,
        indent=2,
    ))

    if error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
