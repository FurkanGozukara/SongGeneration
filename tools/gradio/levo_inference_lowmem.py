import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import OmegaConf

# Ensure project root is importable when this file is executed directly.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from codeclm.models import CodecLM
from codeclm.models import builders
from separator import Separator
from codeclm.utils.offload_profiler import OffloadProfiler, OffloadParamParse

# Import suppression utilities
sys.path.append(APP_DIR)
from utils.suppress_output import suppress_output, disable_verbose_logging
from utils.torch_load import load_torch_file

AUTO_PROMPT_CANDIDATES = (
    os.path.join(APP_DIR, "tools", "new_auto_prompt.pt"),
    os.path.join(APP_DIR, "tools", "new_prompt.pt"),
    os.path.join(APP_DIR, "ckpt", "prompt.pt"),
)
WORKER_STAGE_PREPARE = "prepare"
WORKER_STAGE_LM = "lm"
WORKER_STAGE_DIFFUSION = "diffusion"
WORKER_STAGES = {WORKER_STAGE_PREPARE, WORKER_STAGE_LM, WORKER_STAGE_DIFFUSION}


def _register_omegaconf_resolvers():
    # Use register_resolver for older OmegaConf and register_new_resolver for newer versions.
    if hasattr(OmegaConf, "register_new_resolver"):
        register_method = OmegaConf.register_new_resolver
    else:
        register_method = OmegaConf.register_resolver

    resolver_items = (
        ("eval", lambda x: eval(x)),
        ("concat", lambda *x: [xxx for xx in x for xxx in xx]),
        ("get_fname", lambda: "default"),
        ("load_yaml", lambda x: list(OmegaConf.load(x))),
    )

    for name, resolver in resolver_items:
        try:
            register_method(name, resolver)
        except Exception as e:
            error_str = str(e).lower()
            if not (
                "already registered" in error_str
                or "already exists" in error_str
                or ("resolved" in error_str and "registered" in error_str)
            ):
                raise


def detect_model_version(ckpt_path: str) -> str:
    model_name = os.path.basename(os.path.normpath(ckpt_path)).lower().replace('-', '_')
    if model_name == "songgeneration_v2_large":
        return "v2"
    return "v1"


def detect_language(text: str) -> str:
    if not text:
        return "en"
    chinese_count = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    english_count = len([c for c in text if ('a' <= c.lower() <= 'z')])
    total = chinese_count + english_count
    if total == 0:
        return "en"
    if chinese_count / total >= 0.2:
        return "zh"
    if english_count / total >= 0.5:
        return "en"
    return "en"


def load_auto_prompt_data(auto_prompt_path: os.PathLike = None):
    candidate_paths = [auto_prompt_path] if auto_prompt_path else list(AUTO_PROMPT_CANDIDATES)
    for candidate in candidate_paths:
        if candidate and os.path.exists(candidate):
            return load_torch_file(candidate, map_location='cpu')
    return None


def _choose_prompt_token(prompt_group, language: str):
    if isinstance(prompt_group, dict):
        preferred = prompt_group.get(language)
        if isinstance(preferred, list) and preferred:
            return preferred[np.random.randint(0, len(preferred))]
        for prompts in prompt_group.values():
            if isinstance(prompts, list) and prompts:
                return prompts[np.random.randint(0, len(prompts))]
        return None
    if isinstance(prompt_group, list) and prompt_group:
        return prompt_group[np.random.randint(0, len(prompt_group))]
    return prompt_group


def select_auto_prompt_token(auto_prompt, genre: str, lyric: str):
    if not isinstance(auto_prompt, dict):
        return None
    language = detect_language(lyric)
    if genre == "Auto":
        auto_group = auto_prompt.get("Auto")
        prompt_token = _choose_prompt_token(auto_group, language)
        if prompt_token is not None:
            return prompt_token
        merged_prompts = []
        for value in auto_prompt.values():
            if isinstance(value, dict):
                for prompts in value.values():
                    if isinstance(prompts, list):
                        merged_prompts.extend(prompts)
            elif isinstance(value, list):
                merged_prompts.extend(value)
            elif value is not None:
                merged_prompts.append(value)
        if merged_prompts:
            return merged_prompts[np.random.randint(0, len(merged_prompts))]
        return None
    return _choose_prompt_token(auto_prompt.get(genre), language)


def prepare_condition_inputs(lyric: str, description: str, gen_type: str, version: str):
    normalized_description = (description or ".").strip()
    normalized_lyric = lyric.replace("  ", " ")
    if version != "v1":
        normalized_description = normalized_description.lower() if normalized_description else "."
        if gen_type == "bgm":
            normalized_description = f"[Musicality-very-high], [Pure-Music], {normalized_description}"
            normalized_lyric = "."
        elif "[Musicality-very-high]" not in normalized_description:
            normalized_description = f"[Musicality-very-high], {normalized_description}"
    else:
        if "[Musicality-very-high]" not in normalized_description:
            normalized_description = f"[Musicality-very-high], {normalized_description}"
    return normalized_lyric, normalized_description


def _to_cpu_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _to_cpu_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cpu_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_value(v) for v in value)
    return value


def _move_to_device(value: Any, device: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(v, device) for v in value)
    return value


def _load_inference_cfg(ckpt_path: str):
    _register_omegaconf_resolvers()
    cfg_path = os.path.join(ckpt_path, "config.yaml")
    pt_path = os.path.join(ckpt_path, "model.pt")

    cfg = OmegaConf.load(cfg_path)
    cfg.mode = "inference"
    max_duration = cfg.max_dur
    version = detect_model_version(ckpt_path)
    default_params = dict(
        top_p=0.0,
        record_tokens=True,
        record_window=50,
        extend_stride=5,
        duration=max_duration,
    )
    return cfg, pt_path, version, max_duration, default_params


def _cuda_cleanup(disable_cache_clear: bool):
    gc.collect()
    if torch.cuda.is_available() and not disable_cache_clear:
        torch.cuda.empty_cache()


def _save_stage_payload(path: str, payload: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def _load_stage_payload(path: str) -> Dict[str, Any]:
    payload = load_torch_file(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid stage payload type from {path}: {type(payload)}")
    return payload


def _emit_stage_progress(
    progress_path: str,
    stage: str,
    phase: str,
    stage_progress: float,
    message: str,
    current_step: int = None,
    total_steps: int = None,
    speed: str = None,
    eta_seconds: float = None,
):
    if not progress_path:
        return
    event = {
        "timestamp": time.time(),
        "stage": stage,
        "phase": phase,
        "stage_progress": max(0.0, min(float(stage_progress), 1.0)),
        "message": message,
    }
    if current_step is not None:
        event["current_step"] = int(current_step)
    if total_steps is not None:
        event["total_steps"] = int(total_steps)
    if speed:
        event["speed"] = str(speed)
    if eta_seconds is not None:
        try:
            event["eta_seconds"] = max(0.0, float(eta_seconds))
        except (TypeError, ValueError):
            pass
    try:
        os.makedirs(os.path.dirname(progress_path), exist_ok=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=True) + "\n")
    except Exception:
        # Progress reporting must never crash inference.
        pass


def _read_progress_events(progress_path: str, last_position: int):
    if not progress_path or not os.path.exists(progress_path):
        return [], last_position
    events = []
    try:
        with open(progress_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(last_position)
            while True:
                line = f.readline()
                if not line:
                    break
                last_position = f.tell()
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                if isinstance(event, dict):
                    events.append(event)
    except Exception:
        return [], last_position
    return events, last_position


def _tail_log_file(path: str, max_lines: int = 40) -> str:
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:]).strip()
    except Exception:
        return ""


def _build_worker_env() -> Dict[str, str]:
    env = os.environ.copy()
    pythonpath_additions = [
        os.path.join(APP_DIR, "codeclm", "tokenizer"),
        APP_DIR,
        os.path.join(APP_DIR, "codeclm", "tokenizer", "Flow1dVAE"),
        os.path.join(APP_DIR, "codeclm", "tokenizer"),
    ]
    existing = env.get("PYTHONPATH", "")
    merged_paths = []
    for path in pythonpath_additions + ([p for p in existing.split(os.pathsep) if p] if existing else []):
        if path and path not in merged_paths:
            merged_paths.append(path)
    env["PYTHONPATH"] = os.pathsep.join(merged_paths)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("TRANSFORMERS_CACHE", os.path.join(APP_DIR, "third_party", "hub"))
    return env


def _run_stage_subprocess(
    stage_name: str,
    input_payload: Dict[str, Any],
    output_path: str,
    workspace_dir: str,
    progress_callback=None,
    stage_start: float = 0.0,
    stage_weight: float = 1.0,
    pipeline_start_time: float = None,
    cancellation_token=None,
):
    input_path = os.path.join(workspace_dir, f"{stage_name}_input.pt")
    log_path = os.path.join(workspace_dir, f"{stage_name}.log")
    progress_path = input_payload.get("progress_path")
    _save_stage_payload(input_path, input_payload)

    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--subprocess-stage",
        stage_name,
        "--input",
        input_path,
        "--output",
        output_path,
    ]

    stage_loop_start = time.time()
    last_stage_event = None
    last_stage_event_time = stage_loop_start

    def emit_parent_progress(event: Dict[str, Any]):
        if not progress_callback:
            return
        local_progress = max(0.0, min(float(event.get("stage_progress", 0.0)), 1.0))
        overall_progress = max(0.0, min(stage_start + (local_progress * stage_weight), 0.999))
        elapsed = 0.0
        eta_seconds = None
        if pipeline_start_time is not None:
            elapsed = max(0.0, time.time() - pipeline_start_time)
            if overall_progress > 0:
                eta_seconds = (elapsed * (1.0 - overall_progress)) / overall_progress

        progress_info = {
            "progress": overall_progress,
            "phase": event.get("phase", stage_name),
            "message": event.get("message", ""),
            "current_step": event.get("current_step"),
            "total_steps": event.get("total_steps"),
            "speed": event.get("speed"),
            "elapsed_seconds": elapsed,
            "eta_seconds": eta_seconds,
            "stage": stage_name,
            "stage_progress": local_progress,
        }
        progress_callback(progress_info)

    progress_offset = 0
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=APP_DIR,
            env=_build_worker_env(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        if progress_callback:
            start_event = {
                "phase": stage_name,
                "stage_progress": 0.0,
                "message": f"Starting {stage_name} stage...",
            }
            emit_parent_progress(start_event)
            last_stage_event = start_event
            last_stage_event_time = time.time()
        while proc.poll() is None:
            if progress_path:
                events, progress_offset = _read_progress_events(progress_path, progress_offset)
                for event in events:
                    if event.get("stage") == stage_name:
                        emit_parent_progress(event)
                        last_stage_event = event
                        last_stage_event_time = time.time()

            # Heartbeat updates so long blocking calls never look frozen in console.
            if progress_callback and last_stage_event is not None:
                now = time.time()
                if (now - last_stage_event_time) >= 1.0:
                    last_progress_value = float(last_stage_event.get("stage_progress", 0.0))
                    last_message = str(last_stage_event.get("message") or "")
                    should_heartbeat = (
                        last_progress_value < 0.95
                        and "complete" not in last_message.lower()
                    )
                    if should_heartbeat:
                        heartbeat_event = dict(last_stage_event)
                        base_message = last_message or f"{stage_name} stage running"
                        stage_elapsed = max(0.0, now - stage_loop_start)
                        heartbeat_event["message"] = f"{base_message} (working {stage_elapsed:.0f}s)"
                        emit_parent_progress(heartbeat_event)
                    last_stage_event_time = now

            if cancellation_token and hasattr(cancellation_token, "is_cancelled") and cancellation_token.is_cancelled():
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                return None
            time.sleep(0.2)
        exit_code = proc.returncode
        if progress_path:
            events, progress_offset = _read_progress_events(progress_path, progress_offset)
            for event in events:
                if event.get("stage") == stage_name:
                    emit_parent_progress(event)

    if exit_code != 0:
        stage_log = _tail_log_file(log_path)
        if stage_log:
            raise RuntimeError(f"Subprocess stage '{stage_name}' failed.\n{stage_log}")
        raise RuntimeError(f"Subprocess stage '{stage_name}' failed with exit code {exit_code}.")

    if not os.path.exists(output_path):
        raise RuntimeError(f"Subprocess stage '{stage_name}' did not produce output: {output_path}")

    if progress_callback:
        emit_parent_progress(
            {
                "phase": stage_name,
                "stage_progress": 1.0,
                "message": f"{stage_name.capitalize()} stage complete",
            }
        )

    return output_path


def _stage_prepare_inputs(payload: Dict[str, Any]) -> Dict[str, Any]:
    ckpt_path = payload["ckpt_path"]
    cfg, _, version, _, _ = _load_inference_cfg(ckpt_path)
    progress_path = payload.get("progress_path")

    lyric = payload.get("lyric", "")
    description = payload.get("description")
    prompt_audio_path = payload.get("prompt_audio_path")
    genre = payload.get("genre")
    auto_prompt_path = payload.get("auto_prompt_path")
    gen_type = payload.get("gen_type", "mixed")
    disable_cache_clear = bool(payload.get("disable_cache_clear", False))

    pmt_wav = None
    vocal_wav = None
    bgm_wav = None
    melody_is_wav = True

    _emit_stage_progress(
        progress_path,
        WORKER_STAGE_PREPARE,
        "Processing audio",
        0.02,
        "Initializing conditioning inputs...",
    )

    if prompt_audio_path is not None and os.path.exists(prompt_audio_path):
        _emit_stage_progress(
            progress_path,
            WORKER_STAGE_PREPARE,
            "Processing audio",
            0.12,
            "Loading reference audio separator...",
        )
        with suppress_output():
            separator = Separator()
            audio_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint, cfg)
            audio_tokenizer = audio_tokenizer.eval().cuda()

        _emit_stage_progress(
            progress_path,
            WORKER_STAGE_PREPARE,
            "Processing audio",
            0.30,
            "Separating reference audio...",
        )
        pmt_wav, vocal_wav, bgm_wav = separator.run(prompt_audio_path)
        pmt_wav = pmt_wav.cuda()
        vocal_wav = vocal_wav.cuda()
        bgm_wav = bgm_wav.cuda()

        _emit_stage_progress(
            progress_path,
            WORKER_STAGE_PREPARE,
            "Processing audio",
            0.48,
            "Encoding prompt melody tokens...",
        )
        with torch.no_grad():
            pmt_wav, _ = audio_tokenizer.encode(pmt_wav)

        del audio_tokenizer
        del separator
        _cuda_cleanup(disable_cache_clear)

        _emit_stage_progress(
            progress_path,
            WORKER_STAGE_PREPARE,
            "Processing audio",
            0.67,
            "Loading vocal/BGM tokenizer...",
        )
        with suppress_output():
            seperate_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint_sep, cfg)
            seperate_tokenizer = seperate_tokenizer.eval().cuda()

        _emit_stage_progress(
            progress_path,
            WORKER_STAGE_PREPARE,
            "Processing audio",
            0.85,
            "Encoding vocal and BGM tokens...",
        )
        with torch.no_grad():
            vocal_wav, bgm_wav = seperate_tokenizer.encode(vocal_wav, bgm_wav)

        del seperate_tokenizer
        melody_is_wav = False
        _cuda_cleanup(disable_cache_clear)

    elif genre is not None:
        _emit_stage_progress(
            progress_path,
            WORKER_STAGE_PREPARE,
            "Processing audio",
            0.55,
            "Loading auto prompt tokens...",
        )
        with suppress_output():
            auto_prompt = load_auto_prompt_data(auto_prompt_path)
        prompt_token = select_auto_prompt_token(auto_prompt, genre, lyric)
        if prompt_token is not None:
            if not isinstance(prompt_token, torch.Tensor):
                prompt_token = torch.as_tensor(prompt_token)
            pmt_wav = prompt_token[:, [0], :]
            vocal_wav = prompt_token[:, [1], :]
            bgm_wav = prompt_token[:, [2], :]
            melody_is_wav = False
    else:
        _emit_stage_progress(
            progress_path,
            WORKER_STAGE_PREPARE,
            "Processing audio",
            0.75,
            "No reference audio provided; using text-only conditioning.",
        )

    _emit_stage_progress(
        progress_path,
        WORKER_STAGE_PREPARE,
        "Processing audio",
        0.95,
        "Finalizing conditioning payload...",
    )
    prepared_lyric, prepared_description = prepare_condition_inputs(lyric, description, gen_type, version)

    _emit_stage_progress(
        progress_path,
        WORKER_STAGE_PREPARE,
        "Processing audio",
        1.0,
        "Conditioning ready",
    )

    return {
        "prepared_lyric": prepared_lyric,
        "prepared_description": prepared_description,
        "melody_is_wav": bool(melody_is_wav),
        "pmt_wav": _to_cpu_value(pmt_wav),
        "vocal_wav": _to_cpu_value(vocal_wav),
        "bgm_wav": _to_cpu_value(bgm_wav),
    }


def _stage_generate_tokens(payload: Dict[str, Any]) -> Dict[str, Any]:
    ckpt_path = payload["ckpt_path"]
    cfg, pt_path, version, max_duration, default_params = _load_inference_cfg(ckpt_path)
    prepare_data = _load_stage_payload(payload["prepare_path"])
    progress_path = payload.get("progress_path")

    params = payload.get("params") or {}
    gen_type = payload.get("gen_type", "mixed")
    disable_offload = bool(payload.get("disable_offload", False))
    disable_cache_clear = bool(payload.get("disable_cache_clear", False))
    disable_fp16 = bool(payload.get("disable_fp16", False))
    disable_sequential = bool(payload.get("disable_sequential", False))
    enable_lm_block_swap = bool(payload.get("enable_lm_block_swap", False))
    lm_block_swap_use_pinned = bool(payload.get("lm_block_swap_use_pinned", False))
    lm_blocks_to_swap = payload.get("lm_blocks_to_swap")
    lm_sub_blocks_to_swap = payload.get("lm_sub_blocks_to_swap")

    _emit_stage_progress(
        progress_path,
        WORKER_STAGE_LM,
        "Generating",
        0.05,
        "Loading language model architecture...",
    )
    with suppress_output():
        audiolm = builders.get_lm_model(cfg, version=version)
    _emit_stage_progress(
        progress_path,
        WORKER_STAGE_LM,
        "Generating",
        0.15,
        "Loading LM checkpoint...",
    )
    with suppress_output():
        checkpoint = load_torch_file(pt_path, map_location="cpu")
        audiolm_state_dict = {
            k.replace("audiolm.", ""): v for k, v in checkpoint.items() if k.startswith("audiolm")
        }
        audiolm.load_state_dict(audiolm_state_dict, strict=False)
        audiolm = audiolm.eval()

    offload_audiolm = (
        False
        if (disable_offload or enable_lm_block_swap)
        else (True if "offload" in cfg.keys() and "audiolm" in cfg.offload else False)
    )
    offload_profiler = None
    audiolm_offload_param = None

    if offload_audiolm:
        _emit_stage_progress(
            progress_path,
            WORKER_STAGE_LM,
            "Generating",
            0.28,
            "Applying LM offload profile...",
        )
        with suppress_output():
            audiolm_offload_param = OffloadParamParse.parse_config(audiolm, cfg.offload.audiolm)
            audiolm_offload_param.show()
            offload_profiler = OffloadProfiler(
                device_index=0,
                **(audiolm_offload_param.init_param_dict()),
            )
            offload_profiler.offload_layer(**(audiolm_offload_param.offload_layer_param_dict()))
            offload_profiler.clean_cache_wrapper(**(audiolm_offload_param.clean_cache_param_dict()))
    else:
        lm_dtype = torch.float32 if disable_fp16 else torch.float16
        lm_device = torch.device("cuda:0")
        block_swap_enabled = False
        block_swap_info = {}

        if enable_lm_block_swap and hasattr(audiolm, "enable_block_swap"):
            try:
                max_main_swap = max(0, int(len(audiolm.transformer.model.layers)) - 1)
                max_sub_swap = max(0, int(len(audiolm.transformer2.model.layers)) - 1)
                default_main_swap = min(4, max_main_swap) if max_main_swap > 0 else 0
                default_sub_swap = min(4, max_sub_swap) if max_sub_swap > 0 else 0

                try:
                    main_swap_value = int(lm_blocks_to_swap) if lm_blocks_to_swap is not None else default_main_swap
                except (TypeError, ValueError):
                    main_swap_value = default_main_swap
                try:
                    sub_swap_value = int(lm_sub_blocks_to_swap) if lm_sub_blocks_to_swap is not None else default_sub_swap
                except (TypeError, ValueError):
                    sub_swap_value = default_sub_swap

                if main_swap_value < 0:
                    main_swap_value = default_main_swap
                if sub_swap_value < 0:
                    sub_swap_value = default_sub_swap

                block_swap_info = audiolm.enable_block_swap(
                    transformer_blocks_to_swap=main_swap_value,
                    transformer2_blocks_to_swap=sub_swap_value,
                    device=lm_device,
                    use_pinned_memory=lm_block_swap_use_pinned,
                )
                block_swap_enabled = bool(block_swap_info.get("enabled", False))
            except Exception:
                block_swap_enabled = False
                block_swap_info = {}

        if block_swap_enabled and hasattr(audiolm, "move_to_device_except_swap_blocks"):
            main_swap = int(block_swap_info.get("transformer_blocks_to_swap", 0))
            sub_swap = int(block_swap_info.get("transformer2_blocks_to_swap", 0))
            _emit_stage_progress(
                progress_path,
                WORKER_STAGE_LM,
                "Generating",
                0.28,
                f"Moving LM to GPU with block swap (main={main_swap}, sub={sub_swap}, pinned={lm_block_swap_use_pinned})...",
            )
            audiolm = audiolm.to(lm_dtype)
            audiolm.move_to_device_except_swap_blocks(lm_device)
            audiolm.prepare_block_swap_before_forward()
        else:
            _emit_stage_progress(
                progress_path,
                WORKER_STAGE_LM,
                "Generating",
                0.28,
                "Moving LM to GPU...",
            )
            audiolm = audiolm.cuda().to(lm_dtype)

    _emit_stage_progress(
        progress_path,
        WORKER_STAGE_LM,
        "Generating",
        0.40,
        "Building generation graph...",
    )
    model = CodecLM(
        name="tmp",
        lm=audiolm,
        audiotokenizer=None,
        max_duration=max_duration,
        seperate_tokenizer=None,
    )

    generation_params = {**default_params, **params}
    num_steps = generation_params.pop("num_steps", 50)
    guidance_scale = generation_params.pop("guidance_scale", 1.5)
    chunked = generation_params.pop("chunked", True)
    chunk_size = generation_params.pop("chunk_size", 128)
    extend_stride_value = generation_params.get("extend_stride", default_params.get("extend_stride", 5))
    try:
        extend_stride_value = float(extend_stride_value)
    except (TypeError, ValueError):
        extend_stride_value = default_params.get("extend_stride", 5)
    extend_stride_value = max(0.0, min(extend_stride_value, max_duration))
    model.set_generation_params(**generation_params)

    device = torch.device("cuda:0")
    _emit_stage_progress(
        progress_path,
        WORKER_STAGE_LM,
        "Generating",
        0.50,
        "Preparing conditioning tensors...",
    )
    generate_inp = {
        "lyrics": [prepare_data["prepared_lyric"]],
        "descriptions": [prepare_data["prepared_description"]],
        "melody_wavs": _move_to_device(prepare_data.get("pmt_wav"), device),
        "vocal_wavs": _move_to_device(prepare_data.get("vocal_wav"), device),
        "bgm_wavs": _move_to_device(prepare_data.get("bgm_wav"), device),
        "melody_is_wav": bool(prepare_data.get("melody_is_wav", True)),
    }

    token_start_time = time.time()
    last_emit = {"time": 0.0, "step": -1}

    def token_progress_callback(generated_tokens: int, total_tokens: int):
        now = time.time()
        total_tokens = max(1, int(total_tokens))
        generated_tokens = max(0, min(int(generated_tokens), total_tokens))
        should_emit = (
            generated_tokens >= total_tokens
            or last_emit["step"] < 0
            or (now - last_emit["time"]) >= 0.2
            or (generated_tokens - last_emit["step"]) >= 16
        )
        if not should_emit:
            return
        frac = generated_tokens / float(total_tokens)
        elapsed = max(1e-6, now - token_start_time)
        speed_val = generated_tokens / elapsed
        eta_seconds = (total_tokens - generated_tokens) / speed_val if speed_val > 1e-6 else None
        _emit_stage_progress(
            progress_path,
            WORKER_STAGE_LM,
            "Generating",
            0.55 + (0.40 * frac),
            f"Generating audio tokens {generated_tokens}/{total_tokens}",
            current_step=generated_tokens,
            total_steps=total_tokens,
            speed=f"{speed_val:.2f} tok/s",
            eta_seconds=eta_seconds,
        )
        last_emit["time"] = now
        last_emit["step"] = generated_tokens

    model.set_custom_progress_callback(token_progress_callback)
    _emit_stage_progress(
        progress_path,
        WORKER_STAGE_LM,
        "Generating",
        0.55,
        "Generating audio tokens...",
    )

    if disable_fp16:
        with torch.no_grad():
            tokens = model.generate(**generate_inp, return_tokens=True)
            if offload_audiolm and offload_profiler is not None:
                offload_profiler.reset_empty_cache_mem_line()
    else:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                tokens = model.generate(**generate_inp, return_tokens=True)
                if offload_audiolm and offload_profiler is not None:
                    offload_profiler.reset_empty_cache_mem_line()

    _emit_stage_progress(
        progress_path,
        WORKER_STAGE_LM,
        "Generating",
        0.98,
        "Token generation complete. Finalizing...",
    )

    if offload_audiolm and offload_profiler is not None:
        offload_profiler.stop()

    del model
    del audiolm
    del checkpoint
    if offload_profiler is not None:
        del offload_profiler
    if audiolm_offload_param is not None:
        del audiolm_offload_param
    if not disable_sequential:
        _cuda_cleanup(disable_cache_clear)
    else:
        gc.collect()

    _emit_stage_progress(
        progress_path,
        WORKER_STAGE_LM,
        "Generating",
        1.0,
        "Language-model stage complete",
    )

    return {
        "tokens": _to_cpu_value(tokens),
        "num_steps": int(num_steps),
        "guidance_scale": float(guidance_scale),
        "chunked": bool(chunked),
        "chunk_size": int(chunk_size),
        "extend_stride": float(extend_stride_value),
        "gen_type": gen_type,
    }


def _stage_run_diffusion(payload: Dict[str, Any]) -> Dict[str, Any]:
    ckpt_path = payload["ckpt_path"]
    cfg, _, _, max_duration, _ = _load_inference_cfg(ckpt_path)
    prepare_data = _load_stage_payload(payload["prepare_path"])
    token_data = _load_stage_payload(payload["tokens_path"])
    progress_path = payload.get("progress_path")

    disable_offload = bool(payload.get("disable_offload", False))
    disable_cache_clear = bool(payload.get("disable_cache_clear", False))
    disable_sequential = bool(payload.get("disable_sequential", False))

    _emit_stage_progress(
        progress_path,
        WORKER_STAGE_DIFFUSION,
        "Diffusion",
        0.05,
        "Loading diffusion tokenizer...",
    )
    device = "cuda:0"
    with suppress_output():
        if disable_sequential:
            seperate_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint_sep, cfg)
        else:
            seperate_tokenizer = builders.get_audio_tokenizer_model_cpu(cfg.audio_tokenizer_checkpoint_sep, cfg)
        seperate_tokenizer.model.device = device
        seperate_tokenizer.model.vae = seperate_tokenizer.model.vae.to(device)
        seperate_tokenizer.model.model.device = torch.device(device)
        seperate_tokenizer = seperate_tokenizer.eval()

    offload_wav_tokenizer_diffusion = False if disable_offload else (
        True if "offload" in cfg.keys() and "wav_tokenizer_diffusion" in cfg.offload else False
    )
    sep_offload_profiler = None
    sep_offload_param = None

    if offload_wav_tokenizer_diffusion:
        _emit_stage_progress(
            progress_path,
            WORKER_STAGE_DIFFUSION,
            "Diffusion",
            0.16,
            "Applying diffusion offload profile...",
        )
        with suppress_output():
            sep_offload_param = OffloadParamParse.parse_config(
                seperate_tokenizer, cfg.offload.wav_tokenizer_diffusion
            )
            sep_offload_param.show()
            sep_offload_profiler = OffloadProfiler(
                device_index=0,
                **(sep_offload_param.init_param_dict()),
            )
            sep_offload_profiler.offload_layer(**(sep_offload_param.offload_layer_param_dict()))
            sep_offload_profiler.clean_cache_wrapper(**(sep_offload_param.clean_cache_param_dict()))
    else:
        _emit_stage_progress(
            progress_path,
            WORKER_STAGE_DIFFUSION,
            "Diffusion",
            0.16,
            "Moving diffusion model to GPU...",
        )
        seperate_tokenizer.model.model = seperate_tokenizer.model.model.to(device)

    _emit_stage_progress(
        progress_path,
        WORKER_STAGE_DIFFUSION,
        "Diffusion",
        0.24,
        "Preparing diffusion graph...",
    )
    model = CodecLM(
        name="tmp",
        lm=None,
        audiotokenizer=None,
        max_duration=max_duration,
        seperate_tokenizer=seperate_tokenizer,
    )

    gen_type = token_data.get("gen_type", payload.get("gen_type", "mixed"))
    num_steps = int(token_data.get("num_steps", 50))
    guidance_scale = float(token_data.get("guidance_scale", 1.5))
    chunked = bool(token_data.get("chunked", True))
    chunk_size = int(token_data.get("chunk_size", 128))
    extend_stride = float(token_data.get("extend_stride", 5.0))

    tokens = _move_to_device(token_data["tokens"], device)
    melody_is_wav = bool(prepare_data.get("melody_is_wav", True))
    pmt_wav = _move_to_device(prepare_data.get("pmt_wav"), device)
    vocal_wav = _move_to_device(prepare_data.get("vocal_wav"), device)
    bgm_wav = _move_to_device(prepare_data.get("bgm_wav"), device)

    diffusion_start_time = time.time()
    last_emit = {"time": 0.0, "step": 0}

    def diffusion_chunk_progress(completed_chunks: int, total_chunks: int):
        total_chunks = max(1, int(total_chunks))
        completed_chunks = max(0, min(int(completed_chunks), total_chunks))
        now = time.time()
        should_emit = (
            completed_chunks >= total_chunks
            or last_emit["step"] == 0
            or (now - last_emit["time"]) >= 0.2
            or (completed_chunks - last_emit["step"]) >= 1
        )
        if not should_emit:
            return
        frac = completed_chunks / float(total_chunks)
        elapsed = max(1e-6, now - diffusion_start_time)
        speed_val = completed_chunks / elapsed
        eta_seconds = (total_chunks - completed_chunks) / speed_val if speed_val > 1e-6 else None
        _emit_stage_progress(
            progress_path,
            WORKER_STAGE_DIFFUSION,
            "Diffusion",
            0.30 + (0.66 * frac),
            f"Diffusion chunks {completed_chunks}/{total_chunks}",
            current_step=completed_chunks,
            total_steps=total_chunks,
            speed=f"{speed_val:.2f} chunk/s",
            eta_seconds=eta_seconds,
        )
        last_emit["time"] = now
        last_emit["step"] = completed_chunks

    _emit_stage_progress(
        progress_path,
        WORKER_STAGE_DIFFUSION,
        "Diffusion",
        0.30,
        "Running diffusion decoding...",
    )
    with torch.no_grad():
        if melody_is_wav:
            wav_seperate = model.generate_audio(
                tokens,
                pmt_wav,
                vocal_wav,
                bgm_wav,
                gen_type=gen_type,
                chunked=chunked,
                chunk_size=chunk_size,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                extend_stride=extend_stride,
                progress_callback=diffusion_chunk_progress,
            )
        else:
            wav_seperate = model.generate_audio(
                tokens,
                gen_type=gen_type,
                chunked=chunked,
                chunk_size=chunk_size,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                extend_stride=extend_stride,
                progress_callback=diffusion_chunk_progress,
            )

    _emit_stage_progress(
        progress_path,
        WORKER_STAGE_DIFFUSION,
        "Diffusion",
        0.98,
        "Diffusion decode complete. Finalizing...",
    )

    if offload_wav_tokenizer_diffusion and sep_offload_profiler is not None:
        sep_offload_profiler.reset_empty_cache_mem_line()
        sep_offload_profiler.stop()

    del model
    del seperate_tokenizer
    if sep_offload_profiler is not None:
        del sep_offload_profiler
    if sep_offload_param is not None:
        del sep_offload_param
    _cuda_cleanup(disable_cache_clear)

    final_audio = None
    if isinstance(wav_seperate, (list, tuple)):
        if wav_seperate:
            final_audio = wav_seperate[0]
    elif isinstance(wav_seperate, torch.Tensor):
        # Keep the same behavior as the legacy path (take first batch item when present).
        if wav_seperate.dim() >= 3:
            final_audio = wav_seperate[0]
        else:
            final_audio = wav_seperate
    elif wav_seperate is not None:
        final_audio = wav_seperate

    if final_audio is None:
        raise RuntimeError("Diffusion stage produced empty audio output.")

    _emit_stage_progress(
        progress_path,
        WORKER_STAGE_DIFFUSION,
        "Diffusion",
        1.0,
        "Diffusion stage complete",
    )

    return {"audio": _to_cpu_value(final_audio)}


class LeVoInference(torch.nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()

        # Disable verbose logging at initialization
        disable_verbose_logging()

        torch.backends.cudnn.enabled = False
        self.ckpt_path = ckpt_path

        _register_omegaconf_resolvers()

        cfg_path = os.path.join(ckpt_path, "config.yaml")
        self.pt_path = os.path.join(ckpt_path, "model.pt")

        self.cfg = OmegaConf.load(cfg_path)
        self.cfg.mode = "inference"
        self.max_duration = self.cfg.max_dur
        self.version = detect_model_version(ckpt_path)

        self.default_params = dict(
            top_p=0.0,
            record_tokens=True,
            record_window=50,
            extend_stride=5,
            duration=self.max_duration,
        )

    def forward(
        self,
        lyric: str,
        description: str = None,
        prompt_audio_path: os.PathLike = None,
        genre: str = None,
        auto_prompt_path: os.PathLike = None,
        gen_type: str = "mixed",
        params=None,
        disable_offload=False,
        disable_cache_clear=False,
        disable_fp16=False,
        disable_sequential=False,
        enable_lm_block_swap=False,
        lm_blocks_to_swap=None,
        lm_sub_blocks_to_swap=None,
        lm_block_swap_use_pinned=False,
        progress_callback=None,
        cancellation_token=None,
    ):
        # Check cancellation at start.
        if cancellation_token and hasattr(cancellation_token, "is_cancelled") and cancellation_token.is_cancelled():
            return None

        temp_root = os.path.join(APP_DIR, "temp", "subprocess_pipeline")
        os.makedirs(temp_root, exist_ok=True)
        workspace_dir = tempfile.mkdtemp(prefix="levo_stage_", dir=temp_root)
        pipeline_start_time = time.time()

        try:
            prepare_output_path = os.path.join(workspace_dir, "prepare_output.pt")
            lm_output_path = os.path.join(workspace_dir, "lm_output.pt")
            diffusion_output_path = os.path.join(workspace_dir, "diffusion_output.pt")
            progress_path = os.path.join(workspace_dir, "pipeline_progress.jsonl")

            if progress_callback:
                progress_callback(
                    {
                        "progress": 0.0,
                        "phase": "Initializing",
                        "message": "Starting subprocess pipeline...",
                    }
                )

            prepare_payload = {
                "ckpt_path": self.ckpt_path,
                "lyric": lyric,
                "description": description,
                "prompt_audio_path": prompt_audio_path,
                "genre": genre,
                "auto_prompt_path": auto_prompt_path,
                "gen_type": gen_type,
                "disable_cache_clear": bool(disable_cache_clear),
                "progress_path": progress_path,
            }
            prepare_result = _run_stage_subprocess(
                WORKER_STAGE_PREPARE,
                prepare_payload,
                prepare_output_path,
                workspace_dir,
                progress_callback=progress_callback,
                stage_start=0.0,
                stage_weight=0.18,
                pipeline_start_time=pipeline_start_time,
                cancellation_token=cancellation_token,
            )
            if prepare_result is None:
                return None

            if cancellation_token and hasattr(cancellation_token, "is_cancelled") and cancellation_token.is_cancelled():
                return None

            lm_payload = {
                "ckpt_path": self.ckpt_path,
                "prepare_path": prepare_output_path,
                "params": params or {},
                "gen_type": gen_type,
                "disable_offload": bool(disable_offload),
                "disable_cache_clear": bool(disable_cache_clear),
                "disable_fp16": bool(disable_fp16),
                "disable_sequential": bool(disable_sequential),
                "enable_lm_block_swap": bool(enable_lm_block_swap),
                "lm_blocks_to_swap": lm_blocks_to_swap,
                "lm_sub_blocks_to_swap": lm_sub_blocks_to_swap,
                "lm_block_swap_use_pinned": bool(lm_block_swap_use_pinned),
                "progress_path": progress_path,
            }
            lm_result = _run_stage_subprocess(
                WORKER_STAGE_LM,
                lm_payload,
                lm_output_path,
                workspace_dir,
                progress_callback=progress_callback,
                stage_start=0.18,
                stage_weight=0.47,
                pipeline_start_time=pipeline_start_time,
                cancellation_token=cancellation_token,
            )
            if lm_result is None:
                return None

            if cancellation_token and hasattr(cancellation_token, "is_cancelled") and cancellation_token.is_cancelled():
                return None

            lm_data = _load_stage_payload(lm_output_path)

            diffusion_payload = {
                "ckpt_path": self.ckpt_path,
                "prepare_path": prepare_output_path,
                "tokens_path": lm_output_path,
                "gen_type": gen_type,
                "disable_offload": bool(disable_offload),
                "disable_cache_clear": bool(disable_cache_clear),
                "disable_sequential": bool(disable_sequential),
                "progress_path": progress_path,
            }
            diffusion_result = _run_stage_subprocess(
                WORKER_STAGE_DIFFUSION,
                diffusion_payload,
                diffusion_output_path,
                workspace_dir,
                progress_callback=progress_callback,
                stage_start=0.65,
                stage_weight=0.35,
                pipeline_start_time=pipeline_start_time,
                cancellation_token=cancellation_token,
            )
            if diffusion_result is None:
                return None

            output_data = _load_stage_payload(diffusion_output_path)
            audio = output_data.get("audio")
            if audio is None:
                raise RuntimeError("Subprocess diffusion output is missing generated audio.")

            if progress_callback:
                elapsed = max(0.0, time.time() - pipeline_start_time)
                progress_callback(
                    {
                        "progress": 1.0,
                        "phase": "Complete",
                        "message": f"Audio generation complete ({elapsed:.1f}s)",
                        "elapsed_seconds": elapsed,
                        "eta_seconds": 0.0,
                        "current_step": lm_data.get("num_steps", 0),
                        "total_steps": lm_data.get("num_steps", 0),
                    }
                )

            return audio
        finally:
            shutil.rmtree(workspace_dir, ignore_errors=True)


def _worker_main() -> int:
    parser = argparse.ArgumentParser(description="LeVo subprocess pipeline worker")
    parser.add_argument("--subprocess-stage", choices=sorted(WORKER_STAGES), required=True)
    parser.add_argument("--input", required=True, help="Path to stage input .pt payload")
    parser.add_argument("--output", required=True, help="Path to stage output .pt payload")
    args = parser.parse_args()

    disable_verbose_logging()
    _register_omegaconf_resolvers()

    try:
        payload = _load_stage_payload(args.input)
        if args.subprocess_stage == WORKER_STAGE_PREPARE:
            result = _stage_prepare_inputs(payload)
        elif args.subprocess_stage == WORKER_STAGE_LM:
            result = _stage_generate_tokens(payload)
        elif args.subprocess_stage == WORKER_STAGE_DIFFUSION:
            result = _stage_run_diffusion(payload)
        else:
            raise RuntimeError(f"Unknown subprocess stage: {args.subprocess_stage}")

        _save_stage_payload(args.output, result)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(_worker_main())
