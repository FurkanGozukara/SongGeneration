import os
import os.path
import time
import random
import re
import yaml
import numpy as np
import torch
import scipy.io.wavfile as wavfile
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import threading

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VOCAB_PATH = os.path.join(APP_ROOT, 'conf', 'vocab.yaml')
_DEFAULT_STRUCTURES = [
    '[verse]', '[chorus]', '[bridge]',
    '[intro-short]', '[intro-medium]', '[intro-long]',
    '[inst-short]', '[inst-medium]', '[inst-long]',
    '[outro-short]', '[outro-medium]', '[outro-long]',
    '[silence]'
]
_VOCAL_STRUCTURES = {'[verse]', '[chorus]', '[bridge]'}
_INSTRUMENTAL_ALIASES = {'[instrumental]'}
_STRUCTURE_REPLACEMENTS = {
    '[intro]': '[intro-short]',
    '[inst]': '[inst-short]',
    '[outro]': '[outro-short]'
}
_SANITIZE_PATTERN = re.compile(r"[^\w\s\[\]\-\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\u00c0-\u017f]")
_VALID_STRUCTURES_CACHE: Optional[List[str]] = None


def _load_valid_structures() -> List[str]:
    global _VALID_STRUCTURES_CACHE
    if _VALID_STRUCTURES_CACHE is None:
        try:
            with open(_VOCAB_PATH, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if isinstance(data, list) and data:
                    _VALID_STRUCTURES_CACHE = data
                else:
                    _VALID_STRUCTURES_CACHE = _DEFAULT_STRUCTURES
        except Exception:
            _VALID_STRUCTURES_CACHE = _DEFAULT_STRUCTURES
    return _VALID_STRUCTURES_CACHE


def normalize_lyrics(lyrics: str) -> str:
    """Normalize lyrics to the model's expected format with validation."""
    if not lyrics or not lyrics.strip():
        raise ValueError("Lyrics cannot be empty")

    normalized_text = lyrics
    for src, dst in _STRUCTURE_REPLACEMENTS.items():
        normalized_text = re.sub(re.escape(src), dst, normalized_text, flags=re.IGNORECASE)

    paragraphs = [p.strip() for p in normalized_text.strip().split('\n\n') if p.strip()]
    if not paragraphs:
        raise ValueError("No valid paragraphs found")

    valid_tags_lower = {tag.lower() for tag in _load_valid_structures()}
    valid_tags_lower.update(_INSTRUMENTAL_ALIASES)
    vocal_tags_lower = {tag.lower() for tag in _VOCAL_STRUCTURES}

    normalized_parts: List[str] = []
    has_vocal = False

    for para in paragraphs:
        lines = [line.strip() for line in para.splitlines() if line.strip()]
        if not lines:
            continue

        struct_tag = lines[0].lower()
        if struct_tag not in valid_tags_lower:
            all_tags = _load_valid_structures() + list(_INSTRUMENTAL_ALIASES)
            raise ValueError(f"Invalid structure tag: {struct_tag}. Valid tags: {', '.join(all_tags)}")

        if struct_tag in vocal_tags_lower:
            has_vocal = True
            cleaned_lines = []
            for line in lines[1:]:
                cleaned = _SANITIZE_PATTERN.sub("", line).strip()
                if cleaned:
                    cleaned_lines.append(cleaned)
            if not cleaned_lines:
                raise ValueError(f"Structure {struct_tag} requires lyrics but none found")
            normalized_parts.append(f"{struct_tag} {'.'.join(cleaned_lines)}")
        else:
            if any(line.strip() for line in lines[1:]):
                raise ValueError(f"Structure {struct_tag} should not contain lyrics")
            normalized_parts.append(struct_tag)

    if not normalized_parts:
        raise ValueError("No valid paragraphs found")

    if not has_vocal:
        raise ValueError(f"Lyrics must contain at least one vocal structure: {', '.join(sorted(_VOCAL_STRUCTURES))}")

    return " ; ".join(normalized_parts)


def compose_description_from_params(params: Dict[str, Any]) -> str:
    """Compose the textual description used during batch generation."""
    extra_prompt = (params.get('extra_prompt') or "").strip()
    include_dropdown = bool(params.get('include_dropdown_attributes', False))
    force_extra_prompt = bool(params.get('force_extra_prompt', False))

    base_components: List[str] = []

    if include_dropdown and not force_extra_prompt:
        attribute_values = [
            params.get('gender'),
            params.get('timbre'),
            params.get('genre'),
            params.get('emotion'),
            params.get('instrument'),
        ]
        attribute_parts = [str(value).strip() for value in attribute_values if value]

        bpm_value = params.get('bpm')
        if bpm_value is not None and str(bpm_value).strip():
            try:
                bpm_int = int(float(bpm_value))
            except (TypeError, ValueError):
                bpm_int = bpm_value
            attribute_parts.append(f"the bpm is {bpm_int}")

        if attribute_parts:
            base_components.append(", ".join(attribute_parts))

    if extra_prompt:
        base_components.append(extra_prompt)

    base_text = ", ".join(component for component in base_components if component).strip()

    if not base_text or (force_extra_prompt and not extra_prompt):
        base_text = "."

    if "[Musicality-very-high]" not in base_text:
        base_text = f"[Musicality-very-high], {base_text}"

    return base_text


class CancellationToken:
    """Thread-safe cancellation token for stopping generation"""
    def __init__(self):
        self._cancelled = False
        self._lock = threading.Lock()
    
    def cancel(self):
        with self._lock:
            self._cancelled = True
    
    def is_cancelled(self):
        with self._lock:
            return self._cancelled
    
    def reset(self):
        with self._lock:
            self._cancelled = False

class ProgressTracker:
    """Track and report progress for generation tasks"""
    def __init__(self, callback=None):
        self.callback = callback
        self.start_time = None
        self.total_steps = 0
        self.current_step = 0
        self.current_phase = ""
        self.batch_total = 0
        self.batch_current = 0
        self.preset_total = 0
        self.preset_current = 0
        self.generation_total = 0
        self.generation_current = 0
        
    def start(self, total_steps=1):
        self.start_time = time.time()
        self.total_steps = total_steps
        self.current_step = 0
        
    def update(self, step=None, phase=None, message=None):
        if step is not None:
            self.current_step = step
        if phase is not None:
            self.current_phase = phase
            
        elapsed = time.time() - self.start_time if self.start_time else 0
        progress = self.current_step / self.total_steps if self.total_steps > 0 else 0
        
        # Calculate ETA
        if progress > 0 and elapsed > 0:
            total_time = elapsed / progress
            eta = total_time - elapsed
        else:
            eta = 0
            
        info = {
            'progress': progress,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'phase': self.current_phase,
            'elapsed': elapsed,
            'eta': eta,
            'message': message or "",
            'batch_info': f"{self.batch_current}/{self.batch_total}" if self.batch_total > 0 else "",
            'preset_info': f"{self.preset_current}/{self.preset_total}" if self.preset_total > 0 else "",
            'generation_info': f"{self.generation_current}/{self.generation_total}" if self.generation_total > 0 else ""
        }
        
        if self.callback:
            self.callback(info)
            
        return info
    
    def set_batch_info(self, current, total):
        self.batch_current = current
        self.batch_total = total
        
    def set_preset_info(self, current, total):
        self.preset_current = current
        self.preset_total = total
        
    def set_generation_info(self, current, total):
        self.generation_current = current
        self.generation_total = total

def set_seed(seed):
    """Set random seeds for reproducibility"""
    if seed is not None and seed >= 0:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return True
    return False

def get_next_file_number(output_dir):
    """Get the next available file number in sequence"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        return 1
    
    existing_files = [f for f in os.listdir(output_dir) if f.endswith(('.wav', '.mp3', '.mp4'))]
    if not existing_files:
        return 1
    
    numbers = []
    for f in existing_files:
        try:
            # Remove extension first
            name_without_ext = f.rsplit('.', 1)[0]
            
            # Try to extract number from the beginning of the filename
            # Handle cases like "0018_The 80s Synth-Pop Throwback" or "0018"
            if '_' in name_without_ext:
                # Split by underscore and try to parse the first part
                first_part = name_without_ext.split('_')[0]
                num = int(first_part)
            else:
                # Try to parse the whole name as a number
                num = int(name_without_ext)
            
            numbers.append(num)
        except:
            continue
    
    # Debug output
    if numbers:
        print(f"Found file numbers: {sorted(set(numbers))}")
        print(f"Next file number will be: {max(numbers) + 1}")
    
    return max(numbers) + 1 if numbers else 1

def format_lyrics_for_model(lyrics):
    """Format lyrics according to the model's expected format"""
    return normalize_lyrics(lyrics)

def generate_single_song(model, params, progress_tracker=None, cancellation_token=None):
    """Generate a single song with the given parameters"""
    
    # Check cancellation before starting
    if cancellation_token and cancellation_token.is_cancelled():
        return None
    
    # Update progress
    if progress_tracker:
        progress_tracker.update(phase="Preparing generation", message="Setting up parameters...")
    
    # Set seed if provided
    used_seed = params.get('seed', -1)
    if used_seed is None or used_seed < 0:
        used_seed = random.randint(0, 2147483647)
    set_seed(used_seed)
    
    # Format lyrics
    try:
        formatted_lyrics = normalize_lyrics(params['lyrics'])
    except ValueError as exc:
        raise ValueError(f"Lyrics validation failed: {exc}") from exc
    
    # Prepare generation parameters
    # Convert steps to duration (25 steps per second)
    # No artificial cap - let the model use its full capacity
    duration_from_steps = params['max_gen_length'] / 25.0
    
    try:
        top_k_value = int(params.get('top_k', -1))
    except (TypeError, ValueError):
        top_k_value = -1
    if top_k_value < 0:
        top_k_value = -1

    try:
        top_p_value = float(params.get('top_p', 0.0))
    except (TypeError, ValueError):
        top_p_value = 0.0

    gen_params = {
        'duration': duration_from_steps,
        'num_steps': params['diffusion_steps'],
        'temperature': params['temperature'],
        'top_k': top_k_value,
        'top_p': top_p_value,
        'cfg_coef': params['cfg_coef'],
        'guidance_scale': params['guidance_scale'],
        'use_sampling': params['use_sampling'],
        'extend_stride': params['extend_stride'],
        'chunked': params['chunked'],
        'chunk_size': params['chunk_size'],
        'record_tokens': params['record_tokens'],
        'record_window': params['record_window'],
    }
    
    # Update progress
    if progress_tracker:
        progress_tracker.update(phase="Generating audio", message="Model processing...")
    
    # Check cancellation before model call
    if cancellation_token and cancellation_token.is_cancelled():
        return None
    
    # Call the model
    # Build description string with extra prompt support
    extra_prompt_text = params.get('extra_prompt', '')
    if params.get('force_extra_prompt') and not (extra_prompt_text and extra_prompt_text.strip()):
        print("⚠️ Warning: 'force_extra_prompt' is True but no extra prompt was provided!")
        print("Using fallback placeholder for description...")

    description = compose_description_from_params(params)
    
    # Use audio path if provided
    audio_path = params.get('audio_path', None)
    
    # Create internal progress callback if provided
    internal_progress_callback = None
    if progress_tracker:
        def internal_progress_callback(info):
            progress_tracker.update(phase=info.get('phase', ''), message=info.get('message', ''))
    
    audio_data = model(
        formatted_lyrics,
        description,
        audio_path,
        None,
        params.get('auto_prompt_path'),
        params['gen_type'],
        gen_params,
        disable_offload=params.get('disable_offload', False),
        disable_cache_clear=params.get('disable_cache_clear', False),
        disable_fp16=params.get('disable_fp16', False),
        disable_sequential=params.get('disable_sequential', False),
        progress_callback=internal_progress_callback,
        cancellation_token=cancellation_token
    )
    
    if audio_data is None:
        return None
        
    audio_data = audio_data.cpu().permute(1, 0).float().numpy()
    
    # Check cancellation after generation
    if cancellation_token and cancellation_token.is_cancelled():
        return None
    
    return {
        'audio_data': audio_data,
        'used_seed': used_seed,
        'formatted_lyrics': formatted_lyrics
    }