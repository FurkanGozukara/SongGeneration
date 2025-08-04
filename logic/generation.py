import os
import os.path
import time
import random
import numpy as np
import torch
import scipy.io.wavfile as wavfile
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import threading

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
    # The model expects: "[struct] lyrics ; [struct] lyrics"
    paragraphs = [p.strip() for p in lyrics.strip().split('\n\n') if p.strip()]
    formatted_parts = []
    
    for para in paragraphs:
        lines = [line.strip() for line in para.splitlines() if line.strip()]
        if not lines:
            continue
            
        struct_tag = lines[0].lower()
        
        # Handle sections with lyrics
        if struct_tag in ['[verse]', '[chorus]', '[bridge]']:
            if len(lines) > 1:
                # Join all lyric lines with periods
                lyric_text = '. '.join(lines[1:])
                formatted_parts.append(f"{struct_tag} {lyric_text}")
            else:
                formatted_parts.append(struct_tag)
        else:
            # Instrumental sections don't have lyrics
            formatted_parts.append(struct_tag)
    
    formatted_lyrics = " ; ".join(formatted_parts)
    return formatted_lyrics

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
    formatted_lyrics = format_lyrics_for_model(params['lyrics'])
    
    # Prepare generation parameters
    pattern_delay_offset = 250
    actual_steps = max(params['max_gen_length'] - pattern_delay_offset, 1000)
    duration_from_steps = min(actual_steps / 25.0, 150.0)
    
    gen_params = {
        'duration': duration_from_steps,
        'num_steps': params['diffusion_steps'],
        'temperature': params['temperature'],
        'top_k': params['top_k'],
        'top_p': params['top_p'],
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
    if params.get('force_extra_prompt'):
        # Force mode: use only the extra prompt, ignore dropdowns
        if params.get('extra_prompt', '').strip():
            description = params['extra_prompt'].strip()
        else:
            # Warning: force mode enabled but no extra prompt provided
            print("⚠️ Warning: 'force_extra_prompt' is True but no extra prompt was provided!")
            print("Using minimal description for generation...")
            description = "music"  # Minimal fallback description
    else:
        # Normal mode: use dropdown values with optional extra prompt
        base_description = f"{params['gender']}, {params['timbre']}, {params['genre']}, {params['emotion']}, {params['instrument']}"
        if params.get('extra_prompt', '').strip():
            description = f"{base_description}, {params['extra_prompt'].strip()}"
        else:
            description = base_description
    
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