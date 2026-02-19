import sys
import gradio as gr
import json
from datetime import datetime
import yaml
import time
import os.path as op
import os
import scipy.io.wavfile as wavfile
import soundfile as sf
import subprocess
import platform
from PIL import Image
import numpy as np
import torch
import random
import argparse
import threading
import re
import shutil

# Import and configure output suppression
from utils.suppress_output import suppress_output, disable_verbose_logging

# Add the tools/gradio directory to the path
sys.path.append(op.join(op.dirname(op.abspath(__file__)), 'tools', 'gradio'))
from levo_inference_lowmem import LeVoInference

# Import from logic folder
from logic.generation import CancellationToken, ProgressTracker, set_seed, format_lyrics_for_model, get_next_file_number, generate_single_song
from logic.file_utils import save_metadata, convert_wav_to_mp3, create_video_from_image_and_audio
from logic.batch_processing import BatchProcessor
from logic.ui_progress import GradioProgressTracker, create_progress_callback, format_eta
from logic.preset_manager import PresetManager
from logic.progress_interceptor import intercept_progress
from logic.audio_separator import AudioSeparator
from logic.auto_prompt_manager import AutoPromptManager

EXAMPLE_LYRICS = """
[intro-short]

[verse]
Streetlights flicker in the night
I walk through familiar corners
Memories come flooding like a tide
Your smile so clear in my mind
Can't erase it from my heart
Those sweet moments we once had
Now only I remain to remember

[verse]
My phone screen lights up
It's a message from you
Just a few simple words
Yet they make tears stream down my face
The warmth of your embrace back then
Now feels so distant and far
How I wish to return to the past
To have your company once more

[chorus]
The warmth of memories remains
But you're no longer here
My heart filled with love
Yet pierced by longing and pain
The rhythm of music plays on
But my heart wanders aimlessly
In days without you
How should I continue forward

[outro-short]
""".strip()

# Generation types and auto prompt types from original repo
GENERATION_TYPES = ['mixed', 'vocal', 'bgm', 'separate']
AUTO_PROMPT_TYPES = ['Pop', 'R&B', 'Dance', 'Jazz', 'Folk', 'Rock', 'Chinese Style', 'Chinese Tradition', 'Metal', 'Reggae', 'Chinese Opera', 'Auto']

# Extended generation length support based on model version
# Only two officially supported models
MAX_GENERATION_LENGTHS = {
    'songgeneration_large': 6750,  # ~4m30s - BEST QUALITY (default)
    'songgeneration_base_full': 6750,  # ~4m30s - GOOD QUALITY
}

DEFAULT_MAX_GENERATION_STEPS = max(MAX_GENERATION_LENGTHS.values())
STEPS_PER_SECOND = 25.0
MIN_GENERATION_STEPS = 250
DEFAULT_GENERATION_STEPS = 4500
MIN_DURATION_SECONDS = int(MIN_GENERATION_STEPS / STEPS_PER_SECOND)
DEFAULT_DURATION_SECONDS = int(DEFAULT_GENERATION_STEPS / STEPS_PER_SECOND)
DURATION_SLIDER_UI_MIN = MIN_DURATION_SECONDS

APP_DIR = op.dirname(op.abspath(__file__))

# Parse command line arguments
parser = argparse.ArgumentParser(description='LeVo Song Generation App')
parser.add_argument('--share', action='store_true', help='Share the Gradio app publicly')
args = parser.parse_args()

# Model version detection and validation
def detect_model_version(ckpt_path):
    """Detect and validate model version from checkpoint path"""
    model_name = op.basename(ckpt_path).lower().replace('-', '_')
    supported_models = ['songgeneration_large', 'songgeneration_base_full']
    
    if model_name not in supported_models:
        print(f"Warning: Unknown model version '{model_name}'. Supported: {supported_models}")
        return 'songgeneration_large'  # fallback to Large model (default)
    
    return model_name

# MODEL_VERSION is now set dynamically when model is loaded
# Initialize MODEL_VERSION to None (will be set when model is loaded)
MODEL_VERSION = None

# GPU Memory Detection and Auto Mode Selection
def detect_gpu_memory():
    """Detect available GPU memory and recommend settings"""
    if not torch.cuda.is_available():
        return 0, True  # No GPU, force low memory
    
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    reserved_memory = torch.cuda.memory_reserved(device)
    available_memory = (total_memory - reserved_memory) / 1024 / 1024 / 1024  # GB
    
    print(f"Available GPU memory: {available_memory:.1f}GB")
    
    # Determine if low memory mode should be used based on model version
    # Only two officially supported models
    memory_requirements = {
        'songgeneration_large': {'normal': 28, 'with_audio': 28},  # 22-28GB, BEST QUALITY
        'songgeneration_base_full': {'normal': 18, 'with_audio': 18}  # 12-18GB, GOOD QUALITY
    }
    
    # Use default requirements if MODEL_VERSION is None or not found (default to Large model)
    required = memory_requirements.get(MODEL_VERSION, {'normal': 28, 'with_audio': 28})
    use_low_mem = available_memory < required['normal']
    
    if use_low_mem:
        print(f"Auto-enabling low memory mode (required: {required['normal']}GB, available: {available_memory:.1f}GB)")
    else:
        print(f"Using normal memory mode (available: {available_memory:.1f}GB >= required: {required['normal']}GB)")
    
    return available_memory, use_low_mem

AVAILABLE_MEMORY, AUTO_LOW_MEM = detect_gpu_memory()

# Set environment variables
os.environ['USER'] = 'root'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['TRANSFORMERS_CACHE'] = op.join(APP_DIR, 'third_party', 'hub')

# Set PYTHONPATH
pythonpath_additions = [
    op.join(APP_DIR, 'codeclm', 'tokenizer'),
    APP_DIR,
    op.join(APP_DIR, 'codeclm', 'tokenizer', 'Flow1dVAE'),
    op.join(APP_DIR, 'codeclm', 'tokenizer')
]
current_pythonpath = os.environ.get('PYTHONPATH', '')
os.environ['PYTHONPATH'] = ';'.join(pythonpath_additions) + (';' + current_pythonpath if current_pythonpath else '')

# Disable verbose logging before initializing model
disable_verbose_logging()

# Global model instance (will be loaded dynamically)
MODEL = None
# MODEL_VERSION is already defined above (line 74)
CURRENT_MODEL_PATH = None

def steps_to_seconds(steps_value):
    """Convert generation steps to seconds using pipeline ratio."""
    try:
        return max(0.0, float(steps_value)) / STEPS_PER_SECOND
    except (TypeError, ValueError):
        return 0.0


def seconds_to_steps(seconds_value):
    """Convert seconds to generation steps using pipeline ratio."""
    try:
        return max(0, int(round(float(seconds_value) * STEPS_PER_SECOND)))
    except (TypeError, ValueError):
        return 0


def get_current_model_max_steps():
    """Return the current model's maximum supported generation steps."""
    if MODEL_VERSION and MODEL_VERSION in MAX_GENERATION_LENGTHS:
        return MAX_GENERATION_LENGTHS[MODEL_VERSION]
    if CURRENT_MODEL_PATH:
        model_name = op.basename(CURRENT_MODEL_PATH)
        return MAX_GENERATION_LENGTHS.get(model_name, DEFAULT_MAX_GENERATION_STEPS)
    return DEFAULT_MAX_GENERATION_STEPS


def clamp_steps_to_model(steps_value):
    """Clamp step value to supported range for loaded model."""
    max_steps = get_current_model_max_steps()
    clamped_steps = max(MIN_GENERATION_STEPS, int(round(max(0, steps_value))))
    return min(clamped_steps, max_steps)


def clamp_duration_seconds(duration_value):
    """Clamp duration (seconds) to supported range for loaded model."""
    max_seconds = steps_to_seconds(get_current_model_max_steps())
    min_seconds = MIN_DURATION_SECONDS
    if max_seconds < min_seconds:
        max_seconds = min_seconds
    try:
        duration = float(duration_value)
    except (TypeError, ValueError):
        duration = DEFAULT_DURATION_SECONDS
    duration = max(float(min_seconds), duration)
    return min(duration, max_seconds)


def get_min_duration_seconds():
    """Return smallest song duration allowed by pipeline in seconds."""
    return int(round(steps_to_seconds(MIN_GENERATION_STEPS)))


def format_duration_slider_info(max_steps):
    """Generate user-facing info text for the duration slider."""
    max_steps = max(MIN_GENERATION_STEPS, int(max_steps))
    max_seconds = int(round(steps_to_seconds(max_steps)))
    min_seconds = get_min_duration_seconds()
    if max_seconds < min_seconds:
        max_seconds = min_seconds
    return (
        "Controls generation duration in seconds (≈25 steps per second). "
        f"Model range: {min_seconds}-{max_seconds}s. "
        f"Values below {min_seconds}s snap to the minimum automatically."
    )


def create_duration_slider_update(step_value, max_steps):
    """Helper to create a duration slider update synchronized with step slider."""
    max_steps = max(MIN_GENERATION_STEPS, int(max_steps))
    try:
        requested_steps = int(round(float(step_value)))
    except (TypeError, ValueError):
        requested_steps = DEFAULT_GENERATION_STEPS
    clamped_steps = max(MIN_GENERATION_STEPS, min(requested_steps, max_steps))
    duration_value = int(round(steps_to_seconds(clamped_steps)))
    duration_max = int(round(steps_to_seconds(max_steps)))
    min_seconds = get_min_duration_seconds()
    if duration_max < min_seconds:
        min_seconds = duration_max
    duration_value = max(min_seconds, min(duration_value, duration_max))
    info_text = format_duration_slider_info(max_steps)
    return gr.update(
        value=duration_value,
        minimum=DURATION_SLIDER_UI_MIN,
        maximum=duration_max,
        info=info_text
    )


# Initialize new systems
audio_separator = AudioSeparator()
auto_prompt_manager = AutoPromptManager(APP_DIR)

# Global cancellation token and batch processor
cancellation_token = CancellationToken()
batch_processor = None  # Will be initialized when model is loaded
gradio_progress_tracker = GradioProgressTracker()

def get_available_models():
    """Get list of available model checkpoints"""
    ckpt_dir = op.join(APP_DIR, 'ckpt')
    available_models = []
    
    # Check for supported model versions
    # Order matters: Large model first (default), then Base Full
    # Only these two models are officially supported
    model_dirs = [
        ('songgeneration_large', 'SongGeneration Large - BEST QUALITY (4m30s, 22-28GB VRAM)'),
        ('songgeneration_base_full', 'SongGeneration Base Full - GOOD QUALITY (4m30s, 12-18GB VRAM)')
    ]
    
    for model_dir, description in model_dirs:
        model_path = op.join(ckpt_dir, model_dir)
        config_path = op.join(model_path, 'config.yaml')
        model_file_path = op.join(model_path, 'model.pt')
        
        if os.path.exists(config_path) and os.path.exists(model_file_path):
            available_models.append((model_dir, description, model_path))
    
    return available_models

def load_model(model_path, progress_callback=None):
    """Load a model with proper cleanup of previous model"""
    global MODEL, MODEL_VERSION, CURRENT_MODEL_PATH, batch_processor
    
    if progress_callback:
        progress_callback(0.1, "Cleaning up previous model...")
    
    # Clean up existing model
    if MODEL is not None:
        try:
            print("🧹 Cleaning up previous model...")
            del MODEL
            MODEL = None
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            import gc
            gc.collect()
            print("✅ Previous model cleaned up")
            
        except Exception as e:
            print(f"⚠️ Warning during model cleanup: {e}")
    
    if progress_callback:
        progress_callback(0.3, f"Loading new model from {model_path}...")
    
    # Load new model
    try:
        print(f"🔄 Loading model: {op.basename(model_path)}")
        
        with suppress_output():
            MODEL = LeVoInference(model_path)
        
        MODEL_VERSION = detect_model_version(model_path)
        CURRENT_MODEL_PATH = model_path
        
        # Reinitialize batch processor with new model
        batch_processor = BatchProcessor(MODEL, APP_DIR, MODEL.cfg)
        
        if progress_callback:
            progress_callback(1.0, "Model loaded successfully!")
        
        print(f"✅ Model loaded successfully: {MODEL_VERSION}")
        return True, f"✅ Successfully loaded {MODEL_VERSION}"
        
    except Exception as e:
        error_msg = f"❌ Failed to load model: {str(e)}"
        print(error_msg)
        if progress_callback:
            progress_callback(1.0, error_msg)
        return False, error_msg

def get_model_info(model_path):
    """Get information about a model"""
    if not model_path:
        return "No model selected"
    
    model_name = op.basename(model_path)
    
    # Model specifications - Only two officially supported models
    model_specs = {
        'songgeneration_large': {
            'length': '4m30s (270 seconds)',
            'languages': 'Chinese + English',
            'vram': '22-28GB (RTX 4090, A100)', 
            'quality': '★★★★★ BEST',
            'steps': '6750 steps max',
            'notes': 'Default & Recommended'
        },
        'songgeneration_base_full': {
            'length': '4m30s (270 seconds)',
            'languages': 'Chinese + English', 
            'vram': '12-18GB (RTX 3090, 4080)',
            'quality': '★★★★ GOOD',
            'steps': '6750 steps max',
            'notes': 'For lower VRAM systems'
        }
    }
    
    specs = model_specs.get(model_name, {})
    
    info = f"""
### {specs.get('quality', 'Unknown')} Quality Model

**Model**: `{model_name}`
**Max Length**: {specs.get('length', 'Unknown')}
**Max Steps**: {specs.get('steps', 'Unknown')}
**Languages**: {specs.get('languages', 'Unknown')}
**VRAM Required**: {specs.get('vram', 'Unknown')}
**Note**: {specs.get('notes', 'N/A')}
**Status**: {'✅ **LOADED**' if CURRENT_MODEL_PATH == model_path else '⏳ Not loaded'}
"""
    
    return info

# Load description options from text files
def load_options(filename):
    filepath = op.join(APP_DIR, 'sample', 'description', filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        options = [line.strip() for line in f if line.strip()]
        return sorted(options)  # Sort alphabetically

# Load vocabulary structures for validation
def validate_lyrics_structure(lyrics):
    """Validate and normalize lyrics structure against vocabulary"""
    try:
        normalized = format_lyrics_for_model(lyrics)
    except ValueError as exc:
        return False, str(exc), None
    return True, "Lyrics structure is valid", normalized

def compose_generation_description(
    extra_prompt: str,
    include_dropdown_attributes: bool,
    gender: str,
    timbre: str,
    genre: str,
    emotion: str,
    instrument: str,
    bpm,
    force_extra_prompt: bool,
) -> str:
    """Compose the textual description passed to the model."""
    extra_text = (extra_prompt or "").strip()

    base_components = []
    use_dropdown = include_dropdown_attributes and not force_extra_prompt

    if use_dropdown:
        attribute_parts = []
        for attr_type, attr_value in [
            ("gender", gender),
            ("timbre", timbre),
            ("genre", genre),
            ("emotion", emotion),
            ("instrument", instrument),
        ]:
            formatted = format_attribute_for_prompt(attr_type, attr_value)
            if formatted:
                attribute_parts.append(formatted)

        if bpm is not None and str(bpm).strip():
            try:
                bpm_value = int(float(bpm))
            except (TypeError, ValueError):
                bpm_value = bpm
            attribute_parts.append(f"the bpm is {bpm_value}")

        if attribute_parts:
            base_components.append(", ".join(attribute_parts))

    if extra_text:
        base_components.append(extra_text)

    base_text = ", ".join(component for component in base_components if component).strip()

    if not base_text or force_extra_prompt and not extra_text:
        base_text = "."

    if "[Musicality-very-high]" not in base_text:
        base_text = f"[Musicality-very-high], {base_text}"

    return enhance_prompt_text(base_text)

def format_description_preview(description: str) -> str:
    """Format description preview text for display."""
    return f"**Description Preview:** `{description}`"

def add_none_option(options):
    """Return a new list with 'None' prepended for optional selections."""
    cleaned = [opt for opt in options if opt.lower() != "none"]
    return ["None"] + cleaned


def format_attribute_for_prompt(attribute_type: str, value) -> str:
    """Convert attribute values into richer prompt phrases."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "none":
        return ""
    
    lower_text = text.lower()
    
    if attribute_type == "gender":
        gender_map = {
            "female": "female vocalist",
            "male": "male vocalist",
        }
        return gender_map.get(lower_text, text)
    
    if attribute_type == "timbre":
        return f"{text} timbre"
    
    if attribute_type == "emotion":
        return f"{text} emotion"
    
    if attribute_type == "genre":
        return f"{text} music"
    
    if attribute_type == "instrument":
        return f"{text} instrumentation"
    
    return text


def enhance_prompt_text(prompt_text: str) -> str:
    """Strengthen key tokens so the model adheres to them."""
    if not prompt_text:
        return prompt_text
    
    def replace_gender(match: re.Match) -> str:
        word = match.group(0)
        replacements = {
            "female": "female vocalist",
            "male": "male vocalist",
        }
        replacement = replacements.get(word.lower())
        if not replacement:
            return word
        
        if word.isupper():
            return replacement.upper()
        if word[0].isupper():
            return replacement.capitalize()
        return replacement
    
    gender_pattern = re.compile(r"\b(female|male)\b(?!\s+(vocal|voice|singer))", re.IGNORECASE)
    enhanced = gender_pattern.sub(replace_gender, prompt_text)
    return enhanced


GENRES = add_none_option(load_options('genre.txt'))
INSTRUMENTS = add_none_option(load_options('instrument.txt'))
EMOTIONS = add_none_option(load_options('emotion.txt'))
TIMBRES = add_none_option(load_options('timbre.txt'))
GENDERS = add_none_option(load_options('gender.txt'))

# Preset directory
PRESET_DIR = op.join(APP_DIR, 'presets')
os.makedirs(PRESET_DIR, exist_ok=True)

# Initialize preset manager
preset_manager = PresetManager(PRESET_DIR)


def collect_current_parameters(
    lyrics, genre, instrument, bpm, emotion, timbre, gender, extra_prompt, include_dropdown_attributes, force_extra_prompt,
    audio_path, image_path, save_mp3, seed,
    duration_seconds, max_gen_length, diffusion_steps, temperature, top_k, top_p,
    cfg_coef, guidance_scale, use_sampling, extend_stride,
    gen_type, chunked, chunk_size, record_tokens, record_window,
    disable_offload, disable_cache_clear, disable_fp16, disable_sequential,
    num_generations, loop_presets, randomize_params
):
    """Collect all current parameters into a dictionary"""
    return {
        'lyrics': lyrics,
        'genre': genre,
        'instrument': instrument,
        'bpm': bpm,
        'emotion': emotion,
        'timbre': timbre,
        'gender': gender,
        'extra_prompt': extra_prompt,
        'include_dropdown_attributes': include_dropdown_attributes,
        'force_extra_prompt': force_extra_prompt,
        'audio_path': audio_path,
        'image_path': image_path,
        'save_mp3': save_mp3,
        'seed': seed,
        'duration_seconds': duration_seconds,
        'max_gen_length': max_gen_length,
        'diffusion_steps': diffusion_steps,
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p,
        'cfg_coef': cfg_coef,
        'guidance_scale': guidance_scale,
        'use_sampling': use_sampling,
        'extend_stride': extend_stride,
        'gen_type': gen_type,
        'chunked': chunked,
        'chunk_size': chunk_size,
        'record_tokens': record_tokens,
        'record_window': record_window,
        'disable_offload': disable_offload,
        'disable_cache_clear': disable_cache_clear,
        'disable_fp16': disable_fp16,
        'disable_sequential': disable_sequential,
        'num_generations': num_generations,
        'loop_presets': loop_presets,
        'randomize_params': randomize_params
    }



def output_messages(msg):
    gr.Info(msg)
    print(msg)



def open_output_folder():
    """Open the output folder in the system's file explorer"""
    output_dir = op.join(APP_DIR, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if platform.system() == "Windows":
        os.startfile(output_dir)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", output_dir])
    else:  # Linux and others
        subprocess.run(["xdg-open", output_dir])
    
    return gr.Info(f"Opened output folder: {output_dir}")

def process_struct(struct):
    parts = [{"part": k, "prefix": "", "repeat": v} for k,v in struct]
    for i,part in enumerate(parts):
        if i==0:
            continue
        same_repeat_count = 0
        j = i - 1
        while j>=0:
            if parts[j]["repeat"]==parts[i]["repeat"]:
                same_repeat_count += 1
                j -= 1
            else:
                break
        if same_repeat_count:
            parts[i]["prefix"] = str(same_repeat_count+1)
    all_parts = []
    for part in parts:
        p = f"{part['prefix']}{part['part']}"
        if part['repeat']>1:
            p = f"{p}x{part['repeat']}"
        all_parts.append(p)
    struct_str = "-".join(all_parts)
    return struct_str

def process_history(history):
    songs = []
    for message in history:
        if message["role"]=="assistant" and "song" in message:
            songs.append(message["song"])
    return get_history_html(songs)

def get_history_html(songs):
    if not songs:
        return ""
    
    html = "<h3>Generated Songs</h3>"
    for i, song in enumerate(songs):
        # Skip songs that don't have audio files (failed generations)
        if not song or not song.get('audio'):
            continue
            
        video_html = ""
        if song.get('video'):
            video_html = f"""
            <video controls style="width: 100%; max-width: 400px;">
                <source src="{song['video']}" type="video/mp4">
                Your browser does not support the video element.
            </video>
            """
        
        mp3_html = ""
        if song.get('mp3'):
            mp3_html = f"""
            <p><strong>MP3:</strong> <a href="{song['mp3']}" download>Download MP3</a></p>
            """
        
        html += f"""
        <div style="margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
            <h4>Song {i+1}</h4>
            <p><strong>Lyrics:</strong> {song['lyrics'][:100]}...</p>
            <p><strong>Genre:</strong> {song['genre']}</p>
            <p><strong>Instrument:</strong> {song['instrument']}</p>
            <p><strong>Emotion:</strong> {song['emotion']}</p>
            <p><strong>Timbre:</strong> {song['timbre']}</p>
            <p><strong>Gender:</strong> {song['gender']}</p>
            <p><strong>Time:</strong> {song['time']}</p>
            <audio controls>
                <source src="{song['audio']}" type="audio/wav">
                Your browser does not support the audio element.
            </audio>
            {mp3_html}
            {video_html}
        </div>
        """
    return html


def cancel_generation():
    """Cancel the current generation"""
    cancellation_token.cancel()
    gr.Info("Cancellation requested. Generation will stop after current step.")
    return gr.update(visible=False), gr.update(visible=False)

def run_batch_processing(
    input_folder, output_folder, skip_existing, loop_presets, num_generations,
    lyrics, genre, instrument, bpm, emotion, timbre, gender, extra_prompt, include_dropdown_attributes, force_extra_prompt,
    audio_path, image_path, save_mp3, seed,
    duration_seconds, max_gen_length, diffusion_steps, temperature, top_k, top_p,
    cfg_coef, guidance_scale, use_sampling, extend_stride,
    gen_type, chunked, chunk_size, record_tokens, record_window,
    disable_offload, disable_cache_clear, disable_fp16, disable_sequential,
    randomize_params, auto_prompt_enabled, auto_prompt_type, progress=gr.Progress()
):
    """Run batch processing with progress tracking"""
    # Check if model is loaded
    if MODEL is None:
        error_msg = "❌ No model loaded! Please select and load a model first."
        print(f"❌ ERROR: {error_msg}")
        gr.Error(error_msg)
        return error_msg
    
    if not input_folder or not output_folder:
        error_msg = "Please specify both input and output folders"
        print(f"❌ ERROR: {error_msg}")
        gr.Error(error_msg)
        return error_msg
    
    # Show cancel button
    yield gr.update(visible=True), gr.update(visible=True), "Starting batch processing..."
    
    # Collect base parameters
    base_params = collect_current_parameters(
        lyrics, genre, instrument, bpm, emotion, timbre, gender, extra_prompt, include_dropdown_attributes, force_extra_prompt,
        audio_path, image_path, save_mp3, seed,
        duration_seconds, max_gen_length, diffusion_steps, temperature, top_k, top_p,
        cfg_coef, guidance_scale, use_sampling, extend_stride,
        gen_type, chunked, chunk_size, record_tokens, record_window,
        disable_offload, disable_cache_clear, disable_fp16, disable_sequential,
        num_generations, loop_presets, randomize_params
    )
    
    # Set progress callback
    def batch_progress_callback(info):
        progress_msg = []
        if info.get('batch_info'):
            progress_msg.append(f"File: {info['batch_info']}")
        if info.get('preset_info'):
            progress_msg.append(f"Preset: {info['preset_info']}")
        if info.get('generation_info'):
            progress_msg.append(f"Generation: {info['generation_info']}")
        if info.get('message'):
            progress_msg.append(info['message'])
        if info.get('eta') and info['eta'] > 0:
            progress_msg.append(f"ETA: {format_eta(info['eta'])}")
        
        # Add detailed progress information
        detailed_progress = []
        if info.get('phase'):
            detailed_progress.append(f"Phase: {info['phase']}")
        if info.get('diffusion_progress'):
            detailed_progress.append(f"Diffusion: {info['diffusion_progress']}")
        if info.get('current_step') and info.get('total_steps'):
            from logic.ui_progress import create_progress_bar
            progress_bar = create_progress_bar(info['current_step'], info['total_steps'], width=30)
            detailed_progress.append(progress_bar)
        
        status_text = " | ".join(progress_msg)
        if detailed_progress:
            status_text += "\n" + " | ".join(detailed_progress)
            
        progress(info.get('progress', 0), desc=status_text)
        
        # Update batch status
        return status_text
    
    batch_processor.set_progress_callback(batch_progress_callback)
    
    # Run batch processing
    try:
        results = batch_processor.process_batch(
            input_folder=input_folder,
            output_folder=output_folder,
            base_params=base_params,
            num_generations=num_generations,
            loop_presets=loop_presets,
            skip_existing=skip_existing,
            preset_dir=PRESET_DIR
        )
        
        # Format results
        status_msg = f"**Batch Processing Complete**\\n\\n"
        status_msg += f"- Processed: {results['processed']} files\\n"
        status_msg += f"- Skipped: {results['skipped']} files\\n"
        status_msg += f"- Failed: {results['failed']} files\\n"
        
        if results['cancelled']:
            status_msg += "\\n⚠️ Processing was cancelled"
        
        yield gr.update(visible=False), gr.update(visible=False), status_msg
        
    except Exception as e:
        error_msg = f"Batch processing error: {str(e)}"
        print(f"❌ ERROR: {error_msg}")
        gr.Error(error_msg)
        yield gr.update(visible=False), gr.update(visible=False), error_msg

def validate_audio_file(audio_path, use_separation=False):
    """Validate uploaded audio/video file and extract audio if needed
    
    Args:
        audio_path: Path to audio/video file
        use_separation: Whether to use audio separation for processing
        
    Returns:
        Tuple of (processed_audio_path, error_message)
    """
    if not audio_path:
        return None, None
    
    if not os.path.exists(audio_path):
        return None, "File not found"
    
    try:
        # Check file size (limit to 100MB for video files)
        file_size = os.path.getsize(audio_path) / (1024 * 1024)  # Convert to MB
        if file_size > 100:
            return None, f"File too large ({file_size:.1f}MB). Maximum size is 100MB."
        
        # Get file extension
        file_ext = os.path.splitext(audio_path)[1].lower()
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v']
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wma']
        
        # If it's a video file, extract audio
        if file_ext in video_extensions:
            print(f"Detected video file: {os.path.basename(audio_path)}")
            print("Extracting audio from video...")
            
            # Create temp directory for extracted audio
            temp_dir = os.path.join(APP_DIR, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Generate unique filename for extracted audio
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extracted_audio_path = os.path.join(temp_dir, f"extracted_audio_{timestamp}.wav")
            
            # Use ffmpeg to extract audio
            try:
                # Build ffmpeg command
                cmd = [
                    'ffmpeg',
                    '-i', audio_path,           # Input video file
                    '-vn',                      # No video
                    '-acodec', 'pcm_s16le',     # Audio codec
                    '-ar', '48000',             # Sample rate 48kHz
                    '-ac', '2',                 # Stereo
                    '-y',                       # Overwrite output file
                    extracted_audio_path        # Output audio file
                ]
                
                # Run ffmpeg
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    return None, f"Failed to extract audio from video: {result.stderr}"
                
                print(f"Audio extracted successfully to: {extracted_audio_path}")
                
                # Validate the extracted audio
                import torchaudio
                audio, sr = torchaudio.load(extracted_audio_path)
                duration = audio.shape[-1] / sr
                print(f"Extracted audio duration: {duration:.2f} seconds")
                
                return extracted_audio_path, None
                
            except FileNotFoundError:
                return None, "ffmpeg not found. Please install ffmpeg to use video files."
            except Exception as e:
                return None, f"Error extracting audio from video: {str(e)}"
        
        # If it's an audio file, validate it directly
        elif file_ext in audio_extensions:
            import torchaudio
            audio, sr = torchaudio.load(audio_path)
            duration = audio.shape[-1] / sr
            print(f"Audio file duration: {duration:.2f} seconds")
            return audio_path, None
        
        else:
            return None, f"Unsupported file format: {file_ext}. Supported formats: Audio ({', '.join(audio_extensions)}), Video ({', '.join(video_extensions)})"
            
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

def process_reference_audio(audio_path, auto_prompt_enabled, auto_prompt_type, gen_type):
    """Process reference audio including auto prompt and separation
    
    Args:
        audio_path: Path to uploaded audio file
        auto_prompt_enabled: Whether to use auto prompt
        auto_prompt_type: Type of auto prompt
        gen_type: Generation type (mixed, vocal, bgm, separate)
        
    Returns:
        Tuple of (pmt_wav, vocal_wav, bgm_wav, use_audio, status_message, processed_audio_path)
    """
    # Handle uploaded reference audio first (manual reference takes priority)
    if audio_path:
        try:
            validated_path, error_msg = validate_audio_file(audio_path, use_separation=True)
            if error_msg:
                return None, None, None, False, f"⚠️ Audio validation failed: {error_msg}", None
            
            # Use audio separator for processing
            if audio_separator.is_available():
                full_audio, vocal_audio, bgm_audio = audio_separator.separate_audio(validated_path)
                if full_audio is not None:
                    # Convert to appropriate format for model
                    if full_audio.dim() == 2:
                        full_audio = full_audio[None]
                    if vocal_audio.dim() == 2:
                        vocal_audio = vocal_audio[None]  
                    if bgm_audio.dim() == 2:
                        bgm_audio = bgm_audio[None]
                    
                    return full_audio, vocal_audio, bgm_audio, True, "✓ Reference audio processed with separation", validated_path
                else:
                    return None, None, None, False, "⚠️ Audio separation failed", None
            else:
                # Fallback: use original audio without separation
                import torchaudio
                audio, sr = torchaudio.load(validated_path)
                if sr != 48000:
                    audio = torchaudio.functional.resample(audio, sr, 48000)
                if audio.shape[-1] > 48000 * 10:  # Limit to 10 seconds
                    audio = audio[..., :48000 * 10]
                if audio.dim() == 2:
                    audio = audio[None]
                
                return audio, audio, audio, True, "✓ Reference audio loaded (no separation available)", validated_path
                
        except Exception as e:
            return None, None, None, False, f"⚠️ Reference audio error: {str(e)}", None
    
    # If no manual audio (or not provided), fall back to auto prompt if enabled
    if auto_prompt_enabled and auto_prompt_manager.is_available():
        try:
            prompt_tokens = auto_prompt_manager.get_auto_prompt_tokens(auto_prompt_type)
            if prompt_tokens:
                pmt_wav, vocal_wav, bgm_wav = prompt_tokens
                return pmt_wav, vocal_wav, bgm_wav, True, f"✓ Using auto prompt: {auto_prompt_type}", None
            else:
                return None, None, None, False, f"⚠️ Auto prompt '{auto_prompt_type}' not available", None
        except Exception as e:
            return None, None, None, False, f"⚠️ Auto prompt error: {str(e)}", None
    
    # No reference audio
    return None, None, None, False, "", None

def submit_lyrics(
    lyrics, struct, genre, instrument, bpm, emotion, timbre, gender, extra_prompt, include_dropdown_attributes, force_extra_prompt,
    audio_path, image_path, save_mp3, seed,
    duration_seconds, max_gen_length, diffusion_steps, temperature, top_k, top_p,
    cfg_coef, guidance_scale, use_sampling, extend_stride,
    gen_type, chunked, chunk_size, record_tokens, record_window,
    disable_offload, disable_cache_clear, disable_fp16, disable_sequential,
    num_generations, loop_presets, randomize_params, 
    auto_prompt_enabled, auto_prompt_type,  # New parameters
    history, session, progress=gr.Progress()
):
    # Check if model is loaded
    if MODEL is None:
        error_msg = "❌ **ERROR: No model loaded!**\n\nPlease select a model from the dropdown and click '🔄 Load Selected Model' before generating."
        print(error_msg)
        yield None, None, history, process_history(history), gr.update(visible=False), gr.update(value=error_msg, visible=True)
        return
    
    # Reset cancellation token
    cancellation_token.reset()
    
    # Show cancel button and progress immediately
    yield None, None, history, process_history(history), gr.update(visible=True), gr.update(visible=True)
    
    # Create progress callback
    progress_callback = create_progress_callback(gradio_progress_tracker, progress)
    
    # Validate lyrics structure first
    is_valid, validation_message, normalized_lyrics = validate_lyrics_structure(lyrics)
    if not is_valid:
        error_msg = f"Lyrics validation failed: {validation_message}"
        print(f"❌ ERROR: {error_msg}")
        gr.Error(error_msg)
        yield None, None, history, process_history(history), gr.update(visible=False), gr.update(visible=False)
        return
    
    output_messages(f"✓ Lyrics structure validation passed")
    lyrics = normalized_lyrics
    
    # Limit lyrics length to prevent exceeding token limit
    # Get current model's character limit
    current_model_max = MAX_GENERATION_LENGTHS.get(MODEL_VERSION, 6750) if MODEL_VERSION else 6750
    MAX_CHARS = int((current_model_max / 25.0) * 6.5 * 300 / 3750 * 1500)
    MAX_CHARS = min(MAX_CHARS, 5000)  # Cap at 5000 for Large model
    
    if len(lyrics) > MAX_CHARS:
        lyrics = lyrics[:MAX_CHARS]
        output_messages(f"Lyrics truncated to {MAX_CHARS} characters to fit token limit")
    
    # Log basic info without debug prefix
    print(f"Lyrics length: {len(lyrics)} characters")
    
    # Set seed if provided
    used_seed = seed
    if seed is None or seed < 0:
        # Generate a random seed
        used_seed = random.randint(0, 2147483647)
    
    set_seed(used_seed)
    output_messages(f"Using seed: {used_seed}")
    
    formatted_lyrics = lyrics
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process reference audio (including auto prompt)
    pmt_wav, vocal_wav, bgm_wav, use_audio, audio_status, processed_audio_path = process_reference_audio(
        audio_path, auto_prompt_enabled, auto_prompt_type, gen_type
    )
    
    if audio_status:
        output_messages(audio_status)
    
    # Call the model using the forward method with progress tracking
    # Harmonize duration slider (seconds) with step-based control
    duration_seconds = clamp_duration_seconds(duration_seconds)
    max_gen_length = clamp_steps_to_model(seconds_to_steps(duration_seconds))
    
    # Convert steps to duration (25 steps per second)
    # Large model supports up to 6750 steps (~270 seconds or 4m30s)
    duration_from_steps = steps_to_seconds(max_gen_length)
    
    # Get model's actual max capacity
    current_model_max = get_current_model_max_steps()
    max_duration = steps_to_seconds(current_model_max)
    
    # Cap duration at model's maximum capacity
    duration_from_steps = min(duration_from_steps, max_duration)
    actual_steps = clamp_steps_to_model(seconds_to_steps(duration_from_steps))
    max_gen_length = actual_steps

    # IMPORTANT: song_data["lyrics"] contains the user's lyrics and must NEVER be overwritten
    # when looping through presets. Each preset should use these same lyrics.
    song_data = {
        "lyrics": formatted_lyrics,  # User's lyrics - constant across all preset loops
        "struct": process_struct(struct),
        "genre": genre,
        "instrument": instrument,
        "bpm": bpm,
        "emotion": emotion,
        "timbre": timbre,
        "gender": gender,
        "sample_prompt": use_audio,
        "audio_path": processed_audio_path if processed_audio_path else None,
        "original_audio_path": audio_path,
        "pmt_wav": pmt_wav,
        "vocal_wav": vocal_wav,
        "bgm_wav": bgm_wav,
        "melody_is_wav": bool(processed_audio_path),
        "time": current_time,
        "gen_type": gen_type,
        "auto_prompt_type": auto_prompt_type if auto_prompt_enabled else None,
        "include_dropdown_attributes": include_dropdown_attributes,
        "duration_seconds": duration_from_steps
    }
    
    # Log the actual values for debugging
    output_messages(f"Requested steps: {max_gen_length}, Actual generation: {actual_steps} steps (~{duration_from_steps:.1f}s, {int(duration_from_steps/60)}m{int(duration_from_steps%60)}s)")
    
    if force_extra_prompt and not (extra_prompt and extra_prompt.strip()):
        output_messages("⚠️ Warning: 'Use only extra prompt' is enabled but no extra description was provided!")
        output_messages("Using fallback placeholder for description...")

    description_for_model = compose_generation_description(
        extra_prompt,
        include_dropdown_attributes,
        gender,
        timbre,
        genre,
        emotion,
        instrument,
        bpm,
        force_extra_prompt,
    )

    try:
        top_k_value = int(top_k)
    except (TypeError, ValueError):
        top_k_value = -1
    if top_k_value < 0:
        top_k_value = -1

    try:
        top_p_value = float(top_p)
    except (TypeError, ValueError):
        top_p_value = 0.0

    gen_params = {
        'duration': duration_from_steps,
        'num_steps': diffusion_steps,
        'temperature': temperature,
        'top_k': top_k_value,
        'top_p': top_p_value,
        'cfg_coef': cfg_coef,
        'guidance_scale': guidance_scale,
        'use_sampling': use_sampling,
        'extend_stride': extend_stride,
        'chunked': chunked,
        'chunk_size': chunk_size,
        'record_tokens': record_tokens,
        'record_window': record_window,
    }
    
    # Skip initial generation if we're going to loop through presets
    # The first preset in the loop will handle the generation
    if not loop_presets:
        try:
            # Store initial values before potential randomization
            current_genre = genre
            current_instrument = instrument
            current_emotion = emotion
            current_timbre = timbre
            current_gender = gender
            
            # Use progress interceptor to capture tqdm output
            with intercept_progress(progress_callback):
                description_for_generation = description_for_model
                # Print final prompt to console
                print("\n" + "="*80)
                print("GENERATING SONG WITH PROMPT:")
                print("="*80)
                print(f"Description: {description_for_generation}")
                if song_data["sample_prompt"] and song_data["audio_path"]:
                    print(f"Using reference audio for consistent voice")
                print("="*80 + "\n")
                
                # Prepare generation input based on new format
                generate_input = {
                    'lyrics': [song_data["lyrics"]],
                    'descriptions': [description_for_generation],
                    'melody_wavs': song_data["pmt_wav"],
                    'vocal_wavs': song_data["vocal_wav"],
                    'bgm_wavs': song_data["bgm_wav"],
                    'melody_is_wav': song_data["melody_is_wav"]
                }
                
                # Call model with correct interface matching original
                audio_data = MODEL(
                    lyric=song_data["lyrics"],
                    description=description_for_generation,
                    prompt_audio_path=song_data["audio_path"] if song_data["sample_prompt"] else None,
                    genre=song_data["auto_prompt_type"] if song_data["auto_prompt_type"] else None,
                    auto_prompt_path=auto_prompt_manager.get_fallback_prompt_path(),
                    gen_type=song_data["gen_type"],
                    params=gen_params,
                    disable_offload=disable_offload,
                    disable_cache_clear=disable_cache_clear,
                    disable_fp16=disable_fp16,
                    disable_sequential=disable_sequential,
                    progress_callback=progress_callback,
                    cancellation_token=cancellation_token
                )
            
            if audio_data is None:
                gr.Info("Generation cancelled")
                return None, None, history, process_history(history), gr.update(visible=False), gr.update(visible=False)
                
            audio_data = audio_data.cpu().permute(1, 0).float().numpy()
            
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            print(f"❌ ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            gr.Error(error_msg)
            yield None, None, history, process_history(history), gr.update(visible=False), gr.update(value=error_msg, visible=True)
            return
    else:
        # Initialize audio_data to None when looping presets
        audio_data = None
    
    # Load all presets if loop_presets is enabled
    presets_to_use = [None]  # None means use current settings
    if loop_presets:
        all_presets = preset_manager.get_preset_list()
        if all_presets:
            presets_to_use = all_presets
            output_messages(f"\n{'='*50}")
            output_messages(f"PRESET LOOP MODE: Processing {len(presets_to_use)} presets")
            output_messages(f"Presets to process: {', '.join(presets_to_use)}")
            output_messages(f"{'='*50}\n")
        else:
            output_messages("No presets found, using current settings only")
    
    # Generate multiple songs if requested
    generated_files = []
    total_generations = len(presets_to_use) * num_generations
    generation_count = 0
    
    # Log generation settings
    if num_generations > 1 or randomize_params:
        print(f"Generation settings: {num_generations} generations, randomization {'enabled' if randomize_params else 'disabled'}")
    
    # Get the starting file number once
    output_dir = op.join(APP_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    starting_file_number = get_next_file_number(output_dir)
    file_counter = 0
    
    # Show initial file numbering info
    existing_files = [f for f in os.listdir(output_dir) if f.endswith(('.wav', '.mp3', '.mp4'))]
    if existing_files:
        output_messages(f"Found {len(existing_files)} existing files in output directory")
    output_messages(f"Starting file number: {starting_file_number:04d}")
    
    # Preset loop (outer loop)
    for preset_idx, preset_name in enumerate(presets_to_use):
        # Progress info for preset
        if loop_presets:
            output_messages(f"\n{'='*40}")
            output_messages(f"Processing Preset {preset_idx + 1}/{len(presets_to_use)}: {preset_name if preset_name else 'Current Settings'}")
            output_messages(f"{'='*40}")
        
        # Load preset if specified
        if preset_name:
            output_messages(f"Loading preset '{preset_name}'...")
            preset_data, message = preset_manager.load_preset(preset_name)
            if preset_data:
                # Show preset configuration being loaded
                output_messages(message)
                
                # IMPORTANT: When looping presets, ALWAYS use the user's lyrics, never the preset's lyrics
                # This ensures consistency across all preset generations
                
                # Override current parameters with preset values (EXCLUDING lyrics)
                genre = preset_data.get('genre', genre)
                instrument = preset_data.get('instrument', instrument)
                bpm = preset_data.get('bpm', bpm)
                emotion = preset_data.get('emotion', emotion)
                timbre = preset_data.get('timbre', timbre)
                gender = preset_data.get('gender', gender)
                extra_prompt = preset_data.get('extra_prompt', extra_prompt)
                force_extra_prompt = preset_data.get('force_extra_prompt', force_extra_prompt)
                # Update current_* variables used for generation
                current_genre = genre
                current_instrument = instrument
                current_emotion = emotion
                current_timbre = timbre
                current_gender = gender
                # Also get preset's randomize setting
                randomize_params = preset_data.get('randomize_params', randomize_params)
                
                if force_extra_prompt and extra_prompt:
                    output_messages(f"Using only extra prompt: {extra_prompt}")
                else:
                    output_messages(f"Using preset settings: {current_genre}, {current_instrument}, {current_emotion}, {current_timbre}, {current_gender}")
                output_messages(f"Keeping user's lyrics (not using preset lyrics)")
            else:
                output_messages(f"Failed to load preset {preset_name}, skipping")
                continue
        
        # Initialize current_* variables before the loop
        if not preset_name:  # Only if we're not in preset loop mode
            current_genre = genre
            current_instrument = instrument
            current_emotion = emotion
            current_timbre = timbre
            current_gender = gender
        
        # Generation loop (inner loop)
        for gen_idx in range(num_generations):
            if cancellation_token.is_cancelled():
                gr.Info("Generation cancelled")
                break
            
            generation_count += 1
            
            # Detailed progress info
            if loop_presets or num_generations > 1:
                output_messages(f"\n--- Generation {generation_count}/{total_generations} ---")
                if preset_name:
                    output_messages(f"Preset: {preset_name} ({preset_idx + 1}/{len(presets_to_use)})")
                output_messages(f"Generation: {gen_idx + 1}/{num_generations} within this preset")
                output_messages(f"Files completed: {file_counter}, Next file: {starting_file_number + file_counter:04d}")
            
            # Update progress for multiple generations
            progress_desc = f"Generating {generation_count}/{total_generations}"
            if preset_name:
                progress_desc += f" | Preset: {preset_name} ({preset_idx + 1}/{len(presets_to_use)})"
            if num_generations > 1:
                progress_desc += f" | Generation {gen_idx + 1}/{num_generations}"
            
            # Add progress bar visualization
            from logic.ui_progress import create_progress_bar
            progress_bar = create_progress_bar(generation_count, total_generations, width=30)
            progress_desc += f"\n{progress_bar}"
            
            progress((generation_count / total_generations), desc=progress_desc)
        
            # Generate new seed for each iteration after the first
            if gen_idx > 0:
                used_seed = random.randint(0, 2147483647)
                set_seed(used_seed)
                output_messages(f"Generation {gen_idx + 1}: Using seed: {used_seed}")
            
            # Randomize parameters if enabled (for each generation)
            if randomize_params and gen_idx > 0:  # Keep first generation with original values
                current_genre = random.choice(GENRES)
                current_instrument = random.choice(INSTRUMENTS)
                current_emotion = random.choice(EMOTIONS)
                current_timbre = random.choice(TIMBRES)
                current_gender = random.choice(GENDERS)
                output_messages(f"Randomized: {current_genre}, {current_instrument}, {current_emotion}, {current_timbre}, {current_gender}")
            
            # Generate audio for each preset/generation
            # When loop_presets is True: always generate (including first)
            # When loop_presets is False: skip first generation if we already generated above
            should_generate = loop_presets or (not loop_presets and (preset_name is not None or gen_idx > 0))
            
            # Build model description based on optional extra prompt
            generation_description = compose_generation_description(
                extra_prompt,
                include_dropdown_attributes,
                current_gender,
                current_timbre,
                current_genre,
                current_emotion,
                current_instrument,
                bpm,
                force_extra_prompt,
            )
            
            if should_generate:
                try:
                    with intercept_progress(progress_callback):
                        # Print mode info if using force extra prompt
                        if force_extra_prompt:
                            print("\n" + "="*80)
                            print(f"FORCE EXTRA PROMPT MODE (Generation {generation_count}/{total_generations}) - Ignoring dropdown selections")
                            print("="*80)
                        
                        # Print final prompt to console
                        print("\n" + "="*80)
                        print(f"GENERATING SONG {generation_count}/{total_generations} WITH PROMPT:")
                        print("="*80)
                        print(f"Description: {generation_description}")
                        if preset_name:
                            print(f"Preset: {preset_name}")
                        if song_data["sample_prompt"] and song_data["audio_path"]:
                            print(f"Using reference audio for consistent voice")
                        print("="*80 + "\n")
                        
                        audio_data = MODEL(
                            song_data["lyrics"], 
                            generation_description,
                            song_data["audio_path"] if song_data["sample_prompt"] else None,
                            None,
                            op.join(APP_DIR, "ckpt/prompt.pt"),
                            gen_type,
                            gen_params,
                            disable_offload=disable_offload,
                            disable_cache_clear=disable_cache_clear,
                            disable_fp16=disable_fp16,
                            disable_sequential=disable_sequential,
                            progress_callback=progress_callback,
                            cancellation_token=cancellation_token
                        )
                    
                    if audio_data is None:
                        continue
                        
                    audio_data = audio_data.cpu().permute(1, 0).float().numpy()
                    
                except Exception as e:
                    error_msg = f"Generation {gen_idx + 1} failed: {str(e)}"
                    print(f"❌ ERROR: {error_msg}")
                    import traceback
                    traceback.print_exc()
                    gr.Error(error_msg)
                    output_messages(f"❌ {error_msg}")
                    continue
            
            # Generate separate tracks if requested (runs regardless of should_generate)
            vocal_audio_data = None
            bgm_audio_data = None
            
            if gen_type == 'separate' and audio_data is not None:
                try:
                    # Generate separate vocal and BGM tracks
                    with intercept_progress(progress_callback):
                        print("Generating separate vocal track...")
                        vocal_audio_data = MODEL(
                            lyric=song_data["lyrics"],
                            description=generation_description,
                            prompt_audio_path=song_data["audio_path"],
                            genre=song_data["auto_prompt_type"] if song_data["auto_prompt_type"] else None,
                            auto_prompt_path=auto_prompt_manager.get_fallback_prompt_path(),
                            gen_type='vocal',
                            params=gen_params
                        )
                        
                        if vocal_audio_data is not None:
                            vocal_audio_data = vocal_audio_data.cpu().permute(1, 0).float().numpy()
                        
                        print("Generating separate BGM track...")
                        bgm_audio_data = MODEL(
                            lyric=song_data["lyrics"],
                            description=generation_description,
                            prompt_audio_path=song_data["audio_path"],
                            genre=song_data["auto_prompt_type"] if song_data["auto_prompt_type"] else None,
                            auto_prompt_path=auto_prompt_manager.get_fallback_prompt_path(),
                            gen_type='bgm',
                            params=gen_params
                        )
                        
                        if bgm_audio_data is not None:
                            bgm_audio_data = bgm_audio_data.cpu().permute(1, 0).float().numpy()
                            
                except Exception as e:
                    output_messages(f"⚠️ Separate track generation failed: {str(e)}")
                    vocal_audio_data = None
                    bgm_audio_data = None
            
            # Save the audio with sequential numbering
            # Make sure we have audio_data to save
            if audio_data is not None:
                # Calculate the current file number
                current_file_number = starting_file_number + file_counter
                file_counter += 1
                
                # Construct filename based on preset and generation
                if preset_name and num_generations > 1:
                    base_filename = f"{current_file_number:04d}_{preset_name}_gen{gen_idx+1:03d}"
                elif preset_name:
                    base_filename = f"{current_file_number:04d}_{preset_name}"
                elif num_generations > 1:
                    base_filename = f"{current_file_number:04d}_gen{gen_idx+1:03d}"
                else:
                    base_filename = f"{current_file_number:04d}"
                
                # Save as both WAV and FLAC (FLAC is the original format)
                wav_path = op.join(output_dir, f"{base_filename}.wav")
                flac_path = op.join(output_dir, f"{base_filename}.flac")
                
                # Save WAV for compatibility
                try:
                    print(f"Debug: audio_data shape: {audio_data.shape}, dtype: {audio_data.dtype}, sample_rate: {MODEL.cfg.sample_rate}")
                    wavfile.write(wav_path, MODEL.cfg.sample_rate, audio_data)
                    output_messages(f"✓ Audio saved: {base_filename}.wav")
                except Exception as e:
                    print(f"Warning: Could not save WAV: {e}")
                    print(f"Debug: audio_data type: {type(audio_data)}, shape: {getattr(audio_data, 'shape', 'no shape')}")
                    output_messages(f"⚠️ Failed to save WAV: {base_filename}.wav")
                    continue  # Skip the rest of file saving for this generation
                
                # Save FLAC (original format)
                try:
                    sf.write(flac_path, audio_data, MODEL.cfg.sample_rate, format="FLAC")
                    output_messages(f"✓ FLAC saved: {base_filename}.flac")
                except Exception as e:
                    print(f"Warning: Could not save FLAC: {e}")
                
                # Save separate tracks if generated
                vocal_wav_path = None
                bgm_wav_path = None
                
                if gen_type == 'separate':
                    if vocal_audio_data is not None:
                        vocal_wav_path = op.join(output_dir, f"{base_filename}_vocal.wav")
                        wavfile.write(vocal_wav_path, MODEL.cfg.sample_rate, vocal_audio_data)
                        output_messages(f"✓ Vocal track saved: {base_filename}_vocal.wav")
                    
                    if bgm_audio_data is not None:
                        bgm_wav_path = op.join(output_dir, f"{base_filename}_bgm.wav")
                        wavfile.write(bgm_wav_path, MODEL.cfg.sample_rate, bgm_audio_data)
                        output_messages(f"✓ BGM track saved: {base_filename}_bgm.wav")
                
                # Convert to MP3 if requested
                mp3_path = None
                vocal_mp3_path = None
                bgm_mp3_path = None
                
                if save_mp3:
                    # Main track MP3
                    mp3_path = op.join(output_dir, f"{base_filename}.mp3")
                    if convert_wav_to_mp3(wav_path, mp3_path, '192k'):
                        output_messages(f"✓ MP3 saved: {base_filename}.mp3")
                    else:
                        mp3_path = None
                        output_messages("✗ Failed to convert main track to MP3")
                else:
                    mp3_path = None
                
                # Save reference audio copy if available
                reference_copy_path = None
                reference_source_path = song_data.get("original_audio_path")
                processed_reference_path = song_data.get("audio_path")
                if reference_source_path and os.path.exists(reference_source_path):
                    reference_copy_path = op.join(output_dir, f"{base_filename}_reference_audio{op.splitext(reference_source_path)[1]}")
                    try:
                        shutil.copy(reference_source_path, reference_copy_path)
                        output_messages(f"✓ Reference audio copied: {op.basename(reference_copy_path)}")
                    except Exception as ref_err:
                        reference_copy_path = None
                        output_messages(f"⚠️ Unable to copy reference audio: {ref_err}")
                elif processed_reference_path and os.path.exists(processed_reference_path):
                    reference_copy_path = op.join(output_dir, f"{base_filename}_reference_audio.wav")
                    try:
                        shutil.copy(processed_reference_path, reference_copy_path)
                        output_messages(f"✓ Reference audio copied: {op.basename(reference_copy_path)}")
                    except Exception as ref_err:
                        reference_copy_path = None
                        output_messages(f"⚠️ Unable to copy processed reference audio: {ref_err}")
                    
                    # Separate tracks MP3
                    if gen_type == 'separate':
                        if vocal_wav_path:
                            vocal_mp3_path = op.join(output_dir, f"{base_filename}_vocal.mp3")
                            if convert_wav_to_mp3(vocal_wav_path, vocal_mp3_path, '192k'):
                                output_messages(f"✓ Vocal MP3 saved: {base_filename}_vocal.mp3")
                            else:
                                vocal_mp3_path = None
                        
                        if bgm_wav_path:
                            bgm_mp3_path = op.join(output_dir, f"{base_filename}_bgm.mp3")
                            if convert_wav_to_mp3(bgm_wav_path, bgm_mp3_path, '192k'):
                                output_messages(f"✓ BGM MP3 saved: {base_filename}_bgm.mp3")
                            else:
                                bgm_mp3_path = None
                
                # Create video if image is provided
                video_path = None
                if image_path:
                    video_path = op.join(output_dir, f"{base_filename}.mp4")
                    if create_video_from_image_and_audio(image_path, wav_path, video_path):
                        output_messages(f"✓ Video saved: {base_filename}.mp4")
                    else:
                        video_path = None
                        output_messages("✗ Failed to create video")
                
                # For the first generation, update the main outputs
                if gen_idx == 0 and preset_idx == 0:
                    song_data["audio"] = wav_path
                    song_data["mp3"] = mp3_path
                    song_data["video"] = video_path
                
                # Save metadata (use actual generated values, not original if randomized)
                metadata = collect_current_parameters(
                    lyrics, current_genre, current_instrument, bpm, current_emotion, current_timbre, current_gender, extra_prompt, include_dropdown_attributes, force_extra_prompt,
                    audio_path, image_path, save_mp3, used_seed,
                    duration_from_steps, max_gen_length, diffusion_steps, temperature, top_k, top_p,
                    cfg_coef, guidance_scale, use_sampling, extend_stride,
                    gen_type, chunked, chunk_size, record_tokens, record_window,
                    disable_offload, disable_cache_clear, disable_fp16, disable_sequential,
                    num_generations, loop_presets, randomize_params
                )
                metadata['timestamp'] = current_time
                metadata['model'] = CURRENT_MODEL_PATH
                metadata['generation_index'] = gen_idx + 1
                metadata['total_generations'] = num_generations
                metadata['output_files'] = {
                    'wav': wav_path,
                    'mp3': mp3_path,
                    'mp4': video_path
                }
                
                save_metadata(wav_path, metadata)
                generated_files.append({'wav': wav_path, 'mp3': mp3_path, 'mp4': video_path})
                
                # Show completion status for this generation
                output_messages(f"✓ Generation {generation_count}/{total_generations} complete")
            else:
                output_messages(f"⚠️ No audio data for generation {generation_count}/{total_generations}, skipping file save")
    
    # Update history
    history.append({"role": "user", "content": f"Generate {num_generations} song(s) with lyrics: {lyrics[:50]}..."})
    
    # Final summary
    output_messages(f"\n{'='*50}")
    output_messages(f"GENERATION COMPLETE")
    output_messages(f"Total files generated: {len(generated_files)}")
    if loop_presets:
        output_messages(f"Presets processed: {len([p for p in presets_to_use if p is not None])}")
    output_messages(f"Output directory: {output_dir}")
    output_messages(f"{'='*50}\n")
    
    # Check if cancelled
    if cancellation_token.is_cancelled():
        history.append({"role": "assistant", "content": f"Generation cancelled. Generated {len(generated_files)} song(s) before cancellation.", "song": song_data if generated_files else None})
    else:
        history.append({"role": "assistant", "content": f"Generated {len(generated_files)} song(s) successfully!", "song": song_data})
    
    # Hide cancel button and progress when complete
    yield song_data.get("audio"), song_data.get("video"), history, process_history(history), gr.update(visible=False), gr.update(visible=False)

# Create Gradio interface
with gr.Blocks(title="SECourses LeVo Song Generation App",theme=gr.themes.Soft()) as demo:
    gr.Markdown("# SECourses Premium LeVo Song Generator V9.0 - Up to 4m30s Songs - https://www.patreon.com/posts/135592123")
    
    history = gr.State([])
    session = gr.State({})
    
    # Model Selection Section
    with gr.Row():
        with gr.Column(scale=3):
            available_models = get_available_models()
            model_choices = [f"{desc} ({model_dir})" for model_dir, desc, _ in available_models]
            model_paths = {f"{desc} ({model_dir})": path for model_dir, desc, path in available_models}
            
            model_dropdown = gr.Dropdown(
                label="🎵 Select SongGeneration Model",
                choices=model_choices,
                value=model_choices[0] if model_choices else None,
                info="Large (22-28GB VRAM, BEST) | Base Full (12-18GB VRAM, GOOD) - Both support 4m30s songs"
            )
        
        with gr.Column(scale=2):
            if not model_choices:
                model_info_display = gr.Markdown("""
                ❌ **No models found!**
                
                Please download models first using:
                ```bash
                python Download_Song_Models.py --model model_large
                ```
                or
                ```bash  
                python Download_Song_Models.py --model model_base
                ```
                """)
                load_model_btn = gr.Button("🔄 Refresh Model List", variant="secondary")
            else:
                model_info_display = gr.Markdown("Select a model to see details")
                load_model_btn = gr.Button("🔄 Load Selected Model", variant="primary")
            
            model_status = gr.Markdown("**Status**: No model loaded", elem_id="model-status")
    
    # Add cancel button at the top
    with gr.Row():
        cancel_btn = gr.Button("🛑 Cancel Generation", variant="stop", visible=False)
    
    # Progress display with detailed status
    with gr.Row():
        progress_text = gr.Markdown("", visible=False)
    
    # Main tabs at the top
    with gr.Tabs():
        with gr.TabItem("🎵 Song Generation"):
            with gr.Row():
                with gr.Column(scale=2):
                    lyrics = gr.Textbox(
                        label="Lyrics",
                        placeholder="Enter your lyrics here...",
                        value=EXAMPLE_LYRICS,
                        lines=15,
                        max_lines=20,
                        info="Both models support up to ~5000 characters for 4m30s songs (Large: best quality, Base Full: good quality)"
                    )
                    
                    # Character counter and duration display
                    char_counter = gr.Markdown("0 characters | Estimated duration: 0:00")
            
                    # Generate and Open Folder buttons at the top
                    with gr.Row():
                        submit_btn = gr.Button("🎵 Generate Song", variant="primary")
                        open_folder_btn = gr.Button("📁 Open Output Folder", variant="secondary")
            
                    # Preset controls and duration
                    with gr.Row():
                        with gr.Column(scale=3):
                            with gr.Accordion("Presets", open=True):
                                with gr.Row():
                                    preset_dropdown = gr.Dropdown(
                                        label="Select Preset",
                                        choices=preset_manager.get_preset_list(),
                                        value=None,
                                        interactive=True
                                    )
                                    load_preset_btn = gr.Button("📂 Load Preset", variant="secondary")
                                    refresh_preset_btn = gr.Button("🔄", variant="secondary", scale=0)
                                
                                with gr.Row():
                                    preset_name_input = gr.Textbox(
                                        label="Preset Name",
                                        placeholder="Enter preset name to save...",
                                        scale=3
                                    )
                                    save_preset_btn = gr.Button("💾 Save Preset", variant="secondary", scale=1)
                                
                                with gr.Row():
                                    loop_presets = gr.Checkbox(
                                        label="Loop all presets",
                                        value=False,
                                        info="Generate songs using each saved preset"
                                    )
                                    
                                    randomize_params = gr.Checkbox(
                                        label="Randomize genre, instrument, emotion, timbre & gender",
                                        value=False,
                                        info="Randomly select values from dropdowns for each generation"
                                    )
                        with gr.Column(scale=2):
                            duration_slider = gr.Slider(
                                label="Duration (seconds)",
                                minimum=DURATION_SLIDER_UI_MIN,
                                maximum=int(steps_to_seconds(DEFAULT_MAX_GENERATION_STEPS)),
                                value=DEFAULT_DURATION_SECONDS,
                                step=1,
                                info=format_duration_slider_info(DEFAULT_MAX_GENERATION_STEPS)
                            )
                    
                    struct = gr.JSON(
                        label="Song Structure (Optional - for display only)",
                        value=[
                            ["intro", 1],
                            ["verse", 1],
                            ["bridge", 1],
                            ["inst", 1],
                            ["chorus", 1],
                            ["outro", 1]
                        ],
                        visible=False  # Hide since it's not used by the model
                    )
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Row():
                                genre = gr.Dropdown(
                                    label="Genre",
                                    choices=GENRES,
                                    value="None"
                                )
                                instrument = gr.Dropdown(
                                    label="Instrument",
                                    choices=INSTRUMENTS,
                                    value="None"
                                )
                                bpm = gr.Slider(
                                    label="BPM",
                                    minimum=30,
                                    maximum=200,
                                    value=150,
                                    step=1,
                                    info="Beats per minute"
                                )
                            
                            with gr.Row():
                                emotion = gr.Dropdown(
                                    label="Emotion",
                                    choices=EMOTIONS,
                                    value="None"
                                )
                                timbre = gr.Dropdown(
                                    label="Timbre",
                                    choices=TIMBRES,
                                    value="None"
                                )
                                gender = gr.Dropdown(
                                    label="Gender",
                                    choices=GENDERS,
                                    value="None"
                                )
                        with gr.Column(scale=1):
                            extra_prompt = gr.Textbox(
                                label="Extra Description (Optional)",
                                placeholder="e.g., pop influences, bpm 130, acoustic feel, summer vibes...",
                                lines=4,
                                info="Add any additional musical description or parameters not covered by the dropdowns"
                            )
                            force_extra_prompt = gr.Checkbox(
                                label="Use only extra prompt",
                                value=False,
                                info="Ignore dropdowns and use only the extra description"
                            )
                            include_dropdown_attributes = gr.Checkbox(
                                label="Include dropdown attributes",
                                value=False,
                                info="Append selected gender, timbre, genre, emotion, instrument and BPM to the description"
                            )
                    
                    with gr.Row():
                        gen_type = gr.Radio(
                            label="Generation Type",
                            choices=GENERATION_TYPES,
                            value="mixed",
                            info="mixed: vocals+BGM combined | vocal: vocals only | bgm: music only | separate: vocals and BGM as separate files"
                        )
                    
                    with gr.Row():
                        save_mp3_check = gr.Checkbox(label="Also save as MP3 (192 kbps)", value=True)
                        seed_input = gr.Number(label="Seed (for reproducibility)", value=-1, precision=0, 
                                            info="Use -1 for random, or any positive number for reproducible results")
                    
                    description_preview = gr.Markdown(format_description_preview("[Musicality-very-high], ."))

                    def update_description_preview(
                        extra_prompt_value,
                        include_dropdown_value,
                        force_extra_value,
                        gender_value,
                        timbre_value,
                        genre_value,
                        emotion_value,
                        instrument_value,
                        bpm_value,
                    ):
                        description = compose_generation_description(
                            extra_prompt_value,
                            include_dropdown_value,
                            gender_value,
                            timbre_value,
                            genre_value,
                            emotion_value,
                            instrument_value,
                            bpm_value,
                            force_extra_value,
                        )
                        return format_description_preview(description)

                    description_preview_inputs = [
                        extra_prompt,
                        include_dropdown_attributes,
                        force_extra_prompt,
                        gender,
                        timbre,
                        genre,
                        emotion,
                        instrument,
                        bpm,
                    ]

                    for _component in description_preview_inputs:
                        _component.change(
                            fn=update_description_preview,
                            inputs=description_preview_inputs,
                            outputs=[description_preview]
                        )
                    
                    # Number of generations slider
                    with gr.Row():
                        num_generations = gr.Slider(
                    label="Number of Generations", 
                    minimum=1, 
                    maximum=999, 
                    value=1, 
                    step=1,
                    info="Generate multiple songs with different seeds. Large model can create 999 unique 4m30s songs in one batch!"
                        )
                    
                    # Auto Prompt Audio Selection
                    with gr.Accordion("🎵 Auto Prompt Audio Selection", open=True):
                        with gr.Row():
                            auto_prompt_enabled = gr.Checkbox(
                                label="Use Auto Prompt Audio",
                                value=False,
                                info="Automatically select reference audio based on genre/style"
                            )
                            # Get available types and ensure "Auto" is always included
                            available_types = auto_prompt_manager.get_available_types() if auto_prompt_manager.is_available() else []
                            if available_types and "Auto" not in available_types:
                                available_types = available_types + ["Auto"]
                            dropdown_choices = available_types if available_types else AUTO_PROMPT_TYPES
                            
                            auto_prompt_type = gr.Dropdown(
                                label="Auto Prompt Type",
                                choices=dropdown_choices,
                                value="Auto",
                                visible=False,
                                info="Select musical style for automatic reference audio"
                            )
                        
                        auto_prompt_status = gr.Markdown("", visible=False)
                        
                        def toggle_auto_prompt(enabled):
                            return gr.update(visible=enabled), gr.update(visible=enabled if enabled else False)
                        
                        auto_prompt_enabled.change(
                            fn=toggle_auto_prompt,
                            inputs=[auto_prompt_enabled],
                            outputs=[auto_prompt_type, auto_prompt_status]
                        )
                    
                with gr.Column(scale=1):
                    output_audio = gr.Audio(label="Generated Audio", type="filepath")
                    output_video = gr.Video(label="Generated Video", visible=True)
                    
                    gr.Markdown("---")
                    gr.Markdown("### 🎵 LeVo Models\n**Large** (★★★★★ Best, 22-28GB) | **Base Full** (★★★★ Good, 12-18GB)\nBoth support 4m30s songs • Chinese + English")
                    
                    # Reference audio section (moved above image upload)
                    with gr.Accordion("📁 Reference Audio/Video (Advanced) - Maximum 10 seconds utilized", open=True):
                        with gr.Row():
                            file_upload = gr.File(
                                label="Upload Audio or Video File",
                                file_types=[".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma",
                                           ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"],
                                file_count="single",
                                visible=True
                            )
                            audio_component = gr.Audio(
                                label="Audio Preview (trim to best 10 seconds)",
                                type="filepath",
                                visible=False
                            )
                        audio_path = gr.Textbox(visible=False)
                        upload_status = gr.Markdown("", visible=False)
                        
                        with gr.Accordion("ℹ️ Reference Tips", open=False):
                            gr.Markdown("""
                            **Recommended Usage**
                            - Upload a clean vocal snippet or video clip featuring the target voice
                            - Only the first 10 seconds are used, so trim to the strongest phrase
                            - Keep background noise and heavy effects to a minimum
                            - Supported audio: WAV, MP3, FLAC, M4A, AAC, OGG, WMA
                            - Supported video: MP4, AVI, MOV, MKV, WebM, FLV, WMV, M4V
                            - Maximum file size: 100MB
                            """)
                    
                    # Image input for video generation
                    image_upload = gr.Image(label="Image for Video (optional)", type="filepath")
                    gr.Markdown("""
                    **Video Generation:**
                    - Upload an image to create an MP4 video
                    - Video will use the uploaded image with generated audio
                    - Video encoded with H.264, CRF 17
                    """)
                    
                    
                    current_model_display = gr.Markdown("**Current Model:** Loading...")
                    
                    gr.Markdown("""
                   example
                    """)
                    
                    history_display = gr.HTML()
        
        with gr.TabItem("⚙️ Advanced Settings"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("⚙️ **Advanced Settings** - Optimized defaults for Large model. Modify carefully.")
                
                    # Primary controls
                    with gr.Row():
                        max_gen_length = gr.Slider(
                            label="Max Generation Length (Steps)", 
                            minimum=MIN_GENERATION_STEPS, 
                            maximum=6750,  # Max possible for Large/Base Full models
                            value=4500,  # Default to 3 minutes (good balance for Large model)
                            step=100,
                            info="Controls song length (~25 steps/sec). 4500 (≈3m) is the balanced default; extend to 5200-6000 for full lyric arcs; 6750 (≈4m30s) is the Large/Base max."
                        )
                        diffusion_steps = gr.Slider(
                            label="Diffusion Steps", 
                            minimum=20, 
                            maximum=200, 
                            value=50, 
                            step=10,
                            info="Detail polish. 30 = fast sketch, 50 = default, 70-90 = tighter mixes & stronger reference following, 120+ = ultra clean but ~2× slower."
                        )
                
                    # Sampling parameters
                    with gr.Row():
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.1,
                            maximum=2.0,
                            value=0.8,
                            step=0.1,
                            info="Variability. 0.6-0.8 keeps lyrics tight, 0.8 (default) balances melody & fidelity, 1.0+ invites wilder phrasing but may drift."
                        )
                        top_k = gr.Slider(
                            label="Top-k Sampling",
                            minimum=-1,
                            maximum=250,
                            value=-1,
                            step=1,
                            info="Restricts word choices. -1 keeps full vocab (default). Use 24-64 when you need stricter lyric phrasing or matched cadences; >100 reintroduces freer wording."
                        )
                        top_p = gr.Slider(
                            label="Top-p (Nucleus Sampling)",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.05,
                            info="Alternative variety control. Leave at 0 when relying on top-k; set 0.8-0.9 instead of top-k for smooth variation, 0.6-0.75 when prioritising lyric adherence."
                        )
                
                    # Guidance parameters
                    with gr.Row():
                        cfg_coef = gr.Slider(
                            label="CFG Coefficient",
                            minimum=0.1,
                            maximum=5.0,
                            value=1.5,
                            step=0.1,
                            info="How closely vocals follow your text. 1.5 default; raise to 1.8-2.2 when the model drifts from lyrics; >2.4 can sound brittle; drop to 1.2 for freer improv."
                        )
                        guidance_scale = gr.Slider(
                            label="Diffusion Guidance Scale",
                            minimum=0.5,
                            maximum=5.0,
                            value=1.5,
                            step=0.1,
                            info="Balances clarity vs creativity during diffusion. 1.5 default clarity; 1.8-2.1 locks tighter to reference audio & vocal tone; <1.3 softens for vibe; >2.5 may add hiss."
                        )

                
                with gr.Column():
                    gr.Markdown("### 🎮 VRAM Optimization")
                    gr.Markdown("**Large: 22-28GB | Base Full: 12-18GB**. Disable optimizations on high-VRAM GPUs for faster generation.")
                    with gr.Row():
                        disable_offload = gr.Checkbox(label="Disable Model Offloading", value=False, 
                                                    info="Keep models in VRAM. Enable on RTX 4090/A100 with 24GB+ VRAM for faster generation")
                        disable_cache_clear = gr.Checkbox(label="Disable Cache Clearing", value=False,
                                                        info="Skip CUDA cache clearing between steps. Enable on high-VRAM GPUs (saves ~1-2s per generation)")
                    with gr.Row():
                        disable_fp16 = gr.Checkbox(label="Disable Float16 Autocast", value=False,
                                                 info="Use full precision. Not recommended - Large model requires FP16 (may crash if enabled)")
                        disable_sequential = gr.Checkbox(label="Disable Sequential Loading", value=False,
                                                       info="Load all components simultaneously. Only for 32GB+ VRAM systems")
                    
                    gr.Markdown("### Advanced Generation Options")
                    # Advanced options
                    with gr.Row():
                        use_sampling = gr.Checkbox(
                            label="Use Sampling",
                            value=True,
                            info="Switches between stochastic sampling and greedy argmax inside `LmModel.generate`; leave on for musical variety, disable only for deterministic debugging."
                        )
                        extend_stride = gr.Slider(
                            label="Extend Stride",
                            minimum=1,
                            maximum=30,
                            value=5,
                            step=1,
                            info="Controls the diffusion crossfade overlap (seconds) used when stitching latent windows; higher values smooth transitions, lower values keep sharper segment changes."
                        )
                
                    # Processing options
                    with gr.Row():
                        chunked = gr.Checkbox(
                            label="Chunked Processing",
                            value=True,
                            info="Process VAE audio in chunks. Recommended for Large model (saves VRAM)"
                        )
                        chunk_size = gr.Slider(
                            label="Chunk Size",
                            minimum=64,
                            maximum=256,
                            value=128,
                            step=32,
                            info="Audio chunk size for VAE processing. 128 = optimal for Large model, lower = more memory efficient"
                        )
                
                    # Token recording options
                    with gr.Row():
                        record_tokens = gr.Checkbox(
                            label="Record Tokens",
                            value=True,
                            info="Stores the raw token stream from `LmModel.generate` for metadata/debug analysis; slightly increases VRAM/runtime usage."
                        )
                        record_window = gr.Slider(
                            label="Record Window",
                            minimum=10,
                            maximum=200,
                            value=50,
                            step=10,
                            info="When recording tokens, pass the last N tokens to the sampler (`record_token_pool[-record_window:]`) for repetition checks; higher values mean more context but more memory."
                        )
            
        with gr.TabItem("📁 Batch Processing"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🚀 Batch Processing with Large Model")
                    gr.Markdown("""
                    **Batch Processing:**
                    - Select a folder containing .txt files with prompts
                    - Each .txt file will generate a song (up to 4m30s with Large model)
                    - Output files use the same name as the .txt file
                    - Large model delivers consistent high quality across all batch generations
                    """)
                    with gr.Row():
                        batch_input_folder = gr.Textbox(
                            label="Input Folder Path",
                            placeholder="E:\\prompts\\folder",
                            info="Folder containing .txt files with lyrics"
                        )
                        batch_output_folder = gr.Textbox(
                            label="Output Folder Path",
                            placeholder="E:\\output\\folder",
                            info="Folder to save generated songs"
                        )
                    with gr.Row():
                        skip_existing = gr.Checkbox(
                            label="Skip existing files",
                            value=True,
                            info="Skip generation if output files already exist"
                        )
                    with gr.Row():
                        batch_process_btn = gr.Button("🚀 Start Batch Processing", variant="primary")
                    
                    # Batch status display
                    batch_status = gr.Markdown("", visible=False)
        
        with gr.TabItem("📖 Song Structure Tags"):
            gr.Markdown("## 🎵 Available Song Structure Tags")
            gr.Markdown("Use these tags in your lyrics to control the song structure. **Large model supports complex structures with multiple sections up to 4m30s!**")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### 🎤 Vocal Sections (REQUIRE lyrics):
                    - **[verse]** - Main story/narrative sections (~20-30 seconds)
                      - Use for verses with lyrics
                      - Can have 2-3 verses in a full song
                    
                    - **[chorus]** - Catchy, repeating hook section (~15-20 seconds)
                      - Use for the main hook/refrain with lyrics
                      - Can repeat 3-4 times in a full song
                    
                    - **[bridge]** - Contrasting section (~15-20 seconds)
                      - Use for a different perspective/emotional shift with lyrics
                      - Usually appears once
                    
                    ### 🎸 Instrumental Sections (NO lyrics):
                    
                    **Intro (Opening):**
                    - **[intro-short]** - ~5 seconds opening
                    - **[intro-medium]** - ~10-15 seconds opening (recommended)
                    - **[intro-long]** - ~20-25 seconds opening
                    
                    **Instrumental Breaks:**
                    - **[inst-short]** - ~5 seconds instrumental break
                    - **[inst-medium]** - ~10-15 seconds instrumental break
                    - **[inst-long]** - ~20-25 seconds instrumental break
                    - **[instrumental]** - Alias for instrumental section (works like `[inst-medium]`)
                    
                    **Outro (Ending):**
                    - **[outro-short]** - ~5 seconds ending
                    - **[outro-medium]** - ~10-15 seconds ending (recommended)
                    - **[outro-long]** - ~20-25 seconds ending
                    
                    **Special:**
                    - **[silence]** - ~2 seconds of silence
                    
                    💡 **Large Model Tip**: With 4m30s capacity, create full songs with:
                    `[intro-medium]` + 2-3 `[verse]` + 3-4 `[chorus]` + 1 `[bridge]` + `[inst-medium]` + `[outro-medium]`
                    """)
                
                with gr.Column():
                    gr.Markdown("""
                    ### 🎼 Example Song Structure:
                    ```
                    [intro-medium]
                    
                    [verse]
                    Walking down the street today
                    Sunshine lighting up my way
                    
                    [pre-chorus]
                    And I can feel it building up inside
                    
                    [chorus]
                    This is our moment, we're alive
                    Dancing through the day and night
                    
                    [verse]
                    Every step feels so right
                    Colors bursting into sight
                    
                    [chorus]
                    This is our moment, we're alive
                    Dancing through the day and night
                    
                    [bridge]
                    When the world gets heavy
                    We'll keep moving steady
                    
                    [chorus]
                    This is our moment, we're alive
                    Dancing through the day and night
                    
                    [outro-long]
                    ```
                    
                    ### 💡 Pro Tips:
                    - **Place empty lines between sections** for clarity
                    - **Tags are case-insensitive**: `[VERSE]` = `[verse]`
                    - **Vocal sections MUST have lyrics**: `[verse]`, `[chorus]`, `[bridge]`
                    - **Instrumental sections MUST NOT have lyrics**: All `[intro-*]`, `[inst-*]`, `[outro-*]`, `[silence]`
                    - Use consistent structure for professional results
                    - Instrumental breaks (`[inst-medium]`) work great between verses/chorus
                    
                    ### ⚠️ Important Rules:
                    - ✅ Valid: `[intro-medium]`, `[intro-short]`, `[intro-long]`
                    - ❌ Invalid: `[intro]` alone (must specify length)
                    - ✅ Valid: `[verse]` with lyrics, `[chorus]` with lyrics, `[bridge]` with lyrics
                    - ❌ Invalid: Tags not in the list above (e.g., `[pre-chorus]`, `[rap]`, `[drop]`)
                    
                    ### 🎵 Large Model Advantages:
                    - **4m30s maximum length (6750 steps)** - Create full-length songs
                    - **Best audio quality** - Superior vocal clarity and instrumental separation
                    - **Better coherence** - Maintains musical consistency across long generations
                    - **More sections** - Fit intro, multiple verses/choruses, bridge, instrumental breaks, and outro
                    - **Stable generation** - Reliable quality even at maximum length
                    """)
    
    # Add character counter and duration estimator for lyrics
    def update_char_count_and_duration(text, max_generation_steps):
        char_count = len(text) if text else 0

        # Get current model limits (default to Large model capacity if not loaded)
        current_model_max = MAX_GENERATION_LENGTHS.get(MODEL_VERSION, 6750) if MODEL_VERSION else 6750
        if current_model_max <= 0:
            current_model_max = 6750

        # Approximate character limit (rough estimate: ~6.5 chars per token)
        max_chars = int((current_model_max / 25.0) * 6.5 * 300 / 3750 * 1500)
        max_chars = min(max_chars, 5000)  # Cap at reasonable limit (increased for Large model)

        # Determine target generation steps (25 steps ≈ 1 second in pipeline)
        fallback_steps = min(4500, current_model_max)
        try:
            requested_steps = float(max_generation_steps) if max_generation_steps is not None else fallback_steps
        except (TypeError, ValueError):
            requested_steps = fallback_steps

        if requested_steps <= 0:
            requested_steps = fallback_steps

        requested_steps = max(requested_steps, 0.0)
        effective_steps = min(int(round(requested_steps)), int(current_model_max))

        # Convert steps to duration (actual pipeline uses steps / 25.0 seconds)
        if effective_steps > 0:
            duration_seconds = effective_steps / 25.0
            total_seconds = int(round(duration_seconds))
            minutes, seconds = divmod(total_seconds, 60)

            max_total_seconds = int(round(current_model_max / 25.0))
            max_minutes, max_seconds = divmod(max_total_seconds, 60)

            duration_str = (
                f" | Estimated duration: {minutes}:{seconds:02d} "
                f"({effective_steps} steps, max {max_minutes}:{max_seconds:02d})"
            )

            if requested_steps > current_model_max:
                duration_str += " ⚠️ capped by model"
        else:
            duration_str = " | Estimated duration: 0:00"

        # Character count warning based on current model
        if char_count > max_chars:
            return f"⚠️ {char_count}/{max_chars} characters (will be truncated){duration_str}"
        elif char_count > max_chars * 0.93:  # 93% of limit
            return f"⚠️ {char_count}/{max_chars} characters{duration_str}"
        else:
            return f"{char_count}/{max_chars} characters{duration_str}"

    def live_char_from_duration(duration_value, lyrics_value):
        """Provide responsive character counter updates while user drags duration slider."""
        max_steps = get_current_model_max_steps()
        max_steps = max(MIN_GENERATION_STEPS, max_steps)
        clamped_duration = clamp_duration_seconds(duration_value)
        clamped_duration = max(steps_to_seconds(MIN_GENERATION_STEPS), min(clamped_duration, steps_to_seconds(max_steps)))
        steps = clamp_steps_to_model(seconds_to_steps(clamped_duration))
        return update_char_count_and_duration(lyrics_value, steps)

    def live_char_from_steps(steps_value, lyrics_value):
        """Provide responsive character counter updates while user drags step slider."""
        max_steps = get_current_model_max_steps()
        max_steps = max(MIN_GENERATION_STEPS, max_steps)
        try:
            requested_steps = int(round(float(steps_value)))
        except (TypeError, ValueError):
            requested_steps = DEFAULT_GENERATION_STEPS
        clamped_steps = min(clamp_steps_to_model(requested_steps), max_steps)
        return update_char_count_and_duration(lyrics_value, clamped_steps)

    def sync_duration_slider(duration_value, lyrics_value):
        """Snap duration slider to valid range and sync max generation length when user releases."""
        max_steps = max(MIN_GENERATION_STEPS, get_current_model_max_steps())
        max_duration_seconds = int(round(steps_to_seconds(max_steps)))
        min_seconds = get_min_duration_seconds()
        if max_duration_seconds < min_seconds:
            max_duration_seconds = min_seconds

        clamped_duration = clamp_duration_seconds(duration_value)
        clamped_duration = max(min_seconds, min(clamped_duration, max_duration_seconds))
        clamped_steps = clamp_steps_to_model(seconds_to_steps(clamped_duration))
        clamped_steps = min(max_steps, max(MIN_GENERATION_STEPS, clamped_steps))

        clamped_duration = max(min_seconds, min(steps_to_seconds(clamped_steps), max_duration_seconds))
        char_text = update_char_count_and_duration(lyrics_value, clamped_steps)

        info_text = format_duration_slider_info(max_steps)
        duration_update = gr.update(
            value=int(round(clamped_duration)),
            minimum=DURATION_SLIDER_UI_MIN,
            maximum=max_duration_seconds,
            info=info_text
        )
        steps_update = gr.update(value=clamped_steps, maximum=max_steps)
        return duration_update, steps_update, char_text

    def sync_step_slider(steps_value, lyrics_value):
        """Snap max generation length slider and sync duration when user releases."""
        max_steps = max(MIN_GENERATION_STEPS, get_current_model_max_steps())
        max_duration_seconds = int(round(steps_to_seconds(max_steps)))
        min_seconds = get_min_duration_seconds()
        if max_duration_seconds < min_seconds:
            max_duration_seconds = min_seconds

        try:
            requested_steps = int(round(float(steps_value)))
        except (TypeError, ValueError):
            requested_steps = DEFAULT_GENERATION_STEPS

        clamped_steps = max(MIN_GENERATION_STEPS, min(requested_steps, max_steps))
        clamped_steps = clamp_steps_to_model(clamped_steps)
        clamped_steps = min(max_steps, max(MIN_GENERATION_STEPS, clamped_steps))
        clamped_duration = max(min_seconds, min(steps_to_seconds(clamped_steps), max_duration_seconds))

        char_text = update_char_count_and_duration(lyrics_value, clamped_steps)

        info_text = format_duration_slider_info(max_steps)
        steps_update = gr.update(value=clamped_steps, maximum=max_steps)
        duration_update = gr.update(
            value=int(round(clamped_duration)),
            minimum=DURATION_SLIDER_UI_MIN,
            maximum=max_duration_seconds,
            info=info_text
        )
        return steps_update, duration_update, char_text

    lyrics.change(
        fn=update_char_count_and_duration,
        inputs=[lyrics, max_gen_length],
        outputs=[char_counter]
    )

    max_gen_length.change(
        fn=live_char_from_steps,
        inputs=[max_gen_length, lyrics],
        outputs=[char_counter],
        js=f"(steps, lyrics) => [Math.max({MIN_GENERATION_STEPS}, steps ?? {MIN_GENERATION_STEPS}), lyrics]"
    )

    max_gen_length.release(
        fn=sync_step_slider,
        inputs=[max_gen_length, lyrics],
        outputs=[max_gen_length, duration_slider, char_counter],
        js=f"(steps, lyrics) => [Math.max({MIN_GENERATION_STEPS}, steps ?? {MIN_GENERATION_STEPS}), lyrics]"
    )

    duration_slider.change(
        fn=live_char_from_duration,
        inputs=[duration_slider, lyrics],
        outputs=[char_counter],
        js=f"(duration, lyrics) => [Math.max({MIN_DURATION_SECONDS}, duration ?? {MIN_DURATION_SECONDS}), lyrics]"
    )

    duration_slider.release(
        fn=sync_duration_slider,
        inputs=[duration_slider, lyrics],
        outputs=[duration_slider, max_gen_length, char_counter],
        js=f"(duration, lyrics) => [Math.max({MIN_DURATION_SECONDS}, duration ?? {MIN_DURATION_SECONDS}), lyrics]"
    )

    # Initialize character counter with default lyrics and generation length
    demo.load(
        fn=sync_step_slider,
        inputs=[max_gen_length, lyrics],
        outputs=[max_gen_length, duration_slider, char_counter]
    )

    demo.load(
        fn=update_description_preview,
        inputs=description_preview_inputs,
        outputs=[description_preview]
    )
    
    # Preset functionality
    def handle_save_preset(preset_name, current_preset, *args):
        """Save current settings as a preset"""
        # If no preset name entered, use the currently selected preset
        if not preset_name:
            if current_preset:
                preset_name = current_preset
                gr.Info(f"Overwriting preset: {preset_name}")
            else:
                gr.Info("Please enter a preset name or select an existing preset to overwrite")
                return gr.Dropdown(choices=preset_manager.get_preset_list())
        
        # Collect all parameters - args order matches all_inputs order
        # all_inputs = [lyrics, genre, instrument, bpm, emotion, timbre, gender, extra_prompt, include_dropdown_attributes, force_extra_prompt,
        #              audio_path, image_upload, save_mp3_check, seed_input,
        #              num_generations, loop_presets, randomize_params, ...]
        param_values = list(args)
        
        # Safety check for parameter count (updated for new parameters)
        if len(param_values) != 38:  # Updated count including duration slider and auto prompt parameters
            error_msg = f"Invalid parameter count: expected 38, got {len(param_values)}"
            print(f"❌ ERROR: {error_msg}")
            gr.Error(error_msg)
            return gr.Dropdown(choices=preset_manager.get_preset_list())
        
        # Create preset data matching the order
        preset_data = {
            'lyrics': param_values[0],
            'genre': param_values[1],
            'instrument': param_values[2],
            'bpm': param_values[3],
            'emotion': param_values[4],
            'timbre': param_values[5],
            'gender': param_values[6],
            'extra_prompt': param_values[7],
            'include_dropdown_attributes': param_values[8],
            'force_extra_prompt': param_values[9],
            'audio_path': param_values[10],
            'image_path': param_values[11],  # This is image_upload in all_inputs
            'save_mp3': param_values[12],
            'seed': param_values[13],
            'num_generations': param_values[14],
            'loop_presets': param_values[15],
            'randomize_params': param_values[16],
            'duration_seconds': param_values[17],
            'max_gen_length': param_values[18],
            'diffusion_steps': param_values[19],
            'temperature': param_values[20],
            'top_k': param_values[21],
            'top_p': param_values[22],
            'cfg_coef': param_values[23],
            'guidance_scale': param_values[24],
            'use_sampling': param_values[25],
            'extend_stride': param_values[26],
            'gen_type': param_values[27],
            'chunked': param_values[28],
            'chunk_size': param_values[29],
            'record_tokens': param_values[30],
            'record_window': param_values[31],
            'disable_offload': param_values[32],
            'disable_cache_clear': param_values[33],
            'disable_fp16': param_values[34],
            'disable_sequential': param_values[35],
            'auto_prompt_enabled': param_values[36],
            'auto_prompt_type': param_values[37]
        }
        # Save preset
        
        success, message = preset_manager.save_preset(preset_name, preset_data)
        
        if success:
            preset_manager.set_last_used_preset(preset_name)
            gr.Info(message)
            # Return updated dropdown with new preset selected
            return gr.Dropdown(choices=preset_manager.get_preset_list(), value=preset_name)
        else:
            print(f"❌ ERROR: {message}")
            gr.Error(message)
            return gr.Dropdown(choices=preset_manager.get_preset_list())
    
    def handle_load_preset(preset_name, current_lyrics):
        """Load a preset and update all UI components"""
        if not preset_name:
            return [gr.update()] * 38  # Updated to 38 for all parameters (including new ones)
        
        preset_data, message = preset_manager.load_preset(preset_name)
        if preset_data is None:
            print(f"❌ ERROR: {message}")
            gr.Error(message)
            return [gr.update()] * 38
        
        # Load preset values
        
        # Default values matching UI defaults - use current lyrics as default
        defaults = {
            'lyrics': current_lyrics if current_lyrics else EXAMPLE_LYRICS,
            'genre': 'None',
            'instrument': 'None',
            'bpm': 150,
            'emotion': 'None',
            'timbre': 'None',
            'gender': 'None',
            'extra_prompt': '',
            'include_dropdown_attributes': False,
            'force_extra_prompt': False,
            'audio_path': None,
            'image_path': None,
            'save_mp3': True,
            'seed': -1,
            'duration_seconds': DEFAULT_DURATION_SECONDS,
            'num_generations': 1,
            'loop_presets': False,
            'randomize_params': False,
            'max_gen_length': 4500,  # Default to 3 minutes for Large model
            'diffusion_steps': 50,
            'temperature': 0.8,
            'top_k': -1,
            'top_p': 0.0,
            'cfg_coef': 1.5,
            'guidance_scale': 1.5,
            'use_sampling': True,
            'extend_stride': 5,
            'gen_type': 'mixed',
            'chunked': True,
            'chunk_size': 128,
            'record_tokens': True,
            'record_window': 50,
            'disable_offload': False,
            'disable_cache_clear': False,
            'disable_fp16': False,
            'disable_sequential': False,
            'auto_prompt_enabled': False,
            'auto_prompt_type': 'Auto'
        }
        
        # Apply preset data with defaults for missing values
        values = preset_manager.apply_preset_to_ui(preset_data, defaults)
        
        if 'top_k' in values:
            try:
                values['top_k'] = int(values['top_k'])
            except (TypeError, ValueError):
                values['top_k'] = defaults['top_k']
        else:
            values['top_k'] = defaults['top_k']

        values['include_dropdown_attributes'] = bool(
            values.get('include_dropdown_attributes', defaults['include_dropdown_attributes'])
        )

        # Get current model's max generation length and clamp preset value
        current_max_length = 6750  # Default to Large model capacity
        if MODEL_VERSION and MODEL_VERSION in MAX_GENERATION_LENGTHS:
            current_max_length = MAX_GENERATION_LENGTHS[MODEL_VERSION]
        elif CURRENT_MODEL_PATH:
            model_name = op.basename(CURRENT_MODEL_PATH)
            current_max_length = MAX_GENERATION_LENGTHS.get(model_name, 6750)
        
        # Clamp max_gen_length to current model's maximum and align duration slider
        preset_max_gen = values.get('max_gen_length', defaults['max_gen_length'])
        preset_duration = values.get('duration_seconds')
        
        if preset_duration is not None:
            candidate_steps = clamp_steps_to_model(seconds_to_steps(preset_duration))
        else:
            candidate_steps = clamp_steps_to_model(preset_max_gen)
        
        clamped_max_gen = min(candidate_steps, current_max_length)
        clamped_max_gen = max(MIN_GENERATION_STEPS, clamped_max_gen)
        values['max_gen_length'] = clamped_max_gen
        values['duration_seconds'] = int(round(steps_to_seconds(clamped_max_gen)))
        
        # Validate auto_prompt_type against available choices
        preset_auto_prompt_type = values.get('auto_prompt_type', defaults['auto_prompt_type'])
        available_types = auto_prompt_manager.get_available_types() if auto_prompt_manager.is_available() else []
        if available_types and "Auto" not in available_types:
            available_types = available_types + ["Auto"]
        valid_choices = available_types if available_types else AUTO_PROMPT_TYPES
        validated_auto_prompt_type = preset_auto_prompt_type if preset_auto_prompt_type in valid_choices else "Auto"
        values['auto_prompt_type'] = validated_auto_prompt_type
        
        # Update last used preset
        preset_manager.set_last_used_preset(preset_name)
        gr.Info(message)
        
        duration_slider_max = int(round(steps_to_seconds(current_max_length)))
        duration_slider_min_seconds = get_min_duration_seconds()
        if duration_slider_max < duration_slider_min_seconds:
            duration_slider_min_seconds = duration_slider_max
        duration_value = int(round(steps_to_seconds(values['max_gen_length'])))
        duration_value = max(duration_slider_min_seconds, min(duration_value, duration_slider_max))
        duration_slider_info = format_duration_slider_info(current_max_length)

        # Return updates for all UI components, with max_gen_length clamped and slider max updated
        return [
            values['lyrics'],
            values['genre'],
            values['instrument'],
            values['bpm'],
            values['emotion'],
            values['timbre'],
            values['gender'],
            values['extra_prompt'],
            values['include_dropdown_attributes'],
            values['force_extra_prompt'],
            values['audio_path'],
            values['image_path'],
            values['save_mp3'],
            values['seed'],
            values['num_generations'],
            values['loop_presets'],
            values['randomize_params'],
            gr.update(
                value=duration_value,
                minimum=DURATION_SLIDER_UI_MIN,
                maximum=duration_slider_max,
                info=duration_slider_info
            ),
            gr.update(value=clamped_max_gen, maximum=current_max_length),  # Update slider with clamped value and max
            values['diffusion_steps'],
            values['temperature'],
            values['top_k'],
            values['top_p'],
            values['cfg_coef'],
            values['guidance_scale'],
            values['use_sampling'],
            values['extend_stride'],
            values['gen_type'],
            values['chunked'],
            values['chunk_size'],
            values['record_tokens'],
            values['record_window'],
            values['disable_offload'],
            values['disable_cache_clear'],
            values['disable_fp16'],
            values['disable_sequential'],
            values['auto_prompt_enabled'],
            validated_auto_prompt_type  # Use validated value
        ]
    
    def handle_refresh_presets():
        """Refresh the preset dropdown"""
        return gr.Dropdown(choices=preset_manager.get_preset_list())
    
    # Connect preset handlers
    all_inputs = [
        lyrics, genre, instrument, bpm, emotion, timbre, gender, extra_prompt, include_dropdown_attributes, force_extra_prompt,
        audio_path, image_upload, save_mp3_check, seed_input,
        num_generations, loop_presets, randomize_params, duration_slider, max_gen_length, diffusion_steps, temperature, top_k, top_p,
        cfg_coef, guidance_scale, use_sampling, extend_stride,
        gen_type, chunked, chunk_size, record_tokens, record_window,
        disable_offload, disable_cache_clear, disable_fp16, disable_sequential,
        auto_prompt_enabled, auto_prompt_type  # New parameters
    ]
    
    # Modified to return both dropdown and clear the input field
    def handle_save_and_clear(preset_name, current_preset, *args):
        dropdown_update = handle_save_preset(preset_name, current_preset, *args)
        # Return updates for both dropdown and input field
        return dropdown_update, gr.Textbox(value="")  # Clear the input field
    
    save_preset_btn.click(
        fn=handle_save_and_clear,
        inputs=[preset_name_input, preset_dropdown] + all_inputs,
        outputs=[preset_dropdown, preset_name_input]
    ).then(
        # Automatically load the preset after saving
        fn=handle_load_preset,
        inputs=[preset_dropdown, lyrics],
        outputs=all_inputs
    )
    
    load_preset_btn.click(
        fn=handle_load_preset,
        inputs=[preset_dropdown, lyrics],
        outputs=all_inputs
    )
    
    refresh_preset_btn.click(
        fn=handle_refresh_presets,
        inputs=[],
        outputs=[preset_dropdown]
    )
    
    # Auto-load preset when dropdown selection changes
    preset_dropdown.change(
        fn=handle_load_preset,
        inputs=[preset_dropdown, lyrics],
        outputs=all_inputs
    )
    
    # Handle file upload and component switching
    def process_file_upload(file_path):
        """Process uploaded file and determine component visibility"""
        if not file_path:
            return (
                gr.update(visible=True),   # file_upload
                gr.update(visible=False),  # audio_component
                "",                        # audio_path
                gr.update(visible=False)   # upload_status
            )
        
        # Get file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Define file types
        audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.opus', '.wma']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v']
        
        if file_ext in audio_extensions:
            # Audio file - show audio component with trim feature
            return (
                gr.update(visible=False),   # file_upload
                gr.update(visible=True, value=file_path),  # audio_component
                file_path,                  # audio_path
                gr.update(visible=True, value="✅ Audio file loaded. You can use the trim feature above.")  # upload_status
            )
        elif file_ext in video_extensions:
            # Video file - extract audio and show status
            print(f"Processing video file: {file_path}")
            extracted_path, error_msg = validate_audio_file(file_path)
            
            if error_msg:
                return (
                    gr.update(visible=True),   # file_upload
                    gr.update(visible=False),  # audio_component
                    "",                        # audio_path
                    gr.update(visible=True, value=f"❌ Error: {error_msg}")  # upload_status
                )
            else:
                # Show extracted audio in audio component
                return (
                    gr.update(visible=False),   # file_upload
                    gr.update(visible=True, value=extracted_path),  # audio_component
                    extracted_path,             # audio_path
                    gr.update(visible=True, value="✅ Audio extracted from video. You can use the trim feature above.")  # upload_status
                )
        else:
            return (
                gr.update(visible=True),   # file_upload
                gr.update(visible=False),  # audio_component
                "",                        # audio_path
                gr.update(visible=True, value="❌ Unsupported file format")  # upload_status
            )
    
    def update_audio_path(audio_value):
        """Update the hidden audio path when audio component changes"""
        return audio_value if audio_value else ""
    
    def reset_upload():
        """Reset the upload components"""
        return (
            gr.update(visible=True, value=None),   # file_upload
            gr.update(visible=False, value=None),  # audio_component
            "",                                    # audio_path
            gr.update(visible=False)               # upload_status
        )
    
    # Load last used preset on startup
    def load_initial_preset():
        """Load the last used preset on startup"""
        last_preset = preset_manager.get_last_used_preset()
        if last_preset and last_preset in preset_manager.get_preset_list():
            return gr.Dropdown(value=last_preset)
        return gr.Dropdown(value=None)
    
    demo.load(fn=load_initial_preset, inputs=[], outputs=[preset_dropdown])
    
    submit_btn.click(
        fn=submit_lyrics,
        inputs=[
            lyrics, struct, genre, instrument, bpm, emotion, timbre, gender, extra_prompt, include_dropdown_attributes, force_extra_prompt,
            audio_path, image_upload, save_mp3_check, seed_input,
            duration_slider, max_gen_length, diffusion_steps, temperature, top_k, top_p,
            cfg_coef, guidance_scale, use_sampling, extend_stride,
            gen_type, chunked, chunk_size, record_tokens, record_window,
            disable_offload, disable_cache_clear, disable_fp16, disable_sequential,
            num_generations, loop_presets, randomize_params,
            auto_prompt_enabled, auto_prompt_type,  # New parameters
            history, session
        ],
        outputs=[output_audio, output_video, history, history_display, cancel_btn, progress_text]
    )
    
    # Cancel button handler
    cancel_btn.click(
        fn=cancel_generation,
        inputs=[],
        outputs=[cancel_btn, progress_text]
    )
    
    # File upload handlers
    file_upload.change(
        fn=process_file_upload,
        inputs=[file_upload],
        outputs=[file_upload, audio_component, audio_path, upload_status]
    )
    
    # Update audio path when audio component changes (after trimming)
    audio_component.change(
        fn=update_audio_path,
        inputs=[audio_component],
        outputs=[audio_path]
    )
    
    # Add clear handler to audio component
    audio_component.clear(
        fn=reset_upload,
        inputs=[],
        outputs=[file_upload, audio_component, audio_path, upload_status]
    )
    
    # Batch processing button handler
    batch_process_btn.click(
        fn=run_batch_processing,
        inputs=[
            batch_input_folder, batch_output_folder, skip_existing, loop_presets, num_generations,
            lyrics, genre, instrument, bpm, emotion, timbre, gender, extra_prompt, include_dropdown_attributes, force_extra_prompt,
            audio_path, image_upload, save_mp3_check, seed_input,
            duration_slider, max_gen_length, diffusion_steps, temperature, top_k, top_p,
            cfg_coef, guidance_scale, use_sampling, extend_stride,
            gen_type, chunked, chunk_size, record_tokens, record_window,
            disable_offload, disable_cache_clear, disable_fp16, disable_sequential,
            randomize_params, auto_prompt_enabled, auto_prompt_type  # Add new parameters
        ],
        outputs=[cancel_btn, progress_text, batch_status]
    )
    
    open_folder_btn.click(
        fn=open_output_folder,
        inputs=[],
        outputs=[]
    )
    
    # Model loading handlers
    def handle_model_selection_change(selected_model):
        """Handle model dropdown selection change - shows info only"""
        if not selected_model or selected_model not in model_paths:
            return "Select a model to see details"
        
        model_path = model_paths[selected_model]
        return get_model_info(model_path)
    
    def handle_model_selection_change_with_auto_load(selected_model):
        """Handle model dropdown selection change with auto-loading"""
        if not selected_model or selected_model not in model_paths:
            return "Select a model to see details", "**Status**: No model selected", gr.update(), gr.update()
        
        model_path = model_paths[selected_model]
        info = get_model_info(model_path)
        
        # Check if this model is already loaded
        if CURRENT_MODEL_PATH == model_path and MODEL is not None:
            model_name = op.basename(model_path)
            status = f"**Status**: ✅ {model_name} already loaded"
            max_length = MAX_GENERATION_LENGTHS.get(model_name, 6750)
            
            # Set appropriate default value
            # Both supported models have 6750 max, default to 4500 (3 minutes)
            default_value = 4500
                
            max_gen_update = gr.update(
                maximum=max_length,
                value=default_value,
                info=f"Target generation steps. Model {model_name} supports up to {max_length} steps (~{int(max_length/25)}s or {int(max_length/25/60)}m{int(max_length/25%60)}s)"
            )
            duration_update = create_duration_slider_update(default_value, max_length)
            return info, status, max_gen_update, duration_update
        
        # Auto-load the selected model
        print(f"🔄 Auto-loading selected model: {op.basename(model_path)}")
        success, message = load_model(model_path)
        
        if success:
            model_name = op.basename(model_path)
            status = f"**Status**: ✅ {model_name} loaded successfully"
            max_length = MAX_GENERATION_LENGTHS.get(model_name, 6750)
            
            # Set appropriate default value
            # Both supported models have 6750 max, default to 4500 (3 minutes)
            default_value = 4500
                
            max_gen_update = gr.update(
                maximum=max_length,
                value=default_value,
                info=f"Target generation steps. Model {model_name} supports up to {max_length} steps (~{int(max_length/25)}s or {int(max_length/25/60)}m{int(max_length/25%60)}s)"
            )
            duration_update = create_duration_slider_update(default_value, max_length)
            return info, status, max_gen_update, duration_update
        else:
            status = f"**Status**: ❌ Failed to load model"
            return info, status, gr.update(), gr.update()
    
    def handle_load_model(selected_model, progress=gr.Progress()):
        """Handle model loading with progress tracking"""
        # If no models available, try to refresh the list
        if not model_choices:
            return (
                "❌ No models found. Please refresh the list.",
                "**Status**: No models available",
                gr.update(),
                gr.update()
            )
        
        if not selected_model or selected_model not in model_paths:
            return "❌ Please select a valid model", "**Status**: No model selected", gr.update(), gr.update()
        
        model_path = model_paths[selected_model]
        
        def progress_callback(progress_val, message):
            progress(progress_val, desc=message)
        
        success, message = load_model(model_path, progress_callback)
        
        if success:
            status = f"**Status**: ✅ {op.basename(model_path)} loaded successfully"
            info = get_model_info(model_path)

            # Update max generation length based on loaded model
            model_name = op.basename(model_path)
            max_length = MAX_GENERATION_LENGTHS.get(model_name, 6750)
            # Set a reasonable default value based on model capacity
            # For large/base_full models (6750): default to 4500 (3 minutes)
            # For base models (3750): default to 3000 (2 minutes)
            if max_length >= 6750:
                default_value = 4500  # 3 minutes for large models
            else:
                default_value = 3000  # 2 minutes for base models
            
            max_gen_update = gr.update(
                maximum=max_length,
                value=default_value,
                info=f"Target generation steps. Model {model_name} supports up to {max_length} steps (~{int(max_length/25)}s or {int(max_length/25/60)}m{int(max_length/25%60)}s)"
            )
            duration_update = create_duration_slider_update(default_value, max_length)
            
            return info, status, max_gen_update, duration_update
        else:
            status = f"**Status**: ❌ Failed to load model"
            return message, status, gr.update(), gr.update()
    
    def handle_refresh_models():
        """Refresh the model list"""
        global model_choices, model_paths
        
        # Refresh available models
        available_models = get_available_models()
        model_choices = [f"{desc} ({model_dir})" for model_dir, desc, _ in available_models]
        model_paths = {f"{desc} ({model_dir})": path for model_dir, desc, path in available_models}
        
        if model_choices:
            info = "✅ Models found! Select a model and click 'Load Selected Model'"
            status = "**Status**: Models available"
            dropdown_update = gr.update(choices=model_choices, value=model_choices[0])
        else:
            info = """
            ❌ **No models found!**
            
            Please download models first using:
            ```bash
            python Download_Song_Models.py --model model_large
            ```
            or
            ```bash  
            python Download_Song_Models.py --model model_base
            ```
            """
            status = "**Status**: No models available"
            dropdown_update = gr.update(choices=[], value=None)
        
        return info, status, gr.update(), gr.update(), dropdown_update
    
    # Connect model selection handlers - auto-load when model is selected
    model_dropdown.change(
        fn=handle_model_selection_change_with_auto_load,
        inputs=[model_dropdown],
        outputs=[model_info_display, model_status, max_gen_length, duration_slider]
    )
    
    if model_choices:
        # Normal model loading
        load_model_btn.click(
            fn=handle_load_model,
            inputs=[model_dropdown],
            outputs=[model_info_display, model_status, max_gen_length, duration_slider]
        )
    else:
        # Refresh model list when no models available
        load_model_btn.click(
            fn=handle_refresh_models,
            inputs=[],
            outputs=[model_info_display, model_status, max_gen_length, duration_slider, model_dropdown]
        )
    
    # Auto-load first available model on startup
    def auto_load_first_model():
        """Auto-load the first available model on startup"""
        if model_choices:
            first_model = model_choices[0]
            model_path = model_paths[first_model]
            
            print(f"🔄 Auto-loading first available model: {op.basename(model_path)}")
            success, message = load_model(model_path)
            
            if success:
                status = f"**Status**: ✅ {op.basename(model_path)} auto-loaded"
                info = get_model_info(model_path)
                
                # Update max generation length
                model_name = op.basename(model_path)
                max_length = MAX_GENERATION_LENGTHS.get(model_name, 6750)
                
                # Set appropriate default value
                if max_length >= 6750:
                    default_value = 4500  # 3 minutes for large models
                else:
                    default_value = 3000  # 2 minutes for base models
                    
                max_gen_update = gr.update(
                    maximum=max_length,
                    value=default_value,
                    info=f"Target generation steps. Model {model_name} supports up to {max_length} steps (~{int(max_length/25)}s or {int(max_length/25/60)}m{int(max_length/25%60)}s)"
                )
                duration_update = create_duration_slider_update(default_value, max_length)
                
                return info, status, max_gen_update, duration_update
            else:
                status = f"**Status**: ⚠️ Auto-load failed - please select and load manually"
                return "❌ Auto-load failed. Please select and load a model manually.", status, gr.update(), gr.update()
        else:
            return "❌ No models found. Please download models first.", "**Status**: No models available", gr.update(), gr.update()
    
    demo.load(
        fn=auto_load_first_model,
        inputs=[],
        outputs=[model_info_display, model_status, max_gen_length, duration_slider]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, share=args.share)
