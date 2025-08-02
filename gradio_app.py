import sys
import gradio as gr
import json
from datetime import datetime
import yaml
import time
import re
import os.path as op
import os
import scipy.io.wavfile as wavfile
import subprocess
import platform
from PIL import Image
import numpy as np
import torch
import random

# Add the tools/gradio directory to the path
sys.path.append(op.join(op.dirname(op.abspath(__file__)), 'tools', 'gradio'))
from levo_inference_lowmem import LeVoInference

EXAMPLE_LYRICS = """
[intro-medium]

[verse]
So close, no matter how far
Couldn't be much more from the heart
Forever trusting who we are
And nothing else matters

[verse]
Never opened myself this way
Life is ours, we live it our way
All these words, I don't just say
And nothing else matters

[chorus]
Trust I seek and I find in you
Every day for us something new
Open mind for a different view
And nothing else matters

[bridge]
Never cared for what they do
Never cared for what they know
But I know

[verse]
So close, no matter how far
It couldn't be much more from the heart
Forever trusting who we are
And nothing else matters

[outro-long]
""".strip()

APP_DIR = op.dirname(op.abspath(__file__))

# Default checkpoint path - this should point to the directory with config.yaml and model.pt
DEFAULT_CKPT = op.join(APP_DIR, 'ckpt', 'songgeneration_base')

# Use command line argument if provided, otherwise use default
ckpt_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT

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

MODEL = LeVoInference(ckpt_path)

# Load description options from text files
def load_options(filename):
    filepath = op.join(APP_DIR, 'sample', 'description', filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

GENRES = load_options('genre.txt')
INSTRUMENTS = load_options('instrument.txt')
EMOTIONS = load_options('emotion.txt')
TIMBRES = load_options('timbre.txt')
GENDERS = load_options('gender.txt')

# Preset directory
PRESET_DIR = op.join(APP_DIR, 'presets')
os.makedirs(PRESET_DIR, exist_ok=True)

def get_preset_list():
    """Get list of available presets"""
    if not os.path.exists(PRESET_DIR):
        return []
    presets = [f[:-5] for f in os.listdir(PRESET_DIR) if f.endswith('.json')]
    return sorted(presets)

def save_preset(preset_name, preset_data):
    """Save preset to file"""
    if not preset_name:
        return False, "Please enter a preset name"
    
    preset_path = op.join(PRESET_DIR, f"{preset_name}.json")
    try:
        with open(preset_path, 'w', encoding='utf-8') as f:
            json.dump(preset_data, f, indent=2, ensure_ascii=False)
        return True, f"Preset '{preset_name}' saved successfully"
    except Exception as e:
        return False, f"Error saving preset: {str(e)}"

def load_preset(preset_name):
    """Load preset from file"""
    if not preset_name:
        return None, "No preset selected"
    
    preset_path = op.join(PRESET_DIR, f"{preset_name}.json")
    if not os.path.exists(preset_path):
        return None, f"Preset '{preset_name}' not found"
    
    try:
        with open(preset_path, 'r', encoding='utf-8') as f:
            preset_data = json.load(f)
        return preset_data, f"Preset '{preset_name}' loaded successfully"
    except Exception as e:
        return None, f"Error loading preset: {str(e)}"

def get_last_used_preset():
    """Get the name of the last used preset"""
    last_preset_path = op.join(PRESET_DIR, '_last_used.txt')
    if os.path.exists(last_preset_path):
        try:
            with open(last_preset_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except:
            pass
    return None

def set_last_used_preset(preset_name):
    """Save the name of the last used preset"""
    last_preset_path = op.join(PRESET_DIR, '_last_used.txt')
    try:
        with open(last_preset_path, 'w', encoding='utf-8') as f:
            f.write(preset_name)
    except:
        pass

def save_metadata(file_path, metadata):
    """Save metadata to text file alongside audio file"""
    metadata_path = file_path.rsplit('.', 1)[0] + '.txt'
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write("=== SONG GENERATION METADATA ===\n")
            f.write(f"Generated: {metadata.get('timestamp', 'Unknown')}\n")
            f.write(f"Model: {metadata.get('model', 'Unknown')}\n\n")
            
            f.write("=== USER INPUTS ===\n")
            f.write(f"Lyrics:\n{metadata.get('lyrics', '')}\n\n")
            
            f.write("=== GENERATION PARAMETERS ===\n")
            for key, value in sorted(metadata.items()):
                if key not in ['timestamp', 'model', 'lyrics']:
                    f.write(f"{key}: {value}\n")
            
            f.write("\n=== END OF METADATA ===\n")
        return True
    except Exception as e:
        print(f"Error saving metadata: {e}")
        return False

def collect_current_parameters(
    lyrics, genre, instrument, emotion, timbre, gender,
    sample_prompt, audio_path, image_path, save_mp3, seed,
    max_gen_length, diffusion_steps, temperature, top_k, top_p,
    cfg_coef, guidance_scale, use_sampling, extend_stride,
    gen_type, chunked, chunk_size, record_tokens, record_window,
    disable_offload, disable_cache_clear, disable_fp16, disable_sequential
):
    """Collect all current parameters into a dictionary"""
    return {
        'lyrics': lyrics,
        'genre': genre,
        'instrument': instrument,
        'emotion': emotion,
        'timbre': timbre,
        'gender': gender,
        'sample_prompt': sample_prompt,
        'audio_path': audio_path,
        'image_path': image_path,
        'save_mp3': save_mp3,
        'seed': seed,
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
        'disable_sequential': disable_sequential
    }

def apply_preset_to_ui(preset_data):
    """Convert preset data to UI values, handling missing keys gracefully"""
    # Default values matching UI defaults
    defaults = {
        'lyrics': EXAMPLE_LYRICS,
        'genre': 'electronic',
        'instrument': 'synthesizer and drums',
        'emotion': 'uplifting',
        'timbre': 'bright',
        'gender': 'male',
        'sample_prompt': False,
        'audio_path': None,
        'image_path': None,
        'save_mp3': True,
        'seed': -1,
        'max_gen_length': 3750,
        'diffusion_steps': 50,
        'temperature': 1.0,
        'top_k': 250,
        'top_p': 0.0,
        'cfg_coef': 3.0,
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
        'disable_sequential': False
    }
    
    # Merge preset data with defaults
    if preset_data:
        for key, value in preset_data.items():
            defaults[key] = value
    
    return defaults


def output_messages(msg):
    gr.Info(msg)
    print(msg)

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
            num = int(f.split('.')[0])
            numbers.append(num)
        except:
            continue
    
    return max(numbers) + 1 if numbers else 1

def create_video_from_image_and_audio(image_path, audio_path, output_path, duration=None):
    """Create an MP4 video from an image and audio using ffmpeg"""
    try:
        # Get audio duration if not provided
        if duration is None:
            cmd = [
                'ffmpeg', '-i', audio_path, '-hide_banner', '-loglevel', 'error',
                '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=p=0'
            ]
            result = subprocess.run(['ffprobe'] + cmd[1:], capture_output=True, text=True)
            duration = float(result.stdout.strip())
        
        # Create video with image and audio
        cmd = [
            'ffmpeg', '-y',
            '-loop', '1',
            '-i', image_path,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-crf', '17',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            '-movflags', '+faststart',
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"Error creating video: {e}")
        return False

def convert_wav_to_mp3(wav_path, mp3_path, bitrate='192k'):
    """Convert WAV to MP3 using ffmpeg"""
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', wav_path,
            '-c:a', 'libmp3lame',
            '-b:a', bitrate,
            mp3_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"Error converting to MP3: {e}")
        return False

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

def set_seed(seed):
    """Set random seeds for reproducibility"""
    if seed is not None and seed >= 0:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        # For CUDA determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return True
    return False

def submit_lyrics(
    lyrics, struct, genre, instrument, emotion, timbre, gender,
    sample_prompt, audio_path, image_path, save_mp3, seed,
    max_gen_length, diffusion_steps, temperature, top_k, top_p,
    cfg_coef, guidance_scale, use_sampling, extend_stride,
    gen_type, chunked, chunk_size, record_tokens, record_window,
    disable_offload, disable_cache_clear, disable_fp16, disable_sequential,
    history, session
):
    # Limit lyrics length to prevent exceeding token limit
    # Approximate: ~6.5 characters per token, max 300 tokens, safe limit = 1000 characters
    MAX_CHARS = 1000
    if len(lyrics) > MAX_CHARS:
        lyrics = lyrics[:MAX_CHARS]
        output_messages(f"Lyrics truncated to {MAX_CHARS} characters to fit token limit")
    
    print(f"Lyrics length: {len(lyrics)} characters")
    print(struct)
    
    # Set seed if provided
    used_seed = seed
    if seed is None or seed < 0:
        # Generate a random seed
        used_seed = random.randint(0, 2147483647)
    
    set_seed(used_seed)
    output_messages(f"Using seed: {used_seed}")
    
    # Format lyrics according to the model's expected format
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
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    song_data = {
        "lyrics": formatted_lyrics,
        "struct": process_struct(struct),
        "genre": genre,
        "instrument": instrument, 
        "emotion": emotion,
        "timbre": timbre,
        "gender": gender,
        "sample_prompt": sample_prompt,
        "audio_path": audio_path,
        "time": current_time
    }
    
    # Call the model using the forward method (or __call__)
    # The model expects: lyric, description, prompt_audio_path, genre, auto_prompt_path, gen_type, params
    # Pass generation parameters - only include valid set_generation_params
    # Account for pattern delays (~250 steps) when converting to duration
    # The pattern adds delays, so we need to subtract them from the requested length
    pattern_delay_offset = 250  # Approximate delay from codebook pattern
    actual_steps = max(max_gen_length - pattern_delay_offset, 1000)
    # Convert to duration (assuming frame_rate of 25)
    duration_from_steps = actual_steps / 25.0
    # Ensure duration doesn't exceed model's max_duration (150 seconds)
    duration_from_steps = min(duration_from_steps, 150.0)
    
    # Log the actual values for debugging
    output_messages(f"Requested steps: {max_gen_length}, Actual generation: ~{int(duration_from_steps * 25) + pattern_delay_offset} steps")
    
    gen_params = {
        'duration': duration_from_steps,  # Convert steps to seconds
        'num_steps': diffusion_steps,     # This will be extracted in the inference code
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p,
        'cfg_coef': cfg_coef,
        'guidance_scale': guidance_scale,  # This will be extracted for audio diffusion
        'use_sampling': use_sampling,
        'extend_stride': extend_stride,
        'chunked': chunked,               # This will be extracted for audio generation
        'chunk_size': chunk_size,         # This will be extracted for audio generation
        'record_tokens': record_tokens,
        'record_window': record_window,
    }
    
    audio_data = MODEL(
        song_data["lyrics"], 
        f"{song_data['gender']}, {song_data['timbre']}, {song_data['genre']}, {song_data['emotion']}, {song_data['instrument']}",
        song_data["audio_path"] if song_data["sample_prompt"] else None,
        None,  # genre parameter (None since we're using description)
        op.join(APP_DIR, "ckpt/prompt.pt"),  # auto_prompt_path
        gen_type,  # gen_type from UI
        gen_params,  # params
        disable_offload=disable_offload,
        disable_cache_clear=disable_cache_clear,
        disable_fp16=disable_fp16,
        disable_sequential=disable_sequential
    ).cpu().permute(1, 0).float().numpy()
    
    # Save the audio with sequential numbering
    output_dir = op.join(APP_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    file_number = get_next_file_number(output_dir)
    base_filename = f"{file_number:04d}"
    
    wav_path = op.join(output_dir, f"{base_filename}.wav")
    wavfile.write(wav_path, MODEL.cfg.sample_rate, audio_data)
    output_messages(f"Generated audio saved to: {wav_path}")
    
    # Convert to MP3 if requested
    mp3_path = None
    if save_mp3:
        mp3_path = op.join(output_dir, f"{base_filename}.mp3")
        if convert_wav_to_mp3(wav_path, mp3_path, '192k'):
            output_messages(f"Generated MP3 saved to: {mp3_path}")
        else:
            mp3_path = None
            output_messages("Failed to convert to MP3")
    
    # Create video if image is provided
    video_path = None
    if image_path:
        video_path = op.join(output_dir, f"{base_filename}.mp4")
        if create_video_from_image_and_audio(image_path, wav_path, video_path):
            output_messages(f"Generated video saved to: {video_path}")
        else:
            video_path = None
            output_messages("Failed to create video")
    
    song_data["audio"] = wav_path
    song_data["mp3"] = mp3_path
    song_data["video"] = video_path
    
    # Save metadata
    metadata = collect_current_parameters(
        lyrics, genre, instrument, emotion, timbre, gender,
        sample_prompt, audio_path, image_path, save_mp3, used_seed,  # Note: using used_seed instead of seed
        max_gen_length, diffusion_steps, temperature, top_k, top_p,
        cfg_coef, guidance_scale, use_sampling, extend_stride,
        gen_type, chunked, chunk_size, record_tokens, record_window,
        disable_offload, disable_cache_clear, disable_fp16, disable_sequential
    )
    metadata['timestamp'] = current_time
    metadata['model'] = ckpt_path
    metadata['output_files'] = {
        'wav': wav_path,
        'mp3': mp3_path,
        'mp4': video_path
    }
    
    save_metadata(wav_path, metadata)
    
    history.append({"role": "user", "content": f"Generate a song with lyrics: {lyrics[:50]}..."})
    history.append({"role": "assistant", "content": f"Generated song successfully!", "song": song_data})
    
    return wav_path, video_path, history, process_history(history)

# Create Gradio interface
with gr.Blocks(title="LeVo Song Generation") as demo:
    gr.Markdown("# LeVo Song Generation")
    
    history = gr.State([])
    session = gr.State({})
    
    with gr.Row():
        with gr.Column(scale=2):
            lyrics = gr.Textbox(
                label="Lyrics",
                placeholder="Enter your lyrics here...",
                value=EXAMPLE_LYRICS,
                lines=15,
                max_lines=20,
                info="Maximum 1000 characters to stay within token limit"
            )
            
            # Character counter and duration display
            char_counter = gr.Markdown("0/1000 characters | Estimated duration: 0:00")
            
            # Preset controls
            with gr.Accordion("Presets", open=True):
                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        label="Select Preset",
                        choices=get_preset_list(),
                        value=None,
                        interactive=True
                    )
                    load_preset_btn = gr.Button("Load Preset", variant="secondary")
                    refresh_preset_btn = gr.Button("🔄", variant="secondary", scale=0)
                
                with gr.Row():
                    preset_name_input = gr.Textbox(
                        label="Preset Name",
                        placeholder="Enter preset name to save...",
                        scale=3
                    )
                    save_preset_btn = gr.Button("Save Preset", variant="secondary", scale=1)
            
            with gr.Row():
                submit_btn = gr.Button("Generate Song", variant="primary")
                open_folder_btn = gr.Button("Open Output Folder", variant="secondary")
            
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
                genre = gr.Dropdown(
                    label="Genre",
                    choices=GENRES,
                    value="electronic"
                )
                instrument = gr.Dropdown(
                    label="Instrument",
                    choices=INSTRUMENTS,
                    value="synthesizer and drums"
                )
            
            with gr.Row():
                emotion = gr.Dropdown(
                    label="Emotion",
                    choices=EMOTIONS,
                    value="uplifting"
                )
                timbre = gr.Dropdown(
                    label="Timbre",
                    choices=TIMBRES,
                    value="bright"
                )
                gender = gr.Dropdown(
                    label="Gender",
                    choices=GENDERS,
                    value="male"
                )
            
            with gr.Row():
                save_mp3_check = gr.Checkbox(label="Also save as MP3 (192 kbps)", value=True)
                seed_input = gr.Number(label="Seed (for reproducibility)", value=-1, precision=0, 
                                     info="Use -1 for random, or any positive number for reproducible results")
            
            # Advanced generation settings
            with gr.Accordion("Advanced Generation Settings", open=False):
                gr.Markdown("⚠️ **Warning:** Modifying these values affects generation quality and speed")
                
                # Primary controls
                with gr.Row():
                    max_gen_length = gr.Slider(
                        label="Max Generation Length (Target)", 
                        minimum=2000, 
                        maximum=7500, 
                        value=3750, 
                        step=100,
                        info="Target generation steps. Currently hard-limited to ~4000 steps (150s) by model"
                    )
                    diffusion_steps = gr.Slider(
                        label="Diffusion Steps", 
                        minimum=20, 
                        maximum=200, 
                        value=50, 
                        step=10,
                        info="Number of denoising steps for audio generation. Higher = better quality but slower"
                    )
                
                # Sampling parameters
                with gr.Row():
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        info="Controls randomness. Lower = more focused, Higher = more creative"
                    )
                    top_k = gr.Slider(
                        label="Top-k",
                        minimum=0,
                        maximum=500,
                        value=250,
                        step=10,
                        info="Limits sampling to top k tokens. 0 = disabled"
                    )
                    top_p = gr.Slider(
                        label="Top-p (Nucleus Sampling)",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.05,
                        info="Cumulative probability cutoff. 0 = use top-k instead"
                    )
                
                # Guidance parameters
                with gr.Row():
                    cfg_coef = gr.Slider(
                        label="CFG Coefficient",
                        minimum=1.0,
                        maximum=10.0,
                        value=3.0,
                        step=0.5,
                        info="Classifier-free guidance strength. Higher = stronger conditioning"
                    )
                    guidance_scale = gr.Slider(
                        label="Diffusion Guidance Scale",
                        minimum=0.5,
                        maximum=3.0,
                        value=1.5,
                        step=0.1,
                        info="Audio diffusion guidance. Higher = stronger prompt adherence"
                    )
                
                # Advanced options
                with gr.Row():
                    use_sampling = gr.Checkbox(
                        label="Use Sampling",
                        value=True,
                        info="Enable probabilistic sampling (recommended)"
                    )
                    extend_stride = gr.Slider(
                        label="Extend Stride",
                        minimum=1,
                        maximum=30,
                        value=5,
                        step=1,
                        info="Stride for extended generation (not currently used)"
                    )
                
                # Generation type and processing options
                with gr.Row():
                    gen_type = gr.Radio(
                        label="Generation Type",
                        choices=["mixed", "vocal", "bgm"],
                        value="mixed",
                        info="Generate vocals+BGM (mixed), vocals only, or BGM only"
                    )
                    chunked = gr.Checkbox(
                        label="Chunked Processing",
                        value=True,
                        info="Process audio in chunks to save memory"
                    )
                    chunk_size = gr.Slider(
                        label="Chunk Size",
                        minimum=64,
                        maximum=256,
                        value=128,
                        step=32,
                        info="Size of audio chunks for processing"
                    )
                
                # Token recording options
                with gr.Row():
                    record_tokens = gr.Checkbox(
                        label="Record Tokens",
                        value=True,
                        info="Record generation tokens for analysis"
                    )
                    record_window = gr.Slider(
                        label="Record Window",
                        minimum=10,
                        maximum=200,
                        value=50,
                        step=10,
                        info="Number of tokens to record"
                    )
                gr.Markdown("""
                **Generation Length:** 
                - 2000 steps ≈ 80 seconds  
                - 2500 steps ≈ 100 seconds
                - 3000 steps ≈ 120 seconds
                - 3750+ steps → ~4000 steps (150s hard limit)
                
                ⚠️ **Model Limitation**: Regardless of slider value, generation is 
                hard-capped at ~4000 steps (150 seconds). Extended generation 
                beyond this limit is not currently implemented.
                
                **Diffusion Steps:** Controls audio quality vs speed trade-off
                - 20 steps: Fastest, lower quality
                - 50 steps: Balanced (default)
                - 100+ steps: Best quality, much slower
                """)
            
            # VRAM optimization controls
            with gr.Accordion("VRAM Optimization Settings", open=False):
                gr.Markdown("Disable these optimizations for faster generation on high-VRAM GPUs (24 GB GPUs can disable all)")
                with gr.Row():
                    disable_offload = gr.Checkbox(label="Disable Model Offloading", value=False, 
                                                info="Keep models in VRAM instead of offloading to CPU")
                    disable_cache_clear = gr.Checkbox(label="Disable Cache Clearing", value=False,
                                                    info="Don't clear CUDA cache between steps")
                with gr.Row():
                    disable_fp16 = gr.Checkbox(label="Disable Float16 Autocast", value=False,
                                             info="Disable automatic mixed precision (may cause errors)")
                    disable_sequential = gr.Checkbox(label="Disable Sequential Loading", value=False,
                                                   info="Keep all models in memory simultaneously")
            
            with gr.Row():
                image_upload = gr.Image(label="Image for Video (optional)", type="filepath")
                gr.Markdown("""
                **Video Generation:**
                - Upload an image to create an MP4 video
                - Video will use the uploaded image with generated audio
                - Video encoded with H.264, CRF 17
                """)
            
            # Reference audio section
            with gr.Accordion("Reference Audio (Advanced)", open=False):
                sample_prompt = gr.Checkbox(label="Use Sample Prompt", value=False)
                audio_path = gr.Audio(label="Reference Audio (optional)", type="filepath")
            
        
        with gr.Column(scale=1):
            output_audio = gr.Audio(label="Generated Audio", type="filepath")
            output_video = gr.Video(label="Generated Video", visible=True)
            
            gr.Markdown("---")
            gr.Markdown("Generate high-quality songs with both vocals and accompaniment using AI")
            gr.Markdown(f"**Model checkpoint:** {ckpt_path}")
            
            gr.Markdown("""
            ### Instructions:
            1. Enter your lyrics with structure tags
            2. Select musical preferences (genre, emotion, instruments, etc.)
            3. Click "Generate Song" to create your music!
            
            ### Supported Structure Tags:
            
            **Sections that require lyrics:**
            - `[verse]` - Main verses of the song
            - `[chorus]` - Chorus/hook sections
            - `[bridge]` - Bridge sections
            
            **Instrumental sections (no lyrics needed):**
            - `[intro-short]` - Short intro (0-10 seconds)
            - `[intro-medium]` - Medium intro (10-20 seconds)
            - `[intro-long]` - Long intro (20+ seconds)
            - `[inst-short]` - Short instrumental (0-10 seconds)
            - `[inst-medium]` - Medium instrumental (10-20 seconds)
            - `[inst-long]` - Long instrumental (20+ seconds)
            - `[outro-short]` - Short outro (0-10 seconds)
            - `[outro-medium]` - Medium outro (10-20 seconds)
            - `[outro-long]` - Long outro (20+ seconds)
            - `[silence]` - Silent section
            """)
            
            history_display = gr.HTML()
    
    # Add character counter and duration estimator for lyrics
    def update_char_count_and_duration(text):
        char_count = len(text) if text else 0
        
        # Estimate duration based on structure tags
        duration_seconds = 0
        if text:
            text_lower = text.lower()
            # Count instrumental sections
            duration_seconds += text_lower.count('[intro-short]') * 5
            duration_seconds += text_lower.count('[intro-medium]') * 15
            duration_seconds += text_lower.count('[intro-long]') * 25
            duration_seconds += text_lower.count('[inst-short]') * 5
            duration_seconds += text_lower.count('[inst-medium]') * 15
            duration_seconds += text_lower.count('[inst-long]') * 25
            duration_seconds += text_lower.count('[outro-short]') * 5
            duration_seconds += text_lower.count('[outro-medium]') * 15
            duration_seconds += text_lower.count('[outro-long]') * 25
            duration_seconds += text_lower.count('[silence]') * 2
            
            # Estimate duration for sections with lyrics (verse, chorus, bridge)
            # Count lines for each section type
            import re
            verses = re.findall(r'\[verse\](.*?)(?=\[|$)', text_lower, re.DOTALL)
            choruses = re.findall(r'\[chorus\](.*?)(?=\[|$)', text_lower, re.DOTALL)
            bridges = re.findall(r'\[bridge\](.*?)(?=\[|$)', text_lower, re.DOTALL)
            
            # Approximate 3 seconds per line for verses/bridges, 2.5 for choruses
            for verse in verses:
                lines = len([l for l in verse.strip().split('\n') if l.strip()])
                duration_seconds += lines * 3
            for chorus in choruses:
                lines = len([l for l in chorus.strip().split('\n') if l.strip()])
                duration_seconds += lines * 2.5
            for bridge in bridges:
                lines = len([l for l in bridge.strip().split('\n') if l.strip()])
                duration_seconds += lines * 3
        
        # Format duration
        if duration_seconds > 0:
            minutes = int(duration_seconds // 60)
            seconds = int(duration_seconds % 60)
            duration_str = f" | Estimated duration: {minutes}:{seconds:02d}"
        else:
            duration_str = " | Estimated duration: 0:00"
        
        # Character count warning
        if char_count > 1000:
            return f"⚠️ {char_count}/1000 characters (will be truncated){duration_str}"
        elif char_count > 900:
            return f"⚠️ {char_count}/1000 characters{duration_str}"
        else:
            return f"{char_count}/1000 characters{duration_str}"
    
    lyrics.change(fn=update_char_count_and_duration, inputs=[lyrics], outputs=[char_counter])
    
    # Initialize character counter with default lyrics
    demo.load(fn=update_char_count_and_duration, inputs=[lyrics], outputs=[char_counter])
    
    # Preset functionality
    def handle_save_preset(preset_name, *args):
        """Save current settings as a preset"""
        if not preset_name:
            gr.Info("Please enter a preset name")
            return gr.Dropdown(choices=get_preset_list())
        
        # Collect all parameters
        param_values = list(args[:28])  # All UI parameters (increased to 28)
        param_names = [
            'lyrics', 'genre', 'instrument', 'emotion', 'timbre', 'gender',
            'sample_prompt', 'audio_path', 'image_path', 'save_mp3', 'seed',
            'max_gen_length', 'diffusion_steps', 'temperature', 'top_k', 'top_p',
            'cfg_coef', 'guidance_scale', 'use_sampling', 'extend_stride',
            'gen_type', 'chunked', 'chunk_size', 'record_tokens', 'record_window',
            'disable_offload', 'disable_cache_clear', 'disable_fp16', 'disable_sequential'
        ]
        
        preset_data = dict(zip(param_names, param_values))
        success, message = save_preset(preset_name, preset_data)
        
        if success:
            set_last_used_preset(preset_name)
            gr.Info(message)
            # Return updated dropdown with new preset selected
            return gr.Dropdown(choices=get_preset_list(), value=preset_name)
        else:
            gr.Error(message)
            return gr.Dropdown(choices=get_preset_list())
    
    def handle_load_preset(preset_name):
        """Load a preset and update all UI components"""
        if not preset_name:
            return [gr.update()] * 28  # Updated to 28
        
        preset_data, message = load_preset(preset_name)
        if preset_data is None:
            gr.Error(message)
            return [gr.update()] * 28  # Updated to 28
        
        # Apply preset data with defaults for missing values
        values = apply_preset_to_ui(preset_data)
        
        # Update last used preset
        set_last_used_preset(preset_name)
        gr.Info(message)
        
        # Return updates for all UI components
        return [
            values['lyrics'],
            values['genre'],
            values['instrument'],
            values['emotion'],
            values['timbre'],
            values['gender'],
            values['sample_prompt'],
            values['audio_path'],
            values['image_path'],
            values['save_mp3'],
            values['seed'],
            values['max_gen_length'],
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
            values['disable_sequential']
        ]
    
    def handle_refresh_presets():
        """Refresh the preset dropdown"""
        return gr.Dropdown(choices=get_preset_list())
    
    # Connect preset handlers
    all_inputs = [
        lyrics, genre, instrument, emotion, timbre, gender,
        sample_prompt, audio_path, image_upload, save_mp3_check, seed_input,
        max_gen_length, diffusion_steps, temperature, top_k, top_p,
        cfg_coef, guidance_scale, use_sampling, extend_stride,
        gen_type, chunked, chunk_size, record_tokens, record_window,
        disable_offload, disable_cache_clear, disable_fp16, disable_sequential
    ]
    
    # Modified to return both dropdown and clear the input field
    def handle_save_and_clear(preset_name, *args):
        dropdown_update = handle_save_preset(preset_name, *args)
        # Return updates for both dropdown and input field
        return dropdown_update, gr.Textbox(value="")  # Clear the input field
    
    save_preset_btn.click(
        fn=handle_save_and_clear,
        inputs=[preset_name_input] + all_inputs,
        outputs=[preset_dropdown, preset_name_input]
    ).then(
        # Automatically load the preset after saving
        fn=handle_load_preset,
        inputs=[preset_dropdown],
        outputs=all_inputs
    )
    
    load_preset_btn.click(
        fn=handle_load_preset,
        inputs=[preset_dropdown],
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
        inputs=[preset_dropdown],
        outputs=all_inputs
    )
    
    # Load last used preset on startup
    def load_initial_preset():
        """Load the last used preset on startup"""
        last_preset = get_last_used_preset()
        if last_preset and last_preset in get_preset_list():
            return gr.Dropdown(value=last_preset)
        return gr.Dropdown(value=None)
    
    demo.load(fn=load_initial_preset, inputs=[], outputs=[preset_dropdown])
    
    submit_btn.click(
        fn=submit_lyrics,
        inputs=[
            lyrics, struct, genre, instrument, emotion, timbre, gender,
            sample_prompt, audio_path, image_upload, save_mp3_check, seed_input,
            max_gen_length, diffusion_steps, temperature, top_k, top_p,
            cfg_coef, guidance_scale, use_sampling, extend_stride,
            gen_type, chunked, chunk_size, record_tokens, record_window,
            disable_offload, disable_cache_clear, disable_fp16, disable_sequential,
            history, session
        ],
        outputs=[output_audio, output_video, history, history_display]
    )
    
    open_folder_btn.click(
        fn=open_output_folder,
        inputs=[],
        outputs=[]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)