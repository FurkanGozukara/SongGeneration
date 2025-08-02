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
import argparse
import threading

# Add the tools/gradio directory to the path
sys.path.append(op.join(op.dirname(op.abspath(__file__)), 'tools', 'gradio'))
from levo_inference_lowmem import LeVoInference

# Import from logic folder
from logic.generation import CancellationToken, ProgressTracker, set_seed, format_lyrics_for_model, get_next_file_number, generate_single_song
from logic.file_utils import save_metadata, convert_wav_to_mp3, create_video_from_image_and_audio
from logic.batch_processing import BatchProcessor
from logic.ui_progress import GradioProgressTracker, create_progress_callback, format_eta
from logic.preset_manager import PresetManager

EXAMPLE_LYRICS = """
example
""".strip()

APP_DIR = op.dirname(op.abspath(__file__))

# Parse command line arguments
parser = argparse.ArgumentParser(description='LeVo Song Generation App')
parser.add_argument('checkpoint', nargs='?', default=None, help='Path to checkpoint directory')
parser.add_argument('--share', action='store_true', help='Share the Gradio app publicly')
args = parser.parse_args()

# Default checkpoint path - this should point to the directory with config.yaml and model.pt
DEFAULT_CKPT = op.join(APP_DIR, 'ckpt', 'songgeneration_base')

# Use command line argument if provided, otherwise use default
ckpt_path = args.checkpoint if args.checkpoint else DEFAULT_CKPT

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

# Global cancellation token and batch processor
cancellation_token = CancellationToken()
batch_processor = BatchProcessor(MODEL, APP_DIR, MODEL.cfg)
gradio_progress_tracker = GradioProgressTracker()

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

# Initialize preset manager
preset_manager = PresetManager(PRESET_DIR)


def collect_current_parameters(
    lyrics, genre, instrument, emotion, timbre, gender,
    sample_prompt, audio_path, image_path, save_mp3, seed,
    max_gen_length, diffusion_steps, temperature, top_k, top_p,
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
    lyrics, genre, instrument, emotion, timbre, gender,
    sample_prompt, audio_path, image_path, save_mp3, seed,
    max_gen_length, diffusion_steps, temperature, top_k, top_p,
    cfg_coef, guidance_scale, use_sampling, extend_stride,
    gen_type, chunked, chunk_size, record_tokens, record_window,
    disable_offload, disable_cache_clear, disable_fp16, disable_sequential,
    randomize_params, progress=gr.Progress()
):
    """Run batch processing with progress tracking"""
    if not input_folder or not output_folder:
        gr.Error("Please specify both input and output folders")
        return "Please specify both input and output folders"
    
    # Show cancel button
    yield gr.update(visible=True), gr.update(visible=True), "Starting batch processing..."
    
    # Collect base parameters
    base_params = collect_current_parameters(
        lyrics, genre, instrument, emotion, timbre, gender,
        sample_prompt, audio_path, image_path, save_mp3, seed,
        max_gen_length, diffusion_steps, temperature, top_k, top_p,
        cfg_coef, guidance_scale, use_sampling, extend_stride,
        gen_type, chunked, chunk_size, record_tokens, record_window,
        disable_offload, disable_cache_clear, disable_fp16, disable_sequential,
        num_generations, loop_presets, randomize_params
    )
    
    # Set progress callback
    def batch_progress_callback(info):
        progress_msg = []
        if info.get('batch_info'):
            progress_msg.append(f"Batch: {info['batch_info']}")
        if info.get('preset_info'):
            progress_msg.append(f"Preset: {info['preset_info']}")
        if info.get('generation_info'):
            progress_msg.append(f"Generation: {info['generation_info']}")
        if info.get('message'):
            progress_msg.append(info['message'])
        if info.get('eta') and info['eta'] > 0:
            progress_msg.append(f"ETA: {format_eta(info['eta'])}")
        
        status_text = " | ".join(progress_msg)
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
        gr.Error(error_msg)
        yield gr.update(visible=False), gr.update(visible=False), error_msg

def submit_lyrics(
    lyrics, struct, genre, instrument, emotion, timbre, gender,
    sample_prompt, audio_path, image_path, save_mp3, seed,
    max_gen_length, diffusion_steps, temperature, top_k, top_p,
    cfg_coef, guidance_scale, use_sampling, extend_stride,
    gen_type, chunked, chunk_size, record_tokens, record_window,
    disable_offload, disable_cache_clear, disable_fp16, disable_sequential,
    num_generations, loop_presets, randomize_params, history, session, progress=gr.Progress()
):
    # Reset cancellation token
    cancellation_token.reset()
    
    # Create progress callback
    progress_callback = create_progress_callback(gradio_progress_tracker, progress)
    
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
    
    # Call the model using the forward method with progress tracking
    # Account for pattern delays (~250 steps) when converting to duration
    pattern_delay_offset = 250  # Approximate delay from codebook pattern
    actual_steps = max(max_gen_length - pattern_delay_offset, 1000)
    duration_from_steps = min(actual_steps / 25.0, 150.0)
    
    # Log the actual values for debugging
    output_messages(f"Requested steps: {max_gen_length}, Actual generation: ~{int(duration_from_steps * 25) + pattern_delay_offset} steps")
    
    gen_params = {
        'duration': duration_from_steps,
        'num_steps': diffusion_steps,
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p,
        'cfg_coef': cfg_coef,
        'guidance_scale': guidance_scale,
        'use_sampling': use_sampling,
        'extend_stride': extend_stride,
        'chunked': chunked,
        'chunk_size': chunk_size,
        'record_tokens': record_tokens,
        'record_window': record_window,
    }
    
    try:
        # Store initial values before potential randomization
        current_genre = genre
        current_instrument = instrument
        current_emotion = emotion
        current_timbre = timbre
        current_gender = gender
        
        audio_data = MODEL(
            song_data["lyrics"], 
            f"{current_gender}, {current_timbre}, {current_genre}, {current_emotion}, {current_instrument}",
            song_data["audio_path"] if song_data["sample_prompt"] else None,
            None,  # genre parameter (None since we're using description)
            op.join(APP_DIR, "ckpt/prompt.pt"),  # auto_prompt_path
            gen_type,  # gen_type from UI
            gen_params,  # params
            disable_offload=disable_offload,
            disable_cache_clear=disable_cache_clear,
            disable_fp16=disable_fp16,
            disable_sequential=disable_sequential,
            progress_callback=progress_callback,
            cancellation_token=cancellation_token
        )
        
        if audio_data is None:
            gr.Info("Generation cancelled")
            return None, None, history, process_history(history)
            
        audio_data = audio_data.cpu().permute(1, 0).float().numpy()
        
    except Exception as e:
        gr.Error(f"Generation failed: {str(e)}")
        yield None, None, history, process_history(history), gr.update(visible=False), gr.update(visible=False)
        return
    
    # Show cancel button and progress
    yield None, None, history, process_history(history), gr.update(visible=True), gr.update(visible=True)
    
    # Load all presets if loop_presets is enabled
    presets_to_use = [None]  # None means use current settings
    if loop_presets:
        all_presets = preset_manager.get_preset_list()
        if all_presets:
            presets_to_use = all_presets
            output_messages(f"Looping through {len(presets_to_use)} presets")
        else:
            output_messages("No presets found, using current settings only")
    
    # Generate multiple songs if requested
    generated_files = []
    total_generations = len(presets_to_use) * num_generations
    generation_count = 0
    
    # Preset loop (outer loop)
    for preset_idx, preset_name in enumerate(presets_to_use):
        # Load preset if specified
        if preset_name:
            preset_data, message = preset_manager.load_preset(preset_name)
            if preset_data:
                output_messages(f"Using preset: {preset_name}")
                # Override current parameters with preset values (except lyrics)
                current_lyrics = lyrics  # Always keep user's lyrics
                genre = preset_data.get('genre', genre)
                instrument = preset_data.get('instrument', instrument)
                emotion = preset_data.get('emotion', emotion)
                timbre = preset_data.get('timbre', timbre)
                gender = preset_data.get('gender', gender)
                # Also get preset's randomize setting
                randomize_params = preset_data.get('randomize_params', randomize_params)
            else:
                output_messages(f"Failed to load preset {preset_name}, skipping")
                continue
        
        # Generation loop (inner loop)
        for gen_idx in range(num_generations):
            if cancellation_token.is_cancelled():
                gr.Info("Generation cancelled")
                break
            
            generation_count += 1
            # Update progress for multiple generations
            progress_desc = f"Generating {generation_count}/{total_generations}"
            if preset_name:
                progress_desc += f" | Preset: {preset_name}"
            if num_generations > 1:
                progress_desc += f" | Generation {gen_idx + 1}/{num_generations}"
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
            
            # Re-run generation for subsequent iterations
            if gen_idx > 0:
                try:
                    audio_data = MODEL(
                        song_data["lyrics"], 
                        f"{current_gender}, {current_timbre}, {current_genre}, {current_emotion}, {current_instrument}",
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
                    gr.Error(f"Generation {gen_idx + 1} failed: {str(e)}")
                    continue
            
                # Save the audio with sequential numbering
            output_dir = op.join(APP_DIR, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            file_number = get_next_file_number(output_dir)
            # Construct filename based on preset and generation
            if preset_name and num_generations > 1:
                base_filename = f"{file_number:04d}_{preset_name}_gen{gen_idx+1:03d}"
            elif preset_name:
                base_filename = f"{file_number:04d}_{preset_name}"
            elif num_generations > 1:
                base_filename = f"{file_number:04d}_gen{gen_idx+1:03d}"
            else:
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
            
            # For the first generation, update the main outputs
            if gen_idx == 0:
                song_data["audio"] = wav_path
                song_data["mp3"] = mp3_path
                song_data["video"] = video_path
            
            # Save metadata (use actual generated values, not original if randomized)
            metadata = collect_current_parameters(
                lyrics, current_genre, current_instrument, current_emotion, current_timbre, current_gender,
                sample_prompt, audio_path, image_path, save_mp3, used_seed,
                max_gen_length, diffusion_steps, temperature, top_k, top_p,
                cfg_coef, guidance_scale, use_sampling, extend_stride,
                gen_type, chunked, chunk_size, record_tokens, record_window,
                disable_offload, disable_cache_clear, disable_fp16, disable_sequential,
                num_generations, loop_presets, randomize_params
            )
            metadata['timestamp'] = current_time
            metadata['model'] = ckpt_path
            metadata['generation_index'] = gen_idx + 1
            metadata['total_generations'] = num_generations
            metadata['output_files'] = {
                'wav': wav_path,
                'mp3': mp3_path,
                'mp4': video_path
            }
            
            save_metadata(wav_path, metadata)
            generated_files.append({'wav': wav_path, 'mp3': mp3_path, 'mp4': video_path})
    
    # Update history
    history.append({"role": "user", "content": f"Generate {num_generations} song(s) with lyrics: {lyrics[:50]}..."})
    history.append({"role": "assistant", "content": f"Generated {len(generated_files)} song(s) successfully!", "song": song_data})
    
    # Hide cancel button and progress
    yield song_data["audio"], song_data.get("video"), history, process_history(history), gr.update(visible=False), gr.update(visible=False)

# Create Gradio interface
with gr.Blocks(title="SECourses LeVo Song Generation App V1",theme=gr.themes.Soft()) as demo:
    gr.Markdown("# LeVo Song Generation")
    
    history = gr.State([])
    session = gr.State({})
    
    # Add cancel button at the top
    with gr.Row():
        cancel_btn = gr.Button("🛑 Cancel Generation", variant="stop", visible=False)
    
    # Progress display
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
                        info="Maximum 1000 characters to stay within token limit"
                    )
                    
                    # Character counter and duration display
                    char_counter = gr.Markdown("0/1000 characters | Estimated duration: 0:00")
            
                    # Generate and Open Folder buttons at the top
                    with gr.Row():
                        submit_btn = gr.Button("Generate Song", variant="primary")
                        open_folder_btn = gr.Button("Open Output Folder", variant="secondary")
            
                    # Preset controls
                    with gr.Accordion("Presets", open=True):
                        with gr.Row():
                            preset_dropdown = gr.Dropdown(
                                label="Select Preset",
                                choices=preset_manager.get_preset_list(),
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
                    
                    # Number of generations slider
                    with gr.Row():
                        num_generations = gr.Slider(
                    label="Number of Generations", 
                    minimum=1, 
                    maximum=10, 
                    value=1, 
                    step=1,
                    info="Generate multiple songs with different seeds"
                        )
                    
                    # Reference audio section
                    with gr.Accordion("Reference Audio (Advanced)", open=False):
                        sample_prompt = gr.Checkbox(label="Use Sample Prompt", value=False)
                        audio_path = gr.Audio(label="Reference Audio (optional)", type="filepath")
                
                with gr.Column(scale=1):
                    output_audio = gr.Audio(label="Generated Audio", type="filepath")
                    output_video = gr.Video(label="Generated Video", visible=True)
                    
                    gr.Markdown("---")
                    gr.Markdown("Generate high-quality songs with both vocals and accompaniment using AI")
                    
                    # Image input for video generation
                    image_upload = gr.Image(label="Image for Video (optional)", type="filepath")
                    gr.Markdown("""
                    **Video Generation:**
                    - Upload an image to create an MP4 video
                    - Video will use the uploaded image with generated audio
                    - Video encoded with H.264, CRF 17
                    """)
                    
                    
                    gr.Markdown(f"**Model checkpoint:** {ckpt_path}")
                    
                    gr.Markdown("""
                   example
                    """)
                    
                    history_display = gr.HTML()
        
        with gr.TabItem("⚙️ Advanced Settings"):
            with gr.Row():
                with gr.Column():
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
                
                with gr.Column():
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
                
                with gr.Column():
                    gr.Markdown("### VRAM Optimization")
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
                    
                    gr.Markdown("### Advanced Generation Options")
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
            
        with gr.TabItem("📁 Batch Processing"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Batch Processing")
                    gr.Markdown("""
                    **Batch Processing:**
                    - Select a folder containing .txt files with prompts
                    - Each .txt file will generate a song
                    - Output files use the same name as the .txt file
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
                        batch_process_btn = gr.Button("Start Batch Processing", variant="primary")
                    
                    # Batch status display
                    batch_status = gr.Markdown("", visible=False)
        
        with gr.TabItem("📖 Song Structure Tags"):
            gr.Markdown("## 🎵 Available Song Structure Tags")
            gr.Markdown("Use these tags in your lyrics to control the song structure. Tags should be placed on their own lines.")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### 🎤 Main Sections:
                    - **[intro]** - Opening instrumental or vocal section (~10 seconds)
                    - **[verse]** - Main story/narrative sections (~20-30 seconds)
                    - **[chorus]** - Catchy, repeating hook section (~15-20 seconds)
                    - **[bridge]** - Contrasting section, usually appears once (~15-20 seconds)
                    - **[outro]** - Closing section (~10 seconds)
                    
                    ### 🎸 Variations & Special Sections:
                    - **[pre-chorus]** - Build-up section before chorus (~8-12 seconds)
                    - **[post-chorus]** - Additional hook after chorus (~8-12 seconds)
                    - **[break]** - Instrumental or rhythm break (~8-16 seconds)
                    - **[instrumental]** - Pure instrumental section (~10-20 seconds)
                    - **[interlude]** - Short connecting section (~5-10 seconds)
                    - **[hook]** - Catchy melodic phrase (~4-8 seconds)
                    - **[drop]** - EDM-style beat drop section (~8-16 seconds)
                    - **[buildup]** - Rising energy section (~8-16 seconds)
                    - **[breakdown]** - Stripped-down section (~8-16 seconds)
                    - **[refrain]** - Repeated lyrical phrase (~4-8 seconds)
                    - **[rap]** - Rap/spoken word section (~16-24 seconds)
                    - **[vocal-run]** - Vocal ad-libs/melisma (~4-8 seconds)
                    
                    ### ⏱️ Length Modifiers:
                    Add these after any tag to control section length:
                    - **[intro-short]** - ~5 seconds
                    - **[intro-medium]** or **[intro]** - ~10 seconds (default)
                    - **[intro-long]** - ~15 seconds
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
                    - Place empty lines between sections for clarity
                    - Tags are case-insensitive ([VERSE] = [verse])
                    - Use consistent structure for professional results
                    - Experiment with tag combinations for unique arrangements
                    - Instrumental tags work great for genre-specific breaks
                    - Length modifiers work with most section types
                    """)
    
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
            return gr.Dropdown(choices=preset_manager.get_preset_list())
        
        # Collect all parameters
        param_values = list(args[:32])  # All UI parameters (32 parameters including new checkboxes)
        param_names = [
            'lyrics', 'genre', 'instrument', 'emotion', 'timbre', 'gender',
            'sample_prompt', 'audio_path', 'image_path', 'save_mp3', 'seed',
            'num_generations', 'loop_presets', 'randomize_params', 'max_gen_length', 'diffusion_steps', 'temperature', 'top_k', 'top_p',
            'cfg_coef', 'guidance_scale', 'use_sampling', 'extend_stride',
            'gen_type', 'chunked', 'chunk_size', 'record_tokens', 'record_window',
            'disable_offload', 'disable_cache_clear', 'disable_fp16', 'disable_sequential'
        ]
        
        preset_data = dict(zip(param_names, param_values))
        success, message = preset_manager.save_preset(preset_name, preset_data)
        
        if success:
            preset_manager.set_last_used_preset(preset_name)
            gr.Info(message)
            # Return updated dropdown with new preset selected
            return gr.Dropdown(choices=preset_manager.get_preset_list(), value=preset_name)
        else:
            gr.Error(message)
            return gr.Dropdown(choices=preset_manager.get_preset_list())
    
    def handle_load_preset(preset_name):
        """Load a preset and update all UI components"""
        if not preset_name:
            return [gr.update()] * 32  # Updated to 32 for all parameters
        
        preset_data, message = preset_manager.load_preset(preset_name)
        if preset_data is None:
            gr.Error(message)
            return [gr.update()] * 32
        
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
            'num_generations': 1,
            'loop_presets': False,
            'randomize_params': False,
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
        
        # Apply preset data with defaults for missing values
        values = preset_manager.apply_preset_to_ui(preset_data, defaults)
        
        # Update last used preset
        preset_manager.set_last_used_preset(preset_name)
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
            values['num_generations'],
            values['loop_presets'],
            values['randomize_params'],
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
        return gr.Dropdown(choices=preset_manager.get_preset_list())
    
    # Connect preset handlers
    all_inputs = [
        lyrics, genre, instrument, emotion, timbre, gender,
        sample_prompt, audio_path, image_upload, save_mp3_check, seed_input,
        num_generations, loop_presets, randomize_params, max_gen_length, diffusion_steps, temperature, top_k, top_p,
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
        last_preset = preset_manager.get_last_used_preset()
        if last_preset and last_preset in preset_manager.get_preset_list():
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
            num_generations, loop_presets, randomize_params, history, session
        ],
        outputs=[output_audio, output_video, history, history_display, cancel_btn, progress_text]
    )
    
    # Cancel button handler
    cancel_btn.click(
        fn=cancel_generation,
        inputs=[],
        outputs=[cancel_btn, progress_text]
    )
    
    # Batch processing button handler
    batch_process_btn.click(
        fn=run_batch_processing,
        inputs=[
            batch_input_folder, batch_output_folder, skip_existing, loop_presets, num_generations,
            lyrics, genre, instrument, emotion, timbre, gender,
            sample_prompt, audio_path, image_upload, save_mp3_check, seed_input,
            max_gen_length, diffusion_steps, temperature, top_k, top_p,
            cfg_coef, guidance_scale, use_sampling, extend_stride,
            gen_type, chunked, chunk_size, record_tokens, record_window,
            disable_offload, disable_cache_clear, disable_fp16, disable_sequential,
            randomize_params
        ],
        outputs=[cancel_btn, progress_text, batch_status]
    )
    
    open_folder_btn.click(
        fn=open_output_folder,
        inputs=[],
        outputs=[]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, share=args.share)