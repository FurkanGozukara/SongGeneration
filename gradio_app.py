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
[intro-short]

[verse]
Morning screen lights up my wall,
Notification beats the dawn.
SECourses calling all—
Curious minds to come along.
Pixels paint tomorrow's dreams,
Code and cadence, laser-clean.
We dive where silicon meets art,
Turning questions into sparks.

[bridge]
When the hype feels hollow,
And the clickbait floods the feed,
Turn the dial to knowledge—
SECourses plants the seed.
From zero lines to launch day,
They map the hidden way;
Let the algorithm play,
We'll still out-learn the wave.

[inst-short]

[chorus]
Raise your hands, hit subscribe,
Feel the future come alive.
Generative souls collide
In every frame they upload.
News that cuts like lightning,
Tutorials brightly guiding—
SECourses, we're riding
On the edge of the code.

[verse]
Neon graphs and tensor ties,
Prompt to painting in a blink.
Robots write their lullabies,
While we craft the missing link.
Ethics, updates, nightly builds,
GPU hearts overfilled.
In the chat our questions flow,
Answers bloom in studio glow.

[bridge]
If confusion clouds your sight,
Scroll no farther—stay tonight.
Step-by-step they lift the veil,
Turning theory into trails.
Every glitch another chance
To rehearse the data dance;
Debug dreams, recompile,
Rise again in learning style.

[inst-short]

[chorus]
Raise your hands, hit subscribe,
Feel the future come alive.
Generative souls collide
In every frame they upload.
News that cuts like lightning,
Tutorials brightly guiding—
SECourses, we're riding
On the edge of the code.

[outro-short]
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


def output_messages(msg):
    gr.Info(msg)
    print(msg)

def get_next_file_number(output_dir):
    """Get the next available file number in sequence"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        return 1
    
    existing_files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
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
    history, session
):
    print(lyrics)
    print(struct)
    
    # Set seed if provided
    if set_seed(seed):
        output_messages(f"Using seed: {seed}")
    else:
        output_messages(f"Processing lyrics... (random seed)")
    
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
    audio_data = MODEL(
        song_data["lyrics"], 
        f"{song_data['gender']}, {song_data['timbre']}, {song_data['genre']}, {song_data['emotion']}, {song_data['instrument']}",
        song_data["audio_path"] if song_data["sample_prompt"] else None,
        None,  # genre parameter (None since we're using description)
        op.join(APP_DIR, "ckpt/prompt.pt"),  # auto_prompt_path
        "mixed",  # gen_type
        {}  # params
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
                lines=15
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
            
            sample_prompt = gr.Checkbox(label="Use Sample Prompt", value=False)
            audio_path = gr.Audio(label="Reference Audio (optional)", type="filepath")
            
            with gr.Row():
                save_mp3_check = gr.Checkbox(label="Also save as MP3 (192 kbps)", value=False)
                seed_input = gr.Number(label="Seed (for reproducibility)", value=-1, precision=0, 
                                     info="Use -1 for random, or any positive number for reproducible results")
            
            with gr.Row():
                image_upload = gr.Image(label="Image for Video (optional)", type="filepath")
                gr.Markdown("""
                **Video Generation:**
                - Upload an image to create an MP4 video
                - Video will use the uploaded image with generated audio
                - Video encoded with H.264, CRF 17
                """)
            
            with gr.Row():
                submit_btn = gr.Button("Generate Song", variant="primary")
                open_folder_btn = gr.Button("Open Output Folder", variant="secondary")
        
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
    
    submit_btn.click(
        fn=submit_lyrics,
        inputs=[
            lyrics, struct, genre, instrument, emotion, timbre, gender,
            sample_prompt, audio_path, image_upload, save_mp3_check, seed_input,
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