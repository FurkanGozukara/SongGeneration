import os
import subprocess
import json
from typing import Optional, List, Dict, Any

def save_metadata(file_path: str, metadata: Dict[str, Any]) -> bool:
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

def convert_wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str = '192k') -> bool:
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

def create_video_from_image_and_audio(image_path: str, audio_path: str, output_path: str, duration: Optional[float] = None) -> bool:
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

def scan_folder_for_txt_files(folder_path: str) -> List[str]:
    """Scan folder for txt files and return their paths"""
    if not os.path.exists(folder_path):
        return []
    
    txt_files = []
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            txt_files.append(os.path.join(folder_path, file))
    
    return sorted(txt_files)

def read_prompt_from_txt(txt_path: str) -> str:
    """Read prompt/lyrics from txt file"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading {txt_path}: {e}")
        return ""

def check_if_output_exists(output_folder: str, base_name: str, check_mp3: bool = True, check_mp4: bool = True) -> bool:
    """Check if output files already exist"""
    wav_exists = os.path.exists(os.path.join(output_folder, f"{base_name}.wav"))
    
    if not check_mp3 and not check_mp4:
        return wav_exists
    
    mp3_exists = os.path.exists(os.path.join(output_folder, f"{base_name}.mp3")) if check_mp3 else False
    mp4_exists = os.path.exists(os.path.join(output_folder, f"{base_name}.mp4")) if check_mp4 else False
    
    return wav_exists or mp3_exists or mp4_exists

def load_presets_from_folder(preset_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load all presets from the preset directory"""
    presets = {}
    if not os.path.exists(preset_dir):
        return presets
    
    for file in os.listdir(preset_dir):
        if file.endswith('.json') and not file.startswith('_'):
            preset_name = file[:-5]
            preset_path = os.path.join(preset_dir, file)
            try:
                with open(preset_path, 'r', encoding='utf-8') as f:
                    presets[preset_name] = json.load(f)
            except Exception as e:
                print(f"Error loading preset {preset_name}: {e}")
    
    return presets