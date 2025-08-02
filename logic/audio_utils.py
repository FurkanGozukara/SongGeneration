import subprocess
import os

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

def get_audio_duration(audio_path):
    """Get duration of audio file in seconds"""
    try:
        cmd = [
            'ffprobe', '-i', audio_path, '-hide_banner', '-loglevel', 'error',
            '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=p=0'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return 0