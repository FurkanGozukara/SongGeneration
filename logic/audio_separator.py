import os
import torch
import torchaudio
import numpy as np
from typing import Tuple, Optional

class AudioSeparator:
    """Audio separation utility using Demucs for source separation"""
    
    def __init__(self, dm_model_path=None, dm_config_path=None, gpu_id=0):
        """Initialize the audio separator
        
        Args:
            dm_model_path: Path to Demucs model
            dm_config_path: Path to Demucs config
            gpu_id: GPU device ID
        """
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
            
        # Set default paths if not provided
        if dm_model_path is None:
            dm_model_path = os.path.join(os.path.dirname(__file__), '..', 'third_party', 'demucs', 'ckpt', 'htdemucs.pth')
        if dm_config_path is None:
            dm_config_path = os.path.join(os.path.dirname(__file__), '..', 'third_party', 'demucs', 'ckpt', 'htdemucs.yaml')
            
        self.demucs_model = None
        self.dm_model_path = dm_model_path
        self.dm_config_path = dm_config_path
        
        # Initialize Demucs model if available
        self._init_demucs_model()
    
    def _init_demucs_model(self):
        """Initialize Demucs model for source separation"""
        try:
            if os.path.exists(self.dm_model_path) and os.path.exists(self.dm_config_path):
                # Import Demucs model loader
                try:
                    from third_party.demucs.models.pretrained import get_model_from_yaml
                    self.demucs_model = get_model_from_yaml(self.dm_config_path, self.dm_model_path)
                    self.demucs_model.to(self.device)
                    self.demucs_model.eval()
                    print("✓ Demucs model loaded successfully for audio separation")
                except ImportError:
                    print("⚠️ Demucs not available - audio separation disabled")
                    self.demucs_model = None
            else:
                print("⚠️ Demucs model files not found - audio separation disabled")
                print(f"  Model path: {self.dm_model_path}")
                print(f"  Config path: {self.dm_config_path}")
                self.demucs_model = None
        except Exception as e:
            print(f"⚠️ Failed to initialize Demucs: {e}")
            self.demucs_model = None
    
    def load_audio(self, audio_path: str, max_duration: float = 10.0) -> torch.Tensor:
        """Load and preprocess audio file
        
        Args:
            audio_path: Path to audio file
            max_duration: Maximum duration in seconds (default 10s)
            
        Returns:
            Preprocessed audio tensor
        """
        try:
            audio, sample_rate = torchaudio.load(audio_path)
            
            # Resample to 48kHz if needed
            if sample_rate != 48000:
                audio = torchaudio.functional.resample(audio, sample_rate, 48000)
            
            # Limit to max duration
            max_samples = int(48000 * max_duration)
            if audio.shape[-1] > max_samples:
                audio = audio[..., :max_samples]
            
            # Ensure stereo
            if audio.shape[0] == 1:
                audio = audio.repeat(2, 1)
            elif audio.shape[0] > 2:
                audio = audio[:2]
                
            return audio
            
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None
    
    def separate_audio(self, audio_path: str, output_dir: str = 'tmp') -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Separate audio into vocals, BGM, and full audio
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory for temporary files
            
        Returns:
            Tuple of (full_audio, vocal_audio, bgm_audio)
        """
        if self.demucs_model is None:
            print("⚠️ Demucs not available - returning original audio as full track")
            full_audio = self.load_audio(audio_path)
            if full_audio is not None:
                # Return full audio as both vocal and bgm (no separation)
                return full_audio, full_audio * 0.5, full_audio * 0.5
            return None, None, None
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Load full audio
            full_audio = self.load_audio(audio_path)
            if full_audio is None:
                return None, None, None
            
            # Perform separation using Demucs
            name = os.path.splitext(os.path.basename(audio_path))[0]
            
            # Check if separated files already exist
            vocal_path = os.path.join(output_dir, f"{name}_vocals.flac")
            
            if os.path.exists(vocal_path):
                # Load existing separated audio
                vocal_audio = self.load_audio(vocal_path)
            else:
                # Perform separation
                try:
                    drums_path, bass_path, other_path, vocal_path = self.demucs_model.separate(
                        audio_path, output_dir, device=self.device
                    )
                    
                    # Clean up unwanted files
                    for path in [drums_path, bass_path, other_path]:
                        if os.path.exists(path):
                            os.remove(path)
                    
                    vocal_audio = self.load_audio(vocal_path)
                    
                except Exception as e:
                    print(f"Demucs separation failed: {e}")
                    # Fallback: use original audio
                    vocal_audio = full_audio * 0.7  # Assume vocals are dominant
            
            # Calculate BGM as difference
            if vocal_audio is not None:
                bgm_audio = full_audio - vocal_audio
            else:
                bgm_audio = full_audio * 0.3  # Fallback
                vocal_audio = full_audio * 0.7
            
            return full_audio, vocal_audio, bgm_audio
            
        except Exception as e:
            print(f"Audio separation failed: {e}")
            # Return original audio with basic splitting
            full_audio = self.load_audio(audio_path)
            if full_audio is not None:
                return full_audio, full_audio * 0.7, full_audio * 0.3
            return None, None, None
    
    def is_available(self) -> bool:
        """Check if audio separation is available"""
        return self.demucs_model is not None
