import os
import torch
import numpy as np
from typing import Optional, Dict, List, Tuple
from utils.torch_load import load_torch_file

class AutoPromptManager:
    """Manages automatic prompt audio selection based on genre/style"""
    
    def __init__(self, app_dir: str):
        """Initialize auto prompt manager
        
        Args:
            app_dir: Application directory path
        """
        self.app_dir = app_dir
        self.auto_prompt_path = os.path.join(app_dir, 'tools', 'new_prompt.pt')
        self.ckpt_prompt_path = os.path.join(app_dir, 'ckpt', 'prompt.pt')
        self.auto_prompt_data = None
        # Auto prompt loading is intentionally disabled in the UI flow.
        # The app uses manual reference audio uploads instead.
    
    def _load_auto_prompt_data(self):
        """Load auto prompt data from checkpoint"""
        try:
            # Try loading from tools directory first
            if os.path.exists(self.auto_prompt_path):
                self.auto_prompt_data = load_torch_file(self.auto_prompt_path, map_location='cpu')
                print(f"✓ Auto prompt data loaded from: {self.auto_prompt_path}")
            elif os.path.exists(self.ckpt_prompt_path):
                self.auto_prompt_data = load_torch_file(self.ckpt_prompt_path, map_location='cpu')
                print(f"✓ Auto prompt data loaded from: {self.ckpt_prompt_path}")
            else:
                print("⚠️ Auto prompt data not found - auto prompt selection disabled")
                print(f"  Looked for: {self.auto_prompt_path}")
                print(f"  Looked for: {self.ckpt_prompt_path}")
                self.auto_prompt_data = None
                
        except Exception as e:
            print(f"⚠️ Failed to load auto prompt data: {e}")
            self.auto_prompt_data = None
    
    def get_available_types(self) -> List[str]:
        """Get list of available auto prompt types"""
        if self.auto_prompt_data is None:
            return []
        
        if isinstance(self.auto_prompt_data, dict):
            return list(self.auto_prompt_data.keys())
        else:
            # Default types if data structure is different
            return ['Pop', 'R&B', 'Dance', 'Jazz', 'Folk', 'Rock', 'Chinese Style', 'Chinese Tradition', 'Metal', 'Reggae', 'Chinese Opera', 'Auto']
    
    def get_auto_prompt_tokens(self, prompt_type: str) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Get auto prompt tokens for specified type
        
        Args:
            prompt_type: Type of prompt ('Pop', 'Jazz', etc., or 'Auto' for random selection)
            
        Returns:
            Tuple of (pmt_wav, vocal_wav, bgm_wav) tokens or None
        """
        if self.auto_prompt_data is None:
            return None
        
        try:
            # Handle "Auto" case - randomly select from all available genres
            if prompt_type == "Auto":
                if isinstance(self.auto_prompt_data, dict):
                    # Merge all prompts from all genres
                    all_prompts = []
                    for genre, prompts in self.auto_prompt_data.items():
                        if isinstance(prompts, list):
                            all_prompts.extend(prompts)
                        else:
                            all_prompts.append(prompts)
                    
                    if len(all_prompts) > 0:
                        selected_prompt = all_prompts[np.random.randint(0, len(all_prompts))]
                        if hasattr(selected_prompt, 'shape') and len(selected_prompt.shape) >= 3:
                            pmt_wav = selected_prompt[:, [0], :]
                            vocal_wav = selected_prompt[:, [1], :]
                            bgm_wav = selected_prompt[:, [2], :]
                            return pmt_wav, vocal_wav, bgm_wav
                    return None
            
            if isinstance(self.auto_prompt_data, dict) and prompt_type in self.auto_prompt_data:
                prompt_tokens = self.auto_prompt_data[prompt_type]
                
                # Select random prompt from available options
                if isinstance(prompt_tokens, list) and len(prompt_tokens) > 0:
                    selected_prompt = prompt_tokens[np.random.randint(0, len(prompt_tokens))]
                else:
                    selected_prompt = prompt_tokens
                
                # Extract tokens (assuming format from original repo)
                if hasattr(selected_prompt, 'shape') and len(selected_prompt.shape) >= 3:
                    pmt_wav = selected_prompt[:, [0], :]
                    vocal_wav = selected_prompt[:, [1], :]
                    bgm_wav = selected_prompt[:, [2], :]
                    return pmt_wav, vocal_wav, bgm_wav
                else:
                    print(f"⚠️ Unexpected prompt token format for {prompt_type}")
                    return None
            else:
                print(f"⚠️ Auto prompt type '{prompt_type}' not found")
                return None
                
        except Exception as e:
            print(f"Error getting auto prompt tokens for {prompt_type}: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if auto prompt system is available"""
        return self.auto_prompt_data is not None
    
    def get_fallback_prompt_path(self) -> str:
        """Get fallback prompt path for manual loading"""
        if os.path.exists(self.ckpt_prompt_path):
            return self.ckpt_prompt_path
        elif os.path.exists(self.auto_prompt_path):
            return self.auto_prompt_path
        else:
            # Create a dummy prompt file path
            return os.path.join(self.app_dir, 'ckpt', 'prompt.pt')
