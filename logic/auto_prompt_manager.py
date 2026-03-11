import os
import torch
import numpy as np
from typing import Optional, List, Tuple
from utils.torch_load import load_torch_file


class AutoPromptManager:
    """Manages automatic prompt audio selection based on genre/style"""

    def __init__(self, app_dir: str):
        self.app_dir = app_dir
        self.auto_prompt_path = os.path.join(app_dir, 'tools', 'new_auto_prompt.pt')
        self.legacy_auto_prompt_path = os.path.join(app_dir, 'tools', 'new_prompt.pt')
        self.ckpt_prompt_path = os.path.join(app_dir, 'ckpt', 'prompt.pt')
        self.auto_prompt_data = None
        self._load_auto_prompt_data()

    def _load_auto_prompt_data(self):
        """Load auto prompt data from disk."""
        try:
            for candidate in [self.auto_prompt_path, self.legacy_auto_prompt_path, self.ckpt_prompt_path]:
                if os.path.exists(candidate):
                    self.auto_prompt_data = load_torch_file(candidate, map_location='cpu')
                    print(f"[INFO] Auto prompt data loaded from: {candidate}")
                    break
            else:
                print("[WARN] Auto prompt data not found - auto prompt selection disabled")
                print(f"  Looked for: {self.auto_prompt_path}")
                print(f"  Looked for: {self.legacy_auto_prompt_path}")
                print(f"  Looked for: {self.ckpt_prompt_path}")
                self.auto_prompt_data = None
        except Exception as e:
            print(f"[WARN] Failed to load auto prompt data: {e}")
            self.auto_prompt_data = None

    def get_available_types(self) -> List[str]:
        """Get list of available auto prompt types."""
        if self.auto_prompt_data is None:
            return []

        if isinstance(self.auto_prompt_data, dict):
            return sorted(self.auto_prompt_data.keys())

        return ['Auto', 'Pop', 'Latin', 'Rock', 'Electronic', 'Metal', 'Country', 'R&B/Soul', 'Ballad', 'Jazz', 'World', 'Hip-Hop', 'Funk', 'Soundtrack']

    def get_auto_prompt_tokens(self, prompt_type: str) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Get auto prompt tokens for the specified type."""
        if self.auto_prompt_data is None:
            return None

        try:
            if prompt_type == 'Auto':
                all_prompts = []
                if isinstance(self.auto_prompt_data, dict):
                    for prompts in self.auto_prompt_data.values():
                        if isinstance(prompts, dict):
                            for prompt_list in prompts.values():
                                if isinstance(prompt_list, list):
                                    all_prompts.extend(prompt_list)
                                elif prompt_list is not None:
                                    all_prompts.append(prompt_list)
                        elif isinstance(prompts, list):
                            all_prompts.extend(prompts)
                        elif prompts is not None:
                            all_prompts.append(prompts)
                if not all_prompts:
                    return None
                selected_prompt = all_prompts[np.random.randint(0, len(all_prompts))]
            elif isinstance(self.auto_prompt_data, dict) and prompt_type in self.auto_prompt_data:
                prompt_tokens = self.auto_prompt_data[prompt_type]
                if isinstance(prompt_tokens, dict):
                    prompt_lists = [value for value in prompt_tokens.values() if isinstance(value, list) and value]
                    if not prompt_lists:
                        return None
                    selected_group = prompt_lists[0]
                    selected_prompt = selected_group[np.random.randint(0, len(selected_group))]
                elif isinstance(prompt_tokens, list) and prompt_tokens:
                    selected_prompt = prompt_tokens[np.random.randint(0, len(prompt_tokens))]
                else:
                    selected_prompt = prompt_tokens
            else:
                print(f"[WARN] Auto prompt type '{prompt_type}' not found")
                return None

            if hasattr(selected_prompt, 'shape') and len(selected_prompt.shape) >= 3:
                pmt_wav = selected_prompt[:, [0], :]
                vocal_wav = selected_prompt[:, [1], :]
                bgm_wav = selected_prompt[:, [2], :]
                return pmt_wav, vocal_wav, bgm_wav

            print(f"[WARN] Unexpected prompt token format for {prompt_type}")
            return None
        except Exception as e:
            print(f"Error getting auto prompt tokens for {prompt_type}: {e}")
            return None

    def is_available(self) -> bool:
        return self.auto_prompt_data is not None

    def get_fallback_prompt_path(self) -> str:
        if os.path.exists(self.auto_prompt_path):
            return self.auto_prompt_path
        if os.path.exists(self.legacy_auto_prompt_path):
            return self.legacy_auto_prompt_path
        if os.path.exists(self.ckpt_prompt_path):
            return self.ckpt_prompt_path
        return self.auto_prompt_path
