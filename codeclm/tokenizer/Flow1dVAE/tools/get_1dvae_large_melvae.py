import torch
from tqdm import tqdm
import torchaudio
import sys
from pathlib import Path

# Add ckpt directory to path for third_party imports
ckpt_dir = Path(__file__).parent.parent.parent.parent.parent / "ckpt"
if str(ckpt_dir) not in sys.path:
    sys.path.insert(0, str(ckpt_dir))

from third_party.stable_audio_tools.stable_audio_tools.models.autoencoders import create_autoencoder_from_config
import numpy as np
import os
import json

def get_model(model_config, path):
    with open(model_config) as f:
        model_config = json.load(f)
    state_dict = torch.load(path, map_location='cpu')
    model = create_autoencoder_from_config(model_config)
    model.load_state_dict(state_dict['state_dict'], strict=False)
    return model