import json
import os
from typing import List, Dict, Any, Optional, Tuple

class PresetManager:
    """Manages presets for song generation"""
    
    def __init__(self, preset_dir: str):
        self.preset_dir = preset_dir
        os.makedirs(preset_dir, exist_ok=True)
        
    def get_preset_list(self) -> List[str]:
        """Get list of available presets"""
        if not os.path.exists(self.preset_dir):
            return []
        presets = [f[:-5] for f in os.listdir(self.preset_dir) if f.endswith('.json') and not f.startswith('_')]
        return sorted(presets)
    
    def save_preset(self, preset_name: str, preset_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Save preset to file"""
        if not preset_name:
            return False, "Please enter a preset name"
        
        preset_path = os.path.join(self.preset_dir, f"{preset_name}.json")
        try:
            with open(preset_path, 'w', encoding='utf-8') as f:
                json.dump(preset_data, f, indent=2, ensure_ascii=False)
            return True, f"Preset '{preset_name}' saved successfully"
        except Exception as e:
            return False, f"Error saving preset: {str(e)}"
    
    def load_preset(self, preset_name: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """Load preset from file"""
        if not preset_name:
            return None, "No preset selected"
        
        preset_path = os.path.join(self.preset_dir, f"{preset_name}.json")
        if not os.path.exists(preset_path):
            return None, f"Preset '{preset_name}' not found"
        
        try:
            with open(preset_path, 'r', encoding='utf-8') as f:
                preset_data = json.load(f)
            return preset_data, f"Preset '{preset_name}' loaded successfully"
        except Exception as e:
            return None, f"Error loading preset: {str(e)}"
    
    def get_last_used_preset(self) -> Optional[str]:
        """Get the name of the last used preset"""
        last_preset_path = os.path.join(self.preset_dir, '_last_used.txt')
        if os.path.exists(last_preset_path):
            try:
                with open(last_preset_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except:
                pass
        return None
    
    def set_last_used_preset(self, preset_name: str):
        """Save the name of the last used preset"""
        last_preset_path = os.path.join(self.preset_dir, '_last_used.txt')
        try:
            with open(last_preset_path, 'w', encoding='utf-8') as f:
                f.write(preset_name)
        except:
            pass
    
    def apply_preset_to_ui(self, preset_data: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Convert preset data to UI values, handling missing keys gracefully"""
        # Merge preset data with defaults
        result = defaults.copy()
        if preset_data:
            for key, value in preset_data.items():
                result[key] = value
        return result

# Keep old functions for compatibility
def get_preset_list(preset_dir):
    """Get list of available presets"""
    if not os.path.exists(preset_dir):
        return []
    presets = [f[:-5] for f in os.listdir(preset_dir) if f.endswith('.json')]
    return sorted(presets)

def save_preset(preset_dir, preset_name, preset_data):
    """Save preset to file"""
    if not preset_name:
        return False, "Please enter a preset name"
    
    preset_path = os.path.join(preset_dir, f"{preset_name}.json")
    try:
        with open(preset_path, 'w', encoding='utf-8') as f:
            json.dump(preset_data, f, indent=2, ensure_ascii=False)
        return True, f"Preset '{preset_name}' saved successfully"
    except Exception as e:
        return False, f"Error saving preset: {str(e)}"

def load_preset(preset_dir, preset_name):
    """Load preset from file"""
    if not preset_name:
        return None, "No preset selected"
    
    preset_path = os.path.join(preset_dir, f"{preset_name}.json")
    if not os.path.exists(preset_path):
        return None, f"Preset '{preset_name}' not found"
    
    try:
        with open(preset_path, 'r', encoding='utf-8') as f:
            preset_data = json.load(f)
        return preset_data, f"Preset '{preset_name}' loaded successfully"
    except Exception as e:
        return None, f"Error loading preset: {str(e)}"

def get_last_used_preset(preset_dir):
    """Get the name of the last used preset"""
    last_preset_path = os.path.join(preset_dir, '_last_used.txt')
    if os.path.exists(last_preset_path):
        try:
            with open(last_preset_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except:
            pass
    return None

def set_last_used_preset(preset_dir, preset_name):
    """Save the name of the last used preset"""
    last_preset_path = os.path.join(preset_dir, '_last_used.txt')
    try:
        with open(last_preset_path, 'w', encoding='utf-8') as f:
            f.write(preset_name)
    except:
        pass