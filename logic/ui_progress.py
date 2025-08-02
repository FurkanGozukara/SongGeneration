import time
import gradio as gr
from typing import Optional, Callable, Dict, Any

class GradioProgressTracker:
    """Handles progress updates for Gradio UI"""
    
    def __init__(self):
        self.current_progress = 0
        self.total_steps = 100
        self.message = ""
        self.phase = ""
        self.start_time = None
        self.batch_info = ""
        self.preset_info = ""
        self.generation_info = ""
        self.diffusion_progress = ""
        self.token_progress = ""
        self.detailed_status = ""
        
    def start(self, total_steps=100):
        """Start tracking progress"""
        self.total_steps = total_steps
        self.current_progress = 0
        self.start_time = time.time()
        
    def update(self, progress_info: Dict[str, Any]):
        """Update progress from progress info dictionary"""
        if 'progress' in progress_info:
            self.current_progress = int(progress_info['progress'] * 100)
        if 'message' in progress_info:
            self.message = progress_info['message']
        if 'phase' in progress_info:
            self.phase = progress_info['phase']
        if 'batch_info' in progress_info:
            self.batch_info = progress_info['batch_info']
        if 'preset_info' in progress_info:
            self.preset_info = progress_info['preset_info']
        if 'generation_info' in progress_info:
            self.generation_info = progress_info['generation_info']
        if 'diffusion_progress' in progress_info:
            self.diffusion_progress = progress_info['diffusion_progress']
        if 'token_progress' in progress_info:
            self.token_progress = progress_info['token_progress']
        if 'detailed_status' in progress_info:
            self.detailed_status = progress_info['detailed_status']
            
    def get_progress_text(self) -> str:
        """Get formatted progress text for display"""
        parts = []
        
        # Phase and message
        if self.phase:
            parts.append(f"{self.phase}")
        if self.message:
            parts.append(f"{self.message}")
            
        # Batch/Preset/Generation info
        if self.batch_info:
            parts.append(f"Batch: {self.batch_info}")
        if self.preset_info:
            parts.append(f"Preset: {self.preset_info}")
        if self.generation_info:
            parts.append(f"Generation: {self.generation_info}")
            
        # Detailed progress bars
        if self.diffusion_progress:
            parts.append(f"Diffusion: {self.diffusion_progress}")
        if self.token_progress:
            parts.append(f"Tokens: {self.token_progress}")
            
        # Time elapsed
        if self.start_time:
            elapsed = time.time() - self.start_time
            parts.append(f"Time: {elapsed:.1f}s")
            
        # Add detailed status if available
        if self.detailed_status:
            return f"{' | '.join(parts)}\n{self.detailed_status}"
            
        return " | ".join(parts) if parts else "Processing..."
        
    def get_progress_bar_text(self) -> str:
        """Get progress bar text (shows percentage)"""
        return f"{self.current_progress}%"

def create_progress_callback(progress_tracker: GradioProgressTracker, progress_bar: gr.Progress = None):
    """Create a progress callback function for the model"""
    def callback(progress_info: Dict[str, Any]):
        # Update tracker
        progress_tracker.update(progress_info)
        
        # Update Gradio progress bar if available
        if progress_bar:
            progress = progress_info.get('progress', 0)
            message = progress_tracker.get_progress_text()
            progress_bar(progress, desc=message)
            
        # Also update via gr.Info for visibility
        if 'message' in progress_info and progress_info['message']:
            gr.Info(progress_info['message'])
            
        # Print to console as well
        print(f"Progress: {progress_tracker.get_progress_text()}")
        
    return callback

def format_eta(seconds: float) -> str:
    """Format ETA in human readable format"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """Create a text-based progress bar"""
    if total <= 0:
        return ""
    
    progress = min(current / total, 1.0)
    filled_width = int(width * progress)
    bar = "█" * filled_width + "▒" * (width - filled_width)
    percentage = int(progress * 100)
    
    return f"{percentage:3d}%|{bar}| {current}/{total}"