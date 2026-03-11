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
    console_state = {
        "last_len": 0,
        "completed": False,
    }

    def _format_seconds(seconds: Optional[float]) -> str:
        if seconds is None:
            return "--:--"
        try:
            seconds = max(0.0, float(seconds))
        except (TypeError, ValueError):
            return "--:--"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def _render_console_line(progress_info: Dict[str, Any]) -> str:
        progress = progress_info.get("progress")
        if progress is None:
            progress = progress_tracker.current_progress / 100.0
        progress = max(0.0, min(float(progress), 1.0))
        percent = int(progress * 100)

        phase = progress_info.get("phase") or progress_tracker.phase or "Processing"
        message = progress_info.get("message") or progress_tracker.message or ""

        current = progress_info.get("current_step")
        total = progress_info.get("total_steps")
        step_text = ""
        if current is not None and total:
            step_text = f" | {int(current)}/{int(total)}"

        speed = progress_info.get("speed")
        speed_text = f" | {speed}" if speed else ""

        eta_seconds = progress_info.get("eta_seconds")
        if eta_seconds is None and progress > 0 and progress_tracker.start_time:
            elapsed = max(0.0, time.time() - progress_tracker.start_time)
            eta_seconds = (elapsed * (1.0 - progress)) / progress
        eta_text = f" | ETA {_format_seconds(eta_seconds)}"

        elapsed_seconds = progress_info.get("elapsed_seconds")
        if elapsed_seconds is None and progress_tracker.start_time:
            elapsed_seconds = max(0.0, time.time() - progress_tracker.start_time)
        elapsed_text = f" | Elapsed {_format_seconds(elapsed_seconds)}" if elapsed_seconds is not None else ""

        return f"[{percent:3d}%] {phase}: {message}{step_text}{speed_text}{eta_text}{elapsed_text}"

    def callback(progress_info: Dict[str, Any]):
        # Update tracker
        progress_tracker.update(progress_info)
        
        # Update Gradio progress bar if available
        if progress_bar:
            progress = progress_info.get('progress', 0)
            
            # Build comprehensive message including detailed status
            message_parts = []
            
            # Add phase info
            if progress_info.get('phase'):
                message_parts.append(progress_info['phase'])
                
            # Add detailed progress bar if available
            if progress_info.get('detailed_status'):
                message_parts.append(progress_info['detailed_status'])
            elif progress_info.get('current_step') and progress_info.get('total_steps'):
                # Create our own progress visualization
                current = progress_info['current_step']
                total = progress_info['total_steps']
                percent = int((current / total) * 100)
                bar = create_progress_bar(current, total, width=30)
                message_parts.append(bar)
            
            # Add speed info
            if progress_info.get('speed'):
                message_parts.append(f"Speed: {progress_info['speed']}")
                
            # Add other info from tracker
            tracker_text = progress_tracker.get_progress_text()
            if tracker_text and tracker_text != "Processing...":
                message_parts.append(tracker_text)
            
            # Combine all parts
            message = " | ".join(filter(None, message_parts))
            progress_bar(progress, desc=message)

        # Console rendering: single updating line with percentage + ETA + speed.
        line = _render_console_line(progress_info)
        padded = line
        if console_state["last_len"] > len(line):
            padded += " " * (console_state["last_len"] - len(line))
        print("\r" + padded, end="", flush=True)
        console_state["last_len"] = len(line)

        progress_value = progress_info.get("progress")
        is_complete = False
        try:
            if progress_value is not None and float(progress_value) >= 1.0:
                is_complete = True
        except (TypeError, ValueError):
            is_complete = False

        if progress_info.get("phase") == "Complete":
            is_complete = True

        if is_complete and not console_state["completed"]:
            print("", flush=True)
            console_state["completed"] = True
        elif not is_complete:
            console_state["completed"] = False
        
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
