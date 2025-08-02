"""
Global progress state for sharing between threads
"""
import threading
from typing import Optional, Dict, Any

class ProgressState:
    """Thread-safe progress state storage"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.current_progress = {}
        self.last_update_time = 0
        
    def update(self, progress_info: Dict[str, Any]):
        """Update progress state"""
        with self.lock:
            self.current_progress = progress_info.copy()
            import time
            self.last_update_time = time.time()
    
    def get(self) -> Dict[str, Any]:
        """Get current progress state"""
        with self.lock:
            return self.current_progress.copy()
    
    def clear(self):
        """Clear progress state"""
        with self.lock:
            self.current_progress = {}
            self.last_update_time = 0
    
    def get_formatted_text(self) -> str:
        """Get formatted progress text"""
        with self.lock:
            if not self.current_progress:
                return ""
            
            lines = []
            
            # Phase
            if 'phase' in self.current_progress:
                lines.append(f"Phase: {self.current_progress['phase']}")
            
            # Detailed status (progress bar)
            if 'detailed_status' in self.current_progress:
                lines.append(self.current_progress['detailed_status'])
            elif 'current_step' in self.current_progress and 'total_steps' in self.current_progress:
                current = self.current_progress['current_step']
                total = self.current_progress['total_steps']
                percent = int((current / total) * 100) if total > 0 else 0
                lines.append(f"Progress: {percent}% ({current}/{total})")
            
            # Speed
            if 'speed' in self.current_progress:
                lines.append(f"Speed: {self.current_progress['speed']}")
            
            # Message
            if 'message' in self.current_progress:
                lines.append(self.current_progress['message'])
            
            return "\n".join(lines)

# Global instance
progress_state = ProgressState()