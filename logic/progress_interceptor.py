import sys
import re
import io
from contextlib import contextmanager
from typing import Optional, Callable, Dict, Any

class ProgressInterceptor:
    """Intercepts tqdm progress output and converts it to progress callbacks"""
    
    def __init__(self, callback: Optional[Callable] = None):
        self.callback = callback
        self.original_stdout = None
        self.original_stderr = None
        self.captured_output = io.StringIO()
        self.last_progress_info = {}
        
    def parse_tqdm_output(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse tqdm progress bar output"""
        # Match pattern like: "10%|███████▋ | 365/3750 [00:10<01:31, 37.11it/s]"
        pattern = r'(\d+)%\|[█▏▎▍▌▋▊▉ ]*\| (\d+)/(\d+) \[(\d+:\d+)<(\d+:\d+), ([\d.]+)it/s\]'
        match = re.search(pattern, text)
        
        if match:
            percent = int(match.group(1))
            current = int(match.group(2))
            total = int(match.group(3))
            elapsed = match.group(4)
            eta = match.group(5)
            speed = float(match.group(6))
            
            return {
                'progress': percent / 100.0,
                'current_step': current,
                'total_steps': total,
                'elapsed': elapsed,
                'eta': eta,
                'speed': f"{speed:.2f}it/s",
                'detailed_status': f"{percent}%|{'█' * (percent // 5)}{'░' * (20 - percent // 5)}| {current}/{total} [{elapsed}<{eta}, {speed:.2f}it/s]"
            }
        
        # Also try simpler pattern without time info
        simple_pattern = r'(\d+)%\|[█▏▎▍▌▋▊▉ ]*\| (\d+)/(\d+)'
        match = re.search(simple_pattern, text)
        if match:
            percent = int(match.group(1))
            current = int(match.group(2))
            total = int(match.group(3))
            
            return {
                'progress': percent / 100.0,
                'current_step': current,
                'total_steps': total,
                'detailed_status': f"{percent}%|{'█' * (percent // 5)}{'░' * (20 - percent // 5)}| {current}/{total}"
            }
        
        return None
    
    def write(self, text):
        """Intercept stdout writes"""
        # Write to original stdout
        if self.original_stdout:
            self.original_stdout.write(text)
            self.original_stdout.flush()
        
        # Parse for progress info
        progress_info = self.parse_tqdm_output(text)
        if progress_info and self.callback:
            # Update with phase information based on the total steps and context
            if progress_info['total_steps'] > 1000:
                progress_info['phase'] = 'Generating tokens'
                progress_info['token_progress'] = f"{progress_info['current_step']}/{progress_info['total_steps']}"
                progress_info['message'] = f"Generating audio tokens: {progress_info['current_step']}/{progress_info['total_steps']}"
            elif progress_info['total_steps'] <= 100:
                progress_info['phase'] = 'Diffusion'
                progress_info['diffusion_progress'] = f"{progress_info['current_step']}/{progress_info['total_steps']}"
                progress_info['message'] = f"Running diffusion step {progress_info['current_step']}/{progress_info['total_steps']}"
            else:
                # Could be encoding or other process
                progress_info['phase'] = 'Processing'
                progress_info['message'] = f"Processing: {progress_info['current_step']}/{progress_info['total_steps']}"
            
            self.last_progress_info = progress_info
            self.callback(progress_info)
        
        # Store output
        self.captured_output.write(text)
    
    def flush(self):
        """Flush stdout"""
        if self.original_stdout:
            self.original_stdout.flush()
    
    def __enter__(self):
        """Start intercepting stdout"""
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop intercepting stdout"""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        return False

@contextmanager
def intercept_progress(callback: Callable):
    """Context manager to intercept progress output"""
    interceptor = ProgressInterceptor(callback)
    try:
        yield interceptor
    finally:
        # Ensure stdout is restored
        if interceptor.original_stdout:
            sys.stdout = interceptor.original_stdout
        if interceptor.original_stderr:
            sys.stderr = interceptor.original_stderr