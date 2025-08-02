"""
Wrapper for tqdm to capture progress in Gradio
"""
import sys
from typing import Optional, Callable, Any

# Store the original tqdm before we modify it
_original_tqdm = None
_progress_callback = None

class TqdmWrapper:
    """Wrapper class that mimics tqdm but sends progress to callback"""
    
    def __init__(self, iterable=None, desc=None, total=None, leave=True, 
                 file=None, ncols=None, mininterval=0.1, maxinterval=10.0, 
                 miniters=None, ascii=None, disable=False, unit='it', 
                 unit_scale=False, dynamic_ncols=False, smoothing=0.3, 
                 bar_format=None, initial=0, position=None, postfix=None, 
                 unit_divisor=1000, write_bytes=None, lock_args=None, 
                 nrows=None, colour=None, delay=0, gui=False, **kwargs):
        
        self.iterable = iterable
        self.desc = desc
        self.total = total if total is not None else (len(iterable) if hasattr(iterable, '__len__') else None)
        self.n = initial
        self.disable = disable
        self.unit = unit
        self.leave = leave
        
        # Use original tqdm if available
        global _original_tqdm
        if _original_tqdm is not None:
            self._tqdm = _original_tqdm(
                iterable, desc=desc, total=total, leave=leave,
                file=file, ncols=ncols, mininterval=mininterval,
                maxinterval=maxinterval, miniters=miniters, ascii=ascii,
                disable=disable, unit=unit, unit_scale=unit_scale,
                dynamic_ncols=dynamic_ncols, smoothing=smoothing,
                bar_format=bar_format, initial=initial, position=position,
                postfix=postfix, unit_divisor=unit_divisor,
                write_bytes=write_bytes, lock_args=lock_args,
                nrows=nrows, colour=colour, delay=delay, gui=gui, **kwargs
            )
        else:
            self._tqdm = None
    
    def __iter__(self):
        if self._tqdm is not None:
            # Iterate through the original tqdm
            for item in self._tqdm:
                # Send progress update to callback
                self._send_progress_update()
                yield item
        else:
            # Fallback if no original tqdm
            if self.iterable is not None:
                for i, item in enumerate(self.iterable):
                    self.n = i + 1
                    self._send_progress_update()
                    yield item
    
    def __enter__(self):
        if self._tqdm is not None:
            self._tqdm.__enter__()
        return self
    
    def __exit__(self, *args):
        if self._tqdm is not None:
            self._tqdm.__exit__(*args)
    
    def update(self, n=1):
        if self._tqdm is not None:
            self._tqdm.update(n)
        else:
            self.n += n
        self._send_progress_update()
    
    def close(self):
        if self._tqdm is not None:
            self._tqdm.close()
    
    def _send_progress_update(self):
        """Send progress update to callback"""
        global _progress_callback
        if _progress_callback and self.total:
            # Get current values from wrapped tqdm or self
            if self._tqdm is not None and hasattr(self._tqdm, 'n'):
                current = self._tqdm.n
                total = self._tqdm.total
                elapsed = getattr(self._tqdm, 'elapsed', 0)
                rate = getattr(self._tqdm, 'rate', 0)
            else:
                current = self.n
                total = self.total
                elapsed = 0
                rate = 0
            
            if total and total > 0:
                progress = current / total
                percent = int(progress * 100)
                
                # Create progress bar visualization
                bar_width = 30
                filled = int(bar_width * progress)
                bar = '█' * filled + '░' * (bar_width - filled)
                
                # Format the detailed status
                if rate and rate > 0:
                    speed_str = f"{rate:.2f}it/s"
                else:
                    speed_str = "calculating..."
                
                detailed_status = f"{percent:3d}%|{bar}| {current}/{total} [{speed_str}]"
                
                # Send to callback
                progress_info = {
                    'progress': progress,
                    'current_step': current,
                    'total_steps': total,
                    'detailed_status': detailed_status,
                    'speed': speed_str if rate else None
                }
                
                # Determine phase based on total steps
                if total > 1000:
                    progress_info['phase'] = 'Generating tokens'
                    progress_info['token_progress'] = f"{current}/{total}"
                elif total <= 100:
                    progress_info['phase'] = 'Diffusion'
                    progress_info['diffusion_progress'] = f"{current}/{total}"
                else:
                    progress_info['phase'] = 'Processing'
                
                _progress_callback(progress_info)


def install_tqdm_wrapper(callback: Optional[Callable] = None):
    """Install the tqdm wrapper to capture progress"""
    global _original_tqdm, _progress_callback
    
    # Import tqdm modules
    try:
        import tqdm
        import tqdm.auto
        from tqdm import tqdm as tqdm_func
        
        # Store original if not already stored
        if _original_tqdm is None:
            _original_tqdm = tqdm_func
        
        # Set callback
        _progress_callback = callback
        
        # Replace tqdm in all modules
        tqdm.tqdm = TqdmWrapper
        tqdm.auto.tqdm = TqdmWrapper
        sys.modules['tqdm'].tqdm = TqdmWrapper
        
        # Also try to replace in commonly imported ways
        if 'tqdm.tqdm' in sys.modules:
            sys.modules['tqdm.tqdm'] = TqdmWrapper
        
        return True
    except Exception as e:
        print(f"Failed to install tqdm wrapper: {e}")
        return False


def uninstall_tqdm_wrapper():
    """Restore original tqdm"""
    global _original_tqdm, _progress_callback
    
    if _original_tqdm is not None:
        try:
            import tqdm
            import tqdm.auto
            
            tqdm.tqdm = _original_tqdm
            tqdm.auto.tqdm = _original_tqdm
            sys.modules['tqdm'].tqdm = _original_tqdm
            
            if 'tqdm.tqdm' in sys.modules:
                sys.modules['tqdm.tqdm'] = _original_tqdm
                
        except Exception as e:
            print(f"Failed to restore tqdm: {e}")
    
    _progress_callback = None