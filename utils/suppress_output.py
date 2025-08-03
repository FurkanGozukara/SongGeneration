import os
import sys
import contextlib
import logging
import warnings
import re
from io import StringIO

@contextlib.contextmanager
def suppress_output(suppress_stdout=True, suppress_stderr=True, suppress_warnings=True, allow_progress=True):
    """
    Context manager to suppress stdout, stderr, and warnings.
    
    Args:
        suppress_stdout: Whether to suppress stdout
        suppress_stderr: Whether to suppress stderr  
        suppress_warnings: Whether to suppress warnings
        allow_progress: Whether to allow progress output through (for gradio)
    """
    # Save original streams
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_warning_filters = warnings.filters[:]
    
    # Save original logging levels
    loggers_to_silence = [
        'fairseq',
        'mert_fairseq', 
        'transformers',
        'torch',
        'xformers',
        'stable_audio_tools',
        'codeclm'
    ]
    
    original_levels = {}
    for logger_name in loggers_to_silence:
        logger = logging.getLogger(logger_name)
        original_levels[logger_name] = logger.level
        logger.setLevel(logging.ERROR)
    
    try:
        # Create custom stream that filters out specific messages
        class FilteredStream:
            def __init__(self, original_stream):
                self.original_stream = original_stream
                self.filtered_patterns = [
                    r'all structure tokens:',
                    r'LlamaForCausalLM has generative capabilities',
                    r'CausalLM has generative capabilities',
                    r'\[OffloadParam\]',
                    r'pin:True',
                    r'conditions \[ConditioningAttributes',
                    r'test\.c',
                    r'LINK : fatal error',
                    r'You are using an old version of the checkpointing format',
                    r'NOTE: Redirects are currently not supported',
                    r'fairseq\.tasks\.text_to_speech',
                    r'xformers\.components is deprecated',
                    r'torch\.cuda\.amp\.autocast\(args\.\.\.\) is deprecated',
                    r'torch\.nn\.utils\.weight_norm.*is deprecated',
                    r'offload_module:',
                    r'cpu_mem_gb:',
                    r'pre_copy_step:',
                    r'clean_cache_after_forward:',
                    r'dtype:',
                    r'offload_layer_dict:',
                    r'ignore_layer_list:',
                    r'clean_cache_wrapper:',
                    r'debug:',
                    r'class:',
                    r'value:',
                    r'--- Nested model:',
                    r'Please install tensorboardX',
                ]
                
            def write(self, text):
                # Check if the text matches any filtered pattern
                for pattern in self.filtered_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        return  # Suppress this output
                        
                # Allow progress-related output if enabled
                if allow_progress and any(keyword in text for keyword in ['progress', '%|', 'it/s']):
                    self.original_stream.write(text)
                elif not allow_progress and suppress_stdout:
                    return  # Suppress all stdout if requested
                else:
                    self.original_stream.write(text)
                    
            def flush(self):
                self.original_stream.flush()
                
            def __getattr__(self, name):
                return getattr(self.original_stream, name)
        
        # Use filtered streams
        if suppress_stdout:
            sys.stdout = FilteredStream(old_stdout)
        if suppress_stderr:
            sys.stderr = FilteredStream(old_stderr)
        if suppress_warnings:
            warnings.simplefilter("ignore")
            
        yield
        
    finally:
        # Restore original streams
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        # Restore warning filters
        warnings.filters[:] = old_warning_filters
        
        # Restore logging levels
        for logger_name, level in original_levels.items():
            logging.getLogger(logger_name).setLevel(level)


@contextlib.contextmanager
def quiet_mode():
    """Simple wrapper for suppressing all output"""
    with suppress_output(suppress_stdout=True, suppress_stderr=True, suppress_warnings=True):
        yield


def disable_verbose_logging():
    """
    Disable verbose logging from various libraries used in song generation.
    Call this at the start of your script.
    """
    # Disable transformers warnings
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Disable torch warnings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # Set logging levels
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("fairseq").setLevel(logging.ERROR)
    logging.getLogger("mert_fairseq").setLevel(logging.ERROR)
    logging.getLogger("xformers").setLevel(logging.ERROR)
    logging.getLogger("stable_audio_tools").setLevel(logging.ERROR)
    
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*weight_norm.*")
    warnings.filterwarnings("ignore", message=".*autocast.*")
    warnings.filterwarnings("ignore", message=".*GenerationMixin.*")