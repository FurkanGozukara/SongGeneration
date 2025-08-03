import os
import time
from typing import List, Dict, Any, Optional, Callable
import scipy.io.wavfile as wavfile
from datetime import datetime

from .generation import generate_single_song, CancellationToken, ProgressTracker
from .file_utils import (
    save_metadata, convert_wav_to_mp3, create_video_from_image_and_audio,
    scan_folder_for_txt_files, read_prompt_from_txt, check_if_output_exists,
    load_presets_from_folder, find_matching_image
)

class BatchProcessor:
    """Handle batch processing of songs with preset and generation loops"""
    
    def __init__(self, model, app_dir: str, model_cfg):
        self.model = model
        self.app_dir = app_dir
        self.model_cfg = model_cfg
        self.cancellation_token = CancellationToken()
        self.progress_tracker = None
        
    def set_progress_callback(self, callback: Callable):
        """Set callback for progress updates"""
        self.progress_tracker = ProgressTracker(callback)
        
    def cancel(self):
        """Cancel the current batch processing"""
        self.cancellation_token.cancel()
        
    def reset(self):
        """Reset cancellation state"""
        self.cancellation_token.reset()
        
    def process_batch(
        self,
        input_folder: str,
        output_folder: str,
        base_params: Dict[str, Any],
        num_generations: int = 1,
        loop_presets: bool = False,
        skip_existing: bool = False,
        preset_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a batch of txt files with multiple generations and preset loops
        
        Order of loops: batch > preset > number of generations
        """
        results = {
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'cancelled': False,
            'files': []
        }
        
        # Reset cancellation token
        self.reset()
        
        # Scan for txt files
        txt_files = scan_folder_for_txt_files(input_folder)
        if not txt_files:
            return results
        
        # Load presets if needed
        presets = {'Default': base_params}
        if loop_presets and preset_dir:
            loaded_presets = load_presets_from_folder(preset_dir)
            if loaded_presets:
                # When looping presets, use only the loaded presets
                presets = loaded_presets
                # Add Default preset to the beginning if not already present
                if 'Default' not in presets:
                    presets = {'Default': base_params, **presets}
        
        # Calculate total operations
        total_operations = len(txt_files) * len(presets) * num_generations
        
        # Initialize progress
        if self.progress_tracker:
            self.progress_tracker.start(total_operations)
            self.progress_tracker.set_batch_info(0, len(txt_files))
        
        operation_count = 0
        
        # Main batch loop
        for batch_idx, txt_file in enumerate(txt_files):
            if self.cancellation_token.is_cancelled():
                results['cancelled'] = True
                break
                
            # Update batch progress
            if self.progress_tracker:
                self.progress_tracker.set_batch_info(batch_idx + 1, len(txt_files))
            
            # Read lyrics from txt file
            lyrics = read_prompt_from_txt(txt_file)
            if not lyrics:
                results['failed'] += 1
                continue
            
            base_name = os.path.splitext(os.path.basename(txt_file))[0]
            
            # Check for matching image file
            matching_image = find_matching_image(txt_file)
            if matching_image:
                print(f"Found matching image for {base_name}: {os.path.basename(matching_image)}")
            
            # Preset loop
            preset_names = list(presets.keys())
            for preset_idx, preset_name in enumerate(preset_names):
                if self.cancellation_token.is_cancelled():
                    results['cancelled'] = True
                    break
                    
                # Update preset progress
                if self.progress_tracker:
                    self.progress_tracker.set_preset_info(preset_idx + 1, len(presets))
                
                preset_params = presets[preset_name].copy()
                # Always use lyrics from txt file, ignoring preset lyrics
                preset_params['lyrics'] = lyrics
                
                # Debug: Print preset parameters being used
                print(f"\nUsing preset: {preset_name}")
                print(f"  Genre: {preset_params.get('genre', 'N/A')}")
                print(f"  Instrument: {preset_params.get('instrument', 'N/A')}")
                print(f"  Emotion: {preset_params.get('emotion', 'N/A')}")
                print(f"  Timbre: {preset_params.get('timbre', 'N/A')}")
                print(f"  Gender: {preset_params.get('gender', 'N/A')}")
                
                # Use matching image if found and no image specified in preset or params
                if matching_image and not preset_params.get('image_path'):
                    preset_params['image_path'] = matching_image
                
                # Generation loop
                for gen_idx in range(num_generations):
                    if self.cancellation_token.is_cancelled():
                        results['cancelled'] = True
                        break
                        
                    # Update generation progress
                    if self.progress_tracker:
                        self.progress_tracker.set_generation_info(gen_idx + 1, num_generations)
                        
                    operation_count += 1
                    
                    # Construct output filename
                    if len(presets) > 1 and num_generations > 1:
                        output_name = f"{base_name}_{preset_name}_{gen_idx+1:04d}"
                    elif len(presets) > 1:
                        output_name = f"{base_name}_{preset_name}"
                    elif num_generations > 1:
                        output_name = f"{base_name}_{gen_idx+1:04d}"
                    else:
                        output_name = base_name
                    
                    # Check if should skip
                    if skip_existing and check_if_output_exists(
                        output_folder, output_name, 
                        preset_params.get('save_mp3', True),
                        bool(preset_params.get('image_path'))
                    ):
                        results['skipped'] += 1
                        if self.progress_tracker:
                            self.progress_tracker.update(
                                step=operation_count,
                                message=f"Skipped existing: {output_name}"
                            )
                        continue
                    
                    # Update progress message
                    if self.progress_tracker:
                        self.progress_tracker.update(
                            step=operation_count,
                            phase="Generating",
                            message=f"Processing: {output_name}"
                        )
                    
                    # Generate the song
                    try:
                        result = self._generate_and_save_song(
                            preset_params,
                            output_folder,
                            output_name
                        )
                        
                        if result:
                            results['processed'] += 1
                            results['files'].append(result)
                        else:
                            results['failed'] += 1
                            
                    except Exception as e:
                        print(f"Error processing {output_name}: {e}")
                        results['failed'] += 1
        
        return results
    
    def _generate_and_save_song(
        self,
        params: Dict[str, Any],
        output_folder: str,
        base_name: str
    ) -> Optional[Dict[str, Any]]:
        """Generate and save a single song"""
        
        # Add auto_prompt_path if not present
        if 'auto_prompt_path' not in params:
            params['auto_prompt_path'] = os.path.join(self.app_dir, "ckpt/prompt.pt")
        
        # Generate the song
        generation_result = generate_single_song(
            self.model,
            params,
            self.progress_tracker,
            self.cancellation_token
        )
        
        if not generation_result:
            return None
        
        # Save the audio
        os.makedirs(output_folder, exist_ok=True)
        wav_path = os.path.join(output_folder, f"{base_name}.wav")
        wavfile.write(wav_path, self.model_cfg.sample_rate, generation_result['audio_data'])
        print(f"Generated audio saved to: {wav_path}")
        
        result = {
            'wav': wav_path,
            'mp3': None,
            'mp4': None,
            'base_name': base_name
        }
        
        # Convert to MP3 if requested
        if params.get('save_mp3', True):
            mp3_path = os.path.join(output_folder, f"{base_name}.mp3")
            if convert_wav_to_mp3(wav_path, mp3_path):
                result['mp3'] = mp3_path
                print(f"Generated MP3 saved to: {mp3_path}")
        
        # Create video if image provided
        if params.get('image_path'):
            mp4_path = os.path.join(output_folder, f"{base_name}.mp4")
            if create_video_from_image_and_audio(params['image_path'], wav_path, mp4_path):
                result['mp4'] = mp4_path
        
        # Save metadata
        metadata = params.copy()
        metadata['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata['model'] = params.get('model_path', 'Unknown')
        metadata['output_files'] = result
        metadata['used_seed'] = generation_result['used_seed']
        save_metadata(wav_path, metadata)
        
        return result