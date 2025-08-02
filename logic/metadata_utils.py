def save_metadata(file_path, metadata):
    """Save metadata to text file alongside audio file"""
    metadata_path = file_path.rsplit('.', 1)[0] + '.txt'
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write("=== SONG GENERATION METADATA ===\n")
            f.write(f"Generated: {metadata.get('timestamp', 'Unknown')}\n")
            f.write(f"Model: {metadata.get('model', 'Unknown')}\n\n")
            
            f.write("=== USER INPUTS ===\n")
            f.write(f"Lyrics:\n{metadata.get('lyrics', '')}\n\n")
            
            f.write("=== GENERATION PARAMETERS ===\n")
            for key, value in sorted(metadata.items()):
                if key not in ['timestamp', 'model', 'lyrics', 'output_files']:
                    f.write(f"{key}: {value}\n")
            
            if 'output_files' in metadata:
                f.write("\n=== OUTPUT FILES ===\n")
                for file_type, path in metadata['output_files'].items():
                    if path:
                        f.write(f"{file_type}: {path}\n")
            
            f.write("\n=== END OF METADATA ===\n")
        return True
    except Exception as e:
        print(f"Error saving metadata: {e}")
        return False

def collect_generation_parameters(
    lyrics, genre, instrument, emotion, timbre, gender,
    sample_prompt, audio_path, image_path, save_mp3, seed,
    max_gen_length, diffusion_steps, temperature, top_k, top_p,
    cfg_coef, guidance_scale, use_sampling, extend_stride,
    gen_type, chunked, chunk_size, record_tokens, record_window,
    disable_offload, disable_cache_clear, disable_fp16, disable_sequential
):
    """Collect all current parameters into a dictionary"""
    return {
        'lyrics': lyrics,
        'genre': genre,
        'instrument': instrument,
        'emotion': emotion,
        'timbre': timbre,
        'gender': gender,
        'sample_prompt': sample_prompt,
        'audio_path': audio_path,
        'image_path': image_path,
        'save_mp3': save_mp3,
        'seed': seed,
        'max_gen_length': max_gen_length,
        'diffusion_steps': diffusion_steps,
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p,
        'cfg_coef': cfg_coef,
        'guidance_scale': guidance_scale,
        'use_sampling': use_sampling,
        'extend_stride': extend_stride,
        'gen_type': gen_type,
        'chunked': chunked,
        'chunk_size': chunk_size,
        'record_tokens': record_tokens,
        'record_window': record_window,
        'disable_offload': disable_offload,
        'disable_cache_clear': disable_cache_clear,
        'disable_fp16': disable_fp16,
        'disable_sequential': disable_sequential
    }