import os
import gc
import sys

import torch

import json
import numpy as np
from omegaconf import OmegaConf

from codeclm.trainer.codec_song_pl import CodecLM_PL
from codeclm.models import CodecLM
from codeclm.models import builders

from separator import Separator
from codeclm.utils.offload_profiler import OffloadProfiler, OffloadParamParse

# Import suppression utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.suppress_output import suppress_output, disable_verbose_logging


class LeVoInference(torch.nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        
        # Disable verbose logging at initialization
        disable_verbose_logging()

        torch.backends.cudnn.enabled = False 
        # Use register_resolver for newer OmegaConf versions
        if hasattr(OmegaConf, 'register_new_resolver'):
            register_method = OmegaConf.register_new_resolver
        else:
            register_method = OmegaConf.register_resolver
            
        register_method("eval", lambda x: eval(x))
        register_method("concat", lambda *x: [xxx for xx in x for xxx in xx])
        register_method("get_fname", lambda: 'default')
        register_method("load_yaml", lambda x: list(OmegaConf.load(x)))

        cfg_path = os.path.join(ckpt_path, 'config.yaml')
        self.pt_path = os.path.join(ckpt_path, 'model.pt')

        self.cfg = OmegaConf.load(cfg_path)
        self.cfg.mode = 'inference'
        self.max_duration = self.cfg.max_dur

        self.default_params = dict(
            top_p = 0.0,
            record_tokens = True,
            record_window = 50,
            extend_stride = 5,
            duration = self.max_duration,
        )


    def forward(self, lyric: str, description: str = None, prompt_audio_path: os.PathLike = None, genre: str = None, auto_prompt_path: os.PathLike = None, gen_type: str = "mixed", params = dict(), 
                disable_offload=False, disable_cache_clear=False, disable_fp16=False, disable_sequential=False,
                progress_callback=None, cancellation_token=None):
        
        # Check cancellation at start
        if cancellation_token and hasattr(cancellation_token, 'is_cancelled') and cancellation_token.is_cancelled():
            return None
            
        # Update progress
        if progress_callback:
            progress_callback({'phase': 'Loading models', 'message': 'Initializing audio tokenizers...'})
        
        if prompt_audio_path is not None and os.path.exists(prompt_audio_path):
            try:
                if progress_callback:
                    progress_callback({'phase': 'Processing audio', 'message': 'Loading vocal separation model...'})
                    
                with suppress_output():
                    separator = Separator()
                    
                if progress_callback:
                    progress_callback({'phase': 'Processing audio', 'message': 'Loading audio tokenizer (1/2)...'})
                    
                with suppress_output():
                    audio_tokenizer = builders.get_audio_tokenizer_model(self.cfg.audio_tokenizer_checkpoint, self.cfg)
                    audio_tokenizer = audio_tokenizer.eval().cuda()
                    
                if progress_callback:
                    progress_callback({'phase': 'Processing audio', 'message': 'Separating vocals from background...'})
                    
                pmt_wav, vocal_wav, bgm_wav = separator.run(prompt_audio_path)
                pmt_wav = pmt_wav.cuda()
                vocal_wav = vocal_wav.cuda()
                bgm_wav = bgm_wav.cuda()
                
                if progress_callback:
                    progress_callback({'phase': 'Processing audio', 'message': 'Encoding reference audio...'})
                    
                with torch.no_grad():
                    pmt_wav, _ = audio_tokenizer.encode(pmt_wav)
                    
                del audio_tokenizer
                del separator
                if not disable_cache_clear:
                    torch.cuda.empty_cache()

                if progress_callback:
                    progress_callback({'phase': 'Processing audio', 'message': 'Loading audio tokenizer (2/2)...'})
                    
                with suppress_output():
                    seperate_tokenizer = builders.get_audio_tokenizer_model(self.cfg.audio_tokenizer_checkpoint_sep, self.cfg)
                    seperate_tokenizer = seperate_tokenizer.eval().cuda()
                    
                if progress_callback:
                    progress_callback({'phase': 'Processing audio', 'message': 'Encoding vocal and background separately...'})
                    
                with torch.no_grad():
                    vocal_wav, bgm_wav = seperate_tokenizer.encode(vocal_wav, bgm_wav)
                    
                del seperate_tokenizer
                melody_is_wav = False
                if not disable_cache_clear:
                    torch.cuda.empty_cache()
                    
                if progress_callback:
                    progress_callback({'phase': 'Processing audio', 'message': 'Reference audio processing complete'})
                    
            except Exception as e:
                if progress_callback:
                    progress_callback({'phase': 'Error', 'message': f'Audio processing failed: {str(e)}'})
                raise
        elif genre is not None and auto_prompt_path is not None:
            with suppress_output():
                auto_prompt = torch.load(auto_prompt_path)
            merge_prompt = [item for sublist in auto_prompt.values() for item in sublist]
            if genre == "Auto": 
                prompt_token = merge_prompt[np.random.randint(0, len(merge_prompt))]
            else:
                prompt_token = auto_prompt[genre][np.random.randint(0, len(auto_prompt[genre]))]
            pmt_wav = prompt_token[:,[0],:]
            vocal_wav = prompt_token[:,[1],:]
            bgm_wav = prompt_token[:,[2],:]
            melody_is_wav = False
        else:
            pmt_wav = None
            vocal_wav = None
            bgm_wav = None
            melody_is_wav = True

        # Check cancellation before loading main model
        if cancellation_token and hasattr(cancellation_token, 'is_cancelled') and cancellation_token.is_cancelled():
            return None
            
        # Update progress
        if progress_callback:
            progress_callback({'phase': 'Loading models', 'message': 'Loading language model...'})

        with suppress_output():
            audiolm = builders.get_lm_model(self.cfg)
            checkpoint = torch.load(self.pt_path, map_location='cpu')
            audiolm_state_dict = {k.replace('audiolm.', ''): v for k, v in checkpoint.items() if k.startswith('audiolm')}
            audiolm.load_state_dict(audiolm_state_dict, strict=False)
            audiolm = audiolm.eval()

        offload_audiolm = False if disable_offload else (True if 'offload' in self.cfg.keys() and 'audiolm' in self.cfg.offload else False)
        if offload_audiolm:
            with suppress_output():
                audiolm_offload_param = OffloadParamParse.parse_config(audiolm, self.cfg.offload.audiolm)
                audiolm_offload_param.show()
                offload_profiler = OffloadProfiler(device_index=0, **(audiolm_offload_param.init_param_dict()))
                offload_profiler.offload_layer(**(audiolm_offload_param.offload_layer_param_dict()))
                offload_profiler.clean_cache_wrapper(**(audiolm_offload_param.clean_cache_param_dict()))
        else:
            # Always use float16 for the model to maintain compatibility
            audiolm = audiolm.cuda().to(torch.float16)

        model = CodecLM(name = "tmp",
            lm = audiolm,
            audiotokenizer = None,
            max_duration = self.max_duration,
            seperate_tokenizer = None,
        )
        # Extract parameters that are not for set_generation_params
        num_steps = params.pop('num_steps', 50)
        guidance_scale = params.pop('guidance_scale', 1.5)
        chunked = params.pop('chunked', True)
        chunk_size = params.pop('chunk_size', 128)
        # Merge with defaults and set generation params
        params = {**self.default_params, **params}
        model.set_generation_params(**params)

        generate_inp = {
            'lyrics': [lyric.replace("  ", " ")],
            'descriptions': [description],
            'melody_wavs': pmt_wav,
            'vocal_wavs': vocal_wav,
            'bgm_wavs': bgm_wav,
            'melody_is_wav': melody_is_wav,
        }

        # Check cancellation before generation
        if cancellation_token and hasattr(cancellation_token, 'is_cancelled') and cancellation_token.is_cancelled():
            return None
            
        # Update progress
        if progress_callback:
            progress_callback({'phase': 'Generating', 'message': 'Generating audio tokens...'})

        if disable_fp16:
            with torch.no_grad():
                with suppress_output():
                    tokens = model.generate(**generate_inp, return_tokens=True)
                    if offload_audiolm:
                        offload_profiler.reset_empty_cache_mem_line()
        else:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                with torch.no_grad():
                    with suppress_output():
                        tokens = model.generate(**generate_inp, return_tokens=True)
                        if offload_audiolm:
                            offload_profiler.reset_empty_cache_mem_line()
        
        if offload_audiolm:
            offload_profiler.stop()
            del offload_profiler
            del audiolm_offload_param
        del model
        if not disable_sequential:
            audiolm = audiolm.cpu()
            del audiolm
            del checkpoint
            gc.collect()
            if not disable_cache_clear:
                torch.cuda.empty_cache()

        # Check cancellation before diffusion
        if cancellation_token and hasattr(cancellation_token, 'is_cancelled') and cancellation_token.is_cancelled():
            return None
            
        # Update progress
        if progress_callback:
            progress_callback({'phase': 'Diffusion', 'message': f'Running {num_steps} diffusion steps...'})

        with suppress_output():
            if disable_sequential:
                seperate_tokenizer = builders.get_audio_tokenizer_model(self.cfg.audio_tokenizer_checkpoint_sep, self.cfg)
            else:
                seperate_tokenizer = builders.get_audio_tokenizer_model_cpu(self.cfg.audio_tokenizer_checkpoint_sep, self.cfg)
            device = "cuda:0"
            seperate_tokenizer.model.device = device
            seperate_tokenizer.model.vae = seperate_tokenizer.model.vae.to(device)
            seperate_tokenizer.model.model.device = torch.device(device)
            seperate_tokenizer = seperate_tokenizer.eval()

        offload_wav_tokenizer_diffusion = False if disable_offload else (True if 'offload' in self.cfg.keys() and 'wav_tokenizer_diffusion' in self.cfg.offload else False)
        if offload_wav_tokenizer_diffusion:
            with suppress_output():
                sep_offload_param = OffloadParamParse.parse_config(seperate_tokenizer, self.cfg.offload.wav_tokenizer_diffusion)
                sep_offload_param.show()
                sep_offload_profiler = OffloadProfiler(device_index=0, **(sep_offload_param.init_param_dict()))
                sep_offload_profiler.offload_layer(**(sep_offload_param.offload_layer_param_dict()))
                sep_offload_profiler.clean_cache_wrapper(**(sep_offload_param.clean_cache_param_dict()))
        else:
            seperate_tokenizer.model.model = seperate_tokenizer.model.model.to(device)

        model = CodecLM(name = "tmp",
            lm = None,
            audiotokenizer = None,
            max_duration = self.max_duration,
            seperate_tokenizer = seperate_tokenizer,
        )

        with torch.no_grad():
            with suppress_output():
                if melody_is_wav:
                    wav_seperate = model.generate_audio(tokens, pmt_wav, vocal_wav, bgm_wav, gen_type=gen_type, chunked=chunked, chunk_size=chunk_size, num_steps=num_steps, guidance_scale=guidance_scale)
                else:
                    wav_seperate = model.generate_audio(tokens, gen_type=gen_type, chunked=chunked, chunk_size=chunk_size, num_steps=num_steps, guidance_scale=guidance_scale)

        if offload_wav_tokenizer_diffusion:
            sep_offload_profiler.reset_empty_cache_mem_line()
            sep_offload_profiler.stop()
        if not disable_cache_clear:
            torch.cuda.empty_cache()

        # Update progress
        if progress_callback:
            progress_callback({'phase': 'Complete', 'message': 'Audio generation complete'})

        return wav_seperate[0]
