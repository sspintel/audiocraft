import os
from time import perf_counter

import torch
import torchaudio
import torch._dynamo

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

import openvino.frontend.pytorch.torchdynamo.backend

os.environ["PYTORCH_TRACING_MODE"] = "TORCHFX"
os.environ["OPENVINO_DEVICE"] = "CPU"
os.environ["TORCH_LOGS"] = "+dynamo"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"
# os.environ["OPENVINO_TORCH_CACHE_DIR"] = "/home/adl/SSP/ov_blob_caches"
# os.environ["OPENVINO_TORCH_BACKEND_DEVICE"] = "GPU"

torch._dynamo.config.suppress_errors = True

model_name = 'small' # or 'melody'
gen_type = 'unconditional'
model = MusicGen.get_pretrained(model_name, device='cpu')

# torch function to compile
# model._generate_tokens = torch.compile(model._generate_tokens, backend="openvino")

# generate 8 second clips
model.set_generation_params(duration=8)  

if gen_type == "unconditional":
    start = perf_counter()
    wav = model.generate_unconditional(4)    
    end = perf_counter()
    print(f"Warmup Time: {end-start}"),
else:
    # the samples generated are conditioned based on the descriptions below, 
    # For the melody model, it is also conditioned on an input audio sample
    # descriptions = ['happy rock', 'energetic EDM', 'sad jazz']
    descriptions = ['happy rock']

    if model_name == 'melody':
        # generates using the melody from the given audio and the provided descriptions.
        melody, sr = torchaudio.load('./audiocraft/assets/bach.mp3')
        wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)
    else:
        wav = model.generate(descriptions)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx+1}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

# sum_inf_time = 0
# for i in range(2):
#     start = perf_counter()
#     wav = model.generate_unconditional(1)    
#     end = perf_counter()
#     print(f"{i} Time: {end-start}"),
#     sum_inf_time = sum_inf_time + (end - start)
# avg_inf_time = sum_inf_time / 2
# print(f"Time: {avg_inf_time}"),
