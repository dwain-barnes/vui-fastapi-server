from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
import io, os
import torch
import torchaudio
# ---- VUI initialisation -------------------------------------------------
from vui.model import Vui
from vui.inference import render                # one-shot helper
DEVICE = "cuda" if torch.cuda.is_available() and os.getenv("USE_GPU") else "cpu"

print(f"Loading VUI model on {DEVICE}...")
model = Vui.from_pretrained().to(DEVICE)

# Try to compile for speed, but fall back gracefully if it fails
try:
    print("Attempting to compile model for speed...")
    model.decoder = torch.compile(model.decoder, fullgraph=True)
    print("Model compilation successful!")
except Exception as e:
    print(f"Model compilation failed (this is ok): {e}")
    print("Continuing without compilation...")

print("Warming up model...")
# Warmup run like in the demo
try:
    warmup_audio = render(model, "Hello, this is a warmup test.", max_secs=5)
    print(f"Warmup complete - generated {warmup_audio.shape}")
except Exception as e:
    print(f"Warmup failed: {e}")
    print("Continuing without warmup...")

print(f"VUI model ready on {DEVICE}")
print(f"Model codec sample rate: {model.codec.config.sample_rate}")

# ---- API schema ---------------------------------------------------------
app = FastAPI(
    title="VUI-OpenAI-TTS",
    version="0.1.0",
    openapi_url="/v1/openapi.json",             # keeps spec path identical to OAI
)

class SpeechReq(BaseModel):
    model: str = "vui"
    input: str                                  # ≤4096 chars (OpenAI constraint)
    voice: str | None = None                    # ignored for now
    response_format: str | None = "wav"         # wav/mp3/opus/flac/aac/pcm
    speed: float | None = 1.0
    stream: bool | None = False

@app.post("/v1/audio/speech")
async def create_speech(req: SpeechReq):
    if not (1 <= len(req.input) <= 4096):
        raise HTTPException(400, "input must be 1-4096 characters")

    print(f"Processing text: '{req.input[:50]}...' (length: {len(req.input)})")
    
    try:
        # Use fast inference with minimal parameters
        print("Calling VUI render function...")
        waveform = render(
            model,
            req.input,
            max_secs=30,
            temperature=0.7,  # Add temperature for consistency
            top_k=100,        # Add top_k for consistency  
        )
        
        print(f"Render successful! Waveform type: {type(waveform)}")
        
        # Handle the waveform based on its type
        if isinstance(waveform, tuple):
            # If it's a tuple, extract the audio part (like in some demos)
            print(f"Waveform is tuple with {len(waveform)} elements")
            for i, element in enumerate(waveform):
                print(f"Element {i}: type={type(element)}, shape={getattr(element, 'shape', 'no shape')}")
            
            # Find the audio tensor (usually the largest tensor)
            audio_tensors = [elem for elem in waveform if torch.is_tensor(elem) and elem.numel() > 100]
            if audio_tensors:
                wav = audio_tensors[0]  # Take the first (and likely only) audio tensor
                print(f"Selected audio tensor from tuple: {wav.shape}")
            else:
                raise Exception("No audio tensor found in tuple result")
        else:
            # If it's directly a tensor
            wav = waveform
            print(f"Direct waveform tensor: {wav.shape}")
        
        # Ensure tensor is on CPU
        if wav.is_cuda:
            wav = wav.cpu()
        
        # Handle tensor dimensions - based on demo code pattern
        if wav.dim() == 3:
            # [batch, channels, samples] -> [channels, samples]
            wav = wav.squeeze(0)
        elif wav.dim() == 1:
            # [samples] -> [1, samples] 
            wav = wav.unsqueeze(0)
        
        print(f"Final audio tensor shape: {wav.shape}")
        
        # Get the actual sample rate from the model
        sample_rate = model.codec.config.sample_rate
        print(f"Using sample rate: {sample_rate}")
        
        # Save to buffer
        buf = io.BytesIO()
        torchaudio.save(buf, wav, sample_rate, format=req.response_format)
        
    except Exception as e:
        print(f"VUI render failed with error: {e}")
        print(f"Error type: {type(e)}")
        
        # If render fails, try some fallback approaches
        try:
            print("Trying render with different parameters...")
            # Try with minimal parameters
            waveform = render(model, req.input)
            
            # Same processing as above
            if isinstance(waveform, tuple):
                audio_tensors = [elem for elem in waveform if torch.is_tensor(elem) and elem.numel() > 100]
                wav = audio_tensors[0] if audio_tensors else waveform[0]
            else:
                wav = waveform
                
            if wav.is_cuda:
                wav = wav.cpu()
                
            if wav.dim() == 3:
                wav = wav.squeeze(0)
            elif wav.dim() == 1:
                wav = wav.unsqueeze(0)
                
            sample_rate = model.codec.config.sample_rate
            buf = io.BytesIO()
            torchaudio.save(buf, wav, sample_rate, format=req.response_format)
            
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            raise HTTPException(500, f"VUI render failed: {str(e)}. Fallback failed: {str(e2)}")
    
    audio = buf.getvalue()
    print(f"Audio generated successfully, size: {len(audio)} bytes")
    
    mime = {
        "wav": "audio/wav", "mp3": "audio/mpeg", "opus": "audio/ogg",
        "flac": "audio/flac", "aac": "audio/aac", "pcm": "audio/L16"
    }.get(req.response_format, "application/octet-stream")

    if req.stream:
        # Chunked transfer = identical to OpenAI's streaming guide
        async def streamer():
            for i in range(0, len(audio), 16384):
                yield audio[i:i+16384]
        return StreamingResponse(streamer(), media_type=mime)
    return Response(content=audio, media_type=mime)