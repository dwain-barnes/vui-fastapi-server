      
# scripts/download_pyannote.py
import os
import sys
from pyannote.audio import Pipeline

token_preview = os.getenv("HUGGING_FACE_HUB_TOKEN", "NOT_SET")[:5]
print(f"Attempting to download pyannote VAD pipeline. Token env var starts with: {token_preview}...")

try:
    print("Debug: Inside Python script for pyannote download...")
    # use_auth_token=True tells pyannote to look for HUGGING_FACE_HUB_TOKEN env var
    pipeline = Pipeline.from_pretrained(
        'pyannote/voice-activity-detection',
        use_auth_token=True
    )
    print('PyAnnote voice-activity-detection pipeline (and its dependencies like segmentation) downloaded successfully.')
except Exception as e:
    print(f'ERROR pre-downloading PyAnnote VAD pipeline: {e}')
    print('Ensure you have accepted terms for pyannote/voice-activity-detection AND pyannote/segmentation on Hugging Face AND provided a valid HUGGING_FACE_HUB_TOKEN build argument.')
    sys.exit(1) # Exit with error code to fail the build

    