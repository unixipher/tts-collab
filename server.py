import numpy as np
import torch
from flask import Flask, request, jsonify, send_file
from io import BytesIO
import os

# --- Imports for Hugging Face TTS ---
try:
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
except ImportError:
    print("ERROR: Missing dependencies for Hugging Face TTS.")
    print('Run: pip install "parler-tts @ git+https://github.com/huggingface/parler-tts.git" transformers torch numpy')
    raise

# --- PERFORMANCE OPTIMIZATIONS ---
# Set PyTorch to use all CPU cores
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(os.cpu_count())
print(f"PyTorch configured to use {os.cpu_count()} CPU threads")

# --- 👇 DEFINE YOUR MODEL PATH HERE ---
# Using AI4Bharat's Indic Parler TTS model (supports Indian languages)
# Change to local path like "./tts" if you have the model downloaded locally
MODEL_PATH = "ai4bharat/indic-parler-tts"

print(f"Loading Hugging Face TTS model from: {MODEL_PATH}")
_device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {_device}")

# --- LOAD MODEL ---
_model = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Use FP16 on GPU for speed
    low_cpu_mem_usage=True,  # Optimize memory usage
).to(_device)

# Enable optimizations
if torch.cuda.is_available():
    _model = _model.half()  # FP16 for faster GPU inference
    torch.backends.cudnn.benchmark = True  # Auto-tune cuDNN kernels
    print("GPU optimizations enabled: FP16 precision, cuDNN auto-tuning")
else:
    # CPU optimizations
    torch.set_float32_matmul_precision('high')
    print("CPU optimizations enabled")

# Enable inference mode optimizations
_model.eval()
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 on Ampere GPUs
torch.backends.cudnn.allow_tf32 = True

_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)
# ---
_sample_rate = _model.config.sampling_rate
print(f"Hugging Face TTS model loaded. Sample rate: {_sample_rate}")


# --- Create the Flask App ---
app = Flask(__name__)

@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.get_json()
    text = data.get('text')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    print(f"Synthesizing: {text}")

    try:
        # Use inference mode for maximum speed
        with torch.inference_mode():
            # 1. Define prompt and description
            prompt = text
            # Default description for Indic Parler TTS - customize for different voices/languages
            description = data.get('description', "A clear, natural voice with moderate pace and good pronunciation.")

            # 2. Tokenize inputs
            input_ids = _tokenizer(
                description, return_tensors="pt"
            ).input_ids.to(_device)
            prompt_input_ids = _tokenizer(
                prompt, return_tensors="pt"
            ).input_ids.to(_device)

            # 3. Generate audio with optimizations
            generation = _model.generate(
                input_ids=input_ids, 
                prompt_input_ids=prompt_input_ids,
                do_sample=True,  # Enable sampling for better quality
                temperature=1.0,  # Control randomness
                use_cache=True,  # Use KV-cache for faster generation
            )
            
            # 4. Get NumPy array from tensor
            audio_arr = generation.cpu().numpy().squeeze()
        
        # 5. Normalize from float (-1.0 to 1.0) to int16 PCM (outside inference_mode for NumPy ops)
        if np.max(np.abs(audio_arr)) == 0:
            scaled_audio = np.zeros_like(audio_arr, dtype=np.int16)
        else:
            scaled_audio = (
                (audio_arr / np.max(np.abs(audio_arr))) * 32767
            ).astype(np.int16)

        # 6. Convert int16 NumPy array to raw bytes
        pcm_data = scaled_audio.tobytes()
        
        # 7. Return the raw PCM data
        return send_file(
            BytesIO(pcm_data),
            mimetype='application/octet-stream'
        )

    except Exception as e:
        print(f"Error during synthesis: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Returns the sample rate so the client knows what to expect
    @app.route('/info', methods=['GET'])
    def info():
        return jsonify({"sample_rate": _sample_rate, "channels": 1})
    
    print("Starting Flask server on http://127.0.0.1:5001")
    print("Performance tips:")
    print("  - For production, use: gunicorn -w 4 -b 127.0.0.1:5001 server:app")
    print("  - Or use waitress: pip install waitress && waitress-serve --port=5001 server:app")
    app.run(port=5001, threaded=True)  # Enable threading for concurrent requests