import torch
from diffusers import StableDiffusionPipeline
from flask import Flask, render_template, request, jsonify
import os
from PIL import Image
import base64
from io import BytesIO
import threading
import uuid
import logging
import platform
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Global pipeline and state
pipe = None
generation_in_progress = {}
pipeline_loading = False
pipeline_error = None
device_info = {}

def get_system_info():
    """Collect system configuration information"""
    global device_info
    
    device_info = {
        'python_version': platform.python_version(),
        'platform': platform.system(),
        'platform_release': platform.release(),
        'cpu_count': psutil.cpu_count(logical=False),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 'N/A',
        'ram_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'ram_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        device_info['cuda_device_count'] = torch.cuda.device_count()
        device_info['cuda_device_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        device_info['cuda_device_current'] = torch.cuda.get_device_name(0)
        device_info['gpu_memory_total_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    
    return device_info

def load_pipeline():
    """Load the Z-Image-Turbo pipeline"""
    global pipe, pipeline_loading, pipeline_error
    if pipe is not None or pipeline_loading:
        return
    
    pipeline_loading = True
    try:
        logger.info("Loading pipeline...")
        logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        # Try to load ZImagePipeline from custom source or use alternative
        try:
            from diffusers import ZImagePipeline
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Attempting to load ZImagePipeline...")
            pipe = ZImagePipeline.from_pretrained(
                "Tongyi-MAI/Z-Image-Turbo",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
            )
            pipe.to(device)
            logger.info(f"✓ ZImagePipeline loaded successfully!")
        except (ImportError, AttributeError) as e:
            logger.warning(f"ZImagePipeline not available: {str(e)}")
            logger.warning("Falling back to StableDiffusionPipeline...")
            
            # Fallback to a more commonly available pipeline
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            pipe.to(device)
            logger.info(f"✓ StableDiffusionPipeline loaded successfully!")
        
        logger.info(f"Pipeline ready on {device.upper()}")
        pipeline_loading = False
    except Exception as e:
        logger.error(f"Failed to load pipeline: {str(e)}")
        pipeline_error = str(e)
        pipeline_loading = False

def generate_image_thread(prompt, height, width, num_steps, seed, session_id):
    """Generate image in background thread"""
    try:
        generation_in_progress[session_id] = {
            'status': 'generating',
            'progress': 10
        }
        
        generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))
        
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]
        
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        generation_in_progress[session_id] = {
            'status': 'completed',
            'image': img_str,
            'prompt': prompt
        }
        logger.info(f"Image generation completed for session {session_id}")
    except Exception as e:
        logger.error(f"Image generation error for session {session_id}: {str(e)}")
        generation_in_progress[session_id] = {
            'status': 'error',
            'error': str(e)
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate():
    """API endpoint to generate image"""
    try:
        # Load pipeline on first request if not already loaded
        if pipe is None and not pipeline_loading:
            logger.info("Pipeline not loaded yet, loading now...")
            load_pipeline()
        
        # Check if pipeline is loaded
        if pipe is None:
            if pipeline_error:
                return jsonify({'error': f'Pipeline failed to load: {pipeline_error}'}), 500
            return jsonify({'error': 'Pipeline is loading, please try again in a moment'}), 503
        
        data = request.json
        prompt = data.get('prompt', '')
        height = int(data.get('height', 128))
        width = int(data.get('width', 128))
        num_steps = int(data.get('num_steps', 6))
        seed = int(data.get('seed', 42))
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        # Validate parameters
        if height not in [128, 256]:
            height = 128
        if width not in [128, 256]:
            width = 128
        if num_steps < 1 or num_steps > 20:
            num_steps = 6
        
        session_id = str(uuid.uuid4())
        
        # Start generation in background thread
        thread = threading.Thread(
            target=generate_image_thread,
            args=(prompt, height, width, num_steps, seed, session_id)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({'session_id': session_id})
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<session_id>', methods=['GET'])
def status(session_id):
    """Check generation status"""
    if session_id not in generation_in_progress:
        return jsonify({'status': 'not_found'}), 404
    
    return jsonify(generation_in_progress[session_id])

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok' if pipe is not None else 'loading' if pipeline_loading else 'error',
        'pipeline_loaded': pipe is not None,
        'pipeline_loading': pipeline_loading,
        'pipeline_error': pipeline_error,
        'system_info': device_info
    })

if __name__ == '__main__':
    # Collect system info on startup
    get_system_info()
    logger.info(f"System Info: {device_info}")
    logger.info("=" * 60)
    logger.info("Z-Image-Turbo Web Interface Starting")
    logger.info("=" * 60)
    logger.info(f"CUDA Available: {device_info.get('cuda_available', False)}")
    if device_info.get('cuda_available'):
        logger.info(f"GPU: {device_info.get('cuda_device_current', 'Unknown')}")
    logger.info("=" * 60)
    logger.info("Web server will start on http://0.0.0.0:5000")
    logger.info("Model will be loaded on first generation request")
    logger.info("=" * 60)
    
    # Don't load pipeline on startup - load on first request instead
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
