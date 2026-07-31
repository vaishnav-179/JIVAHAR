import os
import sys
import time
import logging
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

# 1. Configure paths and logging
# Add parent directory to path to allow importing config and ai_module
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config.settings import settings
from ai_module.facade import JivaharAIFacade
from ai_module.gemma import GemmaConfigurationError, GemmaAPIError

# Initialize logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = Path(tempfile.gettempdir()) / "jivahar_sandbox_uploads"
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)

# Initialize single facade instance
# Try loading it immediately, but keep it none if config is broken during startup
# so the UI can still load and guide the user on fixing it.
facade = None
facade_error = None
try:
    facade = JivaharAIFacade()
except Exception as e:
    facade_error = str(e)
    logger.error(f"Failed to initialize Jivahar AI Facade: {e}")


def get_facade():
    """Returns the Jivahar AIFacade instance, re-trying initialization if it failed earlier."""
    global facade, facade_error
    if facade is None:
        try:
            facade = JivaharAIFacade()
            facade_error = None
        except Exception as e:
            facade_error = str(e)
            logger.error(f"Failed to initialize Jivahar AI Facade on retry: {e}")
            raise e
    return facade


@app.route('/')
def index():
    """Renders the main dashboard page."""
    return render_template('index.html')


@app.route('/api/status', methods=['GET'])
def get_status():
    """Checks the status of the local environments and configuration keys."""
    # Check GEMINI_API_KEY
    api_key_configured = False
    api_key_error_msg = ""
    try:
        api_key = settings.GEMINI_API_KEY
        if api_key and api_key != "your_google_ai_studio_api_key_here" and api_key != "mock_key_for_testing":
            api_key_configured = True
    except Exception as e:
        api_key_error_msg = str(e)

    # Check CNN model weights (best_model.pth)
    cnn_model_loaded = False
    cnn_model_error_msg = ""
    try:
        cnn_path = settings.CNN_MODEL_PATH
        if cnn_path.exists():
            # Test instantiation of FoodClassifier
            f = get_facade()
            if f.food_classifier and f.food_classifier.model is not None:
                cnn_model_loaded = True
        else:
            cnn_model_error_msg = f"best_model.pth not found at: {cnn_path}"
    except Exception as e:
        cnn_model_error_msg = str(e)

    # Check FAISS index status
    faiss_index_exists = False
    faiss_index_loaded = False
    faiss_error_msg = ""
    try:
        index_dir = settings.FAISS_INDEX_PATH
        index_file = index_dir / "index.faiss"
        meta_file = index_dir / "metadata.pkl"
        if index_file.exists() and meta_file.exists():
            faiss_index_exists = True
            # Check if vector store can load it
            f = get_facade()
            if f.safety_advisor.vector_store.index is not None or f.safety_advisor.vector_store.load_index():
                faiss_index_loaded = True
        else:
            faiss_error_msg = f"FAISS index files not found in: {index_dir}"
    except Exception as e:
        faiss_error_msg = str(e)

    # Check local Ollama status
    ollama_online = False
    ollama_models = []
    ollama_error_msg = ""
    try:
        import requests
        res = requests.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=2)
        if res.status_code == 200:
            ollama_online = True
            ollama_models = [m["name"] for m in res.json().get("models", [])]
        else:
            ollama_error_msg = f"Ollama returned HTTP status {res.status_code}"
    except Exception as e:
        ollama_error_msg = str(e)

    return jsonify({
        "status": "online",
        "api_key": {
            "configured": api_key_configured,
            "model": settings.GEMINI_MODEL,
            "error": api_key_error_msg
        },
        "cnn_model": {
            "loaded": cnn_model_loaded,
            "path": str(settings.CNN_MODEL_PATH),
            "error": cnn_model_error_msg
        },
        "faiss_index": {
            "exists": faiss_index_exists,
            "loaded": faiss_index_loaded,
            "path": str(settings.FAISS_INDEX_PATH),
            "error": faiss_error_msg
        },
        "ollama": {
            "online": ollama_online,
            "models": ollama_models,
            "host": settings.OLLAMA_HOST,
            "error": ollama_error_msg
        },
        "facade_initialized": (facade is not None),
        "facade_error": facade_error
    })


@app.route('/api/generate', methods=['POST'])
def generate_text():
    """Generates text from Gemma/Gemini based on prompt and parameters."""
    data = request.get_json() or {}
    prompt = data.get('prompt', '').strip()
    model_name = data.get('model_name', '').strip()
    backend = data.get('backend', '').strip()
    system_instruction = data.get('system_instruction', '').strip() or None
    temperature = float(data.get('temperature', 0.2))
    max_tokens = data.get('max_output_tokens')
    if max_tokens:
        max_tokens = int(max_tokens)

    if not prompt:
        return jsonify({"error": "Prompt cannot be empty"}), 400

    # Capture original settings
    original_model = settings.GEMINI_MODEL
    original_backend = settings.LLM_BACKEND
    try:
        if model_name:
            settings.GEMINI_MODEL = model_name
        if backend:
            settings.LLM_BACKEND = backend
            
        f = get_facade()
        
        # Instantiate a new GemmaService to pick up the updated settings
        from ai_module.gemma.gemma_service import GemmaService
        gemma_service = GemmaService()
        
        start_time = time.time()
        response_text = gemma_service.generate_response(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        duration = time.time() - start_time
        
        return jsonify({
            "response": response_text,
            "model_used": settings.GEMINI_MODEL,
            "backend_used": settings.LLM_BACKEND,
            "execution_time_sec": round(duration, 3)
        })
        
    except (GemmaConfigurationError, GemmaAPIError) as g_err:
        return jsonify({"error": str(g_err)}), 500
    except Exception as e:
        logger.exception("Unexpected error during text generation")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
    finally:
        # Restore settings
        settings.GEMINI_MODEL = original_model
        settings.LLM_BACKEND = original_backend


@app.route('/api/classify', methods=['POST'])
def classify_image():
    """Uploads a food image and returns predicted food class and CNN confidence score."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save to temporary path
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    try:
        file.save(temp_file.name)
        temp_file.close()

        f = get_facade()
        start_time = time.time()
        food_name, confidence = f.food_classifier.predict(temp_file.name)
        duration = time.time() - start_time

        return jsonify({
            "food_name": food_name,
            "confidence": round(confidence, 4),
            "execution_time_sec": round(duration, 3)
        })
        
    except FileNotFoundError as fnf:
        return jsonify({"error": f"CNN checkpoint error: {str(fnf)}"}), 500
    except Exception as e:
        logger.exception("Error during CNN classification")
        return jsonify({"error": f"Failed to classify image: {str(e)}"}), 500
    finally:
        # Ensure cleanup of temp file
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)


@app.route('/api/process-donation', methods=['POST'])
def process_donation():
    """Runs the integrated pipeline: CNN classification, Gemma summary, and RAG safety advisor."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    quantity = request.form.get('quantity', '').strip()
    prepared_time = request.form.get('prepared_time', '').strip()
    storage_condition = request.form.get('storage_condition', '').strip()
    model_name = request.form.get('model_name', '').strip()
    backend = request.form.get('backend', '').strip()

    if not quantity or not prepared_time or not storage_condition:
        return jsonify({"error": "Missing required fields (quantity, prepared_time, storage_condition)"}), 400

    # Save file to temp path
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    
    # Capture original settings
    original_model = settings.GEMINI_MODEL
    original_backend = settings.LLM_BACKEND
    try:
        file.save(temp_file.name)
        temp_file.close()

        if model_name:
            settings.GEMINI_MODEL = model_name
        if backend:
            settings.LLM_BACKEND = backend

        f = get_facade()
        start_time = time.time()
        
        # Process the donation using the facade
        payload = f.process_image_donation(
            image_path=temp_file.name,
            quantity=quantity,
            prepared_time=prepared_time,
            storage_condition=storage_condition
        )
        
        duration = time.time() - start_time

        # Serialize retrieved source document chunks
        serialized_sources = [
            {
                "text": chunk.text,
                "source": chunk.source,
                "page": chunk.page
            }
            for chunk in payload.get("safety_sources", [])
        ]

        return jsonify({
            "food_name": payload.get("food_name"),
            "cnn_confidence": round(payload.get("cnn_confidence", 0), 4),
            "summary": payload.get("summary"),
            "safety_advice": payload.get("safety_advice"),
            "safety_sources": serialized_sources,
            "model_used": settings.GEMINI_MODEL,
            "backend_used": settings.LLM_BACKEND,
            "execution_time_sec": round(duration, 3)
        })
        
    except (GemmaConfigurationError, GemmaAPIError) as g_err:
        return jsonify({"error": str(g_err)}), 500
    except Exception as e:
        logger.exception("Error in integrated donation pipeline")
        return jsonify({"error": f"Integrated pipeline failed: {str(e)}"}), 500
    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
        settings.GEMINI_MODEL = original_model
        settings.LLM_BACKEND = original_backend


@app.route('/api/recommend-ngo', methods=['POST'])
def recommend_ngo():
    """Matches donation metrics against NGO policies and returns a RAG-grounded matching explanation."""
    data = request.get_json() or {}
    ngo_name = data.get('ngo_name', '').strip()
    distance_km = data.get('distance_km')
    capacity_kg = data.get('capacity_kg')
    rating = data.get('rating')
    food_details = data.get('food_details', '').strip()
    model_name = data.get('model_name', '').strip()
    backend = data.get('backend', '').strip()

    if not ngo_name or distance_km is None or capacity_kg is None or rating is None or not food_details:
        return jsonify({"error": "Missing matching parameters"}), 400

    try:
        distance_km = float(distance_km)
        capacity_kg = float(capacity_kg)
        rating = float(rating)
    except ValueError:
        return jsonify({"error": "Distance, capacity, and rating must be numeric values"}), 400

    original_model = settings.GEMINI_MODEL
    original_backend = settings.LLM_BACKEND
    try:
        if model_name:
            settings.GEMINI_MODEL = model_name
        if backend:
            settings.LLM_BACKEND = backend

        f = get_facade()
        start_time = time.time()

        # Query FAISS database index for NGO policies and criteria matching
        search_query = f"NGO matching policy, proximity distance, capacity rules, priority rating for {ngo_name}"
        matches = f.chatbot.vector_store.search(search_query, k=2)
        
        context_blocks = []
        sources = []
        for chunk, dist in matches:
            block = f"[Source: {chunk.source}, Page: {chunk.page}]\n{chunk.text}"
            context_blocks.append(block)
            sources.append(chunk)
            
        context_str = "\n\n".join(context_blocks) if context_blocks else None

        # Execute recommendation explanation reasoning
        explanation = f.explain_recommendation(
            ngo_name=ngo_name,
            distance_km=distance_km,
            capacity_kg=capacity_kg,
            rating=rating,
            food_details=food_details,
            context=context_str
        )

        duration = time.time() - start_time
        
        serialized_sources = [
            {
                "text": chunk.text,
                "source": chunk.source,
                "page": chunk.page
            }
            for chunk in sources
        ]

        return jsonify({
            "explanation": explanation,
            "sources": serialized_sources,
            "model_used": settings.GEMINI_MODEL,
            "backend_used": settings.LLM_BACKEND,
            "execution_time_sec": round(duration, 3)
        })
        
    except (GemmaConfigurationError, GemmaAPIError) as g_err:
        return jsonify({"error": str(g_err)}), 500
    except Exception as e:
        logger.exception("Error in NGO recommendation explainer endpoint")
        return jsonify({"error": f"Failed to generate match justification: {str(e)}"}), 500
    finally:
        settings.GEMINI_MODEL = original_model
        settings.LLM_BACKEND = original_backend


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handles chats with the RAG FAQ chatbot using dynamic model override settings."""
    data = request.get_json() or {}
    message = data.get('message', '').strip()
    chat_history = data.get('history', [])
    model_name = data.get('model_name', '').strip()
    backend = data.get('backend', '').strip()

    if not message:
        return jsonify({"error": "Message cannot be empty"}), 400

    original_model = settings.GEMINI_MODEL
    original_backend = settings.LLM_BACKEND
    try:
        if model_name:
            settings.GEMINI_MODEL = model_name
        if backend:
            settings.LLM_BACKEND = backend

        f = get_facade()
        start_time = time.time()
        
        # Jivahar chatbot takes user_message and optional chat_history
        # Returns (response_string, List[DocumentChunk])
        response_text, sources = f.chat(
            user_message=message,
            chat_history=chat_history
        )
        
        duration = time.time() - start_time

        serialized_sources = [
            {
                "text": chunk.text,
                "source": chunk.source,
                "page": chunk.page
            }
            for chunk in sources
        ]

        return jsonify({
            "response": response_text,
            "sources": serialized_sources,
            "model_used": settings.GEMINI_MODEL,
            "backend_used": settings.LLM_BACKEND,
            "execution_time_sec": round(duration, 3)
        })

    except (GemmaConfigurationError, GemmaAPIError) as g_err:
        return jsonify({"error": str(g_err)}), 500
    except Exception as e:
        logger.exception("Error in Jivahar chatbot")
        return jsonify({"error": f"Chatbot error: {str(e)}"}), 500
    finally:
        settings.GEMINI_MODEL = original_model
        settings.LLM_BACKEND = original_backend


@app.route('/api/ingest', methods=['POST'])
def ingest_documents():
    """Trigger the FAISS semantic search index database compilation."""
    try:
        f = get_facade()
        start_time = time.time()
        f.ingest_knowledge_base()
        duration = time.time() - start_time
        return jsonify({
            "success": True,
            "message": "Successfully ingested PDF knowledge base files and rebuilt FAISS vector index.",
            "execution_time_sec": round(duration, 3)
        })
    except Exception as e:
        logger.exception("Failed to build vector index")
        return jsonify({"error": f"Index rebuild failed: {str(e)}"}), 500


if __name__ == '__main__':
    # Auto-ingest if FAISS index files are missing to ensure seamless trial
    try:
        index_dir = settings.FAISS_INDEX_PATH
        index_file = index_dir / "index.faiss"
        meta_file = index_dir / "metadata.pkl"
        if not (index_file.exists() and meta_file.exists()):
            logger.info("FAISS vector database files missing. Initiating automatic PDF ingestion on startup...")
            f = get_facade()
            f.ingest_knowledge_base()
    except Exception as e:
        logger.warning(f"Failed to auto-ingest PDFs during startup (API key might be missing): {e}")

    # Listen on port 5000, localhost
    app.run(host='127.0.0.1', port=5000, debug=True)
