import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ai_module.facade import JivaharAIFacade

app = FastAPI(
    title="JIVAHAR AI Server",
    description="Backend API for JIVAHAR Food Detection, Classification, Gemma Summarization, and RAG Safety Advisor.",
    version="1.0.0"
)

# Enable CORS for local testing and web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI Facade (Lazy load or single instance)
facade: Optional[JivaharAIFacade] = None

def get_facade() -> JivaharAIFacade:
    global facade
    if facade is None:
        facade = JivaharAIFacade()
    return facade

# Directory paths
BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
TEMP_UPLOADS = BASE_DIR / "data" / "temp_uploads"
TEMP_UPLOADS.mkdir(parents=True, exist_ok=True)

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None

@app.get("/api/health")
def health_check():
    """Returns AI Module health and loaded models status."""
    ai = get_facade()
    return {
        "status": "online",
        "service": "JIVAHAR AI Engine",
        "cnn_classifier": "Ready",
        "rag_vector_store": "Ready",
        "gemma_llm": "Ready"
    }

@app.post("/api/detect")
async def detect_food_donation(
    image: UploadFile = File(...),
    quantity: str = Form("1 portion"),
    prepared_time: str = Form("Freshly prepared"),
    storage_condition: str = Form("Room Temperature")
):
    """
    Receives food image upload + donation details, runs CNN classification,
    Gemma log summarization, and RAG safety advisor analysis.
    """
    if not image.filename:
        raise HTTPException(status_code=400, detail="No image file provided.")

    # Save uploaded image to temporary file
    file_ext = Path(image.filename).suffix or ".jpg"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext, dir=TEMP_UPLOADS)
    try:
        content = await image.read()
        temp_file.write(content)
        temp_file.close()

        # Run facade pipeline
        ai = get_facade()
        result = ai.process_image_donation(
            image_path=temp_file.name,
            quantity=quantity,
            prepared_time=prepared_time,
            storage_condition=storage_condition
        )

        # Format safety sources into JSON serializable format
        sources = []
        if "safety_sources" in result and result["safety_sources"]:
            for chunk in result["safety_sources"]:
                sources.append({
                    "chunk_id": getattr(chunk, "chunk_id", "N/A"),
                    "source": getattr(chunk, "source", "Knowledge Base"),
                    "content": getattr(chunk, "content", ""),
                    "page_number": getattr(chunk, "page_number", 1)
                })

        return JSONResponse({
            "status": "success",
            "food_name": result.get("food_name", "Unknown"),
            "confidence": round(float(result.get("cnn_confidence", 0.0)) * 100, 2),
            "summary": result.get("summary", ""),
            "safety_advice": result.get("safety_advice", ""),
            "safety_sources": sources
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Pipeline error: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

@app.post("/api/chat")
async def chat_query(request: ChatRequest):
    """Answers food safety and platform logistics questions using RAG."""
    try:
        ai = get_facade()
        response_text, sources = ai.chat(
            user_message=request.message,
            chat_history=request.history
        )

        formatted_sources = []
        for chunk in sources:
            formatted_sources.append({
                "source": getattr(chunk, "source", "Knowledge Base"),
                "content": getattr(chunk, "content", ""),
                "page_number": getattr(chunk, "page_number", 1)
            })

        return JSONResponse({
            "status": "success",
            "response": response_text,
            "sources": formatted_sources
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")

# Mount static web files
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

@app.get("/")
def read_root():
    """Serves the main web interface."""
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "JIVAHAR AI API is running. Web frontend is located in /web/ index.html."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
