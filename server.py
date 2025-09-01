
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from typing import Dict, Any, Optional, List
import uvicorn

from reasoning_from_scratch.qwen3 import (
    Qwen3Model, 
    Qwen3Tokenizer, 
    QWEN_CONFIG_06_B,
    download_qwen3_small,
    load_hf_weights_into_qwen
)
from reasoning_from_scratch.reasoning_engine import ReasoningEngine
from reasoning_from_scratch.memory_layer import KnowledgeMemoryLayer
from reasoning_from_scratch.ch02 import get_device

app = FastAPI(title="Beast Reasoning LLM API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model components
model = None
tokenizer = None
reasoning_engine = None
knowledge_layer = None
auto_learning_system = None

# Request/Response models
class SolveRequest(BaseModel):
    prompt: str
    method: Optional[str] = "auto"
    max_new_tokens: Optional[int] = 256
    use_context: Optional[bool] = True

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    method: Optional[str] = "auto"

class SolveResponse(BaseModel):
    answer: str
    reasoning: str
    method: str
    confidence: float
    trace: List[str]
    context_used: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    reasoning: str
    method: str
    confidence: float
    session_id: str

class AddDocumentsRequest(BaseModel):
    documents: List[str]

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int  # 1-5 scale
    feedback: Optional[str] = ""

class LearnFromUrlRequest(BaseModel):
    url: str

class AddTopicRequest(BaseModel):
    topic: str


@app.on_event("startup")
async def startup_event():
    """Initialize the model and reasoning components on startup."""
    global model, tokenizer, reasoning_engine, knowledge_layer
    
    print("Initializing Reasoning LLM...")
    
    # Get device
    device = get_device()
    
    # Download and load model
    print("Loading Qwen3 model...")
    model_dir = "models"
    
    try:
        # Try to load existing model
        download_qwen3_small(kind="base", out_dir=model_dir)
        
        # Initialize model and tokenizer
        model = Qwen3Model(QWEN_CONFIG_06_B)
        tokenizer = Qwen3Tokenizer(model_dir)
        
        # Load weights
        weights_path = f"{model_dir}/qwen3_weights.pt"
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            print("Pre-trained weights not found. Using random initialization.")
        
        model.to(device)
        model.eval()
        
        # Initialize knowledge layer
        knowledge_layer = KnowledgeMemoryLayer()
        
        # Initialize reasoning engine with memory layer
        reasoning_engine = ReasoningEngine(model, tokenizer, knowledge_layer)
        
        # Initialize auto-learning system
        try:
            from reasoning_from_scratch.auto_learning import AutoLearningSystem
            auto_learning_system = AutoLearningSystem(knowledge_layer)
            print("Auto-learning system initialized!")
        except ImportError:
            print("Auto-learning dependencies not available. Install with: pip install wikipedia feedparser beautifulsoup4 aiohttp schedule")
            auto_learning_system = None
        
        # Add some default knowledge documents
        default_docs = [
            "Mathematics: Addition, subtraction, multiplication, and division are basic arithmetic operations.",
            "Logic: If-then statements follow the pattern: if premise is true, then conclusion follows.",
            "Problem solving: Break complex problems into smaller, manageable steps.",
            "Reasoning: Good reasoning includes clear steps, logical connections, and verification of results."
        ]
        knowledge_layer.add_knowledge_documents(default_docs)
        
        print("Reasoning LLM initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("The API will start but model-dependent endpoints will not work.")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Beast Reasoning LLM API is running!", "status": "healthy"}


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "reasoning_engine": reasoning_engine is not None,
        "knowledge_layer": knowledge_layer is not None,
        "auto_learning": auto_learning_system is not None
    }


@app.post("/solve", response_model=SolveResponse)
async def solve_problem(request: SolveRequest):
    """Solve a reasoning problem."""
    if not reasoning_engine:
        raise HTTPException(status_code=503, detail="Reasoning engine not initialized")
    
    try:
        # Get context if requested
        context_prompt = ""
        if request.use_context and knowledge_layer:
            context_info = knowledge_layer.get_context_for_question(request.prompt)
            context_prompt = context_info["context_prompt"]
        
        # Enhance prompt with context
        enhanced_prompt = request.prompt
        if context_prompt:
            enhanced_prompt = f"{context_prompt}\n\nQuestion: {request.prompt}"
        
        # Solve the problem
        result = reasoning_engine.solve_query(enhanced_prompt, method=request.method)
        
        # Store in memory
        if knowledge_layer:
            knowledge_layer.add_interaction(
                request.prompt, 
                result["answer"], 
                result["reasoning"], 
                result["method"]
            )
        
        return SolveResponse(
            answer=result["answer"],
            reasoning=result["reasoning"],
            method=result["method"],
            confidence=result["confidence"],
            trace=result["trace"],
            context_used=context_prompt if context_prompt else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error solving problem: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat interface for conversational reasoning."""
    if not reasoning_engine:
        raise HTTPException(status_code=503, detail="Reasoning engine not initialized")
    
    try:
        # Get context for the conversation
        context_prompt = ""
        if knowledge_layer:
            context_info = knowledge_layer.get_context_for_question(request.message)
            context_prompt = context_info["context_prompt"]
        
        # Create conversational prompt
        chat_prompt = f"User: {request.message}\nAssistant: Let me think about this step by step.\n"
        if context_prompt:
            chat_prompt = f"{context_prompt}\n\n{chat_prompt}"
        
        # Solve the problem
        result = reasoning_engine.solve_query(chat_prompt, method=request.method)
        
        # Store in memory
        if knowledge_layer:
            knowledge_layer.add_interaction(
                request.message, 
                result["answer"], 
                result["reasoning"], 
                result["method"]
            )
        
        return ChatResponse(
            response=result["answer"],
            reasoning=result["reasoning"],
            method=result["method"],
            confidence=result["confidence"],
            session_id=request.session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")


@app.post("/add_documents")
async def add_documents(request: AddDocumentsRequest):
    """Add knowledge documents to the system."""
    if not knowledge_layer:
        raise HTTPException(status_code=503, detail="Knowledge layer not initialized")
    
    try:
        knowledge_layer.add_knowledge_documents(request.documents)
        return {"message": f"Added {len(request.documents)} documents successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding documents: {str(e)}")


@app.post("/end_session")
async def end_session():
    """End the current reasoning session."""
    if knowledge_layer:
        knowledge_layer.end_session()
    return {"message": "Session ended successfully"}


@app.post("/start_learning")
async def start_auto_learning():
    """Start the auto-learning process."""
    if not auto_learning_system:
        raise HTTPException(status_code=503, detail="Auto-learning system not available")
    
    try:
        result = await auto_learning_system.learn_from_all_sources()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting auto-learning: {str(e)}")


@app.get("/learning_status")
async def get_learning_status():
    """Get auto-learning status."""
    if not auto_learning_system:
        return {"status": "not_available"}
    
    return auto_learning_system.get_learning_status()


@app.post("/learn_from_url")
async def learn_from_url(request: LearnFromUrlRequest):
    """Learn from a specific URL."""
    if not auto_learning_system:
        raise HTTPException(status_code=503, detail="Auto-learning system not available")
    
    try:
        result = await auto_learning_system.learn_from_url(request.url)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error learning from URL: {str(e)}")


@app.post("/add_topic")
async def add_learning_topic(request: AddTopicRequest):
    """Add a new topic for auto-learning."""
    if not auto_learning_system:
        raise HTTPException(status_code=503, detail="Auto-learning system not available")
    
    try:
        auto_learning_system.add_learning_topic(request.topic)
        return {"message": f"Added topic '{request.topic}' successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding topic: {str(e)}")


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback for continuous improvement."""
    if not knowledge_layer:
        raise HTTPException(status_code=503, detail="Knowledge layer not available")
    
    try:
        knowledge_layer.add_user_feedback(
            request.question, 
            request.answer, 
            request.rating, 
            request.feedback
        )
        return {"message": "Feedback submitted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")


@app.get("/performance_stats")
async def get_performance_stats():
    """Get performance statistics."""
    if not knowledge_layer:
        raise HTTPException(status_code=503, detail="Knowledge layer not available")
    
    try:
        stats = knowledge_layer.get_performance_stats()
        
        # Add reasoning engine stats
        if reasoning_engine:
            stats["method_usage"] = reasoning_engine.method_usage_stats
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting performance stats: {str(e)}")


@app.get("/methods")
async def get_available_methods():
    """Get available reasoning methods."""
    return {
        "methods": [
            "auto",
            "cot",
            "pal", 
            "self_consistency",
            "tot"
        ],
        "descriptions": {
            "auto": "Automatically select the best method",
            "cot": "Chain-of-thought reasoning",
            "pal": "Program-aided language (generates and executes code)",
            "self_consistency": "Multiple reasoning chains with majority voting",
            "tot": "Tree-of-thoughts search"
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
