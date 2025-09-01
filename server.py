
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
from reasoning_from_scratch.memory_layer import AdvancedKnowledgeMemoryLayer
from reasoning_from_scratch.advanced_features import (
    ContinuousLearningEngine, RealtimeReasoningOptimizer, 
    WorldClassReasoningValidator, InnovativeReasoningMethods,
    ExplainabilityEngine, CompetitiveAdvantageTracker
)
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
continuous_learning = None
realtime_optimizer = None
world_class_validator = None
innovative_methods = None
explainability_engine = None
competitive_tracker = None

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
    domain: Optional[str] = None
    domain_enhanced: Optional[bool] = False
    compliance_score: Optional[float] = None
    domain_validation: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    reasoning: str
    method: str
    confidence: float
    session_id: str
    domain: Optional[str] = None
    domain_enhanced: Optional[bool] = False
    compliance_score: Optional[float] = None

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

class DomainExpertRequest(BaseModel):
    question: str
    domain: Optional[str] = None

class ComplianceCheckRequest(BaseModel):
    question: str
    answer: str
    reasoning: str
    domain: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the model and reasoning components on startup."""
    global model, tokenizer, reasoning_engine, knowledge_layer
    
    print("Initializing Reasoning LLM...")
    
    # Get device
    device = get_device()
    
    # Download and load model
    print("Loading Qwen3 model...")
    model_dir = "models/tokenizer.json"
    
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
        
        # Initialize advanced knowledge layer
        knowledge_layer = AdvancedKnowledgeMemoryLayer()
        
        # Initialize reasoning engine with memory layer
        reasoning_engine = ReasoningEngine(model, tokenizer, knowledge_layer)
        
        # Initialize world-class components
        continuous_learning = ContinuousLearningEngine(knowledge_layer)
        realtime_optimizer = RealtimeReasoningOptimizer()
        world_class_validator = WorldClassReasoningValidator()
        innovative_methods = InnovativeReasoningMethods()
        explainability_engine = ExplainabilityEngine()
        competitive_tracker = CompetitiveAdvantageTracker()
        
        print("World-class reasoning components initialized!")
        
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
            context_used=context_prompt if context_prompt else None,
            domain=result.get("domain"),
            domain_enhanced=result.get("domain_enhanced", False),
            compliance_score=result.get("compliance_score"),
            domain_validation=result.get("domain_validation")
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
            session_id=request.session_id,
            domain=result.get("domain"),
            domain_enhanced=result.get("domain_enhanced", False),
            compliance_score=result.get("compliance_score")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")


@app.post("/add_documents")
async def add_documents(request: AddDocumentsRequest):
    """Add knowledge documents to the system."""


@app.get("/world_class_metrics")
async def get_world_class_metrics():
    """Get world-class reasoning metrics and competitive analysis."""
    if not reasoning_engine:
        raise HTTPException(status_code=503, detail="Reasoning engine not initialized")
    
    try:
        metrics = reasoning_engine.get_world_class_metrics()
        competitive_report = competitive_tracker.get_competitive_report()
        
        return {
            "reasoning_metrics": metrics,
            "competitive_analysis": competitive_report,
            "world_class_status": metrics.get("world_class_rating", "Developing"),
            "unique_advantages": competitive_report.get("unique_differentiators", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting world-class metrics: {str(e)}")


@app.post("/innovative_reasoning")
async def innovative_reasoning(request: SolveRequest):
    """Use cutting-edge innovative reasoning methods."""
    if not innovative_methods:
        raise HTTPException(status_code=503, detail="Innovative methods not initialized")
    
    try:
        # Select innovative method based on question complexity
        question_complexity = len(request.prompt.split())
        
        if question_complexity > 30:
            method = "recursive_decomposition"
        elif "analogy" in request.prompt.lower() or "similar" in request.prompt.lower():
            method = "analogical_transfer"
        elif "what if" in request.prompt.lower():
            method = "counterfactual_reasoning"
        else:
            method = "metacognitive_monitoring"
        
        result = innovative_methods.apply_innovative_method(
            request.prompt, method, model, tokenizer
        )
        
        # Validate with world-class standards
        validation = world_class_validator.validate_world_class_reasoning(
            request.prompt, result["reasoning"], result["answer"]
        )
        result.update(validation)
        
        return SolveResponse(
            answer=result["answer"],
            reasoning=result["reasoning"], 
            method=result["method"],
            confidence=result["confidence"],
            trace=result["trace"],
            domain="innovative_ai",
            domain_enhanced=True,
            compliance_score=validation.get("overall_score", 0.85)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with innovative reasoning: {str(e)}")


@app.post("/explain_reasoning")
async def explain_reasoning(question: str, reasoning_result: dict, style: str = "layperson"):
    """Generate advanced explanations of reasoning in different styles."""
    if not explainability_engine:
        raise HTTPException(status_code=503, detail="Explainability engine not initialized")
    
    try:
        explanation = explainability_engine.generate_explanation(reasoning_result, style)
        
        return {
            "explanation": explanation,
            "style": style,
            "available_styles": list(explainability_engine.explanation_styles.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating explanation: {str(e)}")


@app.get("/competitive_analysis")
async def get_competitive_analysis():
    """Get comprehensive competitive analysis and market position."""
    if not competitive_tracker:
        raise HTTPException(status_code=503, detail="Competitive tracker not initialized")
    
    try:
        competitive_report = competitive_tracker.get_competitive_report()
        
        # Add real-time optimization insights
        if realtime_optimizer:
            optimization_insights = realtime_optimizer.get_optimization_insights()
            competitive_report["optimization_performance"] = optimization_insights
        
        return competitive_report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting competitive analysis: {str(e)}")


@app.get("/world_class_validation")
async def get_world_class_validation(question: str, reasoning: str, answer: str):
    """Validate reasoning against world-class standards."""
    if not world_class_validator:
        raise HTTPException(status_code=503, detail="World-class validator not initialized")
    
    try:
        validation = world_class_validator.validate_world_class_reasoning(question, reasoning, answer)
        return validation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating reasoning: {str(e)}")


@app.get("/advanced_analytics")
async def get_advanced_analytics():
    """Get comprehensive advanced analytics and insights."""
    if not knowledge_layer:
        raise HTTPException(status_code=503, detail="Knowledge layer not initialized")
    
    try:
        analytics = knowledge_layer.get_advanced_analytics()
        
        # Add reasoning engine metrics
        if hasattr(reasoning_engine, 'get_world_class_metrics'):
            analytics["world_class_metrics"] = reasoning_engine.get_world_class_metrics()
        
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting advanced analytics: {str(e)}")


@app.post("/start_continuous_learning")
async def start_continuous_learning():
    """Start continuous learning for real-time improvement."""
    if not continuous_learning:
        raise HTTPException(status_code=503, detail="Continuous learning not initialized")
    
    try:
        # Start continuous learning in background
        asyncio.create_task(continuous_learning.continuous_adaptation())
        
        return {
            "message": "Continuous learning started",
            "status": "active",
            "learning_rate": continuous_learning.learning_rate,
            "adaptation_threshold": continuous_learning.adaptation_threshold
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting continuous learning: {str(e)}")

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
            stats["domain_experts"] = reasoning_engine.domain_experts.get_expert_stats()
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting performance stats: {str(e)}")


@app.get("/domain_experts")
async def get_domain_experts():
    """Get available domain experts and their capabilities."""
    if not reasoning_engine:
        raise HTTPException(status_code=503, detail="Reasoning engine not initialized")
    
    try:
        return reasoning_engine.domain_experts.get_expert_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting domain experts: {str(e)}")


@app.post("/domain_prompt")
async def get_domain_prompt(request: DomainExpertRequest):
    """Get domain-specific reasoning prompt for a question."""
    if not reasoning_engine:
        raise HTTPException(status_code=503, detail="Reasoning engine not initialized")
    
    try:
        prompt, domain = reasoning_engine.domain_experts.get_domain_reasoning_prompt(request.question)
        return {
            "prompt": prompt,
            "domain": domain,
            "has_domain_expert": domain is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting domain prompt: {str(e)}")


@app.post("/compliance_check")
async def check_compliance(request: ComplianceCheckRequest):
    """Check reasoning compliance with domain standards."""
    if not reasoning_engine:
        raise HTTPException(status_code=503, detail="Reasoning engine not initialized")
    
    try:
        validation = reasoning_engine.domain_experts.validate_domain_reasoning(
            request.question, request.reasoning, request.answer
        )
        return validation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking compliance: {str(e)}")


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
