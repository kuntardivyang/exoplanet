"""
Chat and LLM endpoints for Exoplanet Detection System
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import APIRouter, HTTPException
from typing import Dict
import uuid

from backend.api.schemas import (
    ChatRequest, ChatResponse, ExplainPredictionRequest,
    CompareRequest
)
from backend.llm.rag_system import RAGSystem, ChatHistory
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

# Initialize router
router = APIRouter(prefix="/chat", tags=["chat"])

# Initialize RAG system (will be done lazily)
rag_system: RAGSystem = None
chat_histories: Dict[str, ChatHistory] = {}


def get_rag_system() -> RAGSystem:
    """Get or initialize RAG system"""
    global rag_system
    if rag_system is None:
        logger.info("Initializing RAG system...")
        rag_system = RAGSystem(ollama_model="llama3.2")
        if rag_system.available:
            # Index knowledge base on first use
            try:
                rag_system.index_knowledge_base()
            except Exception as e:
                logger.error(f"Error indexing knowledge base: {e}")
    return rag_system


@router.post("/query", response_model=ChatResponse)
async def chat_query(request: ChatRequest):
    """
    Chat with the exoplanet AI assistant

    The assistant uses RAG to retrieve relevant context from:
    - Model training results
    - Dataset information
    - Light curve data
    - Exoplanet knowledge base

    Returns contextual answers with sources and follow-up suggestions.
    """
    try:
        rag = get_rag_system()

        if not rag.available:
            raise HTTPException(
                status_code=503,
                detail="Chat system not available. Please ensure Ollama is running and models are installed."
            )

        # Get or create conversation
        conversation_id = request.conversation_id or str(uuid.uuid4())
        if conversation_id not in chat_histories:
            chat_histories[conversation_id] = ChatHistory()

        history = chat_histories[conversation_id]

        # Add user message to history
        history.add_message("user", request.message)

        # Query RAG system
        result = rag.query(
            question=request.message,
            n_context=request.n_context,
            include_context=request.include_context
        )

        # Add assistant response to history
        history.add_message("assistant", result['answer'])

        return ChatResponse(
            answer=result['answer'],
            sources=result.get('sources', []),
            suggestions=result.get('suggestions', []),
            conversation_id=conversation_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain")
async def explain_prediction(request: ExplainPredictionRequest):
    """
    Explain a model prediction using AI

    Provides human-readable explanation of:
    - What the prediction means
    - Confidence level interpretation
    - Key features that influenced the decision
    - Scientific significance
    """
    try:
        rag = get_rag_system()

        if not rag.available:
            raise HTTPException(
                status_code=503,
                detail="Explanation system not available"
            )

        explanation = rag.explain_prediction(
            prediction_result=request.prediction,
            features=request.features
        )

        return {
            "explanation": explanation,
            "prediction": request.prediction
        }

    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_items(request: CompareRequest):
    """
    Compare two datasets, models, or exoplanet objects

    Provides detailed comparison including:
    - Key characteristics
    - Similarities and differences
    - Scientific significance
    - Use case recommendations
    """
    try:
        rag = get_rag_system()

        if not rag.available:
            raise HTTPException(
                status_code=503,
                detail="Comparison system not available"
            )

        if request.comparison_type == "datasets":
            comparison = rag.compare_datasets(request.item1, request.item2)
        else:
            # General comparison
            question = f"Compare {request.item1} and {request.item2}"
            result = rag.query(question)
            comparison = result['answer']

        return {
            "comparison": comparison,
            "item1": request.item1,
            "item2": request.item2,
            "type": request.comparison_type
        }

    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations")
async def get_recommendations(use_case: str):
    """
    Get model recommendations for a specific use case

    Example use cases:
    - "highest accuracy for confirmed planets"
    - "fastest prediction for real-time processing"
    - "best for explaining decisions"
    """
    try:
        rag = get_rag_system()

        if not rag.available:
            raise HTTPException(
                status_code=503,
                detail="Recommendations system not available"
            )

        recommendations = rag.get_model_recommendations(use_case)

        return {
            "recommendations": recommendations,
            "use_case": use_case
        }

    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear conversation history"""
    if conversation_id in chat_histories:
        chat_histories[conversation_id].clear()
        return {"message": f"Conversation {conversation_id} cleared"}
    raise HTTPException(status_code=404, detail="Conversation not found")


@router.get("/status")
async def chat_status():
    """Get chat system status"""
    rag = get_rag_system()

    return {
        "available": rag.available if rag else False,
        "ollama_available": rag.ollama.available if rag else False,
        "vector_store_available": rag.vector_store.available if rag else False,
        "vector_store_stats": rag.vector_store.get_stats() if rag and rag.vector_store.available else {},
        "active_conversations": len(chat_histories)
    }


@router.post("/reindex")
async def reindex_knowledge():
    """Re-index the knowledge base (admin function)"""
    try:
        rag = get_rag_system()

        if not rag.available:
            raise HTTPException(
                status_code=503,
                detail="Cannot reindex: RAG system not available"
            )

        # Clear and reindex
        rag.vector_store.delete_collection()
        rag.vector_store = get_rag_system().vector_store
        rag.index_knowledge_base()

        stats = rag.vector_store.get_stats()

        return {
            "message": "Knowledge base reindexed successfully",
            "stats": stats
        }

    except Exception as e:
        logger.error(f"Reindex error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
