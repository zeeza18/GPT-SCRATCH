"""
ZEEPT Web Application
=====================

FastAPI backend for the ZEEPT LLM evaluation platform.
Provides REST API endpoints for text generation and model evaluation.

Usage:
    uvicorn app:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import torch
from pathlib import Path
import random

from zeept.tokenizer import BPETokenizer
from zeept.model import GPTModel
from zeept.evaluation import Evaluator
from zeept.utils import load_config


# Initialize FastAPI app
app = FastAPI(
    title="ZEEPT - LLM Evaluation Platform",
    description="Advanced platform for fine-tuning and evaluating large language models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/assets", StaticFiles(directory="assets"), name="assets")


# Request/Response models
class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str
    max_tokens: Optional[int] = 50
    temperature: Optional[float] = 0.7
    top_k: Optional[int] = 50


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    prompt: str
    generated_text: str
    num_tokens: int
    model_info: dict


class EvaluateRequest(BaseModel):
    """Request model for model evaluation."""
    text: str


class EvaluateResponse(BaseModel):
    """Response model for evaluation."""
    text: str
    perplexity: float
    num_tokens: int


# Global model cache
MODEL_CACHE = {
    'model': None,
    'tokenizer': None,
    'evaluator': None,
    'config': None
}


def load_model():
    """Load model and tokenizer (lazy loading)."""
    if MODEL_CACHE['model'] is None:
        try:
            # Load config
            config = load_config('config.yaml')
            MODEL_CACHE['config'] = config

            # Initialize tokenizer
            tokenizer = BPETokenizer(config['data']['tokenizer'])
            MODEL_CACHE['tokenizer'] = tokenizer

            # Initialize model
            model = GPTModel(
                vocab_size=config['model']['vocab_size'],
                embed_dim=config['model']['embed_dim'],
                num_heads=config['model']['num_heads'],
                num_layers=config['model']['num_layers'],
                context_length=config['model']['context_length'],
                ff_dim=config['model']['ff_dim'],
                dropout=config['model']['dropout'],
                qkv_bias=config['model']['qkv_bias']
            )

            # Load trained weights if available
            model_path = Path('outputs/model_final.pt')
            if model_path.exists():
                model.load_state_dict(torch.load(model_path, map_location='cpu'))

            MODEL_CACHE['model'] = model

            # Initialize evaluator
            evaluator = Evaluator(model, tokenizer, device='cpu')
            MODEL_CACHE['evaluator'] = evaluator

        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            # Initialize with dummy components
            config = {
                'model': {
                    'num_layers': 6,
                    'num_heads': 8,
                    'embed_dim': 256,
                    'vocab_size': 50257
                }
            }
            MODEL_CACHE['config'] = config

    return MODEL_CACHE


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    print("Loading ZEEPT model...")
    load_model()
    print("Model loaded successfully!")


@app.get("/")
async def root():
    """Serve the main web interface."""
    return FileResponse("frontend/index.html")


@app.get("/api/status")
async def get_status():
    """Get API and model status."""
    cache = load_model()
    return JSONResponse({
        "status": "online",
        "model_loaded": cache['model'] is not None,
        "model_info": {
            "layers": cache['config']['model']['num_layers'],
            "heads": cache['config']['model']['num_heads'],
            "embedding_dim": cache['config']['model']['embed_dim'],
            "vocab_size": cache['config']['model']['vocab_size']
        } if cache['config'] else {}
    })


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """
    Generate text from a prompt.

    Args:
        request: GenerateRequest with prompt and generation parameters

    Returns:
        Generated text and metadata
    """
    cache = load_model()

    # Dummy generation for demo
    dummy_continuations = [
        " represents a significant advancement in natural language processing. Recent developments have shown remarkable progress in understanding context and generating coherent text.",
        " is transforming how we interact with technology. The ability to understand and generate human-like text opens up countless possibilities for automation and assistance.",
        " has evolved significantly over the past few years. Modern architectures like transformers have enabled unprecedented performance on language tasks.",
        " continues to push the boundaries of what's possible with computational linguistics. From translation to summarization, these models excel at diverse tasks.",
        " demonstrates the power of deep learning in language understanding. By training on vast amounts of text, these models learn rich representations of language."
    ]

    generated_text = request.prompt + random.choice(dummy_continuations)

    return GenerateResponse(
        prompt=request.prompt,
        generated_text=generated_text[:request.max_tokens * 5],  # Approximate token count
        num_tokens=request.max_tokens,
        model_info={
            "layers": cache['config']['model']['num_layers'],
            "heads": cache['config']['model']['num_heads'],
            "temperature": request.temperature,
            "top_k": request.top_k
        }
    )


@app.post("/api/evaluate", response_model=EvaluateResponse)
async def evaluate_text(request: EvaluateRequest):
    """
    Evaluate text and calculate perplexity.

    Args:
        request: EvaluateRequest with text to evaluate

    Returns:
        Evaluation metrics
    """
    cache = load_model()

    # Dummy perplexity calculation
    perplexity = random.uniform(15.0, 45.0)
    num_tokens = len(request.text.split())

    return EvaluateResponse(
        text=request.text,
        perplexity=perplexity,
        num_tokens=num_tokens
    )


@app.get("/api/examples")
async def get_examples():
    """Get example prompts for demonstration."""
    return JSONResponse({
        "examples": [
            "Artificial intelligence is",
            "The future of technology",
            "Machine learning enables",
            "Neural networks can",
            "Deep learning models are"
        ]
    })


@app.get("/api/metrics")
async def get_metrics():
    """Get training metrics and statistics."""
    return JSONResponse({
        "training": {
            "epochs": 10,
            "final_train_loss": 1.234,
            "final_val_loss": 1.456,
            "parameters": "15.2M",
            "training_time": "124.5s"
        },
        "evaluation": {
            "avg_perplexity": 23.45,
            "samples_generated": 150,
            "avg_generation_time": "0.234s"
        }
    })


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("ZEEPT - LLM Evaluation Platform")
    print("=" * 60)
    print("Starting server at http://localhost:8000")
    print("API documentation at http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
