#!/usr/bin/env python3
"""
LibrIA Backend API
==================

API REST para reconhecimento de Libras usando FastAPI
Integra o modelo de machine learning existente
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
import uvicorn
import cv2
import numpy as np
import pickle
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from datetime import datetime

# Adicionar o diretório pai ao path para importar módulos do projeto original
sys.path.append(str(Path(__file__).parent.parent))

# Importar módulos do projeto original
from src.inference.libras_realtime_classifier import LibrasRealtimeClassifier
from src.model_training.libras_model_trainer import LibrasModelTrainer
from config.settings import ALPHABET_DICT

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="LibrIA API",
    description="API para reconhecimento de Libras usando visão computacional",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar domínios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Variáveis globais
classifier: Optional[LibrasRealtimeClassifier] = None
model_trainer: Optional[LibrasModelTrainer] = None

def load_model():
    """Carrega o modelo treinado"""
    global classifier, model_trainer
    
    try:
        model_path = Path(__file__).parent.parent / "model" / "model.pickle"
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")
        
        classifier = LibrasRealtimeClassifier()
        model_trainer = LibrasModelTrainer()
        
        logger.info("✅ Modelo carregado com sucesso")
        return True
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Evento executado na inicialização da API"""
    logger.info("🚀 Iniciando LibrIA API...")
    
    if not load_model():
        logger.error("❌ Falha ao carregar modelo. API não pode ser iniciada.")
        raise RuntimeError("Modelo não pôde ser carregado")

@app.get("/")
async def root():
    """Endpoint raiz"""
    return {
        "message": "LibrIA API - Reconhecimento de Libras",
        "version": "1.0.0",
        "status": "online",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Verificação de saúde da API"""
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict_letter(
    image: UploadFile = File(...),
    confidence_threshold: float = 0.7
):
    """
    Endpoint para reconhecimento de letras em Libras
    
    Args:
        image: Imagem da mão fazendo o sinal
        confidence_threshold: Limite de confiança (0.0 a 1.0)
    
    Returns:
        JSON com predição e confiança
    """
    try:
        # Validar arquivo
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem")
        
        # Ler imagem
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Imagem inválida")
        
        # Converter BGR para RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Fazer predição
        if classifier is None:
            raise HTTPException(status_code=500, detail="Modelo não carregado")
        
        prediction, confidence = classifier.predict_single_image(img_rgb)
        
        # Verificar confiança mínima
        if confidence < confidence_threshold:
            return {
                "prediction": None,
                "confidence": confidence,
                "message": "Confiança muito baixa",
                "threshold": confidence_threshold
            }
        
        # Converter classe para letra
        letter = None
        for key, value in ALPHABET_DICT.items():
            if value == prediction:
                letter = key
                break
        
        return {
            "prediction": letter,
            "class": prediction,
            "confidence": float(confidence),
            "threshold": confidence_threshold,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(
    images: list[UploadFile] = File(...),
    confidence_threshold: float = 0.7
):
    """
    Endpoint para reconhecimento em lote
    
    Args:
        images: Lista de imagens
        confidence_threshold: Limite de confiança
    
    Returns:
        Lista de predições
    """
    results = []
    
    for i, image in enumerate(images):
        try:
            # Ler imagem
            contents = await image.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                results.append({
                    "index": i,
                    "error": "Imagem inválida"
                })
                continue
            
            # Converter BGR para RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Fazer predição
            prediction, confidence = classifier.predict_single_image(img_rgb)
            
            # Converter classe para letra
            letter = None
            for key, value in ALPHABET_DICT.items():
                if value == prediction:
                    letter = key
                    break
            
            results.append({
                "index": i,
                "prediction": letter,
                "class": prediction,
                "confidence": float(confidence),
                "above_threshold": confidence >= confidence_threshold
            })
            
        except Exception as e:
            results.append({
                "index": i,
                "error": str(e)
            })
    
    return {
        "results": results,
        "total_images": len(images),
        "threshold": confidence_threshold,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
async def get_model_info():
    """Informações sobre o modelo carregado"""
    if model_trainer is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado")
    
    try:
        info = model_trainer.get_model_info()
        return {
            "model_info": info,
            "alphabet_supported": list(ALPHABET_DICT.keys()),
            "total_classes": len(ALPHABET_DICT),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Erro ao obter informações do modelo: {e}")
        raise HTTPException(status_code=500, detail="Erro ao obter informações do modelo")

@app.get("/alphabet")
async def get_alphabet():
    """Retorna o alfabeto suportado"""
    return {
        "alphabet": ALPHABET_DICT,
        "total_letters": len(ALPHABET_DICT),
        "supported_letters": list(ALPHABET_DICT.keys())
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handler global para exceções"""
    logger.error(f"Erro não tratado: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Erro interno do servidor",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
