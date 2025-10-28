import os
import sys
import subprocess
import logging
import urllib.request
import torch
import time
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if all required Python packages are installed"""
    logger.info("Checking requirements...")
    
    requirements = [
        "torch", "torchaudio", "fastapi", "uvicorn", "websockets", "numpy",
        "scikit-learn", "sqlalchemy", "pydantic", "jinja2", "whisper",
        "sounddevice", "soundfile", "sentence_transformers", "ctransformers"
    ]
    
    missing = []
    for req in requirements:
        try:
            __import__(req)
        except ImportError:
            missing.append(req)
    
    if missing:
        logger.warning(f"Missing required packages: {', '.join(missing)}")
        logger.info("Installing missing requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Requirements installed successfully")
    else:
        logger.info("All requirements are satisfied")

def download_vad_model():
    """Download the Silero VAD model using PyTorch Hub instead of direct URL"""
    model_path = "silero_vad.jit"
    
    if os.path.exists(model_path):
        logger.info(f"Silero VAD model already exists at {model_path}")
        return
    
    logger.info("Downloading Silero VAD model using PyTorch Hub...")
    try:
        # Use torch.hub to download the model instead of direct URL
        torch.hub.set_dir("./models")
        model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad",
                                     model="silero_vad",
                                     force_reload=True,
                                     onnx=False)
        
        # Save the model
        torch.jit.save(model, model_path)
        logger.info(f"Model downloaded and saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Failed to download Silero VAD model using PyTorch Hub: {e}")
        logger.info("Falling back to energy-based VAD - the system will still work but with simpler voice detection")

def download_embedding_models():
    """Download the sentence transformer models for RAG"""
    logger.info("Setting up sentence transformer models...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Download lightweight model for embeddings
        logger.info("Downloading embedding models (this may take a few minutes)...")
        models = [
            "all-MiniLM-L6-v2",  # Fast
            "all-mpnet-base-v2",  # Balanced
            "multi-qa-mpnet-base-dot-v1"  # Best for Q&A
        ]
        
        for model_name in models:
            logger.info(f"Setting up model: {model_name}")
            _ = SentenceTransformer(model_name)
            logger.info(f"Model {model_name} is ready")
            
    except Exception as e:
        logger.error(f"Failed to download embedding models: {e}")
        logger.error("Please try running the script again or download models manually")

def setup_directories():
    """Create necessary directories for the application"""
    directories = ["static", "responses", "embeddings_cache", "templates"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory {directory} is ready")
    
    # Create template redirect file
    template_dir = Path("templates")
    index_html = template_dir / "index.html"
    
    with open(index_html, "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0; url=/static/index.html">
</head>
<body>
    <p>Redirecting to <a href="/static/index.html">AI Companion</a>...</p>
</body>
</html>
        """)
    logger.info("Created index template for redirection")

def setup_database():
    """Initialize the SQLite database"""
    logger.info("Setting up database...")
    
    try:
        from sqlalchemy import create_engine, Column, Integer, String, Text
        from sqlalchemy.ext.declarative import declarative_base
        from sqlalchemy.orm import sessionmaker
        
        Base = declarative_base()
        engine = create_engine("sqlite:///companion.db")
        
        class Conversation(Base):
            __tablename__ = "conversations"
            id = Column(Integer, primary_key=True, index=True)
            session_id = Column(String, index=True)
            timestamp = Column(String)
            user_message = Column(Text)
            ai_message = Column(Text)
            audio_path = Column(String)
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to set up database: {e}")

def check_cuda():
    """Check if CUDA is available for PyTorch"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA is available: {device_name}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        logger.warning("CUDA is not available. The application will run on CPU, which may be very slow")
        logger.warning("For optimal performance, a CUDA-capable GPU is recommended")

def main():
    """Main setup function"""
    logger.info("Starting AI Companion setup...")
    
    # Check for CUDA availability
    check_cuda()
    
    # Check and install requirements
    #check_requirements()
    
    # Create directories
    setup_directories()
    
    # Set up database
    setup_database()
    
    # Download models
    download_vad_model()
    download_embedding_models()
    
    logger.info("Setup completed successfully!")
    logger.info("You can now start the application with:")
    logger.info("   python main.py")

if __name__ == "__main__":
    main()