"""
Logging configuration for the application.
"""
import logging
import sys
from app.core.config import settings

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Suppress noisy warnings from third-party libraries
    logging.getLogger('pdfminer').setLevel(logging.ERROR)  # Suppress pdfminer color warnings (harmless)
    logging.getLogger('pdfplumber').setLevel(logging.ERROR)  # Suppress pdfplumber warnings
    logging.getLogger('arxiv').setLevel(logging.WARNING)  # Only show warnings/errors from arxiv
    logging.getLogger('urllib3').setLevel(logging.WARNING)  # Suppress urllib3 connection pool messages
    
    return logging.getLogger(__name__)
