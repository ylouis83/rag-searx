"""
RAG-SearX 


"""

import logging
import sys
from pathlib import Path
from typing import Optional
from loguru import logger as loguru_logger

from config.settings_simple import get_settings

# 
settings = get_settings()


def setup_logging() -> None:
    """"""
    
    # loguru
    loguru_logger.remove()
    
    # 
    log_file_path = Path(settings.logging.log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # 
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )
    
    # 
    loguru_logger.add(
        sys.stdout,
        format=console_format,
        level=settings.logging.log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    
    # 
    loguru_logger.add(
        settings.logging.log_file,
        format=file_format,
        level=settings.logging.log_level,
        rotation=f"{settings.logging.log_max_size} MB",
        retention=settings.logging.log_backup_count,
        compression="zip",
        backtrace=True,
        diagnose=True,
    )
    
    # logging
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> loguru_logger:
    """
    
    
    Args:
        name:  __name__
        
    Returns:
        
    """
    
    if name:
        return loguru_logger.bind(name=name)
    return loguru_logger


class InterceptHandler(logging.Handler):
    """
    loggingloguru
    """
    
    def emit(self, record):
        # loguru
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        # 
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_stdlib_logging():
    """loggingloguru"""
    
    # 
    root_logger = logging.getLogger()
    
    # 
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 
    root_logger.addHandler(InterceptHandler())
    root_logger.setLevel(logging.DEBUG)
    
    # 
    loggers_to_configure = [
        "uvicorn",
        "uvicorn.access", 
        "fastapi",
        "sqlalchemy",
        "httpx",
        "milvus",
        "qdrant_client",
    ]
    
    for logger_name in loggers_to_configure:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.propagate = True


# 
try:
    setup_logging()
    setup_stdlib_logging()
except Exception as e:
    # 
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("rag_searx.log", encoding="utf-8")
        ]
    )
    print(f"Warning: Failed to setup loguru logging, falling back to stdlib: {e}")


# 
def log_function_call(func_name: str, **kwargs):
    """"""
    logger = get_logger()
    args_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"Calling {func_name}({args_str})")


def log_performance(operation: str, duration: float, **metrics):
    """"""
    logger = get_logger()
    metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
    logger.info(f"Performance: {operation} took {duration:.3f}s [{metrics_str}]")


def log_error_with_context(error: Exception, context: dict):
    """"""
    logger = get_logger()
    logger.error(
        f"Error: {type(error).__name__}: {str(error)} | Context: {context}",
        exc_info=True
    ) 