import os
import json
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Manages configuration persistence for the AI Companion app.
    Saves and loads configuration to avoid re-entering model paths.
    """
    def __init__(self, config_path: str = "config/app_config.json"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to store the configuration file
        """
        self.config_path = config_path
        self.config_dir = os.path.dirname(config_path)
        
        # Create config directory if it doesn't exist
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir, exist_ok=True)
            logger.info(f"Created configuration directory: {self.config_dir}")
    
    def save_config(self, config_data: Dict[str, Any]) -> bool:
        """
        Save configuration data to the config file.
        
        Args:
            config_data: Configuration data to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(self.config_dir, exist_ok=True)
            print(config_data)
            # Verify all reference paths are included
            ref_paths = [
                "reference_audio_path", 
                "reference_audio_path2", 
                "reference_audio_path3"
            ]
            
            # Log which references are being saved
            for path_key in ref_paths:
                if path_key in config_data and config_data[path_key]:
                    logger.info(f"Saving reference path: {path_key}={config_data[path_key]}")
                else:
                    logger.info(f"No {path_key} provided in configuration")
            
            # Save configuration
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def load_config(self) -> Optional[Dict[str, Any]]:
        """
        Load configuration data from the config file.
        
        Returns:
            Dict or None: Configuration data if successful, None otherwise
        """
        if not os.path.exists(self.config_path):
            logger.info(f"Configuration file does not exist at {self.config_path}")
            return None
        
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            # Log which references are being loaded
            ref_paths = [
                "reference_audio_path", 
                "reference_audio_path2", 
                "reference_audio_path3"
            ]
            
            for path_key in ref_paths:
                if path_key in config_data and config_data[path_key]:
                    logger.info(f"Loaded reference path: {path_key}={config_data[path_key]}")
            
            logger.info(f"Configuration loaded from {self.config_path}")
            return config_data
        
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return None

# Helper function to convert Pydantic model to dict
def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    """Convert a Pydantic model to a dictionary suitable for JSON serialization"""
    return json.loads(model.json())