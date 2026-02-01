
import requests
import os
from typing import Optional, Dict, Any

class APIClient:
    """
    Client for the Lego Sorter Inference API.
    """
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
    
    def predict_from_image(self, image_path: str) -> Dict[str, Any]:
        """
        Send an image to the API for prediction.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            JSON response from the API.
            
        Raises:
            IOError: If file not found or API error.
        """
        if not os.path.exists(image_path):
            raise IOError(f"Image file not found: {image_path}")
            
        url = f"{self.base_url}/v1/predict"
        
        try:
            with open(image_path, 'rb') as f:
                files = {'image': ('image.jpg', f, 'image/jpeg')}
                response = requests.post(url, files=files)
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise IOError(f"API request failed: {e}")
