from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from loguru import logger

class BaseModel(nn.Module, ABC):
    """Base class for all PyTorch models in the project."""
    
    def __init__(self, device: str = "cuda"):
        """Initialize the model.
        
        Args:
            device: Device to run the model on ("cuda" or "cpu")
        """
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    def load_weights(self, weights_path: str) -> None:
        """Load model weights from a file.
        
        Args:
            weights_path: Path to the weights file
        """
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            self.load_state_dict(state_dict)
            logger.info(f"Successfully loaded weights from {weights_path}")
        except Exception as e:
            logger.error(f"Error loading weights from {weights_path}: {str(e)}")
            raise
    
    def save_weights(self, weights_path: str) -> None:
        """Save model weights to a file.
        
        Args:
            weights_path: Path to save the weights file
        """
        try:
            torch.save(self.state_dict(), weights_path)
            logger.info(f"Successfully saved weights to {weights_path}")
        except Exception as e:
            logger.error(f"Error saving weights to {weights_path}: {str(e)}")
            raise
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference on input tensor.
        
        Args:
            x: Input tensor
            
        Returns:
            Model predictions
        """
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            return self(x)
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Perform a single training step.
        
        Args:
            x: Input tensor
            y: Target tensor
            criterion: Loss function
            optimizer: Optimizer
            
        Returns:
            Dictionary containing loss and other metrics
        """
        self.train()
        x = x.to(self.device)
        y = y.to(self.device)
        
        optimizer.zero_grad()
        output = self(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        return {"loss": loss.item()} 