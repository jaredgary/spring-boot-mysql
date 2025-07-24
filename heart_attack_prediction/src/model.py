"""
Modelo de red neuronal para predicción de ataques cardíacos.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class HeartAttackPredictor(nn.Module):
    """
    Red neuronal feedforward para predicción de ataques cardíacos.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32], 
                 dropout_rate: float = 0.3):
        """
        Inicializa el modelo.
        
        Args:
            input_dim: Dimensión de entrada (número de características)
            hidden_dims: Lista con dimensiones de capas ocultas
            dropout_rate: Tasa de dropout para regularización
        """
        super(HeartAttackPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Construir capas dinámicamente
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            
        # Capa de salida (clasificación binaria)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Inicializar pesos
        self._initialize_weights()
        
        logger.info(f"Modelo creado: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> 1")
        logger.info(f"Parámetros totales: {self.count_parameters()}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo.
        
        Args:
            x: Tensor de entrada con forma (batch_size, input_dim)
            
        Returns:
            Probabilidades de predicción con forma (batch_size, 1)
        """
        return self.network(x)
        
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Realiza predicciones binarias.
        
        Args:
            x: Tensor de entrada
            threshold: Umbral para clasificación binaria
            
        Returns:
            Predicciones binarias (0 o 1)
        """
        with torch.no_grad():
            probabilities = self.forward(x)
            predictions = (probabilities > threshold).float()
        return predictions
        
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retorna probabilidades de predicción.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Probabilidades de la clase positiva
        """
        with torch.no_grad():
            probabilities = self.forward(x)
        return probabilities
        
    def _initialize_weights(self):
        """Inicializa los pesos del modelo usando Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
                
    def count_parameters(self) -> int:
        """Cuenta el número total de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def get_model_info(self) -> dict:
        """Retorna información del modelo."""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'total_parameters': self.count_parameters(),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.parameters()) / (1024 * 1024)
        }


class SimpleHeartAttackPredictor(nn.Module):
    """
    Versión simplificada del modelo para casos con menos datos.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.2):
        """
        Inicializa el modelo simple.
        
        Args:
            input_dim: Dimensión de entrada
            hidden_dim: Dimensión de la capa oculta
            dropout_rate: Tasa de dropout
        """
        super(SimpleHeartAttackPredictor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
        logger.info(f"Modelo simple creado: {input_dim} -> {hidden_dim} -> {hidden_dim//2} -> 1")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
        
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)


def create_model(input_dim: int, model_type: str = "standard", 
                 hidden_dims: Optional[List[int]] = None, 
                 dropout_rate: float = 0.3) -> nn.Module:
    """
    Factory function para crear modelos.
    
    Args:
        input_dim: Dimensión de entrada
        model_type: Tipo de modelo ("standard" o "simple")
        hidden_dims: Dimensiones de capas ocultas (solo para standard)
        dropout_rate: Tasa de dropout
        
    Returns:
        Modelo de PyTorch
    """
    if model_type == "simple":
        return SimpleHeartAttackPredictor(input_dim, dropout_rate=dropout_rate)
    elif model_type == "standard":
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        return HeartAttackPredictor(input_dim, hidden_dims, dropout_rate)
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")


class ModelConfig:
    """Configuración del modelo."""
    
    def __init__(self, input_dim: int, model_type: str = "standard",
                 hidden_dims: Optional[List[int]] = None, dropout_rate: float = 0.3,
                 learning_rate: float = 0.001, weight_decay: float = 1e-5):
        self.input_dim = input_dim
        self.model_type = model_type
        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
    def create_model(self) -> nn.Module:
        """Crea el modelo basado en la configuración."""
        return create_model(
            self.input_dim, 
            self.model_type, 
            self.hidden_dims, 
            self.dropout_rate
        )