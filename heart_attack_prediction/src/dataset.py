"""
Dataset y DataLoader de PyTorch para predicción de ataques cardíacos.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class HeartAttackDataset(Dataset):
    """
    Dataset de PyTorch para datos de ataques cardíacos.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Inicializa el dataset.
        
        Args:
            X: Características normalizadas
            y: Etiquetas binarias
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
        logger.info(f"Dataset creado: {len(self.X)} muestras, {self.X.shape[1]} características")
        
    def __len__(self) -> int:
        """Retorna el número de muestras en el dataset."""
        return len(self.X)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Obtiene una muestra del dataset.
        
        Args:
            idx: Índice de la muestra
            
        Returns:
            Tuple con características y etiqueta
        """
        return self.X[idx], self.y[idx]
        
    def get_feature_dim(self) -> int:
        """Retorna el número de características."""
        return self.X.shape[1]
        
    def get_class_distribution(self) -> dict:
        """Retorna la distribución de clases."""
        unique, counts = torch.unique(self.y, return_counts=True)
        return {int(cls): int(count) for cls, count in zip(unique, counts)}


def create_data_loaders(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       batch_size: int = 32, shuffle: bool = True,
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    Crea DataLoaders para entrenamiento y prueba.
    
    Args:
        X_train: Características de entrenamiento
        y_train: Etiquetas de entrenamiento
        X_test: Características de prueba
        y_test: Etiquetas de prueba
        batch_size: Tamaño del batch
        shuffle: Si mezclar los datos de entrenamiento
        num_workers: Número de workers para carga de datos
        
    Returns:
        DataLoaders de entrenamiento y prueba
    """
    # Crear datasets
    train_dataset = HeartAttackDataset(X_train, y_train)
    test_dataset = HeartAttackDataset(X_test, y_test)
    
    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"DataLoaders creados:")
    logger.info(f"  Entrenamiento: {len(train_loader)} batches")
    logger.info(f"  Prueba: {len(test_loader)} batches")
    logger.info(f"  Tamaño de batch: {batch_size}")
    
    return train_loader, test_loader


class HeartAttackDataModule:
    """
    Módulo de datos que encapsula la creación de datasets y dataloaders.
    """
    
    def __init__(self, batch_size: int = 32, num_workers: int = 0):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.input_dim: Optional[int] = None
        
    def setup(self, X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray):
        """
        Configura los DataLoaders con los datos proporcionados.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Etiquetas de entrenamiento
            X_test: Características de prueba
            y_test: Etiquetas de prueba
        """
        self.train_loader, self.test_loader = create_data_loaders(
            X_train, y_train, X_test, y_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        
        self.input_dim = X_train.shape[1]
        logger.info(f"DataModule configurado: dimensión de entrada = {self.input_dim}")
        
    def train_dataloader(self) -> DataLoader:
        """Retorna el DataLoader de entrenamiento."""
        if self.train_loader is None:
            raise RuntimeError("DataModule no ha sido configurado. Llama a setup() primero.")
        return self.train_loader
        
    def test_dataloader(self) -> DataLoader:
        """Retorna el DataLoader de prueba."""
        if self.test_loader is None:
            raise RuntimeError("DataModule no ha sido configurado. Llama a setup() primero.")
        return self.test_loader
        
    def get_input_dim(self) -> int:
        """Retorna la dimensión de entrada."""
        if self.input_dim is None:
            raise RuntimeError("DataModule no ha sido configurado. Llama a setup() primero.")
        return self.input_dim