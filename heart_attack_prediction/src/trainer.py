"""
Entrenador para el modelo de predicción de ataques cardíacos.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class HeartAttackTrainer:
    """
    Clase para entrenar y evaluar el modelo de predicción de ataques cardíacos.
    """
    
    def __init__(self, model: nn.Module, device: str = None):
        """
        Inicializa el entrenador.
        
        Args:
            model: Modelo de PyTorch
            device: Dispositivo para entrenamiento ('cuda' o 'cpu')
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Criterio de pérdida para clasificación binaria
        self.criterion = nn.BCELoss()
        
        # Historial de entrenamiento
        self.train_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []
        
        logger.info(f"Entrenador inicializado en dispositivo: {self.device}")
        
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer) -> Tuple[float, float]:
        """
        Entrena el modelo por una época.
        
        Args:
            train_loader: DataLoader de entrenamiento
            optimizer: Optimizador
            
        Returns:
            Tuple con pérdida promedio y precisión de la época
        """
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            features, targets = features.to(self.device), targets.to(self.device)
            targets = targets.unsqueeze(1)  # Añadir dimensión para BCELoss
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calcular métricas
            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)
            
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
        
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Evalúa el modelo en el conjunto de validación.
        
        Args:
            val_loader: DataLoader de validación
            
        Returns:
            Tuple con pérdida promedio y precisión de validación
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                targets = targets.unsqueeze(1)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                predictions = (outputs > 0.5).float()
                correct_predictions += (predictions == targets).sum().item()
                total_samples += targets.size(0)
                
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 100, learning_rate: float = 0.001,
              weight_decay: float = 1e-5, patience: int = 10,
              min_delta: float = 1e-4) -> Dict[str, List[float]]:
        """
        Entrena el modelo completo.
        
        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            epochs: Número de épocas
            learning_rate: Tasa de aprendizaje
            weight_decay: Regularización L2
            patience: Paciencia para early stopping
            min_delta: Mejora mínima para early stopping
            
        Returns:
            Diccionario con historial de entrenamiento
        """
        optimizer = optim.Adam(self.model.parameters(), 
                              lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        logger.info(f"Iniciando entrenamiento por {epochs} épocas")
        
        for epoch in range(epochs):
            # Entrenamiento
            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            
            # Validación
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Guardar métricas
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Scheduler step
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                # Guardar mejor modelo
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
            # Log progreso
            if (epoch + 1) % 10 == 0:
                logger.info(f"Época {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping en época {epoch+1}")
                break
                
        training_time = time.time() - start_time
        logger.info(f"Entrenamiento completado en {training_time:.2f} segundos")
        
        # Cargar mejor modelo
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
            
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        
    def plot_training_curves(self, save_path: Optional[str] = None):
        """
        Genera gráficos de las curvas de entrenamiento.
        
        Args:
            save_path: Ruta para guardar los gráficos
        """
        plt.style.use('seaborn-v0_8')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Gráfico de pérdida
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Entrenamiento', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='Validación', linewidth=2)
        ax1.set_title('Curva de Pérdida', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Pérdida (BCE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de precisión
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Entrenamiento', linewidth=2)
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validación', linewidth=2)
        ax2.set_title('Curva de Precisión', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Precisión')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráficos guardados en: {save_path}")
        
        plt.show()
        
    def save_checkpoint(self, filepath: str, epoch: int, optimizer: optim.Optimizer = None):
        """
        Guarda un checkpoint del modelo.
        
        Args:
            filepath: Ruta para guardar el checkpoint
            epoch: Época actual
            optimizer: Optimizador (opcional)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'model_config': {
                'input_dim': getattr(self.model, 'input_dim', None),
                'hidden_dims': getattr(self.model, 'hidden_dims', None),
                'dropout_rate': getattr(self.model, 'dropout_rate', None)
            }
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint guardado: {filepath}")
        
    def load_checkpoint(self, filepath: str, optimizer: optim.Optimizer = None) -> int:
        """
        Carga un checkpoint del modelo.
        
        Args:
            filepath: Ruta del checkpoint
            optimizer: Optimizador (opcional)
            
        Returns:
            Época del checkpoint cargado
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        epoch = checkpoint['epoch']
        logger.info(f"Checkpoint cargado desde época {epoch}: {filepath}")
        
        return epoch
        
    def get_training_summary(self) -> Dict:
        """
        Retorna un resumen del entrenamiento.
        
        Returns:
            Diccionario con métricas de resumen
        """
        if not self.train_losses:
            return {"error": "No hay datos de entrenamiento disponibles"}
            
        return {
            'total_epochs': len(self.train_losses),
            'final_train_loss': self.train_losses[-1],
            'final_train_accuracy': self.train_accuracies[-1],
            'final_val_loss': self.val_losses[-1],
            'final_val_accuracy': self.val_accuracies[-1],
            'best_val_loss': min(self.val_losses),
            'best_val_accuracy': max(self.val_accuracies),
            'best_epoch': np.argmin(self.val_losses) + 1
        }