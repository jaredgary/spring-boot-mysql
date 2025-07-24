"""
Evaluador para el modelo de predicción de ataques cardíacos.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class HeartAttackEvaluator:
    """
    Clase para evaluar el modelo de predicción de ataques cardíacos.
    """
    
    def __init__(self, model: nn.Module, device: str = None):
        """
        Inicializa el evaluador.
        
        Args:
            model: Modelo entrenado de PyTorch
            device: Dispositivo para evaluación ('cuda' o 'cpu')
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Evaluador inicializado en dispositivo: {self.device}")
        
    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Realiza predicciones en un conjunto de datos.
        
        Args:
            dataloader: DataLoader con los datos a evaluar
            
        Returns:
            Tuple con (etiquetas_reales, probabilidades, predicciones_binarias)
        """
        all_labels = []
        all_probabilities = []
        all_predictions = []
        
        with torch.no_grad():
            for features, labels in dataloader:
                features = features.to(self.device)
                
                # Obtener probabilidades
                outputs = self.model(features)
                probabilities = outputs.cpu().numpy().flatten()
                
                # Convertir a predicciones binarias
                predictions = (probabilities > 0.5).astype(int)
                
                all_labels.extend(labels.numpy())
                all_probabilities.extend(probabilities)
                all_predictions.extend(predictions)
                
        return np.array(all_labels), np.array(all_probabilities), np.array(all_predictions)
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_prob: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas de evaluación completas.
        
        Args:
            y_true: Etiquetas reales
            y_pred: Predicciones binarias
            y_prob: Probabilidades de predicción
            
        Returns:
            Diccionario con todas las métricas
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'specificity': self._calculate_specificity(y_true, y_pred),
            'sensitivity': recall_score(y_true, y_pred, zero_division=0),  # Mismo que recall
        }
        
        return metrics
        
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula especificidad (True Negative Rate).
        
        Args:
            y_true: Etiquetas reales
            y_pred: Predicciones
            
        Returns:
            Especificidad
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return specificity
        
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluación completa del modelo.
        
        Args:
            test_loader: DataLoader de prueba
            
        Returns:
            Diccionario con métricas y resultados
        """
        logger.info("Iniciando evaluación del modelo...")
        
        # Obtener predicciones
        y_true, y_prob, y_pred = self.predict(test_loader)
        
        # Calcular métricas
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        
        # Reporte de clasificación
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        # Log resultados
        logger.info("Resultados de evaluación:")
        for metric, value in metrics.items():
            logger.info(f"  {metric.capitalize()}: {value:.4f}")
            
        return {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions': {
                'y_true': y_true,
                'y_prob': y_prob,
                'y_pred': y_pred
            }
        }
        
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Optional[str] = None):
        """
        Visualiza la matriz de confusión.
        
        Args:
            cm: Matriz de confusión
            save_path: Ruta para guardar el gráfico
        """
        plt.figure(figsize=(8, 6))
        
        # Crear heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Sin Ataque', 'Con Ataque'],
                   yticklabels=['Sin Ataque', 'Con Ataque'])
        
        plt.title('Matriz de Confusión', fontsize=14, fontweight='bold')
        plt.xlabel('Predicción')
        plt.ylabel('Realidad')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Matriz de confusión guardada en: {save_path}")
            
        plt.show()
        
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, 
                      save_path: Optional[str] = None):
        """
        Visualiza la curva ROC.
        
        Args:
            y_true: Etiquetas reales
            y_prob: Probabilidades de predicción
            save_path: Ruta para guardar el gráfico
        """
        # Calcular curva ROC
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        
        # Plotear curva ROC
        plt.plot(fpr, tpr, 'b-', linewidth=2, 
                label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva ROC', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Curva ROC guardada en: {save_path}")
            
        plt.show()
        
    def plot_probability_distribution(self, y_true: np.ndarray, y_prob: np.ndarray,
                                    save_path: Optional[str] = None):
        """
        Visualiza la distribución de probabilidades por clase.
        
        Args:
            y_true: Etiquetas reales
            y_prob: Probabilidades de predicción
            save_path: Ruta para guardar el gráfico
        """
        plt.figure(figsize=(10, 6))
        
        # Separar probabilidades por clase
        prob_class_0 = y_prob[y_true == 0]
        prob_class_1 = y_prob[y_true == 1]
        
        # Crear histogramas
        plt.hist(prob_class_0, bins=30, alpha=0.7, label='Sin Ataque Cardíaco', 
                color='blue', density=True)
        plt.hist(prob_class_1, bins=30, alpha=0.7, label='Con Ataque Cardíaco', 
                color='red', density=True)
        
        plt.axvline(x=0.5, color='black', linestyle='--', 
                   label='Umbral de Decisión (0.5)')
        
        plt.xlabel('Probabilidad Predicha')
        plt.ylabel('Densidad')
        plt.title('Distribución de Probabilidades por Clase', 
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distribución de probabilidades guardada en: {save_path}")
            
        plt.show()
        
    def find_optimal_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        """
        Encuentra el umbral óptimo para clasificación.
        
        Args:
            y_true: Etiquetas reales
            y_prob: Probabilidades de predicción
            
        Returns:
            Diccionario con el umbral óptimo y métricas asociadas
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        # Calcular Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Evaluar con umbral óptimo
        y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
        optimal_metrics = self.calculate_metrics(y_true, y_pred_optimal, y_prob)
        
        logger.info(f"Umbral óptimo encontrado: {optimal_threshold:.3f}")
        
        return {
            'optimal_threshold': optimal_threshold,
            'metrics_at_optimal': optimal_metrics,
            'j_score': j_scores[optimal_idx]
        }
        
    def generate_evaluation_report(self, test_loader: DataLoader, 
                                 save_plots: bool = True, 
                                 plots_dir: str = "results/plots") -> Dict:
        """
        Genera reporte completo de evaluación.
        
        Args:
            test_loader: DataLoader de prueba
            save_plots: Si guardar los gráficos
            plots_dir: Directorio para guardar gráficos
            
        Returns:
            Diccionario con reporte completo
        """
        # Evaluación básica
        results = self.evaluate(test_loader)
        
        # Encontrar umbral óptimo
        y_true = results['predictions']['y_true']
        y_prob = results['predictions']['y_prob']
        y_pred = results['predictions']['y_pred']
        
        optimal_threshold_info = self.find_optimal_threshold(y_true, y_prob)
        
        # Generar gráficos si se solicita
        if save_plots:
            from pathlib import Path
            Path(plots_dir).mkdir(parents=True, exist_ok=True)
            
            self.plot_confusion_matrix(results['confusion_matrix'], 
                                     f"{plots_dir}/confusion_matrix.png")
            self.plot_roc_curve(y_true, y_prob, 
                              f"{plots_dir}/roc_curve.png")
            self.plot_probability_distribution(y_true, y_prob,
                                             f"{plots_dir}/probability_distribution.png")
        else:
            self.plot_confusion_matrix(results['confusion_matrix'])
            self.plot_roc_curve(y_true, y_prob)
            self.plot_probability_distribution(y_true, y_prob)
            
        # Compilar reporte completo
        full_report = {
            **results,
            'optimal_threshold_analysis': optimal_threshold_info,
            'sample_size': len(y_true),
            'class_distribution': {
                'no_attack': int(np.sum(y_true == 0)),
                'attack': int(np.sum(y_true == 1))
            }
        }
        
        return full_report