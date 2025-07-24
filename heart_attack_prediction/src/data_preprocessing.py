"""
Módulo para preprocesamiento de datos del dataset de ataques cardíacos.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Clase para preprocesar datos del dataset de ataques cardíacos.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'heart_attack'
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Carga el dataset desde un archivo CSV.
        
        Args:
            filepath: Ruta al archivo CSV
            
        Returns:
            DataFrame con los datos cargados
        """
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
            return df
        except Exception as e:
            logger.error(f"Error al cargar datos: {e}")
            raise
            
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia el dataset eliminando valores nulos y outliers extremos.
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame limpio
        """
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        # Eliminar filas con valores nulos
        df_clean = df_clean.dropna()
        logger.info(f"Filas eliminadas por valores nulos: {initial_rows - len(df_clean)}")
        
        # Eliminar outliers extremos usando IQR
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        numeric_columns = numeric_columns.drop(self.target_column, errors='ignore')
        
        for col in numeric_columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers_before = len(df_clean)
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            outliers_removed = outliers_before - len(df_clean)
            
            if outliers_removed > 0:
                logger.info(f"Outliers eliminados en {col}: {outliers_removed}")
        
        logger.info(f"Dataset limpio: {len(df_clean)} filas restantes")
        return df_clean
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara las características y etiquetas para el entrenamiento.
        
        Args:
            df: DataFrame limpio
            
        Returns:
            Tuple con características (X) y etiquetas (y)
        """
        # Separar características y target
        if self.target_column not in df.columns:
            raise ValueError(f"Columna target '{self.target_column}' no encontrada")
            
        self.feature_columns = [col for col in df.columns if col != self.target_column]
        
        X = df[self.feature_columns].values
        y = df[self.target_column].values
        
        logger.info(f"Características preparadas: {X.shape}")
        logger.info(f"Distribución de clases: {np.bincount(y)}")
        
        return X, y
        
    def normalize_features(self, X_train: np.ndarray, X_test: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normaliza las características usando StandardScaler.
        
        Args:
            X_train: Características de entrenamiento
            X_test: Características de prueba (opcional)
            
        Returns:
            Características normalizadas
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled, None
        
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Divide los datos en conjuntos de entrenamiento y prueba.
        
        Args:
            X: Características
            y: Etiquetas
            test_size: Proporción del conjunto de prueba
            random_state: Semilla para reproducibilidad
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"División completada:")
        logger.info(f"  Entrenamiento: {X_train.shape[0]} muestras")
        logger.info(f"  Prueba: {X_test.shape[0]} muestras")
        
        return X_train, X_test, y_train, y_test
        
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Obtiene información sobre las características procesadas.
        
        Returns:
            Diccionario con información de características
        """
        return {
            'feature_columns': self.feature_columns,
            'n_features': len(self.feature_columns) if self.feature_columns else 0,
            'scaler_mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
            'scaler_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None
        }
        
    def preprocess_pipeline(self, filepath: str, test_size: float = 0.2, 
                          random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Pipeline completo de preprocesamiento.
        
        Args:
            filepath: Ruta al archivo CSV
            test_size: Proporción del conjunto de prueba
            random_state: Semilla para reproducibilidad
            
        Returns:
            X_train, X_test, y_train, y_test normalizados
        """
        # Cargar y limpiar datos
        df = self.load_data(filepath)
        df_clean = self.clean_data(df)
        
        # Preparar características
        X, y = self.prepare_features(df_clean)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size, random_state)
        
        # Normalizar características
        X_train_scaled, X_test_scaled = self.normalize_features(X_train, X_test)
        
        logger.info("Pipeline de preprocesamiento completado")
        
        return X_train_scaled, X_test_scaled, y_train, y_test