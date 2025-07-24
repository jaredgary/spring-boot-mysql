"""
Pruebas unitarias para el módulo de preprocesamiento de datos.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys

# Añadir directorio padre al path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_preprocessing import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    """Tests para la clase DataPreprocessor."""
    
    def setUp(self):
        """Configuración inicial para cada test."""
        self.preprocessor = DataPreprocessor()
        
        # Crear dataset de prueba
        self.test_data = pd.DataFrame({
            'age': [45, 60, 35, 70, 50, 25, 80],
            'gender': [1, 0, 1, 0, 1, 0, 1],
            'systolic_bp': [120, 140, 110, 160, 130, 100, 180],
            'diastolic_bp': [80, 90, 70, 95, 85, 65, 100],
            'cholesterol': [200, 250, 180, 280, 220, 160, 300],
            'bmi': [25.0, 30.0, 22.0, 35.0, 27.0, 20.0, 40.0],
            'smoker': [0, 1, 0, 1, 0, 0, 1],
            'exercise_hours': [3, 1, 5, 0, 2, 6, 0],
            'alcohol_units': [2, 8, 1, 15, 3, 0, 20],
            'diabetes': [0, 1, 0, 1, 0, 0, 1],
            'family_history': [0, 1, 0, 1, 1, 0, 1],
            'heart_attack': [0, 1, 0, 1, 0, 0, 1]
        })
        
    def test_load_data(self):
        """Test carga de datos desde CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            temp_file = f.name
            
        try:
            df = self.preprocessor.load_data(temp_file)
            
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), len(self.test_data))
            self.assertEqual(list(df.columns), list(self.test_data.columns))
            
        finally:
            os.unlink(temp_file)
            
    def test_load_data_file_not_found(self):
        """Test manejo de archivo no encontrado."""
        with self.assertRaises(Exception):
            self.preprocessor.load_data('archivo_inexistente.csv')
            
    def test_clean_data(self):
        """Test limpieza de datos."""
        # Añadir valores nulos
        dirty_data = self.test_data.copy()
        dirty_data.loc[0, 'age'] = np.nan
        dirty_data.loc[1, 'cholesterol'] = np.nan
        
        clean_data = self.preprocessor.clean_data(dirty_data)
        
        # Verificar que se eliminaron las filas con valores nulos
        self.assertEqual(len(clean_data), len(self.test_data) - 2)
        self.assertFalse(clean_data.isnull().any().any())
        
    def test_clean_data_outliers(self):
        """Test eliminación de outliers extremos."""
        # Crear datos con outliers extremos
        data_with_outliers = self.test_data.copy()
        data_with_outliers.loc[0, 'age'] = 200  # Outlier extremo
        
        clean_data = self.preprocessor.clean_data(data_with_outliers)
        
        # Verificar que el outlier fue eliminado
        self.assertLess(len(clean_data), len(data_with_outliers))
        self.assertTrue(clean_data['age'].max() < 200)
        
    def test_prepare_features(self):
        """Test preparación de características."""
        X, y = self.preprocessor.prepare_features(self.test_data)
        
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(X.shape[0], len(self.test_data))
        self.assertEqual(X.shape[1], len(self.test_data.columns) - 1)  # -1 por target
        self.assertEqual(len(y), len(self.test_data))
        self.assertTrue(all(label in [0, 1] for label in y))
        
    def test_prepare_features_missing_target(self):
        """Test error cuando falta columna target."""
        data_no_target = self.test_data.drop('heart_attack', axis=1)
        
        with self.assertRaises(ValueError):
            self.preprocessor.prepare_features(data_no_target)
            
    def test_normalize_features(self):
        """Test normalización de características."""
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(20, 5)
        
        X_train_norm, X_test_norm = self.preprocessor.normalize_features(X_train, X_test)
        
        # Verificar formas
        self.assertEqual(X_train_norm.shape, X_train.shape)
        self.assertEqual(X_test_norm.shape, X_test.shape)
        
        # Verificar normalización (media aproximadamente 0, std aproximadamente 1)
        self.assertAlmostEqual(np.mean(X_train_norm), 0, places=10)
        self.assertAlmostEqual(np.std(X_train_norm), 1, places=1)
        
    def test_normalize_features_train_only(self):
        """Test normalización solo con datos de entrenamiento."""
        X_train = np.random.randn(100, 5)
        
        X_train_norm, X_test_norm = self.preprocessor.normalize_features(X_train)
        
        self.assertIsNotNone(X_train_norm)
        self.assertIsNone(X_test_norm)
        
    def test_split_data(self):
        """Test división de datos."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y, test_size=0.2)
        
        # Verificar tamaños
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)
        
        # Verificar formas
        self.assertEqual(X_train.shape[1], X.shape[1])
        self.assertEqual(X_test.shape[1], X.shape[1])
        
    def test_get_feature_info(self):
        """Test obtención de información de características."""
        # Preparar datos primero
        X, y = self.preprocessor.prepare_features(self.test_data)
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
        self.preprocessor.normalize_features(X_train, X_test)
        
        info = self.preprocessor.get_feature_info()
        
        self.assertIn('feature_columns', info)
        self.assertIn('n_features', info)
        self.assertIn('scaler_mean', info)
        self.assertIn('scaler_scale', info)
        
        self.assertEqual(info['n_features'], len(self.test_data.columns) - 1)
        self.assertIsNotNone(info['scaler_mean'])
        self.assertIsNotNone(info['scaler_scale'])
        
    def test_preprocess_pipeline(self):
        """Test pipeline completo de preprocesamiento."""
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            temp_file = f.name
            
        try:
            X_train, X_test, y_train, y_test = self.preprocessor.preprocess_pipeline(temp_file)
            
            # Verificar que todos los arrays fueron retornados
            self.assertIsInstance(X_train, np.ndarray)
            self.assertIsInstance(X_test, np.ndarray)
            self.assertIsInstance(y_train, np.ndarray)
            self.assertIsInstance(y_test, np.ndarray)
            
            # Verificar dimensiones
            self.assertEqual(X_train.shape[1], X_test.shape[1])
            self.assertEqual(len(X_train), len(y_train))
            self.assertEqual(len(X_test), len(y_test))
            
            # Verificar normalización
            self.assertAlmostEqual(np.mean(X_train), 0, places=1)
            self.assertAlmostEqual(np.std(X_train), 1, places=1)
            
        finally:
            os.unlink(temp_file)
            
    def test_feature_columns_consistency(self):
        """Test consistencia de columnas de características."""
        X, y = self.preprocessor.prepare_features(self.test_data)
        
        expected_features = [col for col in self.test_data.columns if col != 'heart_attack']
        
        self.assertEqual(self.preprocessor.feature_columns, expected_features)
        self.assertEqual(len(self.preprocessor.feature_columns), X.shape[1])


if __name__ == '__main__':
    unittest.main()