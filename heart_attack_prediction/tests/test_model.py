"""
Pruebas unitarias para el modelo de red neuronal.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Añadir directorio padre al path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import HeartAttackPredictor, SimpleHeartAttackPredictor, create_model, ModelConfig


class TestHeartAttackPredictor(unittest.TestCase):
    """Tests para la clase HeartAttackPredictor."""
    
    def setUp(self):
        """Configuración inicial para cada test."""
        self.input_dim = 11
        self.hidden_dims = [64, 32, 16]
        self.dropout_rate = 0.3
        self.model = HeartAttackPredictor(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate
        )
        
    def test_model_initialization(self):
        """Test inicialización del modelo."""
        self.assertEqual(self.model.input_dim, self.input_dim)
        self.assertEqual(self.model.hidden_dims, self.hidden_dims)
        self.assertEqual(self.model.dropout_rate, self.dropout_rate)
        self.assertIsInstance(self.model.network, nn.Sequential)
        
    def test_model_forward_pass(self):
        """Test forward pass del modelo."""
        batch_size = 32
        input_tensor = torch.randn(batch_size, self.input_dim)
        
        output = self.model(input_tensor)
        
        # Verificar dimensiones de salida
        self.assertEqual(output.shape, (batch_size, 1))
        
        # Verificar que las salidas están entre 0 y 1 (por sigmoid)
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))
        
    def test_model_predict(self):
        """Test función de predicción."""
        input_tensor = torch.randn(10, self.input_dim)
        
        predictions = self.model.predict(input_tensor)
        
        # Verificar dimensiones
        self.assertEqual(predictions.shape, (10, 1))
        
        # Verificar que las predicciones son binarias
        self.assertTrue(torch.all((predictions == 0) | (predictions == 1)))
        
    def test_model_predict_proba(self):
        """Test función de probabilidades."""
        input_tensor = torch.randn(10, self.input_dim)
        
        probabilities = self.model.predict_proba(input_tensor)
        
        # Verificar dimensiones
        self.assertEqual(probabilities.shape, (10, 1))
        
        # Verificar que las probabilidades están entre 0 y 1
        self.assertTrue(torch.all(probabilities >= 0))
        self.assertTrue(torch.all(probabilities <= 1))
        
    def test_count_parameters(self):
        """Test conteo de parámetros."""
        param_count = self.model.count_parameters()
        
        # Verificar que es un número positivo
        self.assertIsInstance(param_count, int)
        self.assertGreater(param_count, 0)
        
        # Verificar cálculo manual
        manual_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(param_count, manual_count)
        
    def test_get_model_info(self):
        """Test información del modelo."""
        info = self.model.get_model_info()
        
        expected_keys = ['input_dim', 'hidden_dims', 'dropout_rate', 'total_parameters', 'model_size_mb']
        for key in expected_keys:
            self.assertIn(key, info)
            
        self.assertEqual(info['input_dim'], self.input_dim)
        self.assertEqual(info['hidden_dims'], self.hidden_dims)
        self.assertEqual(info['dropout_rate'], self.dropout_rate)
        self.assertGreater(info['total_parameters'], 0)
        self.assertGreater(info['model_size_mb'], 0)
        
    def test_model_with_different_dimensions(self):
        """Test modelo con diferentes dimensiones."""
        models_configs = [
            (5, [32]),
            (20, [128, 64]),
            (15, [100, 50, 25, 10])
        ]
        
        for input_dim, hidden_dims in models_configs:
            with self.subTest(input_dim=input_dim, hidden_dims=hidden_dims):
                model = HeartAttackPredictor(input_dim, hidden_dims)
                
                # Test forward pass
                input_tensor = torch.randn(5, input_dim)
                output = model(input_tensor)
                
                self.assertEqual(output.shape, (5, 1))
                self.assertTrue(torch.all(output >= 0))
                self.assertTrue(torch.all(output <= 1))


class TestSimpleHeartAttackPredictor(unittest.TestCase):
    """Tests para la clase SimpleHeartAttackPredictor."""
    
    def setUp(self):
        """Configuración inicial para cada test."""
        self.input_dim = 11
        self.hidden_dim = 64
        self.dropout_rate = 0.2
        self.model = SimpleHeartAttackPredictor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate
        )
        
    def test_simple_model_initialization(self):
        """Test inicialización del modelo simple."""
        self.assertIsInstance(self.model.network, nn.Sequential)
        
    def test_simple_model_forward_pass(self):
        """Test forward pass del modelo simple."""
        batch_size = 16
        input_tensor = torch.randn(batch_size, self.input_dim)
        
        output = self.model(input_tensor)
        
        # Verificar dimensiones de salida
        self.assertEqual(output.shape, (batch_size, 1))
        
        # Verificar que las salidas están entre 0 y 1
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))
        
    def test_simple_model_structure(self):
        """Test estructura del modelo simple."""
        # Verificar que tiene la estructura esperada
        layers = list(self.model.network.children())
        
        # Debe tener: Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear, Sigmoid
        expected_layer_types = [
            nn.Linear, nn.ReLU, nn.Dropout,
            nn.Linear, nn.ReLU, nn.Dropout,
            nn.Linear, nn.Sigmoid
        ]
        
        self.assertEqual(len(layers), len(expected_layer_types))
        
        for layer, expected_type in zip(layers, expected_layer_types):
            self.assertIsInstance(layer, expected_type)


class TestModelFactory(unittest.TestCase):
    """Tests para las funciones factory de modelos."""
    
    def test_create_model_standard(self):
        """Test creación de modelo estándar."""
        input_dim = 11
        hidden_dims = [128, 64, 32]
        dropout_rate = 0.3
        
        model = create_model(
            input_dim=input_dim,
            model_type="standard",
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )
        
        self.assertIsInstance(model, HeartAttackPredictor)
        self.assertEqual(model.input_dim, input_dim)
        self.assertEqual(model.hidden_dims, hidden_dims)
        self.assertEqual(model.dropout_rate, dropout_rate)
        
    def test_create_model_simple(self):
        """Test creación de modelo simple."""
        input_dim = 11
        dropout_rate = 0.2
        
        model = create_model(
            input_dim=input_dim,
            model_type="simple",
            dropout_rate=dropout_rate
        )
        
        self.assertIsInstance(model, SimpleHeartAttackPredictor)
        
    def test_create_model_invalid_type(self):
        """Test error con tipo de modelo inválido."""
        with self.assertRaises(ValueError):
            create_model(input_dim=11, model_type="invalid_type")
            
    def test_create_model_default_params(self):
        """Test creación de modelo con parámetros por defecto."""
        model = create_model(input_dim=11)
        
        self.assertIsInstance(model, HeartAttackPredictor)
        self.assertEqual(model.input_dim, 11)
        self.assertEqual(model.hidden_dims, [128, 64, 32])
        self.assertEqual(model.dropout_rate, 0.3)


class TestModelConfig(unittest.TestCase):
    """Tests para la clase ModelConfig."""
    
    def test_model_config_initialization(self):
        """Test inicialización de ModelConfig."""
        input_dim = 15
        config = ModelConfig(input_dim=input_dim)
        
        self.assertEqual(config.input_dim, input_dim)
        self.assertEqual(config.model_type, "standard")
        self.assertEqual(config.hidden_dims, [128, 64, 32])
        self.assertEqual(config.dropout_rate, 0.3)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.weight_decay, 1e-5)
        
    def test_model_config_custom_params(self):
        """Test ModelConfig con parámetros personalizados."""
        config = ModelConfig(
            input_dim=20,
            model_type="simple",
            hidden_dims=[100, 50],
            dropout_rate=0.4,
            learning_rate=0.01,
            weight_decay=1e-4
        )
        
        self.assertEqual(config.input_dim, 20)
        self.assertEqual(config.model_type, "simple")
        self.assertEqual(config.hidden_dims, [100, 50])
        self.assertEqual(config.dropout_rate, 0.4)
        self.assertEqual(config.learning_rate, 0.01)
        self.assertEqual(config.weight_decay, 1e-4)
        
    def test_model_config_create_model(self):
        """Test creación de modelo desde ModelConfig."""
        config = ModelConfig(input_dim=11, model_type="standard")
        model = config.create_model()
        
        self.assertIsInstance(model, HeartAttackPredictor)
        self.assertEqual(model.input_dim, 11)


class TestModelConsistency(unittest.TestCase):
    """Tests para verificar consistencia del modelo."""
    
    def test_model_reproducibility(self):
        """Test reproducibilidad del modelo."""
        # Fijar semilla
        torch.manual_seed(42)
        model1 = HeartAttackPredictor(input_dim=11, hidden_dims=[64, 32])
        
        torch.manual_seed(42)
        model2 = HeartAttackPredictor(input_dim=11, hidden_dims=[64, 32])
        
        # Los modelos deben tener los mismos pesos iniciales
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.allclose(param1, param2))
            
    def test_model_gradient_flow(self):
        """Test que los gradientes fluyen correctamente."""
        model = HeartAttackPredictor(input_dim=11)
        input_tensor = torch.randn(10, 11, requires_grad=True)
        target = torch.randn(10, 1)
        
        # Forward pass
        output = model(input_tensor)
        
        # Calcular pérdida
        loss = nn.MSELoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Verificar que los gradientes existen
        for param in model.parameters():
            self.assertIsNotNone(param.grad)
            
    def test_model_eval_mode(self):
        """Test modo de evaluación del modelo."""
        model = HeartAttackPredictor(input_dim=11)
        input_tensor = torch.randn(5, 11)
        
        # Modo entrenamiento
        model.train()
        output_train1 = model(input_tensor)
        output_train2 = model(input_tensor)
        
        # En modo entrenamiento con dropout, las salidas pueden diferir
        # (aunque con probabilidad muy baja pueden ser iguales)
        
        # Modo evaluación
        model.eval()
        output_eval1 = model(input_tensor)
        output_eval2 = model(input_tensor)
        
        # En modo evaluación, las salidas deben ser idénticas
        self.assertTrue(torch.allclose(output_eval1, output_eval2))


if __name__ == '__main__':
    unittest.main()