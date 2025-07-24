"""
API Flask para predicción de ataques cardíacos.
"""

from flask import Flask, request, jsonify, render_template_string
import torch
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, List
import sys

# Añadir directorio padre al path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import HeartAttackPredictor
from src.data_preprocessing import DataPreprocessor

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Variables globales para modelo y preprocessor
model = None
preprocessor = None
feature_columns = None

def load_model_and_preprocessor():
    """Carga el modelo entrenado y el preprocessor."""
    global model, preprocessor, feature_columns
    
    try:
        # Cargar checkpoint del modelo
        checkpoint_path = Path(__file__).parent.parent / "models" / "best_model.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_config = checkpoint['model_config']
        
        # Recrear modelo
        model = HeartAttackPredictor(
            input_dim=model_config['input_dim'],
            hidden_dims=model_config['hidden_dims'],
            dropout_rate=model_config['dropout_rate']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Cargar preprocessor
        preprocessor_path = Path(__file__).parent.parent / "models" / "preprocessor.pkl"
        if preprocessor_path.exists():
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
                feature_columns = preprocessor.feature_columns
        else:
            # Crear preprocessor por defecto si no existe
            preprocessor = DataPreprocessor()
            feature_columns = [
                'age', 'gender', 'systolic_bp', 'diastolic_bp', 'cholesterol',
                'bmi', 'smoker', 'exercise_hours', 'alcohol_units', 
                'diabetes', 'family_history'
            ]
            
        logger.info("Modelo y preprocessor cargados exitosamente")
        
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        raise

@app.before_first_request
def initialize():
    """Inicializa el modelo al arrancar la aplicación."""
    load_model_and_preprocessor()

@app.route('/')
def home():
    """Página principal con formulario de predicción."""
    html_form = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Predictor de Ataques Cardíacos</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
            button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; width: 100%; }
            button:hover { background-color: #0056b3; }
            .result { margin-top: 20px; padding: 15px; border-radius: 4px; }
            .high-risk { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .low-risk { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🫀 Predictor de Ataques Cardíacos</h1>
            <form id="predictionForm">
                <div class="form-group">
                    <label for="age">Edad:</label>
                    <input type="number" id="age" name="age" min="18" max="100" required>
                </div>
                
                <div class="form-group">
                    <label for="gender">Género:</label>
                    <select id="gender" name="gender" required>
                        <option value="">Seleccionar...</option>
                        <option value="0">Mujer</option>
                        <option value="1">Hombre</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="systolic_bp">Presión Sistólica (mmHg):</label>
                    <input type="number" id="systolic_bp" name="systolic_bp" min="90" max="200" required>
                </div>
                
                <div class="form-group">
                    <label for="diastolic_bp">Presión Diastólica (mmHg):</label>
                    <input type="number" id="diastolic_bp" name="diastolic_bp" min="60" max="120" required>
                </div>
                
                <div class="form-group">
                    <label for="cholesterol">Colesterol (mg/dL):</label>
                    <input type="number" id="cholesterol" name="cholesterol" min="120" max="350" required>
                </div>
                
                <div class="form-group">
                    <label for="bmi">IMC:</label>
                    <input type="number" id="bmi" name="bmi" step="0.1" min="15" max="45" required>
                </div>
                
                <div class="form-group">
                    <label for="smoker">¿Fumador?:</label>
                    <select id="smoker" name="smoker" required>
                        <option value="">Seleccionar...</option>
                        <option value="0">No</option>
                        <option value="1">Sí</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="exercise_hours">Horas de ejercicio por semana:</label>
                    <input type="number" id="exercise_hours" name="exercise_hours" step="0.5" min="0" max="20" required>
                </div>
                
                <div class="form-group">
                    <label for="alcohol_units">Unidades de alcohol por semana:</label>
                    <input type="number" id="alcohol_units" name="alcohol_units" step="0.5" min="0" max="30" required>
                </div>
                
                <div class="form-group">
                    <label for="diabetes">¿Diabetes?:</label>
                    <select id="diabetes" name="diabetes" required>
                        <option value="">Seleccionar...</option>
                        <option value="0">No</option>
                        <option value="1">Sí</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="family_history">¿Historial familiar de enfermedades cardíacas?:</label>
                    <select id="family_history" name="family_history" required>
                        <option value="">Seleccionar...</option>
                        <option value="0">No</option>
                        <option value="1">Sí</option>
                    </select>
                </div>
                
                <button type="submit">Predecir Riesgo</button>
            </form>
            
            <div id="result"></div>
        </div>
        
        <script>
            document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData(e.target);
                const data = Object.fromEntries(formData.entries());
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        const riskLevel = result.probability > 0.5 ? 'high-risk' : 'low-risk';
                        const riskText = result.probability > 0.5 ? 'ALTO RIESGO' : 'BAJO RIESGO';
                        
                        document.getElementById('result').innerHTML = `
                            <div class="result ${riskLevel}">
                                <h3>Resultado de la Predicción</h3>
                                <p><strong>Riesgo de Ataque Cardíaco:</strong> ${riskText}</p>
                                <p><strong>Probabilidad:</strong> ${(result.probability * 100).toFixed(1)}%</p>
                                <p><strong>Predicción:</strong> ${result.prediction === 1 ? 'Riesgo de ataque cardíaco' : 'Sin riesgo inmediato'}</p>
                            </div>
                        `;
                    } else {
                        document.getElementById('result').innerHTML = `
                            <div class="result high-risk">
                                <h3>Error</h3>
                                <p>${result.error}</p>
                            </div>
                        `;
                    }
                } catch (error) {
                    document.getElementById('result').innerHTML = `
                        <div class="result high-risk">
                            <h3>Error</h3>
                            <p>Error de conexión: ${error.message}</p>
                        </div>
                    `;
                }
            });
        </script>
    </body>
    </html>
    """
    return html_form

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para realizar predicciones."""
    try:
        # Obtener datos del request
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No se proporcionaron datos'})
        
        # Validar características requeridas
        required_features = [
            'age', 'gender', 'systolic_bp', 'diastolic_bp', 'cholesterol',
            'bmi', 'smoker', 'exercise_hours', 'alcohol_units', 
            'diabetes', 'family_history'
        ]
        
        missing_features = [f for f in required_features if f not in data]
        if missing_features:
            return jsonify({
                'success': False, 
                'error': f'Características faltantes: {missing_features}'
            })
        
        # Convertir a array numpy
        features = np.array([[float(data[f]) for f in required_features]])
        
        # Normalizar características si existe preprocessor
        if preprocessor and hasattr(preprocessor, 'scaler'):
            features = preprocessor.scaler.transform(features)
        
        # Realizar predicción
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features)
            probability = model(features_tensor).item()
            prediction = 1 if probability > 0.5 else 0
        
        return jsonify({
            'success': True,
            'probability': probability,
            'prediction': prediction,
            'risk_level': 'alto' if probability > 0.5 else 'bajo'
        })
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Endpoint para predicciones en lote."""
    try:
        data = request.get_json()
        
        if not data or 'samples' not in data:
            return jsonify({'success': False, 'error': 'Formato de datos incorrecto'})
        
        samples = data['samples']
        results = []
        
        for sample in samples:
            # Procesar cada muestra
            required_features = [
                'age', 'gender', 'systolic_bp', 'diastolic_bp', 'cholesterol',
                'bmi', 'smoker', 'exercise_hours', 'alcohol_units', 
                'diabetes', 'family_history'
            ]
            
            features = np.array([[float(sample[f]) for f in required_features]])
            
            if preprocessor and hasattr(preprocessor, 'scaler'):
                features = preprocessor.scaler.transform(features)
            
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features)
                probability = model(features_tensor).item()
                prediction = 1 if probability > 0.5 else 0
            
            results.append({
                'probability': probability,
                'prediction': prediction,
                'risk_level': 'alto' if probability > 0.5 else 'bajo'
            })
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error en predicción batch: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/model_info')
def model_info():
    """Información del modelo."""
    try:
        info = {
            'model_loaded': model is not None,
            'preprocessor_loaded': preprocessor is not None,
            'feature_columns': feature_columns,
            'input_features': len(feature_columns) if feature_columns else 0
        }
        
        if model:
            info.update({
                'model_parameters': model.count_parameters(),
                'model_type': type(model).__name__
            })
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None
    })

if __name__ == '__main__':
    # Cargar modelo si no se ha cargado
    if model is None:
        try:
            load_model_and_preprocessor()
        except Exception as e:
            logger.error(f"No se pudo cargar el modelo: {e}")
            logger.info("La API funcionará sin modelo cargado")
    
    app.run(debug=True, host='0.0.0.0', port=5000)