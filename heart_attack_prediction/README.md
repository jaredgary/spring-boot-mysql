# 🫀 Predictor de Ataques Cardíacos con PyTorch

Un sistema completo de Machine Learning para predecir el riesgo de ataques cardíacos utilizando PyTorch, Flask y técnicas avanzadas de procesamiento de datos.

## 📋 Características

- **Generación de Dataset Sintético**: Crea datos realistas de salud cardiovascular
- **Preprocesamiento Robusto**: Limpieza de datos, normalización y división estratificada
- **Red Neuronal con PyTorch**: Arquitectura personalizable con capas ReLU y dropout
- **Entrenamiento Avanzado**: Early stopping, learning rate scheduling y logging detallado
- **Evaluación Completa**: Métricas, curvas ROC, matrices de confusión y análisis de umbral óptimo
- **API Flask**: Interfaz web interactiva para predicciones en tiempo real
- **Pruebas Unitarias**: Suite completa de tests para garantizar calidad del código

## 🏗️ Estructura del Proyecto

```
heart_attack_prediction/
├── src/                          # Código fuente principal
│   ├── __init__.py
│   ├── data_preprocessing.py     # Preprocesamiento de datos
│   ├── dataset.py               # Dataset y DataLoader de PyTorch
│   ├── model.py                 # Arquitectura de red neuronal
│   ├── trainer.py               # Entrenamiento del modelo
│   └── evaluator.py             # Evaluación y métricas
├── api/                         # API Flask
│   └── app.py                   # Aplicación web
├── tests/                       # Pruebas unitarias
│   ├── __init__.py
│   ├── test_data_preprocessing.py
│   └── test_model.py
├── data/                        # Datos (generado automáticamente)
├── models/                      # Modelos entrenados (generado automáticamente)
├── results/                     # Resultados y gráficos (generado automáticamente)
│   └── plots/
├── generate_dataset.py         # Generador de dataset sintético
├── train.py                    # Script principal de entrenamiento
├── run_tests.py               # Ejecutor de pruebas
├── requirements.txt           # Dependencias
└── README.md                  # Este archivo
```

## 🚀 Instalación

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar el repositorio:**
```bash
git clone <url-del-repositorio>
cd heart_attack_prediction
```

2. **Crear entorno virtual (recomendado):**
```bash
python -m venv venv

# En Linux/Mac:
source venv/bin/activate

# En Windows:
venv\Scripts\activate
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

## 📊 Uso del Sistema

### 1. Entrenamiento del Modelo

#### Entrenamiento Básico
```bash
python train.py
```

#### Entrenamiento con Parámetros Personalizados
```bash
python train.py --generate_new --model_type standard --hidden_dims 256 128 64 --learning_rate 0.001 --dropout_rate 0.3
```

#### Opciones de Entrenamiento
- `--data_path`: Ruta al archivo CSV (default: `data/heart_attack_dataset.csv`)
- `--generate_new`: Generar nuevo dataset sintético
- `--model_type`: Tipo de modelo (`standard` o `simple`)
- `--hidden_dims`: Dimensiones de capas ocultas (ej: `128 64 32`)
- `--dropout_rate`: Tasa de dropout (default: `0.3`)
- `--learning_rate`: Tasa de aprendizaje (default: `0.001`)
- `--weight_decay`: Regularización L2 (default: `1e-5`)

### 2. API Web

#### Iniciar la API
```bash
python api/app.py
```

La API estará disponible en: `http://localhost:5000`

#### Endpoints Disponibles

- **`GET /`**: Interfaz web interactiva
- **`POST /predict`**: Predicción individual
- **`POST /predict_batch`**: Predicciones en lote
- **`GET /model_info`**: Información del modelo
- **`GET /health`**: Estado de la API

#### Ejemplo de Uso con curl
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "gender": 1,
    "systolic_bp": 140,
    "diastolic_bp": 90,
    "cholesterol": 240,
    "bmi": 28.5,
    "smoker": 1,
    "exercise_hours": 2,
    "alcohol_units": 5,
    "diabetes": 0,
    "family_history": 1
  }'
```

### 3. Pruebas Unitarias

#### Ejecutar Todas las Pruebas
```bash
python run_tests.py
```

#### Ejecutar Pruebas Específicas
```bash
python run_tests.py data_preprocessing
python run_tests.py model
```

#### Usando pytest (alternativo)
```bash
pip install pytest
pytest tests/ -v
```

## 📈 Características del Dataset

El sistema utiliza las siguientes características para predecir ataques cardíacos:

| Característica | Descripción | Rango/Valores |
|---------------|-------------|---------------|
| `age` | Edad del paciente | 18-90 años |
| `gender` | Género | 0: Mujer, 1: Hombre |
| `systolic_bp` | Presión arterial sistólica | 90-200 mmHg |
| `diastolic_bp` | Presión arterial diastólica | 60-120 mmHg |
| `cholesterol` | Nivel de colesterol | 120-350 mg/dL |
| `bmi` | Índice de masa corporal | 15-45 |
| `smoker` | Estado de fumador | 0: No, 1: Sí |
| `exercise_hours` | Horas de ejercicio por semana | 0-20 |
| `alcohol_units` | Unidades de alcohol por semana | 0-30 |
| `diabetes` | Diabetes | 0: No, 1: Sí |
| `family_history` | Historial familiar cardíaco | 0: No, 1: Sí |

## 🧠 Arquitectura del Modelo

### Modelo Estándar
- **Entrada**: 11 características
- **Capas ocultas**: [128, 64, 32] neuronas (configurable)
- **Activación**: ReLU
- **Regularización**: Dropout (30%)
- **Salida**: 1 neurona con activación Sigmoid
- **Función de pérdida**: Binary Cross-Entropy

### Modelo Simple
- **Entrada**: 11 características
- **Capas ocultas**: [64, 32] neuronas
- **Activación**: ReLU
- **Regularización**: Dropout (20%)
- **Salida**: 1 neurona con activación Sigmoid

## 📊 Métricas de Evaluación

El sistema proporciona las siguientes métricas:

- **Exactitud (Accuracy)**: Proporción de predicciones correctas
- **Precisión (Precision)**: TP / (TP + FP)
- **Sensibilidad/Recall**: TP / (TP + FN)
- **Especificidad**: TN / (TN + FP)
- **F1-Score**: Media armónica de precisión y recall
- **ROC-AUC**: Área bajo la curva ROC

## 📁 Archivos Generados

Después del entrenamiento, se generan automáticamente:

```
models/
├── best_model.pth           # Modelo entrenado
└── preprocessor.pkl         # Preprocessor para normalización

results/
├── evaluation_results.json  # Métricas en formato JSON
└── plots/
    ├── training_curves.png   # Curvas de entrenamiento
    ├── confusion_matrix.png  # Matriz de confusión
    ├── roc_curve.png        # Curva ROC
    └── probability_distribution.png

training.log                 # Log detallado del entrenamiento
```

## 🔧 Configuración Avanzada

### Variables de Entorno
```bash
export FLASK_ENV=development  # Para modo desarrollo
export FLASK_DEBUG=1          # Para debug de Flask
```

### Configuración de GPU
El sistema detecta automáticamente CUDA y utiliza GPU si está disponible:
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

## 🐛 Solución de Problemas

### Error: Modelo no encontrado
```
FileNotFoundError: Modelo no encontrado: models/best_model.pth
```
**Solución**: Ejecutar primero `python train.py` para entrenar el modelo.

### Error: Módulo no encontrado
```
ModuleNotFoundError: No module named 'src'
```
**Solución**: Ejecutar desde el directorio `heart_attack_prediction/`.

### Error: Dependencias faltantes
```
ImportError: No module named 'torch'
```
**Solución**: Instalar dependencias con `pip install -r requirements.txt`.

## 📚 Ejemplos de Uso

### Generar Dataset Personalizado
```python
from generate_dataset import generate_heart_attack_dataset

# Generar 5000 muestras
df = generate_heart_attack_dataset(n_samples=5000, random_state=123)
df.to_csv('mi_dataset.csv', index=False)
```

### Cargar Modelo Entrenado
```python
import torch
from src.model import HeartAttackPredictor

# Cargar checkpoint
checkpoint = torch.load('models/best_model.pth')
model = HeartAttackPredictor(
    input_dim=checkpoint['model_config']['input_dim'],
    hidden_dims=checkpoint['model_config']['hidden_dims']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## 🤝 Contribución

1. Fork el proyecto
2. Crear rama para nueva característica (`git checkout -b feature/nueva-caracteristica`)
3. Commit los cambios (`git commit -am 'Añadir nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crear Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para detalles.

## 🔗 Enlaces Útiles

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

## 📧 Contacto

Para preguntas o sugerencias, crear un issue en el repositorio.

---

**⚠️ Disclaimer**: Este es un proyecto educativo con datos sintéticos. No debe utilizarse para diagnósticos médicos reales.