#!/usr/bin/env python3
"""
Script de demostración del proyecto Heart Attack Prediction.
Este script muestra las capacidades principales sin requerir dependencias externas.
"""

import random
import json
import os
from pathlib import Path

def create_sample_data():
    """Crea datos de muestra para demostración."""
    sample_features = [
        "age", "gender", "systolic_bp", "diastolic_bp", "cholesterol",
        "bmi", "smoker", "exercise_hours", "alcohol_units", 
        "diabetes", "family_history"
    ]
    
    # Generar pacientes de ejemplo
    patients = [
        {
            "name": "Juan Pérez",
            "age": 55,
            "gender": 1,  # Hombre
            "systolic_bp": 140,
            "diastolic_bp": 90,
            "cholesterol": 240,
            "bmi": 28.5,
            "smoker": 1,
            "exercise_hours": 1,
            "alcohol_units": 8,
            "diabetes": 0,
            "family_history": 1
        },
        {
            "name": "María García",
            "age": 42,
            "gender": 0,  # Mujer
            "systolic_bp": 120,
            "diastolic_bp": 80,
            "cholesterol": 180,
            "bmi": 24.0,
            "smoker": 0,
            "exercise_hours": 4,
            "alcohol_units": 2,
            "diabetes": 0,
            "family_history": 0
        },
        {
            "name": "Carlos López",
            "age": 68,
            "gender": 1,
            "systolic_bp": 160,
            "diastolic_bp": 95,
            "cholesterol": 280,
            "bmi": 32.0,
            "smoker": 1,
            "exercise_hours": 0,
            "alcohol_units": 12,
            "diabetes": 1,
            "family_history": 1
        }
    ]
    
    return patients, sample_features

def simulate_risk_calculation(patient_data):
    """Simula el cálculo de riesgo cardíaco."""
    # Factores de riesgo simplificados
    risk_score = 0
    
    # Edad
    risk_score += max(0, (patient_data["age"] - 40) * 0.02)
    
    # Presión arterial
    if patient_data["systolic_bp"] > 140:
        risk_score += 0.3
    if patient_data["diastolic_bp"] > 90:
        risk_score += 0.2
    
    # Colesterol
    if patient_data["cholesterol"] > 240:
        risk_score += 0.25
    
    # IMC
    if patient_data["bmi"] > 30:
        risk_score += 0.2
    
    # Factores de estilo de vida
    if patient_data["smoker"]:
        risk_score += 0.4
    
    if patient_data["exercise_hours"] < 2:
        risk_score += 0.15
    
    if patient_data["alcohol_units"] > 7:
        risk_score += 0.1
    
    # Condiciones médicas
    if patient_data["diabetes"]:
        risk_score += 0.3
    
    if patient_data["family_history"]:
        risk_score += 0.25
    
    # Género (hombres tienen mayor riesgo)
    if patient_data["gender"] == 1:
        risk_score += 0.1
    
    # Convertir a probabilidad
    probability = min(0.95, max(0.05, 1 / (1 + 2.71828 ** (-5 * (risk_score - 0.5)))))
    
    return {
        "probability": probability,
        "risk_level": "ALTO" if probability > 0.5 else "BAJO",
        "prediction": 1 if probability > 0.5 else 0
    }

def show_project_structure():
    """Muestra la estructura del proyecto."""
    print("🏗️  ESTRUCTURA DEL PROYECTO")
    print("=" * 50)
    
    structure = """
heart_attack_prediction/
├── 📁 src/                    # Código fuente
│   ├── data_preprocessing.py  # Limpieza y normalización
│   ├── dataset.py            # PyTorch Dataset/DataLoader
│   ├── model.py              # Red neuronal
│   ├── trainer.py            # Entrenamiento
│   └── evaluator.py          # Evaluación y métricas
├── 📁 api/                   # API Flask
│   └── app.py               # Interfaz web
├── 📁 tests/                # Pruebas unitarias
│   ├── test_data_preprocessing.py
│   └── test_model.py
├── 📄 train.py              # Script principal
├── 📄 requirements.txt      # Dependencias
└── 📄 README.md            # Documentación
"""
    print(structure)

def show_model_architecture():
    """Muestra la arquitectura del modelo."""
    print("\n🧠 ARQUITECTURA DEL MODELO")
    print("=" * 50)
    
    print("Red Neuronal Feedforward:")
    print("├── Entrada: 11 características")
    print("├── Capa Oculta 1: 128 neuronas + ReLU + Dropout(30%)")
    print("├── Capa Oculta 2: 64 neuronas + ReLU + Dropout(30%)")
    print("├── Capa Oculta 3: 32 neuronas + ReLU + Dropout(30%)")
    print("└── Salida: 1 neurona + Sigmoid")
    print("\nFunción de pérdida: Binary Cross-Entropy")
    print("Optimizador: Adam")
    print("Regularización: L2 + Dropout")

def show_features():
    """Muestra las características del dataset."""
    print("\n📊 CARACTERÍSTICAS DEL DATASET")
    print("=" * 50)
    
    features = [
        ("age", "Edad del paciente", "18-90 años"),
        ("gender", "Género", "0: Mujer, 1: Hombre"),
        ("systolic_bp", "Presión sistólica", "90-200 mmHg"),
        ("diastolic_bp", "Presión diastólica", "60-120 mmHg"),
        ("cholesterol", "Colesterol total", "120-350 mg/dL"),
        ("bmi", "Índice de masa corporal", "15-45"),
        ("smoker", "Fumador", "0: No, 1: Sí"),
        ("exercise_hours", "Ejercicio/semana", "0-20 horas"),
        ("alcohol_units", "Alcohol/semana", "0-30 unidades"),
        ("diabetes", "Diabetes", "0: No, 1: Sí"),
        ("family_history", "Historial familiar", "0: No, 1: Sí")
    ]
    
    for name, desc, range_val in features:
        print(f"• {name:15} {desc:25} {range_val}")

def demonstrate_predictions():
    """Demuestra predicciones con pacientes de ejemplo."""
    print("\n🔮 PREDICCIONES DE EJEMPLO")
    print("=" * 50)
    
    patients, _ = create_sample_data()
    
    for patient in patients:
        print(f"\n👤 Paciente: {patient['name']}")
        print(f"   📊 Datos: Edad={patient['age']}, Género={'M' if patient['gender'] else 'F'}")
        print(f"           PA={patient['systolic_bp']}/{patient['diastolic_bp']}, Col={patient['cholesterol']}")
        print(f"           IMC={patient['bmi']}, Fumador={'Sí' if patient['smoker'] else 'No'}")
        
        # Calcular riesgo
        result = simulate_risk_calculation(patient)
        
        risk_emoji = "🔴" if result["risk_level"] == "ALTO" else "🟢"
        print(f"   {risk_emoji} Riesgo: {result['risk_level']} ({result['probability']:.1%})")

def show_metrics():
    """Muestra las métricas de evaluación."""
    print("\n📈 MÉTRICAS DE EVALUACIÓN")
    print("=" * 50)
    
    metrics = [
        ("Exactitud", "Proporción de predicciones correctas"),
        ("Precisión", "TP / (TP + FP)"),
        ("Sensibilidad", "TP / (TP + FN)"),
        ("Especificidad", "TN / (TN + FP)"),
        ("F1-Score", "Media armónica de precisión y recall"),
        ("ROC-AUC", "Área bajo la curva ROC")
    ]
    
    for name, desc in metrics:
        print(f"• {name:12}: {desc}")

def show_api_demo():
    """Demuestra el uso de la API."""
    print("\n🌐 DEMOSTRACIÓN DE LA API")
    print("=" * 50)
    
    print("Endpoints disponibles:")
    print("• GET  /           - Interfaz web interactiva")
    print("• POST /predict    - Predicción individual")
    print("• POST /predict_batch - Predicciones en lote")
    print("• GET  /model_info - Información del modelo")
    print("• GET  /health     - Estado de la API")
    
    print("\nEjemplo de request JSON:")
    sample_request = {
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
    }
    print(json.dumps(sample_request, indent=2))

def show_installation():
    """Muestra instrucciones de instalación."""
    print("\n🚀 INSTALACIÓN")
    print("=" * 50)
    
    print("1. Clonar repositorio:")
    print("   git clone <url-del-repositorio>")
    print("   cd heart_attack_prediction")
    
    print("\n2. Ejecutar instalador:")
    print("   chmod +x install.sh")
    print("   ./install.sh")
    
    print("\n3. Entrenar modelo:")
    print("   python train.py")
    
    print("\n4. Ejecutar API:")
    print("   python api/app.py")

def main():
    """Función principal de la demostración."""
    print("🫀 PREDICTOR DE ATAQUES CARDÍACOS - DEMOSTRACIÓN")
    print("=" * 60)
    print("Este proyecto implementa un sistema completo de ML para")
    print("predecir el riesgo de ataques cardíacos usando PyTorch.")
    
    # Mostrar diferentes secciones
    show_project_structure()
    show_features()
    show_model_architecture()
    demonstrate_predictions()
    show_metrics()
    show_api_demo()
    show_installation()
    
    print("\n" + "=" * 60)
    print("🎯 CARACTERÍSTICAS IMPLEMENTADAS:")
    print("✅ Generación de dataset sintético realista")
    print("✅ Preprocesamiento completo (limpieza, normalización)")
    print("✅ Red neuronal con PyTorch (capas ReLU, dropout)")
    print("✅ Entrenamiento con early stopping y scheduling")
    print("✅ Evaluación completa (ROC, métricas, gráficos)")
    print("✅ API Flask con interfaz web interactiva")
    print("✅ Pruebas unitarias para código y modelo")
    print("✅ Documentación completa en README.md")
    
    print("\n🚀 Para usar el proyecto completo:")
    print("1. Instalar dependencias: ./install.sh")
    print("2. Entrenar modelo: python train.py")
    print("3. Ejecutar API: python api/app.py")
    print("4. Abrir navegador: http://localhost:5000")

if __name__ == "__main__":
    main()