#!/usr/bin/env python3
"""
Generador de dataset sintético para predicción de ataques cardíacos.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_heart_attack_dataset(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Genera un dataset sintético para predicción de ataques cardíacos.
    
    Args:
        n_samples: Número de muestras a generar
        random_state: Semilla para reproducibilidad
        
    Returns:
        DataFrame con las características y etiquetas
    """
    np.random.seed(random_state)
    
    # Características demográficas
    age = np.random.normal(55, 15, n_samples).clip(18, 90)
    gender = np.random.choice([0, 1], n_samples)  # 0: mujer, 1: hombre
    
    # Características médicas
    systolic_bp = np.random.normal(130, 20, n_samples).clip(90, 200)
    diastolic_bp = np.random.normal(80, 15, n_samples).clip(60, 120)
    cholesterol = np.random.normal(200, 40, n_samples).clip(120, 350)
    bmi = np.random.normal(26, 5, n_samples).clip(15, 45)
    
    # Hábitos de vida
    smoker = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    exercise_hours = np.random.exponential(2, n_samples).clip(0, 20)
    alcohol_units = np.random.exponential(3, n_samples).clip(0, 30)
    
    # Historial médico
    diabetes = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    family_history = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    
    # Crear etiqueta target con lógica realista
    risk_score = (
        0.1 * age +
        0.3 * systolic_bp / 100 +
        0.2 * cholesterol / 100 +
        0.15 * bmi +
        3 * smoker +
        2 * diabetes +
        1.5 * family_history +
        2 * gender -
        0.2 * exercise_hours +
        0.1 * alcohol_units
    )
    
    # Normalizar y convertir a probabilidad
    risk_prob = 1 / (1 + np.exp(-(risk_score - 15) / 5))
    heart_attack = np.random.binomial(1, risk_prob, n_samples)
    
    # Crear DataFrame
    data = pd.DataFrame({
        'age': age,
        'gender': gender,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'cholesterol': cholesterol,
        'bmi': bmi,
        'smoker': smoker,
        'exercise_hours': exercise_hours,
        'alcohol_units': alcohol_units,
        'diabetes': diabetes,
        'family_history': family_history,
        'heart_attack': heart_attack
    })
    
    logger.info(f"Dataset generado: {len(data)} muestras")
    logger.info(f"Distribución de casos positivos: {data['heart_attack'].mean():.2%}")
    
    return data


if __name__ == "__main__":
    # Generar y guardar dataset
    df = generate_heart_attack_dataset()
    df.to_csv("data/heart_attack_dataset.csv", index=False)
    logger.info("Dataset guardado en data/heart_attack_dataset.csv")