#!/usr/bin/env python3
"""
Script principal para entrenar el modelo de predicción de ataques cardíacos.
"""

import logging
import pickle
from pathlib import Path
import argparse

import torch

from src.data_preprocessing import DataPreprocessor
from src.dataset import HeartAttackDataModule
from src.model import HeartAttackPredictor, ModelConfig
from src.trainer import HeartAttackTrainer
from src.evaluator import HeartAttackEvaluator
from generate_dataset import generate_heart_attack_dataset

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def create_directories():
    """Crea directorios necesarios para el proyecto."""
    directories = ['data', 'models', 'results', 'results/plots']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info("Directorios creados")


def prepare_data(data_path: str, generate_new: bool = False) -> tuple:
    """
    Prepara los datos para entrenamiento.
    
    Args:
        data_path: Ruta al archivo de datos
        generate_new: Si generar nuevo dataset
        
    Returns:
        Tuple con datos procesados y preprocessor
    """
    logger.info("Iniciando preparación de datos...")
    
    # Generar dataset si no existe o se solicita
    if generate_new or not Path(data_path).exists():
        logger.info("Generando nuevo dataset...")
        df = generate_heart_attack_dataset(n_samples=10000)
        df.to_csv(data_path, index=False)
        logger.info(f"Dataset guardado en: {data_path}")
    
    # Inicializar preprocessor
    preprocessor = DataPreprocessor()
    
    # Procesar datos
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(data_path)
    
    # Guardar preprocessor para uso posterior
    with open('models/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    logger.info("Preprocessor guardado en models/preprocessor.pkl")
    
    return X_train, X_test, y_train, y_test, preprocessor


def train_model(X_train, X_test, y_train, y_test, model_config: ModelConfig) -> tuple:
    """
    Entrena el modelo.
    
    Args:
        X_train, X_test, y_train, y_test: Datos procesados
        model_config: Configuración del modelo
        
    Returns:
        Tuple con modelo entrenado y entrenador
    """
    logger.info("Iniciando entrenamiento del modelo...")
    
    # Crear data module
    data_module = HeartAttackDataModule(batch_size=32)
    data_module.setup(X_train, y_train, X_test, y_test)
    
    # Crear modelo
    model = model_config.create_model()
    logger.info(f"Modelo creado: {model.get_model_info()}")
    
    # Crear entrenador
    trainer = HeartAttackTrainer(model)
    
    # Entrenar
    training_history = trainer.train(
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.test_dataloader(),
        epochs=100,
        learning_rate=model_config.learning_rate,
        weight_decay=model_config.weight_decay,
        patience=15
    )
    
    # Generar gráficos de entrenamiento
    trainer.plot_training_curves('results/plots/training_curves.png')
    
    # Guardar checkpoint
    trainer.save_checkpoint('models/best_model.pth', len(trainer.train_losses))
    
    # Mostrar resumen
    summary = trainer.get_training_summary()
    logger.info("Resumen del entrenamiento:")
    for key, value in summary.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    return model, trainer, data_module


def evaluate_model(model, data_module) -> dict:
    """
    Evalúa el modelo entrenado.
    
    Args:
        model: Modelo entrenado
        data_module: Módulo de datos
        
    Returns:
        Diccionario con resultados de evaluación
    """
    logger.info("Iniciando evaluación del modelo...")
    
    # Crear evaluador
    evaluator = HeartAttackEvaluator(model)
    
    # Generar reporte completo
    evaluation_results = evaluator.generate_evaluation_report(
        test_loader=data_module.test_dataloader(),
        save_plots=True,
        plots_dir='results/plots'
    )
    
    # Mostrar métricas principales
    metrics = evaluation_results['metrics']
    logger.info("Métricas de evaluación:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Información del umbral óptimo
    optimal_info = evaluation_results['optimal_threshold_analysis']
    logger.info(f"Umbral óptimo: {optimal_info['optimal_threshold']:.3f}")
    
    return evaluation_results


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Entrenar modelo de predicción de ataques cardíacos')
    parser.add_argument('--data_path', default='data/heart_attack_dataset.csv',
                       help='Ruta al archivo de datos')
    parser.add_argument('--generate_new', action='store_true',
                       help='Generar nuevo dataset')
    parser.add_argument('--model_type', default='standard', choices=['standard', 'simple'],
                       help='Tipo de modelo')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128, 64, 32],
                       help='Dimensiones de capas ocultas')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Tasa de dropout')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Tasa de aprendizaje')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Regularización L2')
    
    args = parser.parse_args()
    
    try:
        # Crear directorios
        create_directories()
        
        # Preparar datos
        X_train, X_test, y_train, y_test, preprocessor = prepare_data(
            args.data_path, args.generate_new
        )
        
        # Configurar modelo
        input_dim = X_train.shape[1]
        model_config = ModelConfig(
            input_dim=input_dim,
            model_type=args.model_type,
            hidden_dims=args.hidden_dims,
            dropout_rate=args.dropout_rate,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Entrenar modelo
        model, trainer, data_module = train_model(X_train, X_test, y_train, y_test, model_config)
        
        # Evaluar modelo
        evaluation_results = evaluate_model(model, data_module)
        
        # Guardar resultados de evaluación
        import json
        
        # Convertir arrays numpy a listas para JSON
        results_for_json = {
            'metrics': evaluation_results['metrics'],
            'optimal_threshold': evaluation_results['optimal_threshold_analysis']['optimal_threshold'],
            'sample_size': evaluation_results['sample_size'],
            'class_distribution': evaluation_results['class_distribution'],
            'model_config': {
                'input_dim': input_dim,
                'model_type': args.model_type,
                'hidden_dims': args.hidden_dims,
                'dropout_rate': args.dropout_rate,
                'learning_rate': args.learning_rate,
                'weight_decay': args.weight_decay
            }
        }
        
        with open('results/evaluation_results.json', 'w') as f:
            json.dump(results_for_json, f, indent=2)
        
        logger.info("Entrenamiento completado exitosamente!")
        logger.info("Archivos generados:")
        logger.info("  - models/best_model.pth (modelo entrenado)")
        logger.info("  - models/preprocessor.pkl (preprocessor)")
        logger.info("  - results/plots/ (gráficos)")
        logger.info("  - results/evaluation_results.json (métricas)")
        
        # Mostrar cómo usar la API
        logger.info("\nPara usar la API:")
        logger.info("  cd heart_attack_prediction")
        logger.info("  python api/app.py")
        logger.info("  Abrir http://localhost:5000 en el navegador")
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()