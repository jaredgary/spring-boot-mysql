#!/bin/bash

# Script de instalación para el proyecto Heart Attack Prediction

echo "🫀 Instalador del Predictor de Ataques Cardíacos"
echo "=================================================="

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 no está instalado"
    exit 1
fi

echo "✅ Python3 encontrado: $(python3 --version)"

# Crear entorno virtual
echo "📦 Creando entorno virtual..."
if command -v python3 -m venv --help &> /dev/null; then
    python3 -m venv venv
    if [ $? -eq 0 ]; then
        echo "✅ Entorno virtual creado"
        
        # Activar entorno virtual
        source venv/bin/activate
        
        # Actualizar pip
        echo "🔄 Actualizando pip..."
        pip install --upgrade pip
        
        # Instalar dependencias
        echo "📚 Instalando dependencias..."
        pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
        pip install numpy==1.24.3
        pip install pandas==2.0.3
        pip install scikit-learn==1.3.0
        pip install matplotlib==3.7.1
        pip install seaborn==0.12.2
        pip install flask==2.3.2
        pip install pytest==7.4.0
        
        echo "✅ Dependencias instaladas"
        
        # Crear directorios
        echo "📁 Creando directorios..."
        mkdir -p data models results/plots
        
        echo "🎉 ¡Instalación completada!"
        echo ""
        echo "Para usar el proyecto:"
        echo "1. Activar entorno virtual: source venv/bin/activate"
        echo "2. Entrenar modelo: python train.py"
        echo "3. Ejecutar API: python api/app.py"
        echo "4. Ejecutar tests: python run_tests.py"
        
    else
        echo "❌ Error creando entorno virtual"
        echo "Intentando instalación alternativa..."
        
        # Intentar instalación con --user
        echo "🔄 Instalando con --user..."
        python3 -m pip install --user torch numpy pandas scikit-learn matplotlib seaborn flask pytest
        
        if [ $? -eq 0 ]; then
            echo "✅ Dependencias instaladas con --user"
            mkdir -p data models results/plots
            echo "🎉 ¡Instalación completada!"
        else
            echo "❌ Error en la instalación"
            echo ""
            echo "Instalación manual requerida:"
            echo "pip install torch numpy pandas scikit-learn matplotlib seaborn flask pytest"
        fi
    fi
else
    echo "❌ No se puede crear entorno virtual"
    echo "Instale python3-venv: sudo apt install python3-venv"
fi