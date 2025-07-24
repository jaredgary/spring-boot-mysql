#!/usr/bin/env python3
"""
Script para ejecutar todas las pruebas unitarias del proyecto.
"""

import unittest
import sys
from pathlib import Path

# Añadir directorio del proyecto al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_all_tests():
    """Ejecuta todas las pruebas unitarias."""
    # Descubrir y ejecutar todas las pruebas
    loader = unittest.TestLoader()
    start_dir = project_root / 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Configurar runner con verbosidad
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    
    print("=" * 70)
    print("EJECUTANDO PRUEBAS UNITARIAS")
    print("=" * 70)
    
    # Ejecutar pruebas
    result = runner.run(suite)
    
    # Mostrar resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE PRUEBAS")
    print("=" * 70)
    print(f"Pruebas ejecutadas: {result.testsRun}")
    print(f"Errores: {len(result.errors)}")
    print(f"Fallos: {len(result.failures)}")
    print(f"Omitidas: {len(result.skipped)}")
    
    if result.errors:
        print(f"\nERRORES ({len(result.errors)}):")
        for test, error in result.errors:
            print(f"  - {test}: {error.split(chr(10))[0]}")
            
    if result.failures:
        print(f"\nFALLOS ({len(result.failures)}):")
        for test, failure in result.failures:
            print(f"  - {test}: {failure.split(chr(10))[0]}")
    
    # Determinar éxito
    success = len(result.errors) == 0 and len(result.failures) == 0
    
    if success:
        print("\n🎉 TODAS LAS PRUEBAS PASARON EXITOSAMENTE!")
    else:
        print(f"\n❌ {len(result.errors) + len(result.failures)} PRUEBAS FALLARON")
    
    print("=" * 70)
    
    return success

def run_specific_test(test_file):
    """
    Ejecuta un archivo de pruebas específico.
    
    Args:
        test_file: Nombre del archivo de pruebas (sin extensión)
    """
    loader = unittest.TestLoader()
    
    try:
        # Cargar módulo específico
        module_name = f'tests.{test_file}'
        suite = loader.loadTestsFromName(module_name)
        
        # Ejecutar pruebas
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return len(result.errors) == 0 and len(result.failures) == 0
        
    except Exception as e:
        print(f"Error cargando pruebas de {test_file}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Ejecutar prueba específica
        test_file = sys.argv[1]
        if test_file.startswith('test_'):
            test_file = test_file[5:]  # Remover prefijo 'test_'
        if test_file.endswith('.py'):
            test_file = test_file[:-3]  # Remover extensión '.py'
            
        success = run_specific_test(f'test_{test_file}')
    else:
        # Ejecutar todas las pruebas
        success = run_all_tests()
    
    # Salir con código apropiado
    sys.exit(0 if success else 1)