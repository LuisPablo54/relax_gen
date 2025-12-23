# 🧬 RelaxGEN: Librería de Algoritmos de Optimización Genética y Probabilística

[![PyPI Version](https://img.shields.io/pypi/v/relax-gen?color=blue)](https://pypi.org/project/relax-gen/)
[![License](https://img.shields.io/github/license/LuisPablo54/relax_gen)](https://github.com/LuisPablo54/relax_gen/blob/main/LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/LuisPablo54/relax_gen)](https://github.com/LuisPablo54/relax_gen/commits/main/)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)]()

## 💡 ¿Qué es RelaxGEN?

**RelaxGEN** es una librería Python de alto nivel diseñada para facilitar la implementación y experimentación con diferentes modelos de optimización metaheurística. Ofrece una API unificada para los siguientes paradigmas:

1.  **Algoritmo Genéticos Clásicos:** Basados en codificación binaria y operadores estándar para explotar bloques de construcción, siendo el método más robusto cuando no se conoce la estructura del problema.
2.  **Algoritmo Genéticos Cuánticos (QGA):** Utilizando representación probabilística (Qubits) permitiendo una búsqueda de alta velocidad con poblaciones mínimas.
3.  **Algoritmo de Estimación de Distribución (EDA):** Modelado probabilístico que identifica correlaciones entre variables, diferenciándose por su capacidad de resolver problemas con dependencias complejas.
4.  **Algoritmo de Programación Genética (GP):** Evoluciona estructuras jerárquicas ejecutables donde la a solución es un algoritmo o función matemática capaz de procesar entradas, permitiendo la síntesis automática de código o modelos simbólicos.

> El objetivo principal es proporcionar una herramienta flexible y rápida para la optimización de funciones y el ajuste de modelos complejos.

## 🚀 Instalación

La forma más sencilla de instalar es a través de `pip`:

```bash
pip install relax-gen
```

## Uso Rápido

```bash
import numpy as np
import relax_gen.GEN as rg

def funcion_test(x):
    return (np.sin(5*x) + 1.5*np.sin(2*x)) * np.exp(-0.1 * x**2)

print("Inicio")
menu = rg(funcion_test,  
           population=300, 
           i_min=-2, 
           i_max=2
           )

best_individual = menu.alg_stn_bin()
```

La librería relax-gen te permite elegir entre tres modelos de optimización diferentes. Los parámetros de inicialización varían según el algoritmo seleccionado. 

Más información sobre las distintas funciones se encuentra en la Wiki: 
https://github.com/LuisPablo54/relax_gen/wiki


## 🤝 Contribuciones
¡Las contribuciones son bienvenidas! Si deseas agregar un nuevo algoritmo genético, mejorar la documentación o reportar un error, por favor revisa la guía de contribución.
- Haz un Fork del repositorio.
- Crea una rama.
- Commitea tus cambios.
- Empuja al branch.
- Abre un Pull Request.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.
