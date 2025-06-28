# Tarea 4 - Metaheurísticas: Inferencia Difusa para Clasificación de Spam

Este repositorio implementa un modelo de clasificación de mensajes de texto basado en técnicas de inferencia difusa, inspirado en el paper *“Fuzzy Based Feature Evaluation and Decision Support for Enhancing Spam Email Classification”*. La tarea consistió en replicar de forma aproximada el método descrito en el artículo, comparándolo con un modelo base Naive Bayes tradicional, para evaluar el impacto de la lógica difusa en la predicción de mensajes spam.

Dado que el paper original no publica su dataset, se optó por utilizar el clásico dataset **SMS Spam Collection (`spam.csv`)**, ampliamente referenciado y validado en la comunidad académica para estudios de detección de spam. Este conjunto de datos contiene mensajes de texto reales clasificados como *ham* (no spam) o *spam*, lo que permite replicar el escenario del paper con datos similares en naturaleza y dificultad, manteniendo la validez de la comparación de resultados.

## Tecnologías utilizadas

- Python 3
- Pandas
- Scikit-learn
- Numpy

## Estructura del repositorio

- `fuzzy.py`  
  Implementa la versión con inferencia difusa inspirada en el paper.

- `tarea3.py`  
  Implementa un modelo Naive Bayes tradicional como baseline para comparar.

- `spam.csv`  
  Dataset de SMS públicos con etiquetado ham/spam, usado para entrenar y evaluar ambos modelos.

## Resultados

Se lograron resultados sólidos en términos de precisión y recall, confirmando que los métodos de inferencia difusa pueden mejorar la robustez de la clasificación frente a técnicas tradicionales, especialmente al manejar incertidumbre en datos de texto.

---
