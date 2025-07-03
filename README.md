# Tarea 4 - Metaheurísticas: Optimización de Selección de Características para Detección de Phishing

Este repositorio implementa y compara algoritmos metaheurísticos para la **selección de características** en problemas de **detección de phishing**. El proyecto incluye la implementación de cuatro metaheurísticas clásicas, con un enfoque especial en el **Butterfly Optimization Algorithm (BOA)** y su optimización de parámetros mediante **irace**.

## Objetivos del Proyecto

1. **Implementar metaheurísticas** para selección de características: Simulated Annealing (SA), Hill Climbing (HC), Tabu Search (TS) y Butterfly Optimization Algorithm (BOA)
2. **Optimizar parámetros** del BOA usando irace (Iterated Racing)
3. **Comparar rendimiento** de forma rigurosa y estadísticamente válida
4. **Evaluar eficacia** en datasets de detección de phishing con diferentes números de características


## Datasets

### 1. Phishing Dataset (9 características)
- **Archivo**: `phishing.csv`
- **Características**: 9 features para detección de sitios web de phishing
- **Uso**: Comparación básica y desarrollo inicial

### 2. Training Dataset (30 características)
- **Archivo**: `training.csv`
- **Muestras**: 11,055 instancias
- **Características**: 30 features 
- **Uso**: Evaluación completa y comparación final



### 1. Instalación
```bash
# Clonar el repositorio
git clone https://github.com/Cristobal1202/mh_t4
cd mh_t4

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Ejecución de Comparaciones

#### Dataset de 9 características:
```bash
python fuzzy_phishing_9.py
```

#### Dataset de 30 características:
```bash
python fuzzy_training_30.py
```


## Características Principales

### 1. **Metaheurísticas Implementadas**

- **Simulated Annealing (SA)**: Enfriamiento simulado con control de temperatura
- **Hill Climbing (HC)**: Búsqueda local con reinicializaciones
- **Tabu Search (TS)**: Búsqueda con memoria adaptativa
- **BOA (Optimizado)**: Algoritmo de optimización de mariposas con parámetros ajustados

### 2. **Optimización de Parámetros BOA**

- **Método**: irace (Iterated Racing)
- **Presupuesto**: 200 experimentos
- **Parámetros optimizados**:
  - `sensory_modality`: 0.0997 (vs 0.01 por defecto)
  - `power_exponent`: 0.3619 (vs 0.1 por defecto)
  - `switch_prob`: 0.6418 (vs 0.8 por defecto)

### 3. **Framework de Comparación Rigurosa**

- **Evaluaciones justas**: Mismo presupuesto computacional para todos
- **Validación cruzada**: 5-fold CV para robustez estadística
- **Múltiples ejecuciones**: 10 runs por algoritmo
- **Métricas completas**: Accuracy, Precision, Recall, F1-Score

### 4. **Visualizaciones Avanzadas**

- **Gráficos de convergencia**: Evolución del fitness con intervalos de confianza
- **Gráficos radar**: Comparación multi-métrica
- **Análisis estadístico**: Media ± desviación estándar

## Resultados Principales

### BOA Optimizado vs Metaheurísticas Clásicas

| Algoritmo | Configuración | Características |
|-----------|---------------|----------------|
| **BOA** | Parámetros optimizados por irace | Exploración/explotación balanceada |
| **SA** | Temperatura adaptativa | Búsqueda probabilística |
| **HC** | Reinicializaciones múltiples | Búsqueda local intensiva |
| **TS** | Memoria adaptativa | Diversificación inteligente |

### Ventajas del BOA Optimizado

1. **Mejor convergencia**: Parámetros ajustados científicamente
2. **Mayor robustez**: Menos varianza entre ejecuciones
3. **Exploración eficiente**: Balance optimizado exploración/explotación
4. **Adaptabilidad**: Funciona bien en diferentes espacios de características



