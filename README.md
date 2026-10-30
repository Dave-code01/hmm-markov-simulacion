# Simulación de un Modelo Oculto de Markov (HMM) – Monitoreo de una máquina industrial

Actividad didáctica 2-M2 · IU Digital · Autor: David Castro

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dave-code01/hmm-markov-simulacion/blob/main/Castro_David_Markov.ipynb)

## Problema
Una máquina industrial tiene un **estado interno que no se ve** (Sano, Desgaste, Crítico). Lo único que se mide es el **nivel de vibración** (Baja, Media, Alta). El HMM simula la evolución de la máquina y permite inferir el estado oculto a partir de las vibraciones.

## Parámetros del modelo
Probabilidades iniciales (π): Sano 0.80 · Desgaste 0.18 · Crítico 0.02

| Estado | → Sano | → Desgaste | → Crítico | Vib. Baja | Vib. Media | Vib. Alta |
|---|---|---|---|---|---|---|
| Sano | 0.92 | 0.07 | 0.01 | 0.75 | 0.20 | 0.05 |
| Desgaste | 0.05 | 0.85 | 0.10 | 0.20 | 0.55 | 0.25 |
| Crítico | 0.40 | 0.10 | 0.50 | 0.05 | 0.25 | 0.70 |

Las tres columnas de transición forman la matriz A y las tres de vibración forman la matriz de emisión B.

## Cómo ejecutarlo

**Opción 1: Google Colab (sin instalar nada).** Pulsa el botón "Abrir en Colab" de arriba y luego Entorno de ejecución, Ejecutar todo. Para obtener otra simulación, cambia el valor de `SEMILLA` en la celda de parámetros.

**Opción 2: en tu computador.**
```bash
pip install -r requirements.txt
python hmm_simulacion.py --pasos 200 --semilla 42 --corridas 500
```
Parámetros: `--pasos` (longitud de la secuencia), `--semilla` (reproducibilidad), `--corridas` (réplicas Monte Carlo), `--salida` (carpeta de resultados, por defecto `results`).

## Archivos del repositorio
- `Castro_David_Markov.ipynb`: notebook de Colab con la simulación completa y sus resultados.
- `hmm_simulacion.py`: la misma simulación como script (simulación, algoritmo forward, Viterbi y análisis Monte Carlo).
- `requirements.txt`: dependencias (numpy y matplotlib).
- `secuencia_simulada.csv`: estados ocultos, observaciones, Viterbi y probabilidades filtradas de la simulación con semilla 42.
- `resumen_montecarlo.csv`: métricas del análisis Monte Carlo.
- `salida_consola.txt`: salida del script con semilla 42.
- `secuencia.png`, `filtrado.png`, `montecarlo.png`: gráficas de resultados.

Nota: al ejecutar el script, los resultados se guardan en una carpeta `results/` que se crea automáticamente.

## Resultados (semilla 42, 200 pasos, 500 corridas)
- Precisión media de Viterbi: 81.8 % (desviación 5.2 puntos).
- Las frecuencias simuladas de estados coinciden con la distribución estacionaria teórica (diferencias menores a 1 punto porcentual).
- Las matrices A y B estimadas desde las simulaciones difieren de las reales en menos de 0.01.
