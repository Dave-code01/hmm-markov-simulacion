# Simulación de un Modelo Oculto de Markov (HMM) – Monitoreo de una máquina industrial

Actividad didáctica 2-M2 · IU Digital.

## Problema
Una máquina industrial tiene un **estado interno que no se ve** (Sano, Desgaste, Crítico). Lo único que se mide es el **nivel de vibración** (Baja, Media, Alta). El HMM simula la evolución de la máquina y permite inferir el estado oculto a partir de las vibraciones.

## Parámetros del modelo
| Estado | π | → Sano | → Desgaste | → Crítico | Vib. Baja | Vib. Media | Vib. Alta |
|---|---|---|---|---|---|---|---|
| Sano | 0.80 | 0.92 | 0.07 | 0.01 | 0.75 | 0.20 | 0.05 |
| Desgaste | 0.18 | 0.05 | 0.85 | 0.10 | 0.20 | 0.55 | 0.25 |
| Crítico | 0.02 | 0.40 | 0.10 | 0.50 | 0.05 | 0.25 | 0.70 |

## Ejecución
```bash
pip install -r requirements.txt
python hmm_simulacion.py --pasos 200 --semilla 42 --corridas 500
```
Parámetros: `--pasos` (longitud de la secuencia), `--semilla` (reproducibilidad; cambiarla da otra simulación), `--corridas` (réplicas Monte Carlo), `--salida` (carpeta de resultados).

## Contenido
- `hmm_simulacion.py`: simulación, algoritmo forward (filtrado), Viterbi y análisis Monte Carlo.
- `results/`: `secuencia_simulada.csv`, `resumen_montecarlo.csv`, `salida_consola.txt`, `secuencia.png`, `filtrado.png`, `montecarlo.png`.
- `Apellido_Nombre_Markov.docx`: documento con introducción, metodología, resultados y conclusiones.

## Resultados (semilla 42, 200 pasos, 500 corridas)
- Precisión media de Viterbi: ~81.8 %.
- Las frecuencias simuladas coinciden con la distribución estacionaria teórica (error < 1 punto porcentual).
- Las matrices A y B estimadas desde las simulaciones difieren de las reales en menos de 0.01.
