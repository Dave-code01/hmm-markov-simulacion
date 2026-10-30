"""
Simulación de un Modelo Oculto de Markov (HMM) para el monitoreo de una máquina industrial.

Estados ocultos : Sano, Desgaste, Critico
Observaciones   : Vibracion Baja, Media, Alta
Uso:  python hmm_simulacion.py --pasos 200 --semilla 42 --corridas 500
"""
import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ESTADOS = ["Sano", "Desgaste", "Critico"]
OBSERV = ["Baja", "Media", "Alta"]

# Probabilidades iniciales (pi)
PI = np.array([0.80, 0.18, 0.02])
# Matriz de transición A[i, j] = P(estado_{t+1}=j | estado_t=i)
A = np.array([
    [0.92, 0.07, 0.01],
    [0.05, 0.85, 0.10],
    [0.40, 0.10, 0.50],
])
# Matriz de emisión B[i, k] = P(observacion=k | estado=i)
B = np.array([
    [0.75, 0.20, 0.05],
    [0.20, 0.55, 0.25],
    [0.05, 0.25, 0.70],
])


def validar():
    assert np.isclose(PI.sum(), 1.0)
    assert np.allclose(A.sum(axis=1), 1.0)
    assert np.allclose(B.sum(axis=1), 1.0)


def simular(T, rng):
    """Genera (estados ocultos, observaciones) con distribuciones categóricas."""
    z = np.empty(T, dtype=int)
    x = np.empty(T, dtype=int)
    z[0] = rng.choice(3, p=PI)
    x[0] = rng.choice(3, p=B[z[0]])
    for t in range(1, T):
        z[t] = rng.choice(3, p=A[z[t - 1]])
        x[t] = rng.choice(3, p=B[z[t]])
    return z, x


def forward(x):
    """Algoritmo forward escalado: log-verosimilitud y filtrado P(z_t | x_1..t)."""
    T = len(x)
    alfa = np.zeros((T, 3))
    c = np.zeros(T)
    a = PI * B[:, x[0]]
    c[0] = a.sum(); alfa[0] = a / c[0]
    for t in range(1, T):
        a = (alfa[t - 1] @ A) * B[:, x[t]]
        c[t] = a.sum(); alfa[t] = a / c[t]
    return np.log(c).sum(), alfa


def viterbi(x):
    """Secuencia de estados ocultos más probable dada la observación."""
    T = len(x)
    d = np.zeros((T, 3)); psi = np.zeros((T, 3), dtype=int)
    d[0] = np.log(PI) + np.log(B[:, x[0]])
    for t in range(1, T):
        m = d[t - 1][:, None] + np.log(A)
        psi[t] = m.argmax(axis=0)
        d[t] = m.max(axis=0) + np.log(B[:, x[t]])
    z = np.zeros(T, dtype=int)
    z[-1] = d[-1].argmax()
    for t in range(T - 2, -1, -1):
        z[t] = psi[t + 1, z[t + 1]]
    return z


def estacionaria():
    """Distribución estacionaria teórica de la cadena (autovector izquierdo de A)."""
    w, v = np.linalg.eig(A.T)
    p = np.real(v[:, np.argmin(np.abs(w - 1))])
    return p / p.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasos", type=int, default=200)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--corridas", type=int, default=500)
    ap.add_argument("--salida", default="results")
    args = ap.parse_args()
    validar()
    os.makedirs(args.salida, exist_ok=True)
    rng = np.random.default_rng(args.semilla)

    # ---- 1) Una simulación detallada ----
    z, x = simular(args.pasos, rng)
    ll, alfa = forward(x)
    zv = viterbi(x)
    acc_v = (zv == z).mean()
    acc_f = (alfa.argmax(axis=1) == z).mean()

    print("=== Simulación única ===")
    print("Estados ocultos (primeros 30):", " ".join(ESTADOS[i][0] for i in z[:30]))
    print("Observaciones   (primeras 30):", " ".join(OBSERV[i][0] for i in x[:30]))
    print("Viterbi         (primeros 30):", " ".join(ESTADOS[i][0] for i in zv[:30]))
    print(f"Log-verosimilitud de la secuencia: {ll:.2f}")
    print(f"Precisión Viterbi: {acc_v:.3f} | Precisión filtrado forward: {acc_f:.3f}")

    with open(os.path.join(args.salida, "secuencia_simulada.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "estado_oculto", "observacion", "viterbi", "P_sano", "P_desgaste", "P_critico"])
        for t in range(args.pasos):
            w.writerow([t, ESTADOS[z[t]], OBSERV[x[t]], ESTADOS[zv[t]], *np.round(alfa[t], 4)])

    # ---- 2) Monte Carlo con muchas corridas ----
    accs, frec_est, frec_obs = [], np.zeros(3), np.zeros(3)
    trans = np.zeros((3, 3)); emis = np.zeros((3, 3))
    for _ in range(args.corridas):
        zz, xx = simular(args.pasos, rng)
        accs.append((viterbi(xx) == zz).mean())
        frec_est += np.bincount(zz, minlength=3)
        frec_obs += np.bincount(xx, minlength=3)
        for a_, b_ in zip(zz[:-1], zz[1:]): trans[a_, b_] += 1
        for s_, o_ in zip(zz, xx): emis[s_, o_] += 1
    frec_est /= frec_est.sum(); frec_obs /= frec_obs.sum()
    A_hat = trans / trans.sum(axis=1, keepdims=True)
    B_hat = emis / emis.sum(axis=1, keepdims=True)
    pi_teo = estacionaria()
    obs_teo = pi_teo @ B

    print(f"\n=== Monte Carlo ({args.corridas} corridas x {args.pasos} pasos) ===")
    print(f"Precisión Viterbi media: {np.mean(accs):.3f} (desv. {np.std(accs):.3f})")
    print("Frecuencia empírica de estados :", dict(zip(ESTADOS, [round(float(v),3) for v in frec_est])))
    print("Distribución estacionaria teórica:", dict(zip(ESTADOS, [round(float(v),3) for v in pi_teo])))
    print("Frecuencia empírica de observ. :", dict(zip(OBSERV, [round(float(v),3) for v in frec_obs])))
    print("Observaciones teóricas         :", dict(zip(OBSERV, [round(float(v),3) for v in obs_teo])))
    print("Error máx. |A_hat - A| =", round(float(np.abs(A_hat - A).max()),4))
    print("Error máx. |B_hat - B| =", round(float(np.abs(B_hat - B).max()),4))

    with open(os.path.join(args.salida, "resumen_montecarlo.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "valor"])
        w.writerow(["precision_viterbi_media", round(float(np.mean(accs)), 4)])
        w.writerow(["precision_viterbi_desv", round(float(np.std(accs)), 4)])
        for n, e, t in zip(ESTADOS, frec_est, pi_teo):
            w.writerow([f"frec_empirica_{n}", round(float(e), 4)]); w.writerow([f"estacionaria_teorica_{n}", round(float(t), 4)])
        w.writerow(["error_max_A", round(float(np.abs(A_hat - A).max()), 4)])
        w.writerow(["error_max_B", round(float(np.abs(B_hat - B).max()), 4)])

    # ---- 3) Gráficos ----
    N = min(80, args.pasos); t = np.arange(N)
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].step(t, z[:N], where="mid", color="tab:blue"); ax[0].set_yticks(range(3)); ax[0].set_yticklabels(ESTADOS); ax[0].set_title("Estado oculto real")
    ax[1].step(t, x[:N], where="mid", color="tab:orange"); ax[1].set_yticks(range(3)); ax[1].set_yticklabels(OBSERV); ax[1].set_title("Observación (vibración)")
    ax[2].step(t, zv[:N], where="mid", color="tab:green"); ax[2].set_yticks(range(3)); ax[2].set_yticklabels(ESTADOS)
    ax[2].set_title(f"Estados inferidos por Viterbi (precisión {acc_v:.1%})"); ax[2].set_xlabel("Paso de tiempo")
    plt.tight_layout(); plt.savefig(os.path.join(args.salida, "secuencia.png"), dpi=130); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    idx = np.arange(3); wd = 0.38
    ax[0].bar(idx - wd/2, frec_est, wd, label="Simulada"); ax[0].bar(idx + wd/2, pi_teo, wd, label="Teórica")
    ax[0].set_xticks(idx); ax[0].set_xticklabels(ESTADOS); ax[0].set_title("Distribución de estados"); ax[0].legend()
    ax[1].hist(accs, bins=20, color="tab:green", edgecolor="white"); ax[1].set_title("Precisión de Viterbi por corrida"); ax[1].set_xlabel("Precisión")
    plt.tight_layout(); plt.savefig(os.path.join(args.salida, "montecarlo.png"), dpi=130); plt.close()

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.stackplot(t, alfa[:N].T, labels=ESTADOS, colors=["#4caf50", "#ffb300", "#e53935"])
    ax.set_title("Probabilidad filtrada P(estado | observaciones hasta t)"); ax.set_xlabel("Paso de tiempo"); ax.legend(loc="upper right")
    plt.tight_layout(); plt.savefig(os.path.join(args.salida, "filtrado.png"), dpi=130); plt.close()


if __name__ == "__main__":
    main()
