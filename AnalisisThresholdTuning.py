import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_FILE = "evaluacion_resultados_modelo.csv"
OUTPUT_PLOT = "curva_threshold_tuning.png"


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] No se encuentra el archivo '{INPUT_FILE}'.")
        return

    print(
        f"[*] Cargando columnas 'label' y 'probabilidad_dga' desde '{INPUT_FILE}'..."
    )
    # Cargamos únicamente las columnas necesarias para optimizar memoria
    df = pd.read_csv(INPUT_FILE, usecols=["label", "probabilidad_dga"])

    y_true = df["label"].values
    y_prob = df["probabilidad_dga"].values

    total_dga = int((y_true == 1).sum())
    total_benign = int((y_true == 0).sum())
    total_samples = len(y_true)

    print(
        f"[+] Datos cargados: {total_samples:,} muestras (Benignos: {total_benign:,}, DGAs: {total_dga:,})\n"
    )

    # Definir la lista de umbrales a evaluar
    thresholds = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ]

    metricas_tabla = []
    fpr_list, fnr_list, f1_list, precision_list, recall_list = (
        [],
        [],
        [],
        [],
        [],
    )

    print("=" * 85)
    print(
        f"{'Umbral':^8} | {'FP (Bloqueos)':^14} | {'FPR (%)':^9} | {'FN (Fugas)':^12} | {'FNR (%)':^9} | {'Precision (%)':^13} | {'F1-Score':^8}"
    )
    print("=" * 85)

    for th in thresholds:
        # Predicción binaria según el umbral actual
        y_pred = (y_prob >= th).astype(int)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        fpr = (fp / total_benign) * 100
        fnr = (fn / total_dga) * 100
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        recall = (tp / total_dga * 100) if total_dga > 0 else 0.0
        f1 = (
            (2 * (precision / 100) * (recall / 100))
            / ((precision / 100) + (recall / 100))
            if (precision + recall) > 0
            else 0.0
        )

        fpr_list.append(fpr)
        fnr_list.append(fnr)
        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)

        metricas_tabla.append({
            "Umbral": th,
            "FP": fp,
            "FPR": fpr,
            "FN": fn,
            "FNR": fnr,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        })

        print(
            f"  {th:.2f}   | {fp:>14,} | {fpr:>8.2f}% | {fn:>12,} | {fnr:>8.2f}% | {precision:>12.2f}% | {f1:>8.4f}"
        )

    print("=" * 85)

    # --------------------------------------------------------------------------
    # GENERAR GRÁFICA DE COMPENSACIÓN (TRADE-OFF)
    # --------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(
        thresholds,
        fpr_list,
        marker="o",
        color="crimson",
        label="Tasa Falsos Positivos (FPR %)",
        linewidth=2,
    )
    plt.plot(
        thresholds,
        fnr_list,
        marker="s",
        color="darkorange",
        label="Tasa Falsos Negativos (FNR %)",
        linewidth=2,
    )
    plt.plot(
        thresholds,
        [f * 100 for f in f1_list],
        marker="^",
        color="navy",
        linestyle="--",
        label="F1-Score (%)",
        linewidth=2,
    )

    plt.title(
        "Impacto del Umbral de Decisión en las Tasas de Error (XGBoost)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Umbral de Decisión (Threshold)", fontsize=11)
    plt.ylabel("Porcentaje (%)", fontsize=11)
    plt.xticks(thresholds)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"\n[+] Gráfica comparativa guardada como: '{OUTPUT_PLOT}'")


if __name__ == "__main__":
    main()