import math
import os
import re
from collections import Counter
import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------
# 1. FUNCIONES MATEMÁTICAS Y LÉXICAS
# ------------------------------------------------------------------------------
def shannon_entropy(s):
    """Calcula la entropía de Shannon de una cadena."""
    if not s:
        return 0.0
    prob = [float(s.count(c)) / len(s) for c in dict(Counter(s))]
    return -sum([p * math.log(p, 2) for p in prob])


def max_consecutive_pattern(s, pattern):
    """Cuenta la racha máxima de caracteres que coinciden con una RegEx."""
    matches = re.findall(pattern, s)
    if not matches:
        return 0
    return max(len(m) for m in matches)


def extract_domain_features(sld):
    """Recibe la cadena del SLD y devuelve un diccionario con las métricas numéricas."""
    sld = str(sld).lower()
    length = len(sld)

    if length == 0:
        return {
            "length": 0,
            "entropy": 0,
            "vowel_ratio": 0,
            "digit_ratio": 0,
            "consonant_ratio": 0,
            "special_ratio": 0,
            "max_consecutive_consonants": 0,
            "max_consecutive_digits": 0,
            "hex_ratio": 0,
        }

    # Conteos básicos
    vowels = len(re.findall(r"[aeiou]", sld))
    digits = len(re.findall(r"[0-9]", sld))
    special = len(re.findall(r"[^a-z0-9]", sld))
    consonants = len(re.findall(r"[bcdfghjklmnpqrstvwxyz]", sld))
    hex_chars = len(re.findall(r"[0-9a-f]", sld))

    # Rachas máximas consecutivas
    max_cons = max_consecutive_pattern(sld, r"[bcdfghjklmnpqrstvwxyz]+")
    max_digs = max_consecutive_pattern(sld, r"[0-9]+")

    return {
        "length": length,
        "entropy": round(shannon_entropy(sld), 4),
        "vowel_ratio": round(vowels / length, 4),
        "digit_ratio": round(digits / length, 4),
        "consonant_ratio": round(consonants / length, 4),
        "special_ratio": round(special / length, 4),
        "max_consecutive_consonants": max_cons,
        "max_consecutive_digits": max_digs,
        "hex_ratio": round(hex_chars / length, 4),
    }


# ------------------------------------------------------------------------------
# 2. PROCESAMIENTO POR BLOQUES (PIPELINE)
# ------------------------------------------------------------------------------
INPUT_FILE = "dataset_dga_vs_benigno_final.csv"
OUTPUT_FILE = "dataset_features_ready.csv"
CHUNK_SIZE = 100000  # Procesa de 100,000 en 100,000 para optimizar RAM


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] No se encuentra el archivo '{INPUT_FILE}'.")
        return

    print(
        f"[*] Iniciando la extracción de características desde '{INPUT_FILE}'..."
    )

    first_chunk = True
    total_processed = 0

    # Leer el CSV por partes
    for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):
        chunk = chunk.dropna(subset=["sld"])

        # Aplicar extracción de características
        features_list = [extract_domain_features(s) for s in chunk["sld"]]
        df_features = pd.DataFrame(features_list)

        # Re-asociar las columnas originales
        df_combined = pd.concat(
            [
                chunk[["sld", "label"]].reset_index(drop=True),
                df_features.reset_index(drop=True),
            ],
            axis=1,
        )

        # Guardar de forma incremental
        if first_chunk:
            df_combined.to_csv(OUTPUT_FILE, index=False, mode="w")
            first_chunk = False
        else:
            df_combined.to_csv(OUTPUT_FILE, index=False, mode="a", header=False)

        total_processed += len(df_combined)
        print(
            f"    [+] Procesados {total_processed:,} dominios...",
            end="\r",
            flush=True,
        )

    print("\n" + "=" * 65)
    print(" [!] EXTRACTOR FINALIZADO CON ÉXITO")
    print(f" [!] Total de registros procesados: {total_processed:,}")
    print(f" [!] Dataset numérico listo en: {os.path.abspath(OUTPUT_FILE)}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()