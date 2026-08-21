import math
import os
import re
from collections import Counter
import pandas as pd

# Matriz de frecuencias normalizadas de bigramas y trigramas comunes en la web
COMMON_BIGRAMS = {
    "th": 0.0388,
    "he": 0.0368,
    "in": 0.0228,
    "er": 0.0217,
    "an": 0.0214,
    "re": 0.0175,
    "nd": 0.0157,
    "on": 0.0141,
    "en": 0.0138,
    "at": 0.0133,
    "ou": 0.0128,
    "ed": 0.0127,
    "ha": 0.0127,
    "to": 0.0117,
    "or": 0.0115,
    "it": 0.0113,
    "is": 0.0111,
    "hi": 0.0109,
    "es": 0.0109,
    "ng": 0.0105,
    "co": 0.0100,
    "de": 0.0095,
    "al": 0.0090,
    "te": 0.0085,
    "se": 0.0080,
    "la": 0.0075,
    "me": 0.0070,
    "ro": 0.0065,
}

COMMON_TRIGRAMS = {
    "the": 0.0181,
    "and": 0.0073,
    "ing": 0.0072,
    "her": 0.0036,
    "hat": 0.0035,
    "his": 0.0035,
    "tha": 0.0033,
    "ere": 0.0031,
    "for": 0.0028,
    "ent": 0.0027,
    "ion": 0.0026,
    "ter": 0.0026,
    "was": 0.0025,
    "you": 0.0025,
    "ith": 0.0024,
    "ver": 0.0024,
    "all": 0.0023,
    "wit": 0.0023,
    "thi": 0.0023,
    "com": 0.0022,
    "app": 0.0020,
    "net": 0.0019,
    "ser": 0.0018,
}


def shannon_entropy(s):
    if not s:
        return 0.0
    prob = [float(s.count(c)) / len(s) for c in dict(Counter(s))]
    return -sum([p * math.log(p, 2) for p in prob])


def max_consecutive_pattern(s, pattern):
    matches = re.findall(pattern, s)
    return max(len(m) for m in matches) if matches else 0


def calculate_ngram_scores(sld):
    """Calcula la frecuencia promedio de bi-gramas y tri-gramas."""
    clean_s = re.sub(r"[^a-z]", "", sld)
    n = len(clean_s)

    if n < 2:
        return 0.0, 0.0

    # Bigramas
    bigrams = [clean_s[i : i + 2] for i in range(n - 1)]
    bigram_score = sum(COMMON_BIGRAMS.get(bg, 0.0001) for bg in bigrams) / len(
        bigrams
    )

    # Trigramas
    if n < 3:
        trigram_score = 0.0
    else:
        trigrams = [clean_s[i : i + 3] for i in range(n - 2)]
        trigram_score = sum(
            COMMON_TRIGRAMS.get(tg, 0.00005) for tg in trigrams
        ) / len(trigrams)

    return round(bigram_score, 6), round(trigram_score, 6)


def vowel_consonant_transitions(sld):
    """Calcula la tasa de alternancia entre vocales y consonantes."""
    clean_s = re.sub(r"[^a-z]", "", sld)
    if len(clean_s) < 2:
        return 0.0

    vowels = set("aeiou")
    transitions = 0
    for i in range(len(clean_s) - 1):
        c1_is_vowel = clean_s[i] in vowels
        c2_is_vowel = clean_s[i + 1] in vowels
        if c1_is_vowel != c2_is_vowel:
            transitions += 1

    return round(transitions / (len(clean_s) - 1), 4)


def extract_domain_features_v2(sld):
    sld = str(sld).lower()
    length = len(sld)

    if length == 0:
        return {
            "length": 0,
            "entropy": 0.0,
            "vowel_ratio": 0.0,
            "digit_ratio": 0.0,
            "consonant_ratio": 0.0,
            "special_ratio": 0.0,
            "max_consecutive_consonants": 0,
            "max_consecutive_digits": 0,
            "hex_ratio": 0.0,
            "bigram_score": 0.0,
            "trigram_score": 0.0,
            "vc_transition_rate": 0.0,
        }

    vowels = len(re.findall(r"[aeiou]", sld))
    digits = len(re.findall(r"[0-9]", sld))
    special = len(re.findall(r"[^a-z0-9]", sld))
    consonants = len(re.findall(r"[bcdfghjklmnpqrstvwxyz]", sld))
    hex_chars = len(re.findall(r"[0-9a-f]", sld))

    max_cons = max_consecutive_pattern(sld, r"[bcdfghjklmnpqrstvwxyz]+")
    max_digs = max_consecutive_pattern(sld, r"[0-9]+")

    bigram_score, trigram_score = calculate_ngram_scores(sld)
    vc_rate = vowel_consonant_transitions(sld)

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
        "bigram_score": bigram_score,
        "trigram_score": trigram_score,
        "vc_transition_rate": vc_rate,
    }


INPUT_FILE = "dataset_dga_vs_benigno_final.csv"
OUTPUT_FILE = "dataset_features_v2_ready.csv"
CHUNK_SIZE = 100000


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] No se encuentra el archivo '{INPUT_FILE}'.")
        return

    print(
        f"[*] Iniciando extracción V2 con métricas de Wordiness/N-Gramas desde '{INPUT_FILE}'..."
    )

    first_chunk = True
    total_processed = 0

    for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):
        chunk = chunk.dropna(subset=["sld"])

        features_list = [extract_domain_features_v2(s) for s in chunk["sld"]]
        df_features = pd.DataFrame(features_list)

        df_combined = pd.concat(
            [
                chunk[["sld", "label"]].reset_index(drop=True),
                df_features.reset_index(drop=True),
            ],
            axis=1,
        )

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
    print(" [!] EXTRACTOR V2 FINALIZADO CON ÉXITO")
    print(f" [!] Total registros: {total_processed:,}")
    print(f" [!] Archivo guardado en: {os.path.abspath(OUTPUT_FILE)}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()