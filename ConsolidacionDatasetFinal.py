import os
import pandas as pd

# Archivos de entrada
FILE_DGA = "dataset_dga_consolidado.csv"
FILE_TRANCO = "tranco_sld_output.csv"

# Archivo maestro de salida
OUTPUT_FINAL = "dataset_dga_vs_benigno_final.csv"

def main():
    if not os.path.exists(FILE_DGA):
        print(f"[ERROR] No se encontró el archivo '{FILE_DGA}'.")
        return
    if not os.path.exists(FILE_TRANCO):
        print(f"[ERROR] No se encontró el archivo '{FILE_TRANCO}'.")
        return

    # 1. Cargar Clase 1 (DGA)
    print(f"[*] Cargando Clase 1 (DGA) desde '{FILE_DGA}'...")
    df_dga = pd.read_csv(FILE_DGA)
    
    # Asegurar columnas sld y label
    df_dga = df_dga[['sld']].dropna().drop_duplicates()
    df_dga = df_dga[df_dga['sld'].astype(str).str.len() > 1]
    df_dga['label'] = 1
    total_dga = len(df_dga)
    print(f"[+] Total de muestras DGA (Clase 1): {total_dga:,}")

    # 2. Cargar Clase 0 (Benigno)
    print(f"[*] Cargando Clase 0 (Tranco) desde '{FILE_TRANCO}'...")
    df_tranco = pd.read_csv(FILE_TRANCO)

    # Normalizar columna sld
    col_sld = 'sld' if 'sld' in df_tranco.columns else df_tranco.columns[-1]
    df_benign = df_tranco[[col_sld]].rename(columns={col_sld: 'sld'}).dropna().drop_duplicates()
    df_benign = df_benign[df_benign['sld'].astype(str).str.len() > 1].reset_index(drop=True)
    
    total_tranco = len(df_benign)
    print(f"[*] Total de dominios Tranco limpios disponibles: {total_tranco:,}")

    # 3. MUESTREO ESTRATIFICADO (Opción B)
    print("[*] Aplicando Muestreo Estratificado por estratos de popularidad...")
    NUM_ESTRATOS = 5
    mues_por_estrato = total_dga // NUM_ESTRATOS

    # Crear quintiles según la posición en el ranking
    df_benign['estrato'] = pd.qcut(df_benign.index, q=NUM_ESTRATOS, labels=False)

    muestras_estratificadas = []
    for estrato_id in range(NUM_ESTRATOS):
        grupo = df_benign[df_benign['estrato'] == estrato_id]
        n_tomar = min(len(grupo), mues_por_estrato)
        muestras_estratificadas.append(grupo.sample(n=n_tomar, random_state=42))

    df_benign_sampled = pd.concat(muestras_estratificadas, ignore_index=True)
    df_benign_sampled = df_benign_sampled[['sld']].copy()
    df_benign_sampled['label'] = 0
    
    total_benign = len(df_benign_sampled)
    print(f"[+] Total de muestras Benignas seleccionadas (Clase 0): {total_benign:,}")

    # 4. Consolidación y Mezcla (Shuffle)
    print("[*] Uniendo Clase 0 y Clase 1 y realizando Shuffle...")
    df_final = pd.concat([df_dga[['sld', 'label']], df_benign_sampled[['sld', 'label']]], ignore_index=True)
    
    # Desordenar aleatoriamente
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    # 5. Guardar dataset maestro
    print(f"[*] Guardando dataset balanceado en '{OUTPUT_FINAL}'...")
    df_final.to_csv(OUTPUT_FINAL, index=False)

    print("\n" + "="*65)
    print(" [!] DATASET CONSOLIDADO BALANCEADO GENERADO CON ÉXITO")
    print(f" [!] Total registros: {len(df_final):,} (1:1 perfect balance)")
    print(f" [!] Clase 1 (DGA): {total_dga:,}")
    print(f" [!] Clase 0 (Benigno - Estratificado): {total_benign:,}")
    print(f" [!] Archivo listo: {os.path.abspath(OUTPUT_FINAL)}")
    print("="*65 + "\n")

if __name__ == "__main__":
    main()