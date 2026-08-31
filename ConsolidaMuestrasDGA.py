import glob
import os
import pandas as pd
import tldextract

# Configuración
CARPETA_INPUT = "./"  # Ruta donde están los archivos .csv de DGAs
OUTPUT_FILE = "dataset_dga_consolidado.csv"
LIMITE_POR_FAMILIA = 10000


def extraer_sld(domain):
    """Extrae el SLD limpio del dominio."""
    try:
        extracted = tldextract.extract(str(domain))
        return extracted.domain
    except:
        return None


def procesar_dgas():
    # Buscar todos los archivos que terminen en _dga.csv
    patron = os.path.join(CARPETA_INPUT, "*_dga.csv")
    archivos_csv = glob.glob(patron)

    if not archivos_csv:
        print("[!] No se encontraron archivos *_dga.csv en el directorio.")
        return

    print(
        f"[*] Se encontraron {len(archivos_csv)} archivos de familias DGA.\n"
    )

    lista_df_familias = []

    for filepath in archivos_csv:
        nombre_archivo = os.path.basename(filepath)
        # Extraer el nombre de la familia (ej. 'abcbot_dga.csv' -> 'abcbot')
        familia = nombre_archivo.replace("_dga.csv", "")

        try:
            # Leer el archivo CSV
            # Nota: Si los CSV no tienen cabecera, se asume que la columna 0 o 1 contiene el dominio.
            df_temp = pd.read_csv(filepath, header=None, low_memory=False)

            # Detectar la columna que contiene los dominios (la primera columna de texto)
            col_domain = df_temp.columns[0]
            for col in df_temp.columns:
                if (
                    df_temp[col]
                    .astype(str)
                    .str.contains(r"\.", regex=True)
                    .any()
                ):
                    col_domain = col
                    break

            # Limpiar nulos y duplicados del archivo original
            df_temp = df_temp[[col_domain]].dropna().drop_duplicates()
            df_temp.columns = ["domain_raw"]

            # Muestrear hasta 10,000 elementos (o el total si son menos)
            num_disponibles = len(df_temp)
            if num_disponibles > LIMITE_POR_FAMILIA:
                df_sampled = df_temp.sample(
                    n=LIMITE_POR_FAMILIA, random_state=42
                ).copy()
            else:
                df_sampled = df_temp.copy()

            # Extraer SLD
            df_sampled["sld"] = df_sampled["domain_raw"].apply(extraer_sld)

            # Limpieza básica de SLD
            df_sampled = df_sampled.dropna(subset=["sld"])
            df_sampled = df_sampled[df_sampled["sld"].str.len() > 1]
            df_sampled["family"] = familia
            df_sampled["label"] = 1  # 1 = DGA

            lista_df_familias.append(
                df_sampled[["sld", "family", "label", "domain_raw"]]
            )

            print(
                f"[OK] {familia:<20} | Procesados: {len(df_sampled):>6} dominios (Disponibles originalmente: {num_disponibles})"
            )

        except Exception as e:
            print(f"[ERROR] No se pudo procesar {nombre_archivo}: {e}")

    # Consolidar todas las familias en un solo DataFrame
    if lista_df_familias:
        df_final = pd.concat(lista_df_familias, ignore_index=True)

        # Eliminar posibles duplicados de SLD entre familias si fuera necesario
        df_final = df_final.drop_duplicates(subset=["sld"]).reset_index(
            drop=True
        )

        # Guardar CSV resultante
        df_final.to_csv(OUTPUT_FILE, index=False)
        print("\n" + "=" * 60)
        print(f"[+] Proceso completado exitosamente.")
        print(f"[+] Total de dominios DGA consolidados: {len(df_final)}")
        print(f"[+] Archivo guardado en: {os.path.abspath(OUTPUT_FILE)}")
        print("=" * 60)


if __name__ == "__main__":
    procesar_dgas()