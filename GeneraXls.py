import os
import time
import pandas as pd
from playwright.sync_api import sync_playwright

URLS_FILE = "urls_dgas.txt"
EXCEL_FILE = "caracteristicas_dga.xlsx"
LOGIN_URL = "https://dgarchive.caad.fkie.fraunhofer.de/site/families.html"
PAUSA_SEGUNDOS = 35

# Modifica con tus credenciales
HTTP_USER = "juan_jose_castro_ceballos"
HTTP_PASS = "xxxxxxxxxx"

def parse_urls_file(filepath):
    dga_list = []
    if not os.path.exists(filepath):
        print(f"[ERROR] No se encontró el archivo '{filepath}'.")
        return dga_list

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                parts = line.split(":", 1)
                dga_name = parts[0].strip()
                url = parts[1].strip()
                if not url.startswith("http") and "http" in line:
                    url = line[line.index("http"):]
                dga_list.append((dga_name, url))
    return dga_list

def main():
    dga_items = parse_urls_file(URLS_FILE)
    if not dga_items:
        print("[X] No hay URLs para procesar. Abortando.")
        return

    total = len(dga_items)
    print(f"[+] Se cargaron {total} URLs desde '{URLS_FILE}'.\n")

    excel_data = []

    with sync_playwright() as p:
        print("[*] Iniciando navegador Chromium...")
        browser = p.chromium.launch(headless=False) 
        
        context = browser.new_context(
            viewport={'width': 1280, 'height': 1020},
            http_credentials={'username': HTTP_USER, 'password': HTTP_PASS}
        )
        page = context.new_page()

        print(f"[*] Navegando a la página de inicio para autenticación: {LOGIN_URL}")
        try:
            page.goto(LOGIN_URL, timeout=60000)
            print("[+] Autenticación exitosa.")
        except Exception as e:
            print(f"[!] Nota sobre la navegación inicial: {e}")

        for index, (dga_name, url) in enumerate(dga_items, start=1):
            print(f"[{index}/{total}] Extrayendo metadatos: {dga_name} -> {url}")

            # Valores por defecto en caso de no encontrarse en la ficha
            dga_active_since = "N/A"
            sld_length = "N/A"
            regex = "N/A"
            num_seeds = "N/A"
            num_domains = "N/A"

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                
                # --- EXTRAER DATOS DE LA TABLA HTML ---
                # 1. DGA active since
                active_row = page.locator("tr:has(td:text-is('DGA active since:')) td").nth(1)
                if active_row.count() > 0:
                    dga_active_since = active_row.inner_text().strip()

                # 2. SLD length
                sld_row = page.locator("tr:has(td:text-is('SLD length:')) td").nth(1)
                if sld_row.count() > 0:
                    sld_length = sld_row.inner_text().strip()

                # 3. Regex
                regex_row = page.locator("tr:has(td:text-is('Regex:')) td").nth(1)
                if regex_row.count() > 0:
                    regex = regex_row.inner_text().strip()

                # 4. #Seeds
                seeds_row = page.locator("tr:has(td:text-is('#Seeds:')) td").nth(1)
                if seeds_row.count() > 0:
                    num_seeds = seeds_row.inner_text().strip()

                # 5. #Domains
                domains_row = page.locator("tr:has(td:text-is('#Domains:')) td").nth(1)
                if domains_row.count() > 0:
                    num_domains = domains_row.inner_text().strip()

                print(f"    [OK] Datos extraídos correctamente.")

            except Exception as e:
                print(f"    [ERROR] Falló la extracción para '{dga_name}': {e}")

            # Guardar el registro estructurado
            excel_data.append({
                "Nombre DGA": dga_name,
                "DGA Active Since": dga_active_since,
                "SLD Length": sld_length,
                "Regex": regex,
                "#Seeds": num_seeds,
                "#Domains": num_domains
            })

            if index < total:
                print(f"    [*] Esperando {PAUSA_SEGUNDOS} segundos...")
                time.sleep(PAUSA_SEGUNDOS)

        # --- GENERAR ARCHIVO EXCEL ---
        print(f"\n[*] Generando archivo Excel '{EXCEL_FILE}'...")
        df = pd.DataFrame(excel_data)
        df.to_excel(EXCEL_FILE, index=False)

        print(f"[+] finalizado. Archivo guardado en: {os.path.abspath(EXCEL_FILE)}")
        browser.close()

if __name__ == "__main__":
    main()