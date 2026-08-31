import time
import requests
import json

API_URL = "http://127.0.0.1:8000/predict"

# Simulación de eventos DNS capturados por Suricata / Snort
suricata_dns_events = [
    {
        "timestamp": "2026-08-22T12:00:01.120Z",
        "src_ip": "192.168.1.45",
        "dest_ip": "8.8.8.8",
        "proto": "UDP",
        "dns": {"rrname": "google.com", "rrtype": "A"}
    },
    {
        "timestamp": "2026-08-22T12:00:03.450Z",
        "src_ip": "192.168.1.110",
        "dest_ip": "1.1.1.1",
        "proto": "UDP",
        "dns": {"rrname": "cmid1s1zeiu.life", "rrtype": "A"}  # DGA BumbleBee
    },
    {
        "timestamp": "2026-08-22T12:00:05.890Z",
        "src_ip": "192.168.1.80",
        "dest_ip": "8.8.4.4",
        "proto": "UDP",
        "dns": {"rrname": "applebankservice.biz", "rrtype": "A"}  # DGA Diccionario
    },
    {
        "timestamp": "2026-08-22T12:00:07.010Z",
        "src_ip": "192.168.1.15",
        "dest_ip": "8.8.8.8",
        "proto": "UDP",
        "dns": {"rrname": "app-cluster-us-east.org", "rrtype": "A"}
    }
]

def main():
    print("=" * 80)
    print(" [*] INICIANDO SIMULADOR DE FLUJO NIDS (SNORT / SURICATA) -> API MICROSERVICIO")
    print("=" * 80)

    for event in suricata_dns_events:
        query_domain = event["dns"]["rrname"]
        src_host = event["src_ip"]
        
        print(f"\n[!] Paquete DNS interceptado: Host={src_host} -> Query={query_domain}")
        
        # Enviar petición a la API
        try:
            start_t = time.time()
            response = requests.post(API_URL, json={"domain": query_domain}, timeout=3)
            elapsed_ms = (time.time() - start_t) * 1000
            
            if response.status_code == 200:
                data = response.json()
                prob = data["dga_probability"]
                action = data["action_recommended"]
                is_dga = data["is_dga"]
                
                print(f"    [API Inferencia]: Latencia={elapsed_ms:.2f}ms | Prob DGA={prob*100:.2f}% | SLD='{data['sld']}'")
                print(f"    [Acción NIDS]: {action}")
                
                if is_dga:
                    print(f"    >>> [ALERTA ROJA] Bloqueando IP de origen {src_host} en iptables / Firewall")
                else:
                    print(f"    >>> [PASS] Tráfico permitido en el switch/router")
            else:
                print(f"    [ERROR API]: Código {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("    [ERROR]: No se pudo conectar con el microservicio. ¿Está uvicorn en ejecución?")
            break

        time.time()
        time.sleep(1)

if __name__ == "__main__":
    main()