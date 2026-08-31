import json
import requests
import sys

API_URL = "http://127.0.0.1:8000/predict"

# Simulación de un evento de log recibido por el agente Wazuh (Sysmon Event ID 22: DNS Query)
wazuh_event_alert = {
    "agent": {"id": "003", "name": "workstation-finance", "ip": "10.0.0.55"},
    "rule": {"id": "100201", "level": 5, "description": "Consulta DNS sospechosa detectada en endpoint"},
    "data": {
        "sysmon": {
            "event_id": "22",
            "process_name": "powershell.exe",
            "query_name": "x831a0plz9v1.biz"  # Dominio generado por malware
        }
    }
}

def consultar_api_dga(domain):
    payload = {"domain": domain}
    try:
        res = requests.post(API_URL, json=payload, timeout=2)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        return {"error": str(e)}
    return None

def main():
    print("=" * 80)
    print(" [*] SIMULADOR DE WAZUH CUSTOM INTEGRATION (SYSMON DNS EVENT -> DGA API)")
    print("=" * 80)

    agent_name = wazuh_event_alert["agent"]["name"]
    agent_ip = wazuh_event_alert["agent"]["ip"]
    process = wazuh_event_alert["data"]["sysmon"]["process_name"]
    suspect_domain = wazuh_event_alert["data"]["sysmon"]["query_name"]

    print(f"[*] Alerta recibida en Wazuh Manager:")
    print(f"    - Endpoint: {agent_name} ({agent_ip})")
    print(f"    - Proceso Ejecutado: {process}")
    print(f"    - Dominio Consultado: {suspect_domain}")
    print("[*] Consultando Microservicio XGBoost V3...")

    resultado_api = consultar_api_dga(suspect_domain)

    if not resultado_api or "error" in resultado_api:
        print("[ERROR] Falló la comunicación con el microservicio DGA.")
        return

    # Construir Alerta desde Wazuh (Wazuh Alert Level Escalation)
    prob = resultado_api.get("dga_probability", 0.0)
    is_dga = resultado_api.get("is_dga", False)
    desglose = resultado_api.get("word_breakdown", [])
    
    nivel_alerta = 12 if prob >= 0.85 else (8 if is_dga else 3)

    wazuh_enriched_output = {
        "wazuh_alert": {
            "timestamp": "2026-08-22T12:05:00Z",
            "rule": {
                "id": "100205",
                "level": nivel_alerta,
                "description": f"DGA Inferencia ML: {resultado_api.get('action_recommended')}"
            },
            "agent": wazuh_event_alert["agent"],
            "dga_intelligence": {
                "sld": resultado_api.get("sld"),
                "dga_probability": prob,
                "entropy": resultado_api.get("entropy"),
                "word_breakdown": desglose,
                "status": "MALICIOUS" if is_dga else "CLEAN"
            },
            "active_response": "ISOLATE_HOST" if prob >= 0.85 else "LOG_ONLY"
        }
    }

    print("\n[+] Evento generado por Wazuh Integration:")
    print(json.dumps(wazuh_enriched_output, indent=4))

if __name__ == "__main__":
    main()