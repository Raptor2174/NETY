

def get_modules_status():
    return [
        {"type": "CCM", "name": "TPM", "status": "running"},
        {"type": "LCM", "name": "ESM", "status": "inactive"},
        {"type": "BCM", "name": "IMCM", "status": "running"},
        {"type": "CBM", "name": "LM", "status": "idle"},
    ]
