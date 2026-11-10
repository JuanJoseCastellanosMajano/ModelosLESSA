from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATASET = RAIZ / "files" / "dataset.csv"
MODELO = RAIZ / "model" / "lessa.joblib"
REFERENCIAS = RAIZ / "model" / "referencias.joblib"

# Cambia aquí las señas (validadas con la comunidad sorda)
SENAS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",
]

# 0 = webcam. Con la ESP32-CAM: "http://IP_ESP32:81/stream"
FUENTE = 0