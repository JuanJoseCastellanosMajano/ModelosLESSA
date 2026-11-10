import csv
import sys
import time
import cv2
from config import SENAS, FUENTE, DATASET
from utils import crear_detector, extraer, dibujar_todo, estado

PERSONA = sys.argv[1] if len(sys.argv) > 1 else "p1"
INTERVALO = 0.1   # segundos entre muestras (10 por segundo)

DATASET.parent.mkdir(parents=True, exist_ok=True)


def contar():
    cuentas = {s: 0 for s in SENAS}
    if DATASET.exists():
        with open(DATASET) as f:
            for fila in csv.reader(f):
                if fila and fila[0] in cuentas:
                    cuentas[fila[0]] += 1
    return cuentas


cuentas = contar()
actual, grabando, ultimo = 0, False, 0

cap = cv2.VideoCapture(FUENTE)
holistic = crear_detector()

with open(DATASET, "a", newline="") as f:
    escritor = csv.writer(f)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vector, res = extraer(frame, holistic)
        dibujar_todo(frame, res)

        if grabando and vector and time.time() - ultimo >= INTERVALO:
            escritor.writerow([SENAS[actual], PERSONA] + vector)
            cuentas[SENAS[actual]] += 1
            ultimo = time.time()

        txt = f"{SENAS[actual]} ({cuentas[SENAS[actual]]}) - {'GRABANDO' if grabando else 'PAUSA'}"
        cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Persona: {PERSONA}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, estado(res), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Recolectar", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("s"):
            grabando = not grabando
        elif k == ord("n"):
            actual, grabando = (actual + 1) % len(SENAS), False
        elif k == ord("p"):
            actual, grabando = (actual - 1) % len(SENAS), False
        elif ord("1") <= k <= ord("9") and k - ord("1") < len(SENAS):
            actual, grabando = k - ord("1"), False
        f.flush()

cap.release()
cv2.destroyAllWindows()
print(cuentas)