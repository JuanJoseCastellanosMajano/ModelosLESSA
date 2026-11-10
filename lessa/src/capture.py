import cv2
import joblib
import numpy as np
from config import FUENTE, MODELO, REFERENCIAS
from feedback import generar_consejos
from utils import crear_detector, extraer, dibujar_todo, estado

modelo = joblib.load(MODELO)
refs = joblib.load(REFERENCIAS)
OBJETIVO = "A"   # pon por ejemplo "A" para practicar esa seña y ver consejos

cap = cv2.VideoCapture(FUENTE)
holistic = crear_detector()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    vector, res = extraer(frame, holistic)
    dibujar_todo(frame, res)

    texto = "Sin mano"
    if vector:
        probs = modelo.predict_proba(np.array([vector]))[0]
        i = probs.argmax()
        texto = f"{modelo.classes_[i]} ({probs[i]:.0%})" if probs[i] > 0.7 else "No estoy seguro"
        if OBJETIVO:
            for n, c in enumerate(generar_consejos(vector, refs[OBJETIVO])[:3]):
                cv2.putText(frame, c, (10, 110 + 28 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    cv2.putText(frame, texto, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
    cv2.putText(frame, estado(res), (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imshow("Prueba", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()