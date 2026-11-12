import cv2
import joblib
import numpy as np
from config import FUENTE, MODELO, REFERENCIAS
from feedback import generar_consejos
from utils import crear_detector, extraer, dibujar_todo, estado

modelo = joblib.load(MODELO)
refs = joblib.load(REFERENCIAS)
LETRAS = sorted(refs.keys())   # todas las letras que el modelo conoce

# Teclas: N / P = siguiente / anterior letra, 0 = modo libre (sin objetivo),
# A-Z = ir directo a esa letra (si existe), Q = salir
indice = None   # None = modo libre

cap = cv2.VideoCapture(FUENTE)
holistic = crear_detector()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    vector, res = extraer(frame, holistic)
    dibujar_todo(frame, res)

    objetivo = LETRAS[indice] if indice is not None else None
    texto, detectada = "Sin mano", None

    if vector:
        probs = modelo.predict_proba(np.array([vector]))[0]
        i = probs.argmax()
        if probs[i] > 0.7:
            detectada = str(modelo.classes_[i])
            texto = f"{detectada} ({probs[i]:.0%})"
        else:
            texto = "No estoy seguro"

        if objetivo:
            consejos = generar_consejos(vector, refs[objetivo])
            for n, c in enumerate(consejos[:3]):
                cv2.putText(frame, c, (10, 150 + 28 * n),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            if detectada == objetivo:
                cv2.putText(frame, "CORRECTA", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    modo = f"Practicar: {objetivo}" if objetivo else "Modo libre"
    cv2.putText(frame, texto, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
    cv2.putText(frame, modo, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, estado(res), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imshow("Prueba", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    elif k == ord("0"):
        indice = None
    elif k == ord("n"):
        indice = 0 if indice is None else (indice + 1) % len(LETRAS)
    elif k == ord("p"):
        indice = len(LETRAS) - 1 if indice is None else (indice - 1) % len(LETRAS)
    elif 32 <= k < 127 and chr(k).upper() in LETRAS and chr(k).lower() not in ("n", "p", "q"):
        indice = LETRAS.index(chr(k).upper())

cap.release()
cv2.destroyAllWindows()