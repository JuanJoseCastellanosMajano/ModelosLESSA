import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
dibujar = mp.solutions.drawing_utils

# 63 (mano izq) + 63 (mano der) + 3 (posición de una mano respecto a la otra)
N_FEATURES = 63 + 63 + 3

DEDOS = {
    "pulgar": [1, 2, 3, 4],
    "índice": [5, 6, 7, 8],
    "medio": [9, 10, 11, 12],
    "anular": [13, 14, 15, 16],
    "meñique": [17, 18, 19, 20],
}
NOMBRES_DEDOS = list(DEDOS)


class Resultado:
    """Guarda las manos detectadas: {'izq': landmarks o None, 'der': ...}"""
    def __init__(self, izq=None, der=None):
        self.izq = izq
        self.der = der


def crear_detector():
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,          # 0 = más rápido, 1 = más preciso
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _mano(lms):
    """63 valores de una mano, relativos a la muñeca y escalados."""
    if lms is None:
        return [0.0] * 63
    lm = lms.landmark
    b = lm[0]
    pts = [(p.x - b.x, p.y - b.y, p.z - b.z) for p in lm]
    escala = max(abs(c) for p in pts for c in p) or 1
    return [c / escala for p in pts for c in p]


def _tam(lms):
    """Tamaño de la mano en la imagen (muñeca a base del dedo medio)."""
    lm = lms.landmark
    return np.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y) or 1e-6


def extraer(frame, hands):
    """Devuelve (vector de 129 valores o None, Resultado).
    El vector es None si no se detecta ninguna mano."""
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    r = Resultado()

    if res.multi_hand_landmarks and res.multi_handedness:
        mejores = {}  # por etiqueta, nos quedamos con la de mayor confianza
        for lms, h in zip(res.multi_hand_landmarks, res.multi_handedness):
            c = h.classification[0]
            if c.label not in mejores or c.score > mejores[c.label][0]:
                mejores[c.label] = (c.score, lms)
        # Sin voltear la imagen, MediaPipe etiqueta al revés (espejo).
        # Es consistente en todo el proyecto, así que no afecta al modelo.
        r.izq = mejores.get("Left", (0, None))[1]
        r.der = mejores.get("Right", (0, None))[1]

    if r.izq is None and r.der is None:
        return None, r

    if r.izq is not None and r.der is not None:
        tam = (_tam(r.izq) + _tam(r.der)) / 2
        a, b = r.izq.landmark[0], r.der.landmark[0]
        rel = [(b.x - a.x) / tam, (b.y - a.y) / tam, (b.z - a.z) / tam]
    else:
        rel = [0.0, 0.0, 0.0]

    return _mano(r.izq) + _mano(r.der) + rel, r


def dibujar_todo(frame, res):
    for h in (res.izq, res.der):
        if h is not None:
            dibujar.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)


def estado(res):
    return (f"Izq:{'SI' if res.izq else 'no'}  "
            f"Der:{'SI' if res.der else 'no'}")


def _angulo(a, b, c):
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def angulos_mano(v63):
    """5 ángulos (uno por dedo). ~180° = dedo recto, menos = doblado."""
    pts = np.array(v63, dtype=float).reshape(21, 3)
    out = []
    for i in DEDOS.values():
        a, b, c, d = pts[i[0]], pts[i[1]], pts[i[2]], pts[i[3]]
        out.append((_angulo(a, b, c) + _angulo(b, c, d)) / 2)
    return np.array(out)