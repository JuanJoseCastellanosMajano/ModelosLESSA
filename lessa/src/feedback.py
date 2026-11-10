import numpy as np
from utils import angulos_mano, NOMBRES_DEDOS

UMBRAL = 25  # grados de diferencia para avisar


def generar_consejos(vector, ref):
    """Compara el vector del usuario con la referencia de la seña objetivo."""
    v = np.array(vector, dtype=float)
    consejos = []
    for nombre, ini, lado in (("izq", 0, "izquierda"), ("der", 63, "derecha")):
        bloque = v[ini:ini + 63]
        usa_mano = np.abs(bloque).sum() > 0

        if ref[f"activa_{nombre}"] < 0.5:
            if usa_mano:
                consejos.append(f"Esta seña no usa la mano {lado}: bájala.")
            continue
        if not usa_mano:
            consejos.append(f"Muestra la mano {lado} frente a la cámara.")
            continue

        dif = angulos_mano(bloque) - ref[nombre]
        for i, dedo in enumerate(NOMBRES_DEDOS):
            if dif[i] > UMBRAL:
                consejos.append(f"Mano {lado}: dobla más el {dedo}.")
            elif dif[i] < -UMBRAL:
                consejos.append(f"Mano {lado}: extiende más el {dedo}.")

    return consejos or ["¡Muy bien! Tu seña se ve correcta."]