# Aprende LESSA con IA y ESP32-CAM

Sistema que reconoce señas del alfabeto de la Lengua de Señas Salvadoreña (LESSA) a partir de video, y da consejos sobre cómo mejorar la posición de los dedos. Proyecto universitario de programación de artefactos con ESP32.

## Cómo funciona

```
ESP32-CAM --WiFi (stream MJPEG)--> PC con Python --> Resultado (seña, confianza, consejos)
 (captura)                          MediaPipe Hands
                                    + clasificador (Random Forest)
                                    + reglas de consejos
```

1. **Captura:** la ESP32-CAM transmite video por WiFi (o se usa una webcam para pruebas).
2. **Detección de manos:** MediaPipe Hands localiza hasta 2 manos y devuelve 21 puntos (landmarks) por mano.
3. **Normalización:** los puntos se expresan relativos a la muñeca y se escalan, así la seña se reconoce sin importar la posición o la distancia a la cámara.
4. **Clasificación:** un Random Forest (scikit-learn) entrenado con muestras propias decide qué letra es.
5. **Consejos:** se calcula el ángulo de cada dedo y se compara con el promedio de la seña de referencia. Si un dedo difiere más de 25 grados, se indica "dobla más / extiende más el dedo".

El modelo no entrena sobre imágenes sino sobre coordenadas de puntos. Por eso necesita pocos datos y no depende tanto de la iluminación ni del fondo.

## Vector de características (129 valores por muestra)

| Bloque | Valores |
|---|---|
| Mano izquierda (21 puntos x, y, z) | 63 |
| Mano derecha (21 puntos x, y, z) | 63 |
| Posición de una muñeca respecto a la otra | 3 |

Si una mano no aparece, su bloque queda en ceros.

## Estructura del proyecto

```
lessa/
├── files/dataset.csv        # muestras: seña, persona, 129 valores
├── model/
│   ├── lessa.joblib         # modelo entrenado
│   └── referencias.joblib   # ángulos de referencia por seña
├── src/
│   ├── config.py            # señas, cámara y rutas
│   ├── utils.py             # MediaPipe, normalización, ángulos
│   ├── recollect.py         # recolección de datos
│   ├── entrenamiento.py     # entrenamiento y evaluación
│   ├── feedback.py          # generación de consejos
│   └── capture.py           # prueba en vivo
└── README.md
```

## Instalación (Linux)

MediaPipe aún no soporta Python 3.14, por lo que se usa Python 3.12 aislado con [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd lessa
uv venv --python 3.12
source .venv/bin/activate
uv pip install "mediapipe==0.10.14" "numpy<2" scikit-learn pandas joblib
```

Si `cv2.imshow` falla en Linux: `sudo apt install libxcb-cursor0 libgl1`.

## Uso

Todos los comandos se ejecutan desde la raíz del proyecto con el entorno activo.

**1. Recolectar datos** (una ejecución por persona):

```bash
python src/recollect.py ana
python src/recollect.py luis
```

Teclas: `N` / `P` siguiente o anterior seña, `S` iniciar o pausar grabación, `Q` salir. Mientras se graba, conviene mover un poco la mano (distancia y ángulo). Recomendado: unas 200 muestras por seña y al menos 4 personas distintas.

**2. Entrenar:**

```bash
python src/entrenamiento.py
```

Imprime precisión por seña y matriz de confusión. Si hay 3 o más personas, el conjunto de prueba se separa **por persona**, para medir cómo se comporta con gente que el modelo nunca vio.

**3. Probar en vivo:**

```bash
python src/capture.py
```

Para ver consejos, poner `OBJETIVO = "A"` (la seña a practicar) en `capture.py`.

## Usar la ESP32-CAM

1. Cargar el ejemplo `CameraWebServer` (placa AI Thinker ESP32-CAM, WiFi de 2.4 GHz) y anotar la IP del monitor serie.
2. Verificar que `http://IP:81/stream` abre en el navegador.
3. En `src/config.py`: `FUENTE = "http://IP:81/stream"`.
4. **Volver a recolectar y reentrenar con la ESP32-CAM**, porque su imagen es distinta a la de una webcam y el modelo debe entrenarse con lo que verá en uso real.

## Limitaciones conocidas

- Solo reconoce **señas estáticas** (una pose). Las letras que llevan movimiento necesitan un modelo de secuencia y quedan como mejora futura.
- No usa la cara, por rendimiento. Se pierden las expresiones faciales y la ubicación de la seña respecto al rostro.
- La calidad depende de la diversidad del dataset: pocas personas producen un modelo que solo reconoce a quienes lo entrenaron.
- Los consejos se basan solo en el ángulo de los dedos, no en la orientación de la palma.
- Las señas deben validarse con la comunidad sorda salvadoreña. No se deben usar datasets de otras lenguas de señas (como ASL), porque son lenguas distintas.

## Resultados

_Completar tras entrenar: número de muestras, personas, precisión global y matriz de confusión._

## Mejoras futuras

- Modelo de secuencias (LSTM/GRU) para letras con movimiento y palabras.
- Aplicación web con FastAPI para usarlo desde el celular.
- Reincorporar la cara con un detector más liviano.
- Validación de las señas con la comunidad sorda y pruebas con usuarios reales.

## Tecnologías

Python 3.12, MediaPipe, OpenCV, scikit-learn, ESP32-CAM.
