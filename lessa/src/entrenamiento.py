import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from config import DATASET, MODELO, REFERENCIAS
from utils import angulos_mano

MODELO.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATASET, header=None, dtype={0: str, 1: str})
y = df[0].to_numpy()
personas = df[1].to_numpy()
X = df.iloc[:, 2:].to_numpy(dtype=float)

print(f"Muestras: {len(df)} | Personas: {len(set(personas))}")
print(df[0].value_counts().to_string(), "\n")

# Evaluar con personas que el modelo nunca vio
if len(set(personas)) >= 3:
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
                  .split(X, y, groups=personas))
    print("Evaluación separada por persona (la real)")
else:
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=42)
    print("AVISO: menos de 3 personas, evaluación optimista. Recolecta más personas.")

modelo = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
modelo.fit(X[tr], y[tr])
pred = modelo.predict(X[te])
print(classification_report(y[te], pred, zero_division=0))
print("Matriz de confusión (filas = real, columnas = predicho):")
print(modelo.classes_)
print(confusion_matrix(y[te], pred, labels=modelo.classes_))

# Modelo final con todos los datos
modelo.fit(X, y)
joblib.dump(modelo, MODELO)

# Referencias para los consejos: ángulos promedio de cada dedo por seña
refs = {}
for sena in sorted(set(y)):
    datos = X[y == sena]
    ref = {}
    for nombre, ini in (("izq", 0), ("der", 63)):
        bloque = datos[:, ini:ini + 63]
        activas = np.abs(bloque).sum(axis=1) > 0
        ref[f"activa_{nombre}"] = float(activas.mean())
        ref[nombre] = (np.mean([angulos_mano(r) for r in bloque[activas]], axis=0)
                       if activas.any() else np.zeros(5))
    refs[sena] = ref
joblib.dump(refs, REFERENCIAS)
print("\nGuardados:", MODELO, "y", REFERENCIAS)