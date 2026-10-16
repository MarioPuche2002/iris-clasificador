"""Clasificador de flores Iris con Random Forest."""

import matplotlib
matplotlib.use("Agg")  # Guarda las gráficas como imagen (sin ventana)
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split


def main():
    # 1. Cargar los datos
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    print(f"Datos: {X.shape[0]} flores, {X.shape[1]} variables")

    # 2. Dividir en entrenamiento (80 %) y prueba (20 %)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Entrenar el modelo
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    # 4. Evaluar
    y_pred = modelo.predict(X_test)
    print(f"Exactitud: {accuracy_score(y_test, y_pred):.2%}")

    # 5. Guardar la matriz de confusión
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=iris.target_names
    )
    plt.title("Matriz de confusión - Iris")
    plt.savefig("matriz_confusion.png", dpi=150, bbox_inches="tight")
    print("Gráfica guardada en matriz_confusion.png")


if __name__ == "__main__":
    main()