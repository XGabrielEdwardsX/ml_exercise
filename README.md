# Explicación del proyecto

El objetivo principal de este proyecto es desarrollar un modelo de Machine Learning que prediga si un producto en la plataforma de mercado libre es nuevo o usado. Este proyecto utiliza datos de productos listados en la plataforma, que incluyen diversas características como el precio, las cantidades disponibles, el tipo de producto, el modo de compra y otros atributos relacionados con la publicación del producto. (48 columnas en total)

A través de la Exploración y Limpieza de Datos (EDA), se seleccionaron las características más relevantes y se transformaron para asegurar que fueran adecuadas para alimentar los modelos de clasificación. Posteriormente, se entrenaron varios modelos (Logistic Regression, Random Forest, XGBoost, Decision Tree) para determinar cuál ofrece el mejor rendimiento en términos de precisión, AUC y F1.

# 1. Estructura del proyecto


| Ruta                     | Descripción                              |
|--------------------------|------------------------------------------|
| `data/`                  | Contiene el CSV limpio generado tras EDA |
| `data/cleaned_data.csv`  | Archivo de datos procesados              |
| `notebooks/`             | Contiene los notebooks del proyecto      |
| `notebooks/eda.ipynb`    | Notebook de exploración y limpieza       |
| `notebooks/train.ipynb`  | Notebook de entrenamiento y evaluación   |
| `pdf/`                   | Contiene documentos de referencia        |
| `pdf/guia.pdf`           | Documento con el enunciado               |
| `venv/`                  | Entorno virtual (no se sube a Git  y tu debes crear el tuyo)       |
| `requirements.txt`       | Lista de dependencias del proyecto       |
| `.gitignore`             | Archivo de exclusión para Git            |
| `README.md`              | Documentación (estas aquí)    |


# 2. Creación y activación del entorno

```bash
python -m venv venv
venv\Scripts\activate  # Windows CMD
source venv/bin/activate  # macOS / Linux bash
pip install -r requirements.txt
```

`requirements.txt` incluye:

```
pandas
scikit-learn
xgboost
matplotlib
```

# 3. Notebook `eda.ipynb` – Exploración y limpieza

| Paso | Descripción | Resultado clave |
| --- | --- | --- |
| 1. Carga inicial | Se leen 100 000 registros de `MLA_100k.jsonlines`. | 48 columnas originales. |
| 2. Selección de columnas | Se preservan 12 variables que pueden ser relevantes | Precio, cantidades, flags, tipo de listing, modo de compra, moneda, estado y condición. |
| 3. Desanidar `shipping` | Del diccionario `shipping` se extraen dos flags: `shipping_local_pick_up` y `shipping_free`. | Dos columnas binarias (0/1). |
| 4. Conversión de booleanos | `automatic_relist`, `shipping_*` → `int` para usarse como numéricas. | Coherencia de tipos. |
| 5. Eliminación de columnas con baja varianza | `site_id`, `currency_id`, `status` (>95 % un único valor) se descartan. | Se evitan dummies inútiles. |
| 6. One-Hot Encoding | `listing_type_id` (7 niveles) y `buying_mode` (3 niveles) se codifican ⇒ 10 dummies. | DataFrame final de 18 columnas. |
| 7. Exportación | Se guarda `data/cleaned_data.csv`. | Archivo listo para modelar. |

# 4. Notebook `train.ipynb` – Entrenamiento y evaluación

Los datos se leen inicialmente desde `../data/cleaned_data.csv`, generado en el notebook `eda.ipynb`. Este archivo contiene 17 features (7 numéricas originales + 10 dummies) y 1 columna objetivo `condition` (1 = nuevo, 0 = usado).

| Sección | Detalle |
|---------|---------|
| Importaciones | `pandas`, `scikit-learn` (model_selection, metrics, linear_model, ensemble, tree), `xgboost`. |
| Partición 70 / 30 | `train_test_split(stratify=y)` asegura misma proporción nuevo/usado en ambos conjuntos. |
| Modelos evaluados | Logistic Regression (LR), Random Forest (RF), XGBoost (XGB) y Decision Tree (DT). |
| Métricas impresas | AUC (ranking global), accuracy (exactitud), precisión, recall, F1. |
| Resultados | XGB ⇒ AUC 0.92, accuracy 0.832, F1 0.82.<br>RF ⇒ AUC 0.90, F1 0.82.<br>DT ⇒ AUC 0.88, F1 0.82.<br>LR ⇒ AUC 0.86, F1 0.58. |
| Visualización | Curvas ROC individuales confirman la jerarquía XGB > RF > DT > LR. |
| Feature Importance (XGB) | `price` domina la predicción; le siguen `sold_quantity`, `initial_quantity` y `available_quantity`. |

# 5. Conclusiones

- **Limpieza efectiva**: pasamos de 48 columnas crudas a 18 features útiles sin perder registros.
- **Modelo ganador**: XGBoost ofrece la mejor combinación de AUC y F1, superando a RF y LR.
- **Variables clave**: el precio y las cantidades vendidas/disponibles concentran la señal predictiva.
