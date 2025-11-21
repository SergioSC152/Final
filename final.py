
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, silhouette_score, roc_curve, auc)

np.random.seed(42)
n = 500

# Variables base
Habitaciones = np.random.randint(1, 6, n)
Banos = np.random.randint(1, 4, n)
Superficie_m2 = np.random.normal(120, 40, n).clip(40, 300)
Antiguedad = np.random.randint(0, 50, n)
Garaje = np.random.randint(0, 2, n)  # 0/1
Distrito = np.random.choice(["A", "B", "C", "D"], n, p=[0.25, 0.35, 0.25, 0.15])
Calidad = np.random.choice(["Baja", "Media", "Alta"], n, p=[0.2, 0.5, 0.3])

# Construcción del precio con factores realistas
base = (
    50000 +
    Habitaciones * 15000 +
    Banos * 10000 +
    Superficie_m2 * 120 -
    Antiguedad * 800 +
    Garaje * 10000
)
dist_factor = {"A": 45000, "B": 25000, "C": 10000, "D": 5000}
qual_factor = {"Baja": -15000, "Media": 10000, "Alta": 30000}

Precio = (
    base +
    np.array([dist_factor[d] for d in Distrito]) +
    np.array([qual_factor[q] for q in Calidad]) +
    np.random.normal(0, 15000, n)
).clip(30000)  # mínimo 30000

# DataFrame con nombres en español
df = pd.DataFrame({
    "Precio": Precio,
    "Habitaciones": Habitaciones,
    "Banos": Banos,
    "Superficie_m2": Superficie_m2,
    "Antiguedad": Antiguedad,
    "Garaje": Garaje,
    "Distrito": Distrito,
    "Calidad": Calidad
})

# Etiqueta binaria 'alto_precio' (para clasificación)
df["alto_precio"] = (df["Precio"] > df["Precio"].quantile(0.75)).astype(int)

print("Dataset sintético creado. Primeras filas:")
print(df.head())

# ----------------------------
# LIMPIEZA E IMPUTACIÓN (si necesario)
# ----------------------------
# (Aquí los datos ya están completos; incluimos imputadores por buenas prácticas)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

imputer_num = SimpleImputer(strategy='median')
imputer_cat = SimpleImputer(strategy='most_frequent')

df[num_cols] = imputer_num.fit_transform(df[num_cols])
if len(cat_cols) > 0:
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

# ----------------------------
# INGENIERÍA DE VARIABLES
# ----------------------------
# precio por m2
df["precio_m2"] = (df["Precio"] / df["Superficie_m2"]).replace(np.inf, np.nan)
df["precio_m2"] = df["precio_m2"].fillna(df["precio_m2"].median())

# Dummies (one-hot) para categóricas (Distrito, Calidad)
df_proc = pd.get_dummies(df, columns=["Distrito", "Calidad"], drop_first=True)

# ----------------------------
# FEATURES Y TARGETS
# ----------------------------
features = [c for c in df_proc.columns if c not in ("Precio", "alto_precio")]
X = df_proc[features].copy()
y_reg = df_proc["Precio"].copy()
y_clf = df_proc["alto_precio"].copy()

# ----------------------------
# ESCALADO
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# DIVISIÓN TRAIN / TEST
# ----------------------------
X_train, X_test, yreg_train, yreg_test = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)
_, _, yclf_train, yclf_test = train_test_split(X_scaled, y_clf, test_size=0.2, random_state=42)

# ----------------------------
# MODELADO - REGRESIÓN
# ----------------------------
rf_reg = RandomForestRegressor(n_estimators=300, random_state=42)
lr = LinearRegression()

rf_reg.fit(X_train, yreg_train)
lr.fit(X_train, yreg_train)

y_pred_rf = rf_reg.predict(X_test)
y_pred_lr = lr.predict(X_test)

mae_rf = mean_absolute_error(yreg_test, y_pred_rf)
mse_rf = mean_squared_error(yreg_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(yreg_test, y_pred_rf)

mae_lr = mean_absolute_error(yreg_test, y_pred_lr)
r2_lr = r2_score(yreg_test, y_pred_lr)

print("\nREGRESIÓN - Random Forest -> MAE: {:.2f}, RMSE: {:.2f}, R2: {:.3f}".format(mae_rf, rmse_rf, r2_rf))
print("REGRESIÓN - Linear Regression -> MAE: {:.2f}, R2: {:.3f}".format(mae_lr, r2_lr))

# ----------------------------
# MODELADO - CLASIFICACIÓN COMPARATIVA
# ----------------------------
models_clf = {
    "RegLog": LogisticRegression(max_iter=2000, random_state=42),
    "Arbol": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42)
}

clf_results = {}
for name, model in models_clf.items():
    model.fit(X_train, yclf_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(yclf_test, y_pred)
    prec = precision_score(yclf_test, y_pred, zero_division=0)
    rec = recall_score(yclf_test, y_pred, zero_division=0)
    f1 = f1_score(yclf_test, y_pred, zero_division=0)
    auc_score = roc_auc_score(yclf_test, y_proba) if y_proba is not None else None

    clf_results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc_score}
    print(f"{name} - Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}, AUC: {None if auc_score is None else round(auc_score,3)}")

# ----------------------------
# PCA + K-MEANS
# ----------------------------
pca = PCA(n_components=0.90, svd_solver='full')  # conservar 90% varianza
X_pca = pca.fit_transform(X_scaled)
n_comp = X_pca.shape[1]
print("\nPCA - componentes retenidos:", n_comp)
print("Varianza explicada acumulada (primeros componentes):", np.round(pca.explained_variance_ratio_.cumsum()[:n_comp], 3))

k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_pca)
sil_score = silhouette_score(X_pca, clusters)
print("KMeans - k:", k, "Silhouette score:", round(sil_score, 3))

# Añadir clusters al df original (versión con nombres en español)
df["cluster"] = clusters

# ----------------------------
# IMPORTANCIA DE VARIABLES Y GUARDADO
# ----------------------------
feat_importances = rf_reg.feature_importances_
imp_df = pd.DataFrame({"feature": features, "importance": feat_importances}).sort_values(by="importance", ascending=False)
imp_df.to_csv("feature_importances_rf_reg.csv", index=False)

pd.DataFrame(clf_results).T.to_csv("classification_results_summary.csv")
df.to_csv("housing_with_clusters.csv", index=False)

print("\nArchivos guardados: feature_importances_rf_reg.csv, classification_results_summary.csv, housing_with_clusters.csv")

# ----------------------------
# GRÁFICOS (se muestran si ejecutas interactivamente)
# ----------------------------
# 1) Real vs Predicho (Regresión RF)
plt.figure(figsize=(8,6))
plt.scatter(yreg_test, y_pred_rf, alpha=0.6)
plt.plot([yreg_test.min(), yreg_test.max()], [yreg_test.min(), yreg_test.max()], 'k--')
plt.xlabel("Precio real")
plt.ylabel("Precio predicho (RF)")
plt.title("Real vs Predicho - Random Forest (Regresión)")
plt.show()

# 2) ROC - RandomForest clasificación (si aplica)
best_clf = models_clf["RandomForest"]
if hasattr(best_clf, "predict_proba"):
    y_proba = best_clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(yclf_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - RandomForest (AUC = {roc_auc:.3f})")
    plt.show()
