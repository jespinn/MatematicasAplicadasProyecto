import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
from collections import defaultdict, Counter
import json

try:
    from kmodes.kmodes import KModes
    KMODES_AVAILABLE = True
except Exception:
    KMODES_AVAILABLE = False

st.set_page_config(page_title="DataLab 3er Corte", layout="wide")
st.title("Proyecto Aplicadas 3er Corte")

with st.sidebar:
    st.header("Panel de Control")
    algo = st.selectbox(
        "Selecciona el algoritmo / operación",
        [
            "Normalización",
            "Discretización",
            "Árbol de decisión",
            "K-Means",
            "K-Modes (categórico)"
        ]
    )
    st.divider()
    decimales = st.number_input("Decimales para mostrar", 0, 10, 3)

uploaded = st.file_uploader("Cargar CSV", type=["csv"])

def read_csv(file):
    if file is None:
        return pd.DataFrame()
    return pd.read_csv(
        file,
        sep=None,
        engine="python",
        na_values=["", " ", "?", "NA", "NaN", "nan"],
        keep_default_na=True
    )

if uploaded is None:
    st.stop()

df = read_csv(uploaded).copy()
if df.empty:
    st.stop()

st.dataframe(df.head(20))
st.write("Tipos de datos")
st.write(df.dtypes)
st.write("Faltantes")
st.write(df.isna().sum())
st.divider()

def numeric_cols(df_):
    cols = []
    for c in df_.columns:
        serie = pd.to_numeric(df_[c], errors="coerce")
        if serie.notna().any():
            cols.append(c)
    return cols

def categorical_cols(df_):
    return [c for c in df_.columns if df_[c].dtype == object or pd.api.types.is_categorical_dtype(df_[c])]

def chi_merge(df_, col, target, max_bins=6, chi_critical=3.84):
    data = df_[[col, target]].dropna().copy()
    if data.empty:
        return pd.DataFrame(columns=["min", "max"])
    data = data.sort_values(col)
    clases = sorted(data[target].unique())
    tabla = data.groupby(col)[target].value_counts().unstack(fill_value=0)
    bins = tabla.reset_index().rename(columns={col: "min"})
    bins["max"] = bins["min"]
    bins = bins[["min", "max"] + clases]
    while len(bins) > max_bins:
        chi_vals = []
        for i in range(len(bins) - 1):
            fila1 = bins.iloc[i][clases].to_numpy()
            fila2 = bins.iloc[i + 1][clases].to_numpy()
            obs = np.vstack([fila1, fila2])
            fila_tot = obs.sum(axis=1, keepdims=True)
            col_tot = obs.sum(axis=0, keepdims=True)
            total = obs.sum()
            esp = fila_tot * col_tot / total
            chi2 = ((obs - esp) ** 2 / esp).sum()
            chi_vals.append((chi2, i))
        chi_min, idx = min(chi_vals, key=lambda x: x[0])
        bins.loc[idx, "max"] = bins.loc[idx + 1, "max"]
        for c in clases:
            bins.loc[idx, c] = bins.loc[idx, c] + bins.loc[idx + 1, c]
        bins = bins.drop(idx + 1).reset_index(drop=True)
    while True:
        if len(bins) <= 1:
            break
        chi_vals = []
        for i in range(len(bins) - 1):
            fila1 = bins.iloc[i][clases].to_numpy()
            fila2 = bins.iloc[i + 1][clases].to_numpy()
            obs = np.vstack([fila1, fila2])
            fila_tot = obs.sum(axis=1, keepdims=True)
            col_tot = obs.sum(axis=0, keepdims=True)
            total = obs.sum()
            esp = fila_tot * col_tot / total
            chi2 = ((obs - esp) ** 2 / esp).sum()
            chi_vals.append((chi2, i))
        chi_min, idx = min(chi_vals, key=lambda x: x[0])
        if chi_min > chi_critical:
            break
        bins.loc[idx, "max"] = bins.loc[idx + 1, "max"]
        for c in clases:
            bins.loc[idx, c] = bins.loc[idx, c] + bins.loc[idx + 1, c]
        bins = bins.drop(idx + 1).reset_index(drop=True)
    return bins

class DecisionTree:
    def __init__(self):
        self.tree = {}

    def calcular_desorden_arbol(self, datos, columna, columna_objetivo):
        n_total = len(datos)
        grupos = defaultdict(list)
        for fila in datos:
            grupos[fila[columna]].append(fila)
        desorden_total = 0
        categorias_objetivo = list(set([fila[columna_objetivo] for fila in datos]))
        for valor, grupo in grupos.items():
            n_rama = len(grupo)
            conteo_categorias = Counter([fila[columna_objetivo] for fila in grupo])
            desorden_rama = 0
            for categoria in categorias_objetivo:
                count = conteo_categorias.get(categoria, 0)
                if n_rama > 0:
                    p = count / n_rama
                    if p > 0:
                        desorden_rama += -p * math.log(p, len(categorias_objetivo))
            contribucion = (n_rama / n_total) * desorden_rama
            desorden_total += contribucion
        return desorden_total, grupos

    def construir_arbol(self, datos, columnas, target, profundidad=0):
        categorias = [fila[target] for fila in datos]
        if len(set(categorias)) == 1:
            return categorias[0]
        if len(columnas) == 0:
            return Counter(categorias).most_common(1)[0][0]
        mejor_desorden = float("inf")
        mejor_col = None
        mejor_particiones = None
        for columna in columnas:
            if columna != target:
                desorden, particiones = self.calcular_desorden_arbol(datos, columna, target)
                if desorden < mejor_desorden:
                    mejor_desorden = desorden
                    mejor_col = columna
                    mejor_particiones = particiones
        if mejor_col is None:
            return Counter(categorias).most_common(1)[0][0]
        arbol = {mejor_col: {}}
        nuevas_columnas = [c for c in columnas if c != mejor_col]
        for valor, subconjunto in mejor_particiones.items():
            if len(subconjunto) == 0:
                arbol[mejor_col][valor] = Counter(categorias).most_common(1)[0][0]
            else:
                subcategorias = [fila[target] for fila in subconjunto]
                if len(set(subcategorias)) == 1:
                    arbol[mejor_col][valor] = subcategorias[0]
                else:
                    arbol[mejor_col][valor] = self.construir_arbol(
                        subconjunto, nuevas_columnas, target, profundidad + 1
                    )
        return arbol

    def entrenar(self, datos, target):
        columnas = list(datos[0].keys())
        columnas = [c for c in columnas if c != target]
        self.tree = self.construir_arbol(datos, columnas, target)

    def generar_reglas(self, arbol=None, regla_actual=""):
        if arbol is None:
            arbol = self.tree
        reglas = []
        if isinstance(arbol, str):
            if regla_actual:
                reglas.append(f"{regla_actual} = {arbol}")
            return reglas
        if isinstance(arbol, dict):
            for caracteristica, valores in arbol.items():
                if isinstance(valores, dict):
                    for valor, subarbol in valores.items():
                        nueva_regla = f"{regla_actual} AND " if regla_actual else ""
                        nueva_regla += f"{caracteristica} = {valor}"
                        if nueva_regla.startswith("AND "):
                            nueva_regla = nueva_regla[4:]
                        if isinstance(subarbol, str):
                            reglas.append(f"SI {nueva_regla} ENTONCES Categoria = {subarbol}")
                        else:
                            reglas.extend(self.generar_reglas(subarbol, nueva_regla))
        return reglas

if algo == "Normalización":
    num_cols = numeric_cols(df)
    if not num_cols:
        st.error("No existen columnas numéricas para normalizar.")
        st.stop()

    cols = st.multiselect("Columnas a normalizar", num_cols, default=num_cols)
    if not cols:
        st.error("Selecciona columnas válidas.")
        st.stop()

    metodo = st.radio("Método", ["Min-Max [0,1]", "Z-Score", "Log"], horizontal=True)
    out = df.copy()

    cols_validas = []
    for c in cols:
        serie = pd.to_numeric(out[c], errors="coerce")
        if serie.notna().any():
            out[c] = serie
            cols_validas.append(c)

    if not cols_validas:
        st.error("Las columnas seleccionadas no tienen valores numéricos válidos para normalizar.")
        st.stop()

    cols = cols_validas
    X = out[cols]

    if X.shape[0] == 0 or X.shape[1] == 0:
        st.error("No hay datos suficientes (filas/columnas) para normalizar.")
        st.stop()

    if metodo == "Min-Max [0,1]":
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(X.values.astype(float))
        out[cols] = scaled
    elif metodo == "Z-Score":
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X.values.astype(float))
        out[cols] = scaled
    else:
        for c in cols:
            x = out[c].astype(float)
            shift = 0.0
            if (x <= 0).any():
                shift = abs(x.min()) + 1e-9
            out[c] = np.log(x + shift)

    st.dataframe(out.round(decimales))

elif algo == "Discretización":
    num_cols = numeric_cols(df)
    if not num_cols:
        st.warning("No hay columnas numéricas para discretizar")
        st.stop()
    col = st.selectbox("Columna numérica", num_cols)
    metodo = st.radio("Método", ["Equal-Width", "Equal-Frequency", "ChiMerge"], horizontal=True)
    out = df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce")
    if metodo in ["Equal-Width", "Equal-Frequency"]:
        k = st.number_input("Número de intervalos", 2, 50, 5)
        if metodo == "Equal-Width":
            out[col + "_bin"] = pd.cut(out[col], bins=int(k), duplicates="drop")
        else:
            out[col + "_bin"] = pd.qcut(out[col], q=int(k), duplicates="drop")
        st.dataframe(out[[col, col + "_bin"]].head(200))
    else:
        posibles_targets = [c for c in df.columns if c != col]
        if not posibles_targets:
            st.warning("No hay columnas objetivo disponibles")
            st.stop()
        target = st.selectbox("Columna objetivo", posibles_targets)
        max_bins = st.number_input("Máximo de intervalos", 2, 50, 6)
        chi_crit = st.number_input("Valor crítico", 0.0, 100.0, 3.84, step=0.01)
        bins_chi = chi_merge(out, col, target, max_bins=int(max_bins), chi_critical=chi_crit)
        if bins_chi.empty:
            st.warning("No se pudieron generar bins con ChiMerge")
            st.stop()
        def asignar_bin(valor, bins_df):
            if pd.isna(valor):
                return np.nan
            for _, row in bins_df.iterrows():
                if valor >= row["min"] and valor <= row["max"]:
                    return f"[{row['min']}, {row['max']}]"
            return np.nan
        out[col + "_chi"] = out[col].apply(lambda v: asignar_bin(v, bins_chi))
        st.dataframe(out[[col, target, col + "_chi"]].head(200))

elif algo == "Árbol de decisión":
    target = st.selectbox("Columna objetivo", df.columns)
    posibles_features = [c for c in df.columns if c != target]
    features = st.multiselect("Columnas de entrada", posibles_features, default=posibles_features)
    if not features:
        st.stop()
    work = df[features + [target]].copy()
    work = work.replace(["", " ", "?"], np.nan)
    work = work.fillna("NA")
    registros = work.to_dict(orient="records")
    if len(registros) == 0:
        st.stop()
    dt = DecisionTree()
    dt.entrenar(registros, target)
    reglas = dt.generar_reglas()
    st.subheader("Reglas generadas:")
    for i, regla in enumerate(reglas, 1):
        st.write(f"Regla {i}: {regla}")
    st.subheader("Árbol en formato JSON:")
    st.code(json.dumps(dt.tree, indent=2, ensure_ascii=False), language="json")

elif algo == "K-Means":
    num_cols = numeric_cols(df)
    if not num_cols:
        st.warning("No hay columnas numéricas para K-Means")
        st.stop()
    cols = st.multiselect("Columnas numéricas", num_cols, default=num_cols)
    if not cols:
        st.stop()
    k = st.number_input("Número de clusters", 2, 20, 3)
    init = st.selectbox("Inicialización", ["k-means++", "random"])
    out = df.copy()
    out[cols] = out[cols].apply(pd.to_numeric, errors="coerce")
    X = out[cols].fillna(out[cols].median(numeric_only=True))
    if X.shape[0] == 0 or X.shape[1] == 0:
        st.error("No hay datos suficientes para K-Means.")
        st.stop()
    model = KMeans(n_clusters=int(k), init=init, n_init="auto", random_state=42)
    labels = model.fit_predict(X)
    out["cluster_kmeans"] = labels
    st.dataframe(out.head(200))

elif algo == "K-Modes (categórico)":
    if not KMODES_AVAILABLE:
        st.warning("KModes no está disponible. Instala: pip install kmodes")
        st.stop()
    cat_cols = categorical_cols(df)
    if not cat_cols:
        st.warning("No hay columnas categóricas para K-Modes")
        st.stop()
    cols = st.multiselect("Columnas categóricas", cat_cols, default=cat_cols)
    if not cols:
        st.stop()
    k = st.number_input("Número de clusters", 2, 20, 3)
    out = df.copy()
    X = out[cols].astype(str).fillna("NA")
    km = KModes(n_clusters=int(k), init="Huang", n_init=5, random_state=42)
    labels = km.fit_predict(X)
    out["cluster_kmodes"] = labels
    st.dataframe(out.head(200))
