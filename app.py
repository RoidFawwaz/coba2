import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Analisis Diabetes - SVM",
    layout="wide"
)

st.title("Analisis Diabetes Menggunakan Support Vector Machine (SVM)")

# =====================================================
# LOAD DATA (SEKALI SAJA)
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_excel("data/data_diabetes.xlsx")
    df.columns = df.columns.str.strip()
    return df

data = load_data()

TARGET = "Outcome"
X = data.drop(TARGET, axis=1)
y = data[TARGET]

# =====================================================
# TRAIN MODEL (SEKALI SAJA)
# =====================================================
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel="rbf", random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    return model, scaler, X_test, y_test, y_pred

svm_model, scaler, X_test, y_test, y_pred = train_model(X, y)

# =====================================================
# TAB
# =====================================================
tab_data, tab_prediksi = st.tabs([
    "📊 Data & Analisis",
    "🧠 Prediksi Diabetes"
])

# =====================================================
# TAB 1 — DATA & ANALISIS
# =====================================================
with tab_data:

    st.subheader("1. Dataset")
    st.dataframe(data.head(10), use_container_width=True)

    st.subheader("2. Informasi Dataset")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Jumlah Baris & Kolom")
        st.write(data.shape)

    with col2:
        st.write("Jumlah Nilai Unik")
        st.write(data.nunique())

    st.subheader("3. Statistik Deskriptif")
    st.dataframe(data.describe(), use_container_width=True)

    # =========================
    # OUTLIER IQR
    # =========================
    st.subheader("4. Analisis Outlier (IQR)")

    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
    outlier_info = {}

    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_info[col] = len(data[(data[col] < lower) | (data[col] > upper)])

    st.json(outlier_info)

    st.subheader("5. Boxplot Sebelum IQR Capping")
    for col in numeric_cols:
        fig, ax = plt.subplots()
        ax.boxplot(data[col])
        ax.set_title(col)
        st.pyplot(fig)

    # =========================
    # IQR CAPPING
    # =========================
    st.subheader("6. Penanganan Outlier (IQR Capping)")
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        data[col] = data[col].clip(lower, upper)

    st.success("Outlier berhasil ditangani")

    st.subheader("7. Evaluasi Model SVM")

    accuracy = accuracy_score(y_test, y_pred)
    st.metric("Akurasi Model", f"{accuracy:.2%}")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Tidak Diabetes", "Diabetes"]
    )
    disp.plot(ax=ax, values_format="d")
    st.pyplot(fig)

    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report, use_container_width=True)

    import pandas as pd

# Ambil hanya kolom numerik
corr_matrix = data.corr()

print("Matriks Korelasi Antar Atribut:")
print(corr_matrix)

st.subheader("Korelasi Antar Atribut")

corr_matrix = data.corr(numeric_only=True)

st.write("Matriks Korelasi Antar Atribut:")
st.dataframe(corr_matrix, use_container_width=True)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Heatmap Korelasi Antar Atribut")
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

fitur_pilih = [
    'Kadar gula darah',
    'Indeks massa tubuh',
    'Age',
    'Tekanan darah',
    'Outcome'
]

sns.pairplot(
    data[fitur_pilih],
    hue='Outcome',
    diag_kind='kde',
    plot_kws={'alpha': 0.6, 's': 30}
)

plt.suptitle(
    "Pairplot Antar Atribut Utama pada Data SVM",
    y=1.02,
    fontsize=14
)
plt.show()

import seaborn as sns

st.subheader("Heatmap Korelasi Antar Atribut")

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    ax=ax
)

ax.set_title("Heatmap Korelasi Antar Atribut")
st.pyplot(fig)

st.subheader("Pairplot Antar Atribut Utama")

fitur_pilih = [
    'Kadar gula darah',
    'Indeks massa tubuh',
    'Age',
    'Tekanan darah',
    'Outcome'
]

fig = sns.pairplot(
    data[fitur_pilih],
    hue='Outcome',
    diag_kind='kde',
    plot_kws={'alpha': 0.6, 's': 30}
)

fig.fig.suptitle(
    "Pairplot Antar Atribut Utama pada Data SVM",
    y=1.02,
    fontsize=14
)

st.pyplot(fig.fig)


# =====================================================
# TAB 2 — PREDIKSI
# =====================================================
with tab_prediksi:

    st.subheader("Prediksi Diabetes")

    st.markdown("### Input Manual Data Pasien")

    with st.form("form_prediksi"):
        input_data = {}

        for col in X.columns:
            input_data[col] = st.number_input(
                col,
                value=float(X[col].mean())
            )

        submit = st.form_submit_button("Prediksi")

    if submit:
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        pred = svm_model.predict(input_scaled)[0]

        hasil = "Diabetes" if pred == 1 else "Tidak Diabetes"
        st.success(f"Hasil Prediksi: **{hasil}**")

    st.markdown("### Prediksi dari File (CSV / Excel)")

    uploaded_file = st.file_uploader(
        "Upload file data pasien",
        type=["csv", "xlsx"]
    )

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df_input = pd.read_csv(uploaded_file)
        else:
            df_input = pd.read_excel(uploaded_file)

        st.dataframe(df_input.head(), use_container_width=True)

        if set(X.columns).issubset(df_input.columns):
            df_input = df_input[X.columns]
            preds = svm_model.predict(scaler.transform(df_input))
            df_input["Prediksi"] = np.where(
                preds == 1,
                "Diabetes",
                "Tidak Diabetes"
            )

            st.success("Prediksi berhasil")
            st.dataframe(df_input, use_container_width=True)
        else:
            st.error("Kolom file tidak sesuai dengan data training")

st.markdown("---")
st.caption("Aplikasi Streamlit – Analisis Diabetes dengan SVM")
