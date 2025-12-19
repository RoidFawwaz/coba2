import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# =========================
# SETUP AWAL
# =========================
st.set_page_config(page_title="SVM App", layout="wide")

menu = st.sidebar.selectbox(
    "Menu",
    ["🏠 Home", "🧹 Preprocessing", "🤖 Modeling"]
)

# =========================
# HOME
# =========================
if menu == "🏠 Home":
    st.title("🏠 Home")
    st.write("Aplikasi Klasifikasi Menggunakan SVM")
    st.info("Silakan upload dan preprocessing data terlebih dahulu.")

# =========================
# PREPROCESSING
# =========================
elif menu == "🧹 Preprocessing":
    st.title("🧹 Preprocessing Data")

    uploaded_file = st.file_uploader("Upload Dataset (Excel)", type=["xlsx"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)

        st.subheader("📄 Data Awal")
        st.dataframe(df.head())

        # ---------------------------
        # 1. Pilih Target
        # ---------------------------
        target_col = st.selectbox("Pilih Kolom Target (y)", df.columns)

        # ---------------------------
        # 2. Tangani Missing Value
        # ---------------------------
        st.subheader("🧼 Missing Value")

        if df.isnull().sum().sum() > 0:
            st.warning("⚠️ Terdapat missing value")

            metode = st.radio(
                "Metode Penanganan Missing Value",
                ["Drop baris", "Isi dengan mean"]
            )

            if metode == "Drop baris":
                df = df.dropna()
            else:
                for col in df.select_dtypes(include=np.number):
                    df[col] = df[col].fillna(df[col].mean())
        else:
            st.success("✅ Tidak ada missing value")

        # ---------------------------
        # 3. Encoding Data Kategorikal
        # ---------------------------
        st.subheader("🔡 Encoding Data")

        cat_cols = df.select_dtypes(include="object").columns.tolist()
        cat_cols = [c for c in cat_cols if c != target_col]

        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
            st.info(f"Encoding kolom: {cat_cols}")

        # ---------------------------
        # 4. Pisahkan X dan y
        # ---------------------------
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Pastikan target kategorikal
        if y.dtype == "object":
            y = pd.factorize(y)[0]

        # ---------------------------
        # 5. Simpan ke session_state
        # ---------------------------
        st.session_state['X'] = X
        st.session_state['y'] = y

        st.success("✅ Preprocessing selesai & data siap untuk Modeling")

        st.write("📌 Shape X:", X.shape)
        st.write("📌 Distribusi Target:")
        st.bar_chart(pd.Series(y).value_counts())

# =========================
# MODELING SVM
# =========================
elif menu == "🤖 Modeling":
    st.title("🤖 Modeling SVM")

    if 'X' not in st.session_state or 'y' not in st.session_state:
        st.warning("⚠️ Data belum siap. Silakan ke menu Preprocessing dulu.")
    else:
        X = st.session_state['X']
        y = st.session_state['y']

        with st.expander("⚙️ Konfigurasi Parameter Model", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                test_size = st.slider(
                    "Ukuran Data Test (%)",
                    10, 90, 40
                ) / 100

            with col2:
                kernel = st.selectbox(
                    "Kernel SVM",
                    ["linear", "rbf", "poly", "sigmoid"]
                )

        if st.button("🚀 Mulai Training Model"):
            with st.spinner("Sedang melatih model..."):
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=test_size,
                        random_state=42,
                        stratify=y
                    )
                except ValueError:
                    st.warning("Stratify gagal, lanjut tanpa stratify.")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=test_size,
                        random_state=42
                    )

                # Scaling
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Training
                model = SVC(kernel=kernel, probability=True)
                model.fit(X_train_scaled, y_train)

                # Prediksi
                y_pred = model.predict(X_test_scaled)

                # Simpan hasil
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = y_pred

                acc = accuracy_score(y_test, y_pred)

                st.success("✅ Training Selesai!")
                st.metric("🎯 Akurasi Model", f"{acc*100:.2f}%")
