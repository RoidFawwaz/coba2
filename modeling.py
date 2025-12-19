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

# Contoh menu (biar ga error)
menu = st.sidebar.selectbox(
    "Menu",
    ["🏠 Home", "🧹 Preprocessing", "🤖 Modeling"]
)

# =========================
# DUMMY PREPROCESSING
# (SIMULASI BIAR MODELING BISA JALAN)
# =========================
if menu == "🧹 Preprocessing":
    st.title("🧹 Preprocessing (Dummy)")

    # Contoh data klasifikasi
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    st.session_state['X'] = X
    st.session_state['y'] = y

    st.success("✅ Data preprocessing selesai (dummy dataset).")
    st.write(X.head())

# =========================
# MODELING SVM
# =========================
elif menu == "🤖 Modeling":
    st.title("🤖 Modeling SVM")

    if 'X' not in st.session_state or 'y' not in st.session_state:
        st.warning("⚠️ Data belum siap. Silakan ke Preprocessing dulu.")
    else:
        X = st.session_state['X']
        y = st.session_state['y']

        with st.expander("⚙️ Konfigurasi Parameter Model", expanded=True):
            col_p1, col_p2 = st.columns(2)

            with col_p1: 
                test_size = st.slider(
                    "Ukuran Data Test (%)",
                    min_value=10,
                    max_value=90,
                    value=40
                ) / 100
                st.caption("Semakin besar, data latih semakin sedikit.")

            with col_p2: 
                kernel = st.selectbox(
                    "Kernel SVM",
                    ["linear", "rbf", "poly", "sigmoid"]
                )
                st.caption("RBF biasanya bagus untuk data kompleks.")

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
                    st.warning("⚠️ Stratify gagal (target kontinu?), lanjut tanpa stratify.")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=test_size,
                        random_state=42
                    )

                # Scaling
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Training Model
                model = SVC(kernel=kernel, probability=True)
                model.fit(X_train_scaled, y_train)

                # Prediksi
                y_pred = model.predict(X_test_scaled)

                # Simpan ke session_state
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['X_test'] = X_test_scaled
                st.session_state['X_test_raw'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = y_pred
                st.session_state['X_train_raw'] = X_train
                st.session_state['y_train'] = y_train

                # Output
                acc = accuracy_score(y_test, y_pred)

                st.success("✅ Training Selesai!")
                st.metric("🎯 Akurasi Model", f"{acc*100:.2f}%")

# =========================
# HOME
# =========================
else:
    st.title("🏠 Home")
    st.write("Aplikasi SVM dengan Streamlit")
