# Generated from: uuu.ipynb
# Converted at: 2025-12-16T12:13:09.492Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(
    page_title="Analisis Diabetes - SVM",
    layout="wide"
)

st.title("Analisis Diabetes - SVM")

data=pd.read_excel("data/data_diabetes.xlsx")
data.columns = data.columns.str.strip()
st.set_page_config(
    page_title="Analisis Diabetes - SVM",
    layout="wide"
)

st.title("Analisis Diabetes Menggunakan Support Vector Machine (SVM)")

data.head(10)

st.subheader("1. Memuat Dataset")

data = pd.read_excel("data/data_diabetes.xlsx")
data.columns = data.columns.str.strip()

st.success("Dataset berhasil dimuat")
st.dataframe(data.head(10), use_container_width=True)

data.info()

st.subheader("2. Informasi Dataset")

col1, col2 = st.columns(2)

with col1:
    st.write("Jumlah Baris & Kolom")
    st.write(data.shape)

with col2:
    st.write("Jumlah Nilai Unik per Kolom")
    st.write(data.nunique())

st.subheader("Statistik Deskriptif")
st.dataframe(data.describe(), use_container_width=True)

print(data.nunique())

print(data.describe())

numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
hasil_iqr = {}

for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    hasil_iqr[col] = {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound
    }

for col, info in hasil_iqr.items():
    print(f"\n=== {col} ===")
    for k, v in info.items():
        print(f"{k}: {v}")


numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

outlier_counts = {}

for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
    outlier_counts[col] = len(outliers)


for col, count in outlier_counts.items():
    print(f"Kolom {col}: {count} outlier")

st.subheader("3. Analisis Outlier (IQR)")

numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns

outlier_counts = {}

for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_counts[col] = len(data[(data[col] < lower) | (data[col] > upper)])

st.write("Jumlah outlier per kolom:")
st.json(outlier_counts)

numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    plt.figure(figsize=(5, 4))
    plt.boxplot(data[col].dropna())
    plt.title(f"Boxplot {col}")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

st.subheader("4. Boxplot Sebelum Penanganan Outlier")

for col in numeric_cols:
    fig, ax = plt.subplots()
    ax.boxplot(data[col].dropna())
    ax.set_title(f"Boxplot {col}")
    ax.set_ylabel(col)
    st.pyplot(fig)

numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns


for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data[col] = data[col].clip(lower_bound, upper_bound)

print("Outlier berhasil ditangani menggunakan IQR Capping!")

st.subheader("5. Penanganan Outlier (IQR Capping)")

for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    data[col] = data[col].clip(lower, upper)

st.success("Outlier berhasil ditangani menggunakan IQR Capping")

for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    count = len(data[(data[col] < lower) | (data[col] > upper)])
    print(f"Kolom {col}: {count} outlier tersisa")


for col in numeric_cols:
    plt.figure(figsize=(5, 5))
    plt.boxplot(data[col])
    plt.title(f"Boxplot Kolom: {col}")
    plt.ylabel(col)
    plt.grid(True)
    plt.show()

st.subheader("6. Boxplot Setelah Penanganan Outlier")

for col in numeric_cols:
    fig, ax = plt.subplots()
    ax.boxplot(data[col])
    ax.set_title(f"Boxplot {col}")
    ax.set_ylabel(col)
    st.pyplot(fig)


data.duplicated().sum()


TARGET = "Outcome"

if TARGET not in data.columns:
    raise ValueError(f"Kolom target '{TARGET}' tidak ditemukan. Kolom tersedia: {list(data.columns)}")

X = data.drop(TARGET, axis=1)
y = data[TARGET]



from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Ukuran X_train:", X_train.shape)
print("Ukuran X_test :", X_test.shape)
print("Ukuran y_train:", y_train.shape)
print("Ukuran y_test :", y_test.shape)

st.subheader("7. Persiapan Data Modeling")

TARGET = "Outcome"

if TARGET not in data.columns:
    st.error(f"Kolom target '{TARGET}' tidak ditemukan")
    st.stop()

X = data.drop(TARGET, axis=1)
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

st.write("Ukuran Data:")
st.write({
    "X_train": X_train.shape,
    "X_test": X_test.shape,
    "y_train": y_train.shape,
    "y_test": y_test.shape
})


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.svm import SVC

svm_model = SVC(kernel='rbf', random_state=42)


svm_model.fit(X_train_scaled, y_train)


y_pred = svm_model.predict(X_test_scaled)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


accuracy = accuracy_score(y_test, y_pred)
print("Akurasi Model SVM:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Tidak Diabetes', 'Diabetes'])
disp.plot(values_format='d')
plt.title("Confusion Matrix - SVM")
plt.show()

st.subheader("8. Training Model SVM")

svm_model = SVC(kernel="rbf", random_state=42)
svm_model.fit(X_train_scaled, y_train)

st.success("Model SVM berhasil dilatih")

error = 1 - accuracy

plt.figure(figsize=(6,5))
plt.bar(['Accuracy', 'Error'], [accuracy, error])
plt.title("Akurasi vs Error Model SVM")
plt.ylabel("Nilai")
plt.show()


report = classification_report(y_test, y_pred, output_dict=True)

st.subheader("9. Evaluasi Model")

y_pred = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

st.metric("Akurasi Model", f"{accuracy:.2%}")

import pandas as pd
df_report = pd.DataFrame(report).transpose()

plt.figure(figsize=(10,4))
plt.plot(df_report['precision'], label='Precision')
plt.plot(df_report['recall'], label='Recall')
plt.plot(df_report['f1-score'], label='F1-Score')
plt.title("Classification Report Metrics")
plt.xlabel("Kelas / Rata-rata")
plt.ylabel("Nilai")
plt.legend()
plt.grid(True)
plt.show()

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Tidak Diabetes", "Diabetes"]
)
disp.plot(ax=ax, values_format="d")
st.pyplot(fig)

st.subheader("Classification Report")

report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
st.dataframe(df_report, use_container_width=True)

st.subheader("Visualisasi Precision, Recall, F1-Score")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_report["precision"], label="Precision")
ax.plot(df_report["recall"], label="Recall")
ax.plot(df_report["f1-score"], label="F1-Score")
ax.set_title("Classification Metrics")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# =========================================================
# PREDIKSI DIABETES (APLIKASI INTERAKTIF)
# =========================================================
st.markdown("---")
st.subheader("10. Prediksi Diabetes")

st.write(
    "Masukkan data pasien untuk memprediksi apakah berpotensi **Diabetes** atau **Tidak Diabetes** "
    "menggunakan model SVM yang telah dilatih."
)

# =========================================================
# INPUT MANUAL
# =========================================================
st.markdown("### 🔹 Input Manual Data Pasien")

with st.form("form_prediksi"):
    input_data = {}

    for col in X.columns:
        min_val = float(X[col].min())
        max_val = float(X[col].max())
        mean_val = float(X[col].mean())

        input_data[col] = st.number_input(
            label=col,
            min_value=min_val,
            max_value=max_val,
            value=mean_val
        )

    submit_btn = st.form_submit_button("Prediksi")

if submit_btn:
    input_df = pd.DataFrame([input_data])

    # Scaling input
    input_scaled = scaler.transform(input_df)

    # Prediksi
    pred = svm_model.predict(input_scaled)[0]

    hasil = "Diabetes" if pred == 1 else "Tidak Diabetes"

    if hasattr(svm_model, "decision_function"):
        score = svm_model.decision_function(input_scaled)[0]
        confidence = abs(score)
    else:
        confidence = None

    st.success(f"Hasil Prediksi: **{hasil}**")

    if confidence is not None:
        st.info(f"Skor Kepercayaan Model: {confidence:.2f}")

# =========================================================
# UPLOAD FILE (CSV / EXCEL)
# =========================================================
st.markdown("### 🔹 Prediksi dari File (CSV / Excel)")

uploaded_file = st.file_uploader(
    "Upload file berisi data pasien",
    type=["csv", "xlsx"]
)

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        input_file_df = pd.read_csv(uploaded_file)
    else:
        input_file_df = pd.read_excel(uploaded_file)

    st.write("Preview Data:")
    st.dataframe(input_file_df.head(), use_container_width=True)

    # Validasi kolom
    missing_cols = set(X.columns) - set(input_file_df.columns)

    if missing_cols:
        st.error(f"Kolom berikut tidak ditemukan di file: {missing_cols}")
    else:
        input_file_df = input_file_df[X.columns]

        input_scaled = scaler.transform(input_file_df)
        preds = svm_model.predict(input_scaled)

        hasil_prediksi = input_file_df.copy()
        hasil_prediksi["Prediksi"] = np.where(preds == 1, "Diabetes", "Tidak Diabetes")

        st.success("Prediksi berhasil dilakukan")
        st.dataframe(hasil_prediksi, use_container_width=True)

        # Ringkasan
        st.markdown("### Ringkasan Hasil")
        st.write(hasil_prediksi["Prediksi"].value_counts())

st.markdown("---")
st.caption("Aplikasi Streamlit – Analisis Diabetes dengan SVM")