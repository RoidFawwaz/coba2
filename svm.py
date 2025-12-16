# Generated from: SVM.ipynb
# Converted at: 2025-12-16T12:54:15.550Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

data=pd.read_excel("data/data_diabetes.xlsx")
st
# Melihat 5 baris pertama
data.head(10)

data.info()

  # Nilai unik
print(data.nunique())

# Statistik deskriptif (numerik)
print(data.describe())

# Kolom numerik & kategorikal
numerik = data.select_dtypes(include=['int64','float64']).columns
kategorikal = data.select_dtypes(include=['object']).columns

print("Kolom numerik:", numerik)
print("Kolom kategorikal:", kategorikal)

import pandas as pd

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

# Tampilkan hasil
for col, info in hasil_iqr.items():
    print(f"\n=== {col} ===")
    for k, v in info.items():
        print(f"{k}: {v}")


import pandas as pd

# Cari semua kolom numerik
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

# Tampilkan hasil
for col, count in outlier_counts.items():
    print(f"Kolom {col}: {count} outlier")




# Ambil kolom numerik
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    plt.figure(figsize=(5, 4))
    plt.boxplot(data[col].dropna())
    plt.title(f"Boxplot {col}")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()


# Cari semua kolom numerik
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

# Lakukan IQR capping pada setiap kolom numerik
for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1

    # Tentukan batas bawah & atas
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Lakukan capping (nilai di luar batas akan diganti)
    data[col] = data[col].clip(lower_bound, upper_bound)

print("Outlier berhasil ditangani menggunakan IQR Capping!")

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


data.isnull().sum()


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
    test_size=0.2,      # 20% data untuk pengujian
    random_state=42,    # supaya hasil konsisten
    stratify=y          # membagi data seimbang antara 0 & 1
)

print("Ukuran X_train:", X_train.shape)
print("Ukuran X_test :", X_test.shape)
print("Ukuran y_train:", y_train.shape)
print("Ukuran y_test :", y_test.shape)


from sklearn.preprocessing import StandardScaler

# Buat scaler
scaler = StandardScaler()

# Fit + transform pada X_train, hanya transform pada X_test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.svm import SVC

# Buat model SVM (kernel RBF yang paling umum digunakan)
svm_model = SVC(kernel='rbf', random_state=42)

# Latih model
svm_model.fit(X_train_scaled, y_train)


y_pred = svm_model.predict(X_test_scaled)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Akurasi
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi Model SVM:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))