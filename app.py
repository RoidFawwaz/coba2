# Generated from: uuu.ipynb
# Converted at: 2025-12-16T12:13:09.492Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_excel("data/data diabetes.xlsx")

data.head(10)

data.info()


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



numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    plt.figure(figsize=(5, 4))
    plt.boxplot(data[col].dropna())
    plt.title(f"Boxplot {col}")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()



numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns


for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

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


data.duplicated().sum()


data.write(data.columns)
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

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



from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Tidak Diabetes', 'Diabetes'])
disp.plot(values_format='d')
plt.title("Confusion Matrix - SVM")
plt.show()


error = 1 - accuracy

plt.figure(figsize=(6,5))
plt.bar(['Accuracy', 'Error'], [accuracy, error])
plt.title("Akurasi vs Error Model SVM")
plt.ylabel("Nilai")
plt.show()


report = classification_report(y_test, y_pred, output_dict=True)

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