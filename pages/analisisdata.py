import streamlit as st
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Judul aplikasi
st.title("Klasifikasi Jenis Beras Cammeo dan Osmancik")

# Header untuk bagian dataset
st.header("Dataset")

# Membaca dataset
df = pd.read_csv('Rice_dataset.csv', sep=';')
st.write("Tampilan Dataset:")
st.dataframe(df.head())  # Menampilkan 5 baris pertama dataset

# Menambahkan tabel penjelasan fitur
st.subheader("Penjelasan Fitur Dataset")
feature_description = {
    "Fitur": [
        "Area",
        "Perimeter",
        "Major Axis Length",
        "Minor Axis Length",
        "Eccentricity",
        "Convex Area",
        "Extent"
    ],
    "Penjelasan": [
        "Mengembalikan jumlah piksel dalam batas butiran beras",
        "Menghitung keliling dengan menghitung jarak antar piksel di sekitar batas bulir beras",
        "Garis terpanjang yang dapat ditarik pada bulir padi, yaitu jarak sumbu utama",
        "Garis terpendek yang dapat ditarik pada butiran beras, yaitu jarak sumbu kecil",
        "Mengukur seberapa bulat elips, yang memiliki momen yang sama dengan butiran beras",
        "Mengembalikan jumlah piksel kulit cembung terkecil dari wilayah yang dibentuk oleh butiran beras",
        "Mengembalikan rasio wilayah yang dibentuk oleh butiran beras terhadap kotak pembatas"
    ]
}
feature_df = pd.DataFrame(feature_description)

# Menampilkan tabel penjelasan fitur
st.write("Tabel Penjelasan Fitur:")
st.dataframe(feature_df)


# Pengecekan Missing Value
st.header("Pengecekan Missing Value")
missing_values = df.isnull().sum()
st.write("Jumlah missing value pada setiap kolom:")
st.write(missing_values)

# Jika ada missing values, tampilkan opsi untuk mengisi nilai yang hilang
if missing_values.any():
    st.warning("Dataset ini memiliki missing values.")
    st.write("Mengisi missing values dengan median kolom...")
    df.fillna(df.median(), inplace=True)
    st.success("Missing values telah diisi dengan median kolom.")
else:
    st.success("Tidak ada missing values dalam dataset.")

# Deteksi dan hapus outlier dengan LOF
st.header("Deteksi dan Penghapusan Outlier")
x = df.drop('Class', axis=1)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.040)
y_pred = lof.fit_predict(x)
anomaly_scores = lof.negative_outlier_factor_
df['LOF_Prediksi'] = y_pred
df['LOF_Skor_Anomali'] = anomaly_scores

# Plot LOF Scores
st.subheader("LOF Anomaly Scores")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(np.arange(len(anomaly_scores)), anomaly_scores, c=(y_pred == -1), cmap='coolwarm', edgecolor='k')
ax.axhline(y=-1.3, color='r', linestyle='--')
ax.set_title("LOF Anomaly Scores")
ax.set_xlabel("Indeks")
ax.set_ylabel("Skor Anomali LOF")
st.pyplot(fig)

# Filter data tanpa outlier
df_cleaned = df[df['LOF_Prediksi'] != -1].reset_index(drop=True)
df_cleaned = df_cleaned.drop(columns=['LOF_Prediksi', 'LOF_Skor_Anomali'])
st.write("Dataset setelah penghapusan outlier:")
st.dataframe(df_cleaned.head())

# Memisahkan fitur dan kelas
x = df_cleaned.drop(['Class'], axis=1)
y = df_cleaned['Class']

# Normalisasi data
st.header("Normalisasi Data")
scaler = MinMaxScaler()
x_normalized = scaler.fit_transform(x)
x = pd.DataFrame(x_normalized, columns=x.columns)
st.write("Hasil Normalisasi Data:")
st.dataframe(x.head())

# Simpan hasil normalisasi ke file CSV
x.to_csv("normalisasi_rice.csv", index=False)
st.success("Hasil normalisasi disimpan sebagai 'normalisasi_rice.csv'.")

# Split data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0, shuffle=True)

# Decision Tree
# Header utama
st.header("Pemodelan menggunakan Decision Tree")

# Decision Tree
with st.expander("Decision Tree"):
    st.subheader("Decision Tree")
    
    # Melatih model
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train, y_train)
    pred_dt = decision_tree.predict(X_test)
    
    # Menampilkan akurasi
    st.write("**Akurasi Decision Tree:**", round(accuracy_score(y_test, pred_dt) * 100, 2), "%")
    
    # Menampilkan Classification Report dalam bentuk tabel
    report = classification_report(y_test, pred_dt, output_dict=True)  # Mengubah laporan ke bentuk dict
    report_df = pd.DataFrame(report).transpose()  # Mengonversi ke DataFrame
    st.dataframe(report_df)  # Menampilkan DataFrame sebagai tabel


# Confusion Matrix Decision Tree
conf_matrix_dt = confusion_matrix(y_test, pred_dt)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Cammeo', 'Osmancik'], 
            yticklabels=['Cammeo', 'Osmancik'], ax=ax)
ax.set_title('Confusion Matrix - Decision Tree')
st.pyplot(fig)

# K-Nearest Neighbors
st.header("Perbandingan Model")

with st.expander("K-Nearest Neighbors (KNN)"):
    st.subheader("K-Nearest Neighbors (KNN)")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    pred_knn = knn.predict(X_test)
    
    # Menampilkan Akurasi
    st.write("**Akurasi KNN:**", round(accuracy_score(y_test, pred_knn) * 100, 2), "%")
    
    # Menampilkan Classification Report dalam bentuk teks
    st.text("Classification Report:")
    st.code(classification_report(y_test, pred_knn), language="text")

# Naive Bayes
with st.expander("Naive Bayes"):
    st.subheader("Naive Bayes")
    gaussian_nb = GaussianNB()
    gaussian_nb.fit(X_train, y_train)
    pred_nb = gaussian_nb.predict(X_test)
    
    # Menampilkan Akurasi
    st.write("**Akurasi Naive Bayes:**", round(accuracy_score(y_test, pred_nb) * 100, 2), "%")
    
    # Menampilkan Classification Report dalam bentuk teks
    st.text("Classification Report:")
    st.code(classification_report(y_test, pred_nb), language="text")


# Perbandingan Akurasi
st.header("Perbandingan Akurasi Model")
model_names = ['Decision Tree', 'KNN', 'Naive Bayes']
accuracies = [
    accuracy_score(y_test, pred_dt),
    accuracy_score(y_test, pred_knn),
    accuracy_score(y_test, pred_nb),
]
comparison_df = pd.DataFrame({'Model': model_names, 'Accuracy': accuracies})
st.dataframe(comparison_df)

# Visualisasi Perbandingan Akurasi
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Accuracy', y='Model', data=comparison_df, hue='Model', palette='viridis', ax=ax, legend=False)

ax.set_title('Perbandingan Akurasi Model')
st.pyplot(fig)
