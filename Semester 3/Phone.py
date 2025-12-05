import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score,precision_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

url = "https://raw.githubusercontent.com/ahnaf0806/dataset/main/phones.csv"
data = pd.read_csv(url)
print(data.head())
data.info()
data.describe()

print(data.isnull().values.any())
data = data.fillna(data.mean(numeric_only=True))
print(data.isnull().sum())

features = ['specs_score', 'rating', 'display frequency (in Hz)',]

x = data[features]
y = data['price']

discret = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
y_binned = discret.fit_transform(y.values.reshape(-1, 1)).flatten()
print('Data Target Setelah diskritisasi: \n')

# Untuk Mengetahui nilai unik hasil diskrit

nilai_unik = set(y_binned)
print('nilai unik dari y_binned:\n', nilai_unik)

# Nilai Detail dari hasil  diskrit

for bin in range(4):
    bin_mask = (y_binned == bin)
    print(f"Bin {bin}: {y[bin_mask]}")


# bagi dataset menjadi data latih dan data uji

x_train, x_test, y_train, y_test = train_test_split(x, y_binned, test_size=0.2, random_state=42)
print("Bentuk Dari X_Train:", x_train.shape)
print("Bentuk Dari Y_Train:", x_test.shape)
print("Bentuk Dari X_Test:",y_train.shape)
print("Bentuk Dari Y_Test:", y_test.shape)

# menghitung baseline performance (prediksi kelas mayoritas)
# Mencari kelas mayoritas
y_train = y_train.astype(int)
# konfersi nilai data agar dapat dihitung
mayoritas = np.bincount(y_train).argmax()
# prediksi kelas mayoritas pada testing
y_predik_baseline = np.full_like(y_test, mayoritas)

# inisialisasi model
model = DecisionTreeClassifier(random_state=42)
# latih Model menggunakan data training
model.fit(x_train, y_train)
# Prediksi data test
y_predik = model.predict(x_test)

# Hitung Metrik Evaluasi
akurasi = accuracy_score(y_test, y_predik)
precision = precision_score(y_test, y_predik, average='weighted')
label = ['Low', 'Medium', 'High', 'Very High']

# Menghitung Akurasi Baseline
baseline_akurasi = accuracy_score(y_test, y_predik_baseline)
print("\nAkurasi Baseline :",baseline_akurasi)
print(f"Decicion Tree Accurasy: {akurasi}")
print(f"Decicion Tree Precicion: {precision}\n")

# Tampilkan Matrix

matrix = confusion_matrix(y_test, y_predik)
print(f'Confusion Matrix:\n{matrix}')

sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label, yticklabels=label)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Visual Tree
plt.figure(figsize=(30, 20))
plot_tree(
    model,
    feature_names=features,
    class_names=label,
    filled=True,
    # rounded=True,
    fontsize=5
)
plt.title("Decision Tree")
plt.show()


# Tampilkan Classification report
print(classification_report(y_test, y_predik))