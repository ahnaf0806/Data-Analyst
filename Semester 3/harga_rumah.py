from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import re

url = "https://raw.githubusercontent.com/ahnaf0806/Data-Analyst/refs/heads/main/Semester%203/Dataset/rumah-yogya.csv"
data = pd.read_csv(url)


# ====== MERUBAH DATA TYPE MENJADI NUMERIK ======
def convert(x):
    if pd.isna(x):
        return np.nan
    number = re.findall(r"\d+", str(x))
    return float("".join(number)) if number else np.nan

data["surface_area"]  = data["surface_area"].apply(convert)
data["building_area"] = data["building_area"].apply(convert)

# ====== CLEAN PRICE ======
def clean_price(x):
    if pd.isna(x):
        return np.nan
    
    s = str(x).lower().replace("rp", "").strip()

    num_match = re.search(r"[\d.,]+", s)
    if not num_match:
        return np.nan

    num_str = num_match.group().replace(".", "").replace(",", ".")
    val = float(num_str)

    if "miliar" in s:
        val *= 1_000_000_000
    elif "juta" in s:
        val *= 1_000_000
    else:
        val *= 1_000_000 

    return val

data["price"] = data["price"].apply(clean_price)
data["price"] = pd.to_numeric(data["price"], errors="coerce")

# ====== DROP PRICE NaN DULU (WAJIB SEBELUM ASTYPE) ======
data = data.dropna(subset=["price"])

# ====== BARU BOLEH JADI INT ======
data["price"] = data["price"].round(0).astype("int64")

print(data["price"].head())

num_col = ["bed", "bath", "carport", "surface_area", "building_area", "price"]
for col in num_col:
    data[col] = pd.to_numeric(data[col], errors="coerce")


# =========================================
# MEMBERSIHKAN DATA YANG HILANG
# =========================================
data = data.dropna(subset=['price'])

prediktor = ["bed", "bath", "carport", "surface_area", "building_area"]

# CEK RATA RATA DATA PREDIKTOR
mean_prediktor = data[prediktor].mean()
print("\n RATA RATA DATA PREDIKTOR")
print(mean_prediktor)

# ISI DATA YANG HILANG DENGAN RATA RATA
data[prediktor] = data[prediktor].fillna(mean_prediktor)

print("\n INFORMASI DATA SETELAH MENGISI DATA YANG HILANG")
print(data.info())

print(f"\n{data.isnull().sum()}")


# X dan y (pastikan ini sudah hasil cleaning + imputasi)
X = data[["bed","bath","carport","surface_area","building_area"]]
y = data["price"]

# kalau mau pakai log target:
# y = np.log1p(data["price"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=8),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=200)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    results.append([name, mse, rmse, r2, mape])

results_df = pd.DataFrame(results, columns=["Model","MSE","RMSE","R2","MAPE"])
results_df = results_df.sort_values(by="RMSE")  # makin kecil makin bagus
print(results_df)
