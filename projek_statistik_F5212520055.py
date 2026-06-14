import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# Set gaya visualisasi agar terlihat lebih rapi dan profesional
sns.set_theme(style="whitegrid")

# ==========================================
# 1. LOAD & PREPROCESSING DATA
# ==========================================
file_name = "993a40a1-f148-4630-9a71-3743599f6784 (1).csv"
df = pd.read_csv(file_name)

luas_panen_row = df[df['Uraian'] == 'Luas Panen']
produktivitas_row = df[df['Uraian'] == 'Produktivitas']
produksi_row = df[df['Uraian'] == 'Jumlah Produksi']

years = ['2021', '2022', '2023', '2024', '2025']

def clean_val(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip().replace(',', '.')
    return float(val_str)

structured_data = []
for yr in years:
    lp = clean_val(luas_panen_row[yr].values[0])
    prod = clean_val(produktivitas_row[yr].values[0])
    prod_amount = clean_val(produksi_row[yr].values[0])
    structured_data.append({
        'Tahun': yr,
        'Luas_Panen': lp,
        'Produktivitas': prod,
        'Jumlah_Produksi': prod_amount
    })

df_reg = pd.DataFrame(structured_data)

# ==========================================
# 2. SEPARASI VARIABEL & PEMODELAN
# ==========================================
X = df_reg[['Luas_Panen', 'Produktivitas']]
Y = df_reg['Jumlah_Produksi']

reg_model = LinearRegression()
reg_model.fit(X, Y)

# Menghitung nilai prediksi untuk membuat diagram perbandingan
df_reg['Prediksi_Produksi'] = reg_model.predict(X)

# ==========================================
# 3. KODE UNTUK PEMBUATAN DIAGRAM / GRAFIK
# ==========================================

# --- DIAGRAM 1: LINE PLOT (Aktual vs Prediksi) ---
plt.figure(figsize=(8, 5))
plt.scatter(df_reg['Tahun'], df_reg['Jumlah_Produksi'], color='darkblue', label='Data Aktual (Asli)', s=120, zorder=3)
plt.plot(df_reg['Tahun'], df_reg['Prediksi_Produksi'], color='crimson', marker='o', linestyle='--', linewidth=2, label='Garis Estimasi Regresi')

plt.title('Diagram Estimasi Model: Jumlah Produksi Jagung (Aktual vs Prediksi)', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Tahun Analisis', fontsize=11)
plt.ylabel('Jumlah Produksi (Ton)', fontsize=11)
plt.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig('diagram_aktual_vs_prediksi.png', dpi=300) # Menyimpan grafik ke komputer
plt.show() # Menampilkan diagram di layar/notebook

# --- DIAGRAM 2: HEATMAP (Matriks Korelasi) ---
plt.figure(figsize=(6, 4))
correlation_matrix = df_reg[['Luas_Panen', 'Produktivitas', 'Jumlah_Produksi']].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".4f", linewidths=0.5, cbar=True)
plt.title('Diagram Matriks Korelasi Antar Variabel', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('diagram_matriks_korelasi.png', dpi=300)
plt.show()

# --- DIAGRAM 3: BAR PLOT (Tingkat Pengaruh Koefisien Regresi) ---
plt.figure(figsize=(7, 4))
variabel_bebas = ['Luas Panen (X1)', 'Produktivitas (X2)']
koefisien_nilai = reg_model.coef_

colors = ['teal', 'coral']
plt.barh(variabel_bebas, koefisien_nilai, color=colors, height=0.5, edgecolor='black')
plt.axvline(0, color='black', linestyle='-', linewidth=0.8) # Garis tengah nol

plt.title('Diagram Batang Koefisien Regresi (Tingkat Kontribusi Variabel)', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Nilai Koefisien Regresi (Bobot Pengaruh)', fontsize=11)
plt.ylabel('Variabel Independen', fontsize=11)
for index, value in enumerate(koefisien_nilai):
    plt.text(value, index, f'  {value:.4f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('diagram_bobot_koefisien.png', dpi=300)
plt.show()

print("\n[INFO] Semua diagram berhasil dibuat, ditampilkan, dan disimpan dalam format gambar PNG.")