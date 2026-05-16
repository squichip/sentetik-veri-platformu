"""
============================================================
 Akıllı Veri Üretimi - Faz 1: Veri Keşfi & Anomali Üretimi
============================================================
 
 Bu script:
 1. İtalya FCD verisini yükler ve analiz eder
 2. Normal yörüngeleri görselleştirir
 3. Anomali injection uygular (spike, drift, dropout, freeze)
 4. Normal vs anomali karşılaştırması yapar
 5. Sonuçları kaydeder
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI olmadan çalıştır
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# AYARLAR
# ============================================================
DATA_DIR = "data/fcd-italy"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Güzel grafik stili
plt.style.use('dark_background')
COLORS = {
    'normal': '#10b981',
    'spike': '#ef4444',
    'drift': '#f59e0b',
    'dropout': '#8b5cf6',
    'freeze': '#3b82f6',
    'bg': '#0f172a',
    'grid': '#1e293b',
    'text': '#f1f5f9'
}

# ============================================================
# 1. VERİ YÜKLEME
# ============================================================
print("=" * 60)
print("🚗 AKILLI VERİ ÜRETİMİ - FCD VERİ ANALİZİ")
print("=" * 60)

cities = {
    'Milano': 'milan_fcd_prep.csv',
    'Napoli': 'naples_fcd_prep.csv',
    'Roma': 'rome_fcd_prep.csv',
    'Torino': 'turin_fcd_prep.csv'
}

city_data = {}
total_rows = 0

print("\n📂 Veri Yükleniyor...")
for city, filename in cities.items():
    filepath = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(filepath)
    city_data[city] = df
    total_rows += len(df)
    print(f"  ✅ {city}: {len(df):,} yörünge, {df.shape[1]} sütun")

print(f"\n📊 Toplam: {total_rows:,} yörünge")

# ============================================================
# 2. VERİ YAPISI ANALİZİ
# ============================================================
print("\n" + "=" * 60)
print("📐 VERİ YAPISI ANALİZİ")
print("=" * 60)

# Milano verisini örnek olarak incele
sample_df = city_data['Milano']
columns = sample_df.columns.tolist()

print(f"\nSütunlar ({len(columns)} adet):")
print(f"  İlk sütun: {columns[0]} (index)")
print(f"  Veri sütunları: {columns[1]} ... {columns[-1]}")
print(f"\nHer yörünge = 20 zaman noktası × 2 koordinat (x, y)")
print(f"Zaman aralığı: t-8'den t+11'e (toplam 20 adım, 60 sn aralıklı)")

# x ve y sütunlarını ayır
x_cols = [c for c in columns if c.startswith('x(')]
y_cols = [c for c in columns if c.startswith('y(')]
print(f"\nX sütunları: {len(x_cols)} adet")
print(f"Y sütunları: {len(y_cols)} adet")

# İstatistikler
print("\n📊 Koordinat İstatistikleri (Milano):")
all_x = sample_df[x_cols].values.flatten()
all_y = sample_df[y_cols].values.flatten()
print(f"  X aralığı: [{all_x.min():.4f}, {all_x.max():.4f}]")
print(f"  Y aralığı: [{all_y.min():.4f}, {all_y.max():.4f}]")
print(f"  X ortalama: {all_x.mean():.4f}, std: {all_x.std():.4f}")
print(f"  Y ortalama: {all_y.mean():.4f}, std: {all_y.std():.4f}")

# ============================================================
# 3. NORMAL YÖRÜNGE GÖRSELLEŞTİRME
# ============================================================
print("\n" + "=" * 60)
print("🗺️ NORMAL YÖRÜNGE GÖRSELLEŞTİRME")
print("=" * 60)

fig = plt.figure(figsize=(20, 16), facecolor=COLORS['bg'])
fig.suptitle('İtalya FCD Verisi - Normal Araç Yörüngeleri', 
             fontsize=20, fontweight='bold', color=COLORS['text'], y=0.98)

for idx, (city, df) in enumerate(city_data.items()):
    ax = fig.add_subplot(2, 2, idx + 1, facecolor=COLORS['bg'])
    
    # Rastgele 200 yörünge seç
    sample_indices = np.random.choice(len(df), min(200, len(df)), replace=False)
    
    for i in sample_indices:
        row = df.iloc[i]
        x_vals = [row[c] for c in x_cols]
        y_vals = [row[c] for c in y_cols]
        ax.plot(x_vals, y_vals, color=COLORS['normal'], alpha=0.15, linewidth=0.8)
    
    ax.set_title(f'{city} ({len(df):,} yörünge)', fontsize=14, 
                 color=COLORS['text'], fontweight='bold')
    ax.set_xlabel('X (normalize)', fontsize=10, color=COLORS['text'])
    ax.set_ylabel('Y (normalize)', fontsize=10, color=COLORS['text'])
    ax.tick_params(colors=COLORS['text'], labelsize=8)
    ax.grid(True, alpha=0.1, color=COLORS['grid'])
    ax.set_facecolor(COLORS['bg'])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, '01_normal_trajectories.png'), 
            dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("  ✅ Kaydedildi: outputs/01_normal_trajectories.png")

# ============================================================
# 4. TEK YÖRÜNGE DETAYLI ANALİZ
# ============================================================
print("\n📈 Tek Yörünge Zaman Serisi Analizi...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=COLORS['bg'])
fig.suptitle('Tek Yörünge - X ve Y Zaman Serileri', 
             fontsize=16, fontweight='bold', color=COLORS['text'])

# 4 farklı yörünge tipi
sample_row = sample_df.iloc[100]
x_vals = np.array([sample_row[c] for c in x_cols])
y_vals = np.array([sample_row[c] for c in y_cols])
time_steps = list(range(len(x_cols)))

# X zaman serisi
ax = axes[0, 0]
ax.plot(time_steps, x_vals, color=COLORS['normal'], linewidth=2, marker='o', markersize=4)
ax.set_title('X Koordinatı - Zaman Serisi', color=COLORS['text'], fontweight='bold')
ax.set_xlabel('Zaman Adımı (60s aralıklı)', color=COLORS['text'])
ax.set_ylabel('X (normalize)', color=COLORS['text'])
ax.axvline(x=8, color='#ffffff40', linestyle='--', label='t=0 (şimdi)')
ax.legend(facecolor=COLORS['bg'], edgecolor=COLORS['grid'])
ax.grid(True, alpha=0.1)
ax.set_facecolor(COLORS['bg'])
ax.tick_params(colors=COLORS['text'])

# Y zaman serisi
ax = axes[0, 1]
ax.plot(time_steps, y_vals, color=COLORS['drift'], linewidth=2, marker='o', markersize=4)
ax.set_title('Y Koordinatı - Zaman Serisi', color=COLORS['text'], fontweight='bold')
ax.set_xlabel('Zaman Adımı (60s aralıklı)', color=COLORS['text'])
ax.set_ylabel('Y (normalize)', color=COLORS['text'])
ax.axvline(x=8, color='#ffffff40', linestyle='--', label='t=0 (şimdi)')
ax.legend(facecolor=COLORS['bg'], edgecolor=COLORS['grid'])
ax.grid(True, alpha=0.1)
ax.set_facecolor(COLORS['bg'])
ax.tick_params(colors=COLORS['text'])

# 2D yörünge
ax = axes[1, 0]
ax.plot(x_vals, y_vals, color=COLORS['normal'], linewidth=2, marker='o', markersize=5)
ax.plot(x_vals[0], y_vals[0], 'o', color='#10b981', markersize=12, label='Başlangıç', zorder=5)
ax.plot(x_vals[-1], y_vals[-1], 's', color='#ef4444', markersize=12, label='Bitiş', zorder=5)
ax.plot(x_vals[8], y_vals[8], '^', color='#fbbf24', markersize=12, label='t=0 (şimdi)', zorder=5)
ax.set_title('2D Yörünge', color=COLORS['text'], fontweight='bold')
ax.set_xlabel('X', color=COLORS['text'])
ax.set_ylabel('Y', color=COLORS['text'])
ax.legend(facecolor=COLORS['bg'], edgecolor=COLORS['grid'])
ax.grid(True, alpha=0.1)
ax.set_facecolor(COLORS['bg'])
ax.tick_params(colors=COLORS['text'])

# Hız profili (ardışık noktalar arası mesafe)
dx = np.diff(x_vals)
dy = np.diff(y_vals)
speed = np.sqrt(dx**2 + dy**2)

ax = axes[1, 1]
ax.plot(range(len(speed)), speed, color=COLORS['spike'], linewidth=2, marker='o', markersize=4)
ax.fill_between(range(len(speed)), speed, alpha=0.2, color=COLORS['spike'])
ax.set_title('Hız Profili (ardışık mesafe)', color=COLORS['text'], fontweight='bold')
ax.set_xlabel('Zaman Adımı', color=COLORS['text'])
ax.set_ylabel('Δ mesafe / adım', color=COLORS['text'])
ax.grid(True, alpha=0.1)
ax.set_facecolor(COLORS['bg'])
ax.tick_params(colors=COLORS['text'])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_single_trajectory_analysis.png'), 
            dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("  ✅ Kaydedildi: outputs/02_single_trajectory_analysis.png")

# ============================================================
# 5. ANOMALİ INJECTION
# ============================================================
print("\n" + "=" * 60)
print("⚡ ANOMALİ INJECTION (Sentetik Anomali Üretimi)")
print("=" * 60)

def inject_spike(x, y, num_spikes=2, magnitude=0.5):
    """Ani sıçrama anomalisi - sensör okuma hatası"""
    x_anom = x.copy()
    y_anom = y.copy()
    spike_indices = np.random.choice(len(x), num_spikes, replace=False)
    for idx in spike_indices:
        x_anom[idx] += np.random.choice([-1, 1]) * magnitude * np.random.uniform(0.5, 1.5)
        y_anom[idx] += np.random.choice([-1, 1]) * magnitude * np.random.uniform(0.5, 1.5)
    return x_anom, y_anom, spike_indices

def inject_drift(x, y, start_idx=10, drift_rate=0.02):
    """Yavaş kayma anomalisi - GPS/IMU kalibrasyon bozulması"""
    x_anom = x.copy()
    y_anom = y.copy()
    for i in range(start_idx, len(x)):
        drift_amount = (i - start_idx) * drift_rate
        angle = np.random.uniform(0, 2 * np.pi)
        x_anom[i] += drift_amount * np.cos(angle)
        y_anom[i] += drift_amount * np.sin(angle)
    return x_anom, y_anom

def inject_dropout(x, y, dropout_ratio=0.25):
    """Veri kaybı anomalisi - sensör sinyal kopması"""
    x_anom = x.copy()
    y_anom = y.copy()
    n_dropout = int(len(x) * dropout_ratio)
    dropout_start = np.random.randint(3, len(x) - n_dropout - 2)
    dropout_indices = list(range(dropout_start, dropout_start + n_dropout))
    x_anom[dropout_indices] = np.nan
    y_anom[dropout_indices] = np.nan
    return x_anom, y_anom, dropout_indices

def inject_freeze(x, y, freeze_start=8, freeze_duration=7):
    """Donma anomalisi - sensör kilitlenmesi"""
    x_anom = x.copy()
    y_anom = y.copy()
    frozen_x = x[freeze_start]
    frozen_y = y[freeze_start]
    for i in range(freeze_start, min(freeze_start + freeze_duration, len(x))):
        x_anom[i] = frozen_x
        y_anom[i] = frozen_y
    return x_anom, y_anom

def inject_noise_amplification(x, y, start_idx=5, noise_factor=0.08):
    """Gürültü artışı anomalisi - elektromanyetik parazit"""
    x_anom = x.copy()
    y_anom = y.copy()
    for i in range(start_idx, len(x)):
        noise_level = noise_factor * (1 + (i - start_idx) * 0.3)
        x_anom[i] += np.random.normal(0, noise_level)
        y_anom[i] += np.random.normal(0, noise_level)
    return x_anom, y_anom

# Örnek yörünge seç
np.random.seed(42)
sample_idx = 500
sample_row = sample_df.iloc[sample_idx]
x_normal = np.array([sample_row[c] for c in x_cols])
y_normal = np.array([sample_row[c] for c in y_cols])

# Anomalileri üret
anomalies = {}
x_spike, y_spike, spike_pts = inject_spike(x_normal, y_normal, num_spikes=3, magnitude=0.4)
anomalies['Spike (Ani Sıçrama)'] = (x_spike, y_spike, COLORS['spike'])

x_drift, y_drift = inject_drift(x_normal, y_normal, start_idx=8, drift_rate=0.025)
anomalies['Drift (Kayma)'] = (x_drift, y_drift, COLORS['drift'])

x_drop, y_drop, drop_pts = inject_dropout(x_normal, y_normal, dropout_ratio=0.3)
anomalies['Dropout (Veri Kaybı)'] = (x_drop, y_drop, COLORS['dropout'])

x_freeze, y_freeze = inject_freeze(x_normal, y_normal, freeze_start=7, freeze_duration=8)
anomalies['Freeze (Donma)'] = (x_freeze, y_freeze, COLORS['freeze'])

x_noise, y_noise = inject_noise_amplification(x_normal, y_normal, start_idx=5, noise_factor=0.06)
anomalies['Noise (Gürültü Artışı)'] = (x_noise, y_noise, '#06b6d4')

# ============================================================
# 6. ANOMALİ GÖRSELLEŞTİRME - ANA ÇIKTI
# ============================================================
print("\n🎨 Anomali Görselleştirmesi Oluşturuluyor...")

fig = plt.figure(figsize=(22, 18), facecolor=COLORS['bg'])
fig.suptitle('Sensör Füzyon Anomali Injection - Normal vs Anomali Yörüngeler', 
             fontsize=20, fontweight='bold', color=COLORS['text'], y=0.99)

# Alt başlık
fig.text(0.5, 0.965, 'Otonom Araç GPS/IMU Sensör Verisi Üzerinde Sentetik Anomali Üretimi',
         fontsize=12, color='#94a3b8', ha='center')

gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

# --- 2D Yörünge Grafikleri (üst satır) ---

# Normal
ax = fig.add_subplot(gs[0, 0], facecolor=COLORS['bg'])
ax.plot(x_normal, y_normal, color=COLORS['normal'], linewidth=2.5, marker='o', markersize=4, zorder=2)
ax.plot(x_normal[0], y_normal[0], 'o', color='#10b981', markersize=10, zorder=5)
ax.plot(x_normal[-1], y_normal[-1], 's', color='#ef4444', markersize=10, zorder=5)
ax.set_title('✅ Normal Yörünge', color=COLORS['normal'], fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.1)
ax.tick_params(colors=COLORS['text'], labelsize=7)

# Her anomali türü
positions = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]
for (row, col), (name, (x_a, y_a, color)) in zip(positions, anomalies.items()):
    ax = fig.add_subplot(gs[row, col], facecolor=COLORS['bg'])
    # Normal (gri arka plan)
    ax.plot(x_normal, y_normal, color='#ffffff20', linewidth=1.5, linestyle='--', zorder=1)
    # Anomali
    ax.plot(x_a, y_a, color=color, linewidth=2.5, marker='o', markersize=4, zorder=2)
    ax.plot(x_a[0], y_a[0], 'o', color='#10b981', markersize=10, zorder=5)
    valid_last = ~np.isnan(x_a[-1]) if not isinstance(x_a[-1], float) or not np.isnan(x_a[-1]) else False
    if not np.isnan(x_a[-1]):
        ax.plot(x_a[-1], y_a[-1], 's', color='#ef4444', markersize=10, zorder=5)
    ax.set_title(f'⚠️ {name}', color=color, fontweight='bold', fontsize=11)
    ax.grid(True, alpha=0.1)
    ax.tick_params(colors=COLORS['text'], labelsize=7)

# --- X Zaman Serisi Karşılaştırması (alt bölüm) ---
ax_x = fig.add_subplot(gs[1, 2:], facecolor=COLORS['bg'])
time = list(range(len(x_normal)))
ax_x.plot(time, x_normal, color=COLORS['normal'], linewidth=2.5, label='Normal', zorder=5)
ax_x.plot(time, x_spike, color=COLORS['spike'], linewidth=1.5, alpha=0.8, label='Spike', linestyle='-')
ax_x.plot(time, x_drift, color=COLORS['drift'], linewidth=1.5, alpha=0.8, label='Drift', linestyle='-')
ax_x.plot(time, x_freeze, color=COLORS['freeze'], linewidth=1.5, alpha=0.8, label='Freeze', linestyle='-')
ax_x.plot(time, x_noise, color='#06b6d4', linewidth=1.5, alpha=0.8, label='Noise', linestyle='-')
ax_x.set_title('X Koordinatı - Tüm Anomali Türleri Karşılaştırması', 
               color=COLORS['text'], fontweight='bold', fontsize=12)
ax_x.set_xlabel('Zaman Adımı', color=COLORS['text'])
ax_x.set_ylabel('X (normalize)', color=COLORS['text'])
ax_x.legend(facecolor=COLORS['bg'], edgecolor=COLORS['grid'], fontsize=9, ncol=5, loc='upper right')
ax_x.grid(True, alpha=0.1)
ax_x.tick_params(colors=COLORS['text'])

# --- Hız Profili Karşılaştırması (en alt) ---
ax_speed = fig.add_subplot(gs[2, :2], facecolor=COLORS['bg'])

# Normal hız
dx_n = np.diff(x_normal); dy_n = np.diff(y_normal)
speed_n = np.sqrt(dx_n**2 + dy_n**2)
ax_speed.plot(range(len(speed_n)), speed_n, color=COLORS['normal'], linewidth=2, label='Normal')

# Spike hız
dx_s = np.diff(x_spike); dy_s = np.diff(y_spike)  
speed_s = np.sqrt(dx_s**2 + dy_s**2)
ax_speed.plot(range(len(speed_s)), speed_s, color=COLORS['spike'], linewidth=1.5, alpha=0.8, label='Spike')

# Drift hız
dx_d = np.diff(x_drift); dy_d = np.diff(y_drift)
speed_d = np.sqrt(dx_d**2 + dy_d**2)
ax_speed.plot(range(len(speed_d)), speed_d, color=COLORS['drift'], linewidth=1.5, alpha=0.8, label='Drift')

ax_speed.set_title('Hız Profili - Anomali Tespiti İçin Önemli Özellik', 
                   color=COLORS['text'], fontweight='bold', fontsize=12)
ax_speed.set_xlabel('Zaman Adımı', color=COLORS['text'])
ax_speed.set_ylabel('Hız (Δmesafe/adım)', color=COLORS['text'])
ax_speed.legend(facecolor=COLORS['bg'], edgecolor=COLORS['grid'])
ax_speed.grid(True, alpha=0.1)
ax_speed.tick_params(colors=COLORS['text'])

# --- Anomali İstatistikleri (sağ alt) ---
ax_stats = fig.add_subplot(gs[2, 2:], facecolor=COLORS['bg'])
ax_stats.axis('off')

stats_text = """
╔══════════════════════════════════════════════════╗
║          ANOMALİ INJECTION İSTATİSTİKLERİ        ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  📊 Kaynak Veri:                                 ║
║     • Milano FCD: 48,888 yörünge                 ║
║     • Toplam 4 şehir: 498,699 yörünge            ║
║     • Her yörünge: 20 nokta (x,y)                ║
║                                                  ║
║  ⚡ Üretilen Anomali Türleri:                    ║
║     1. Spike  - Ani sensör sıçraması             ║
║     2. Drift  - GPS/IMU kalibrasyon kayması      ║
║     3. Dropout - Sinyal kaybı                    ║
║     4. Freeze - Sensör kilitlenmesi              ║
║     5. Noise  - Elektromanyetik parazit           ║
║                                                  ║
║  🎯 Sonraki Adım:                                ║
║     GAN/LLM ile bu anomalilerin                  ║
║     binlerce varyasyonunu üretmek                ║
║                                                  ║
╚══════════════════════════════════════════════════╝
"""
ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
             fontsize=10, color=COLORS['text'], fontfamily='monospace',
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a2235', edgecolor='#2d3a4f'))

plt.savefig(os.path.join(OUTPUT_DIR, '03_anomaly_injection_comparison.png'), 
            dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("  ✅ Kaydedildi: outputs/03_anomaly_injection_comparison.png")

# ============================================================
# 7. TOPLU ANOMALİ ÜRETİMİ (Demo)
# ============================================================
print("\n" + "=" * 60)
print("🏭 TOPLU ANOMALİ ÜRETİMİ (Demo)")
print("=" * 60)

anomaly_types = ['spike', 'drift', 'dropout', 'freeze', 'noise']
n_samples = 100  # Demo için 100 normal yörüngeden anomali üret

generated_data = {
    'normal': [],
    'spike': [],
    'drift': [],
    'dropout': [],
    'freeze': [],
    'noise': []
}

np.random.seed(123)
sample_indices = np.random.choice(len(sample_df), n_samples, replace=False)

for idx in sample_indices:
    row = sample_df.iloc[idx]
    x = np.array([row[c] for c in x_cols])
    y = np.array([row[c] for c in y_cols])
    
    # Normal
    generated_data['normal'].append(np.concatenate([x, y]))
    
    # Anomaliler
    x_s, y_s, _ = inject_spike(x, y, num_spikes=np.random.randint(1, 4), 
                                magnitude=np.random.uniform(0.2, 0.6))
    generated_data['spike'].append(np.concatenate([x_s, y_s]))
    
    x_d, y_d = inject_drift(x, y, start_idx=np.random.randint(5, 12), 
                             drift_rate=np.random.uniform(0.01, 0.04))
    generated_data['drift'].append(np.concatenate([x_d, y_d]))
    
    x_dr, y_dr, _ = inject_dropout(x, y, dropout_ratio=np.random.uniform(0.15, 0.4))
    generated_data['dropout'].append(np.concatenate([x_dr, y_dr]))
    
    x_f, y_f = inject_freeze(x, y, freeze_start=np.random.randint(4, 12),
                              freeze_duration=np.random.randint(4, 10))
    generated_data['freeze'].append(np.concatenate([x_f, y_f]))
    
    x_n, y_n = inject_noise_amplification(x, y, start_idx=np.random.randint(3, 8),
                                           noise_factor=np.random.uniform(0.03, 0.10))
    generated_data['noise'].append(np.concatenate([x_n, y_n]))

# Sonuçları DataFrame'e çevir ve kaydet
for label, data_list in generated_data.items():
    df_out = pd.DataFrame(data_list)
    # NaN içerenleri -999 ile doldur (dropout)
    df_out = df_out.fillna(-999)
    df_out.insert(0, 'label', label)
    filepath = os.path.join(OUTPUT_DIR, f'generated_{label}.csv')
    df_out.to_csv(filepath, index=False)
    print(f"  ✅ {label}: {len(data_list)} örnek → {filepath}")

# Tümünü birleştir
all_data = []
for label, data_list in generated_data.items():
    for row in data_list:
        all_data.append([label] + list(row))

df_all = pd.DataFrame(all_data)
df_all.columns = ['label'] + [f'feat_{i}' for i in range(len(all_data[0]) - 1)]
df_all.to_csv(os.path.join(OUTPUT_DIR, 'combined_dataset.csv'), index=False)
print(f"\n  📦 Birleşik veri seti: {len(df_all)} örnek → outputs/combined_dataset.csv")

# Sınıf dağılımı
print("\n📊 Sınıf Dağılımı:")
for label in ['normal', 'spike', 'drift', 'dropout', 'freeze', 'noise']:
    count = len(generated_data[label])
    print(f"  {label:10s}: {count:5d} örnek ({count/len(df_all)*100:.1f}%)")

# ============================================================
# 8. SINIF DAĞILIMI GÖRSELLEŞTİRME
# ============================================================
print("\n🎨 Sınıf Dağılımı Grafiği...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=COLORS['bg'])

# Bar chart
labels_list = ['Normal', 'Spike', 'Drift', 'Dropout', 'Freeze', 'Noise']
counts = [len(generated_data[l.lower()]) for l in labels_list]
colors_list = [COLORS['normal'], COLORS['spike'], COLORS['drift'], 
               COLORS['dropout'], COLORS['freeze'], '#06b6d4']

ax = axes[0]
bars = ax.bar(labels_list, counts, color=colors_list, edgecolor='white', linewidth=0.5)
ax.set_title('Üretilen Veri - Sınıf Dağılımı', color=COLORS['text'], fontweight='bold', fontsize=14)
ax.set_ylabel('Örnek Sayısı', color=COLORS['text'])
ax.set_facecolor(COLORS['bg'])
ax.tick_params(colors=COLORS['text'])
ax.grid(True, alpha=0.1, axis='y')
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            str(count), ha='center', va='bottom', color=COLORS['text'], fontweight='bold')

# Pie chart - anomali vs normal
ax = axes[1]
normal_count = counts[0]
anomaly_count = sum(counts[1:])
ax.pie([normal_count, anomaly_count], 
       labels=['Normal', 'Anomali'],
       colors=[COLORS['normal'], COLORS['spike']],
       autopct='%1.1f%%',
       startangle=90,
       textprops={'color': COLORS['text'], 'fontweight': 'bold'},
       wedgeprops={'edgecolor': COLORS['bg'], 'linewidth': 2})
ax.set_title('Normal vs Anomali Oranı', color=COLORS['text'], fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_class_distribution.png'), 
            dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("  ✅ Kaydedildi: outputs/04_class_distribution.png")

# ============================================================
# ÖZET
# ============================================================
print("\n" + "=" * 60)
print("🎉 FAZ 1 TAMAMLANDI!")
print("=" * 60)
print(f"""
📁 Oluşturulan Dosyalar (outputs/ klasöründe):
  • 01_normal_trajectories.png     - 4 şehir normal yörünge haritası
  • 02_single_trajectory_analysis.png - Tek yörünge detaylı analizi
  • 03_anomaly_injection_comparison.png - ANOMALİ İNJECTION ANA ÇIKTI
  • 04_class_distribution.png      - Sınıf dağılımı grafikleri
  • generated_normal.csv           - Normal veri örnekleri
  • generated_spike.csv            - Spike anomali örnekleri
  • generated_drift.csv            - Drift anomali örnekleri
  • generated_dropout.csv          - Dropout anomali örnekleri
  • generated_freeze.csv           - Freeze anomali örnekleri
  • generated_noise.csv            - Noise anomali örnekleri
  • combined_dataset.csv           - Birleşik etiketli veri seti

🎯 Sonraki Adımlar:
  1. Bu seed verileri üzerine GAN ile çoğaltma
  2. Waymo verisinden 3D sensör anomalileri ekleme
  3. Anomali detection modeli eğitimi
  4. Fidelity + Utility doğrulaması
""")
