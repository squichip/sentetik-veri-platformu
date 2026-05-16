"""
============================================================
 Akıllı Veri Üretimi - Faz 4: Feature Engineering + 
 İyileştirilmiş Anomali Tespit Modeli
============================================================
 
 Sorun: Ham x,y koordinatlarıyla model anomali ayırt edemiyor
 Çözüm: Yörüngeden ANLAMLI özellikler çıkar (hız, ivme, 
         sarsıntı, eğrilik, istatistiksel anomali sinyalleri)
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, recall_score, precision_score, 
                              classification_report, confusion_matrix)
import os, warnings, json
warnings.filterwarnings('ignore')

OUTPUT_DIR = "outputs"
np.random.seed(42)

plt.style.use('dark_background')
COLORS = {
    'seed': '#10b981', 'gan': '#f472b6', 'improved': '#fbbf24',
    'bg': '#0f172a', 'text': '#f1f5f9', 'grid': '#1e293b',
    'normal': '#10b981', 'spike': '#ef4444', 'drift': '#f59e0b',
    'dropout': '#8b5cf6', 'freeze': '#3b82f6', 'noise': '#06b6d4'
}

LABEL_MAP = {'normal': 0, 'spike': 1, 'drift': 2, 'dropout': 3, 'freeze': 4, 'noise': 5}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}

print("=" * 60)
print("⚡ FAZ 4: FEATURE ENGINEERING + MODEL İYİLEŞTİRME")
print("=" * 60)

# ============================================================
# 1. FEATURE ENGINEERING FONKSİYONLARI
# ============================================================
print("\n🔧 Feature Engineering Fonksiyonları Tanımlanıyor...")

def extract_features(trajectory):
    """
    40 boyutlu ham x,y verisinden 35+ anlamlı özellik çıkar.
    
    trajectory: [x0,x1,...,x19, y0,y1,...,y19] = 40 değer
    """
    x = trajectory[:20]
    y = trajectory[20:]
    
    features = {}
    
    # --- 1. HIZ PROFİLİ ---
    dx = np.diff(x)
    dy = np.diff(y)
    speed = np.sqrt(dx**2 + dy**2)  # 19 değer
    
    features['speed_mean'] = np.mean(speed)
    features['speed_std'] = np.std(speed)
    features['speed_max'] = np.max(speed)
    features['speed_min'] = np.min(speed)
    features['speed_range'] = np.max(speed) - np.min(speed)
    features['speed_cv'] = np.std(speed) / (np.mean(speed) + 1e-8)  # Varyasyon katsayısı
    
    # --- 2. İVME PROFİLİ ---
    acceleration = np.diff(speed)  # 18 değer
    features['accel_mean'] = np.mean(acceleration)
    features['accel_std'] = np.std(acceleration)
    features['accel_max'] = np.max(np.abs(acceleration))
    features['accel_range'] = np.max(acceleration) - np.min(acceleration)
    
    # --- 3. SARSINTI (JERK) ---
    jerk = np.diff(acceleration)  # 17 değer
    features['jerk_mean'] = np.mean(np.abs(jerk))
    features['jerk_max'] = np.max(np.abs(jerk))
    features['jerk_std'] = np.std(jerk)
    
    # --- 4. SPIKE TESPİT ÖZELLİKLERİ ---
    # Ani sıçrama = hız profili Q3+1.5*IQR üstünde
    q1, q3 = np.percentile(speed, [25, 75])
    iqr = q3 - q1
    n_outliers = np.sum(speed > q3 + 1.5 * iqr)
    features['n_speed_outliers'] = n_outliers
    features['max_speed_zscore'] = (np.max(speed) - np.mean(speed)) / (np.std(speed) + 1e-8)
    
    # Ardışık noktalar arası maximum sıçrama
    features['max_single_jump'] = np.max(speed)
    features['jump_ratio'] = np.max(speed) / (np.median(speed) + 1e-8)
    
    # --- 5. DRIFT TESPİT ÖZELLİKLERİ ---
    # Başlangıç noktasından uzaklık trendi
    dist_from_start = np.sqrt((x - x[0])**2 + (y - y[0])**2)
    # Lineer trend katsayısı (drift = sürekli artan mesafe)
    t = np.arange(len(x))
    if np.std(t) > 0:
        trend_coef = np.corrcoef(t, dist_from_start)[0, 1]
    else:
        trend_coef = 0
    features['drift_trend'] = trend_coef if not np.isnan(trend_coef) else 0
    
    # İkinci yarı vs ilk yarı mesafe
    first_half_dist = np.mean(dist_from_start[:10])
    second_half_dist = np.mean(dist_from_start[10:])
    features['half_dist_ratio'] = second_half_dist / (first_half_dist + 1e-8)
    
    # --- 6. FREEZE TESPİT ÖZELLİKLERİ ---
    # Ardışık aynı/çok yakın değerler
    near_zero_speed = np.sum(speed < 1e-6)
    features['n_zero_speed'] = near_zero_speed
    features['zero_speed_ratio'] = near_zero_speed / len(speed)
    
    # En uzun durağan (freeze) periyodu
    max_freeze = 0
    current_freeze = 0
    for s in speed:
        if s < 1e-6:
            current_freeze += 1
            max_freeze = max(max_freeze, current_freeze)
        else:
            current_freeze = 0
    features['max_freeze_length'] = max_freeze
    
    # --- 7. DROPOUT TESPİT ÖZELLİKLERİ ---
    # Tekrarlanan değer bölgeleri (dropout sırasında son değer tekrarlanır)
    x_changes = np.abs(np.diff(x))
    y_changes = np.abs(np.diff(y))
    n_static = np.sum((x_changes < 1e-8) & (y_changes < 1e-8))
    features['n_static_points'] = n_static
    
    # Statik sonrası ani sıçrama
    if n_static > 0 and n_static < len(speed) - 1:
        for i in range(len(speed) - 1):
            if speed[i] < 1e-6 and i + 1 < len(speed) and speed[i + 1] > 0:
                features['post_static_jump'] = speed[i + 1]
                break
        else:
            features['post_static_jump'] = 0
    else:
        features['post_static_jump'] = 0
    
    # --- 8. NOISE TESPİT ÖZELLİKLERİ ---
    # Yön değişim sıklığı (gürültülü sinyal sürekli yön değiştirir)
    direction_changes_x = np.sum(np.diff(np.sign(dx)) != 0)
    direction_changes_y = np.sum(np.diff(np.sign(dy)) != 0)
    features['direction_changes'] = direction_changes_x + direction_changes_y
    features['direction_change_ratio'] = (direction_changes_x + direction_changes_y) / (2 * len(dx))
    
    # Sinyal gürültü oranı (SNR)
    features['x_snr'] = np.mean(np.abs(x)) / (np.std(x) + 1e-8)
    features['y_snr'] = np.mean(np.abs(y)) / (np.std(y) + 1e-8)
    
    # --- 9. GENEL İSTATİSTİKLER ---
    features['total_distance'] = np.sum(speed)
    features['net_displacement'] = np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2)
    features['path_efficiency'] = features['net_displacement'] / (features['total_distance'] + 1e-8)
    
    # Entropy (karmaşıklık)
    speed_norm = speed / (np.sum(speed) + 1e-8)
    speed_norm = speed_norm[speed_norm > 0]
    features['speed_entropy'] = -np.sum(speed_norm * np.log(speed_norm + 1e-10))
    
    return features

# Test
test_feat = extract_features(np.random.randn(40))
FEATURE_NAMES = list(test_feat.keys())
N_FEATURES = len(FEATURE_NAMES)
print(f"  ✅ {N_FEATURES} özellik çıkarılacak per yörünge")
print(f"  📋 Özellikler: {', '.join(FEATURE_NAMES[:10])}...")

# ============================================================
# 2. VERİ HAZIRLAMA (Büyük ölçekli)
# ============================================================
print("\n📂 Büyük ölçekli veri hazırlanıyor...")

# Anomali fonksiyonları
def inject_spike(x, y):
    xa, ya = x.copy(), y.copy()
    for _ in range(np.random.randint(1, 4)):
        idx = np.random.randint(0, len(x))
        xa[idx] += np.random.choice([-1, 1]) * np.random.uniform(0.2, 0.6)
        ya[idx] += np.random.choice([-1, 1]) * np.random.uniform(0.2, 0.6)
    return np.concatenate([xa, ya])

def inject_drift(x, y):
    xa, ya = x.copy(), y.copy()
    start = np.random.randint(5, 12)
    rate = np.random.uniform(0.01, 0.04)
    angle = np.random.uniform(0, 2 * np.pi)
    for i in range(start, len(x)):
        xa[i] += (i - start) * rate * np.cos(angle)
        ya[i] += (i - start) * rate * np.sin(angle)
    return np.concatenate([xa, ya])

def inject_freeze(x, y):
    xa, ya = x.copy(), y.copy()
    start = np.random.randint(4, 12)
    dur = np.random.randint(4, 10)
    for i in range(start, min(start + dur, len(x))):
        xa[i] = xa[start]; ya[i] = ya[start]
    return np.concatenate([xa, ya])

def inject_noise(x, y):
    xa, ya = x.copy(), y.copy()
    start = np.random.randint(3, 8)
    factor = np.random.uniform(0.03, 0.1)
    for i in range(start, len(x)):
        xa[i] += np.random.normal(0, factor * (1 + (i - start) * 0.25))
        ya[i] += np.random.normal(0, factor * (1 + (i - start) * 0.25))
    return np.concatenate([xa, ya])

def inject_dropout_interp(x, y):
    xa, ya = x.copy(), y.copy()
    n = int(len(x) * np.random.uniform(0.2, 0.4))
    start = np.random.randint(3, len(x) - n - 2)
    for i in range(start, start + n):
        xa[i] = xa[start - 1]; ya[i] = ya[start - 1]
    xa[start + n] += np.random.uniform(-0.2, 0.2)
    ya[start + n] += np.random.uniform(-0.2, 0.2)
    return np.concatenate([xa, ya])

generators = {1: inject_spike, 2: inject_drift, 3: inject_dropout_interp, 4: inject_freeze, 5: inject_noise}

# --- 4 şehirden veri yükle ---
cities = {
    'Milano': 'data/fcd-italy/milan_fcd_prep.csv',
    'Roma': 'data/fcd-italy/rome_fcd_prep.csv',
    'Torino': 'data/fcd-italy/turin_fcd_prep.csv',
    'Napoli': 'data/fcd-italy/naples_fcd_prep.csv'
}

x_cols = None
all_clean_rows = {}

for city, path in cities.items():
    df = pd.read_csv(path)
    if x_cols is None:
        x_cols = [c for c in df.columns if c.startswith('x(')]
        y_cols = [c for c in df.columns if c.startswith('y(')]
    clean = df[~df[x_cols + y_cols].isna().any(axis=1)]
    all_clean_rows[city] = clean
    print(f"  📍 {city}: {len(clean):,} temiz yörünge")

def make_raw(row):
    x = np.array([row[c] for c in x_cols])
    y = np.array([row[c] for c in y_cols])
    return np.concatenate([x, y])

# --- EĞİTİM SETİ: Milano + Roma + Torino ---
N_TRAIN = 300  # Her sınıf için
train_raw = []
train_labels = []

train_pool = pd.concat([all_clean_rows['Milano'], all_clean_rows['Roma'], all_clean_rows['Torino']])
train_pool = train_pool.sample(frac=1, random_state=42).reset_index(drop=True)

# Normal
for i in range(N_TRAIN):
    train_raw.append(make_raw(train_pool.iloc[i]))
    train_labels.append(0)

# Anomaliler
for cid, gen_func in generators.items():
    for i in range(N_TRAIN):
        row = train_pool.iloc[N_TRAIN + (cid-1)*N_TRAIN + i]
        x = np.array([row[c] for c in x_cols])
        y = np.array([row[c] for c in y_cols])
        train_raw.append(gen_func(x, y))
        train_labels.append(cid)

X_train_raw = np.array(train_raw, dtype=np.float32)
y_train = np.array(train_labels)
print(f"\n  ✅ Eğitim seti (seed): {len(X_train_raw)} örnek (3 şehir)")

# --- TEST SETİ: Napoli (hiç görülmemiş şehir) ---
N_TEST = 100
test_raw = []
test_labels = []

napoli = all_clean_rows['Napoli'].sample(frac=1, random_state=99).reset_index(drop=True)

for i in range(N_TEST):
    test_raw.append(make_raw(napoli.iloc[i]))
    test_labels.append(0)

for cid, gen_func in generators.items():
    for i in range(N_TEST):
        row = napoli.iloc[N_TEST + (cid-1)*N_TEST + i]
        x = np.array([row[c] for c in x_cols])
        y = np.array([row[c] for c in y_cols])
        test_raw.append(gen_func(x, y))
        test_labels.append(cid)

X_test_raw = np.array(test_raw, dtype=np.float32)
y_test = np.array(test_labels)
print(f"  ✅ Test seti: {len(X_test_raw)} örnek (Napoli)")

# --- GAN VERİSİ ---
gan_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'gan_generated_dataset.csv'))
X_gan_raw = gan_df.drop('label', axis=1).values.astype(np.float32)
y_gan = gan_df['label'].map(LABEL_MAP).values
print(f"  ✅ GAN verisi: {len(X_gan_raw)} örnek")

# ============================================================
# 3. FEATURE EXTRACTION
# ============================================================
print("\n🔬 Feature Extraction...")

def batch_extract(data):
    features_list = []
    for row in data:
        feat = extract_features(row)
        features_list.append(list(feat.values()))
    return np.array(features_list, dtype=np.float32)

X_train_feat = batch_extract(X_train_raw)
X_test_feat = batch_extract(X_test_raw)
X_gan_feat = batch_extract(X_gan_raw)

# NaN temizle
X_train_feat = np.nan_to_num(X_train_feat, nan=0, posinf=1e6, neginf=-1e6)
X_test_feat = np.nan_to_num(X_test_feat, nan=0, posinf=1e6, neginf=-1e6)
X_gan_feat = np.nan_to_num(X_gan_feat, nan=0, posinf=1e6, neginf=-1e6)

print(f"  ✅ Eğitim: {X_train_feat.shape} | Test: {X_test_feat.shape} | GAN: {X_gan_feat.shape}")

# StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_test_scaled = scaler.transform(X_test_feat)
X_gan_scaled = scaler.transform(X_gan_feat)

# Birleşik
X_aug_scaled = np.vstack([X_train_scaled, X_gan_scaled])
y_aug = np.concatenate([y_train, y_gan])

# ============================================================
# 4. ÜÇ DENEY
# ============================================================
print("\n" + "=" * 60)
print("🧪 ÜÇ DENEY: Raw vs Feature Engineering vs FE+GAN")
print("=" * 60)

# Ham veri için de scaler
scaler_raw = StandardScaler()
X_train_raw_s = scaler_raw.fit_transform(X_train_raw)
X_test_raw_s = scaler_raw.transform(X_test_raw)
X_gan_raw_s = scaler_raw.transform(X_gan_raw)
X_aug_raw_s = np.vstack([X_train_raw_s, X_gan_raw_s])

models_config = {
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=2, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=1000, early_stopping=True, random_state=42)
}

from sklearn.base import clone

all_results = {}

for model_name, model_template in models_config.items():
    print(f"\n  🤖 {model_name}:")
    
    # Deney 1: Ham veri, sadece seed
    m1 = clone(model_template)
    m1.fit(X_train_raw_s, y_train)
    pred1 = m1.predict(X_test_raw_s)
    
    # Deney 2: Feature Engineering, sadece seed
    m2 = clone(model_template)
    m2.fit(X_train_scaled, y_train)
    pred2 = m2.predict(X_test_scaled)
    
    # Deney 3: Feature Engineering + GAN
    m3 = clone(model_template)
    m3.fit(X_aug_scaled, y_aug)
    pred3 = m3.predict(X_test_scaled)
    
    f1_1 = f1_score(y_test, pred1, average='macro')
    f1_2 = f1_score(y_test, pred2, average='macro')
    f1_3 = f1_score(y_test, pred3, average='macro')
    
    r1 = recall_score(y_test, pred1, average='macro')
    r2 = recall_score(y_test, pred2, average='macro')
    r3 = recall_score(y_test, pred3, average='macro')
    
    p1 = precision_score(y_test, pred1, average='macro')
    p2 = precision_score(y_test, pred2, average='macro')
    p3 = precision_score(y_test, pred3, average='macro')
    
    imp_fe = ((f1_2 - f1_1) / (f1_1 + 1e-8)) * 100
    imp_gan = ((f1_3 - f1_2) / (f1_2 + 1e-8)) * 100
    imp_total = ((f1_3 - f1_1) / (f1_1 + 1e-8)) * 100
    
    all_results[model_name] = {
        'raw_f1': f1_1, 'fe_f1': f1_2, 'fegan_f1': f1_3,
        'raw_recall': r1, 'fe_recall': r2, 'fegan_recall': r3,
        'raw_prec': p1, 'fe_prec': p2, 'fegan_prec': p3,
        'imp_fe': imp_fe, 'imp_gan': imp_gan, 'imp_total': imp_total,
        'pred_raw': pred1, 'pred_fe': pred2, 'pred_fegan': pred3
    }
    
    s1 = "✅" if f1_2 > f1_1 else "❌"
    s2 = "✅" if f1_3 > f1_2 else "❌"
    
    print(f"     {'':20s} | {'Ham Veri':>10s} | {'+ FE':>10s} | {'+ FE + GAN':>10s}")
    print(f"     {'-'*20} | {'-'*10} | {'-'*10} | {'-'*10}")
    print(f"     {'F1 Score':20s} | {f1_1:>10.4f} | {f1_2:>10.4f} {s1} | {f1_3:>10.4f} {s2}")
    print(f"     {'Recall':20s} | {r1:>10.4f} | {r2:>10.4f}   | {r3:>10.4f}")
    print(f"     {'Precision':20s} | {p1:>10.4f} | {p2:>10.4f}   | {p3:>10.4f}")
    print(f"     {'FE İyileşme':20s} |            | {imp_fe:>+9.1f}%  |")
    print(f"     {'GAN İyileşme':20s} |            |            | {imp_gan:>+9.1f}%")
    print(f"     {'TOPLAM İyileşme':20s} |            |            | {imp_total:>+9.1f}%")

# ============================================================
# 5. EN İYİ MODEL DETAYLI ANALİZ
# ============================================================
print("\n" + "=" * 60)
print("🏆 EN İYİ MODEL DETAYLI ANALİZ")
print("=" * 60)

best = max(all_results.keys(), key=lambda k: all_results[k]['fegan_f1'])
r = all_results[best]
print(f"\n  🏆 En iyi: {best} (F1: {r['fegan_f1']:.4f})")

print(f"\n  --- Feature Engineering + GAN ---")
print(classification_report(y_test, r['pred_fegan'], 
                            target_names=[LABEL_NAMES[i] for i in range(6)]))

# ============================================================
# 6. GÖRSELLEŞTİRME
# ============================================================
print("\n🎨 Final grafikleri oluşturuluyor...")

fig = plt.figure(figsize=(24, 18), facecolor=COLORS['bg'])
fig.suptitle('Akıllı Veri Üretimi - Final Sonuçlar', 
             fontsize=20, fontweight='bold', color=COLORS['text'], y=0.99)
fig.text(0.5, 0.965, 'Ham Veri → Feature Engineering → GAN Augmentation Karşılaştırması',
         fontsize=12, color='#94a3b8', ha='center')

gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# --- Satır 1: 3 model F1 karşılaştırması ---
model_names = list(all_results.keys())
x_pos = np.arange(len(model_names))
width = 0.25

ax = fig.add_subplot(gs[0, :2], facecolor=COLORS['bg'])
vals1 = [all_results[m]['raw_f1'] for m in model_names]
vals2 = [all_results[m]['fe_f1'] for m in model_names]
vals3 = [all_results[m]['fegan_f1'] for m in model_names]

b1 = ax.bar(x_pos - width, vals1, width, color='#475569', label='Ham Veri', edgecolor='white', linewidth=0.5)
b2 = ax.bar(x_pos, vals2, width, color=COLORS['seed'], label='+ Feature Engineering', edgecolor='white', linewidth=0.5)
b3 = ax.bar(x_pos + width, vals3, width, color=COLORS['improved'], label='+ FE + GAN', edgecolor='white', linewidth=0.5)

for bars, vals in [(b1, vals1), (b2, vals2), (b3, vals3)]:
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', color=COLORS['text'], fontsize=9, fontweight='bold')

ax.set_title('F1 Score - 3 Aşamalı İyileşme', color=COLORS['text'], fontweight='bold', fontsize=14)
ax.set_ylabel('F1 Score (Macro)', color=COLORS['text'])
ax.set_xticks(x_pos)
ax.set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=10)
ax.legend(facecolor=COLORS['bg'], edgecolor=COLORS['grid'], fontsize=10)
ax.set_facecolor(COLORS['bg'])
ax.tick_params(colors=COLORS['text'])
ax.grid(True, alpha=0.1, axis='y')
ax.set_ylim(0, 1.1)

# İyileşme yüzdesi
ax = fig.add_subplot(gs[0, 2], facecolor=COLORS['bg'])
total_imps = [all_results[m]['imp_total'] for m in model_names]
bar_colors = [COLORS['seed'] if imp > 0 else COLORS['spike'] for imp in total_imps]
bars = ax.bar(model_names, total_imps, color=bar_colors, edgecolor='white', linewidth=0.5)
ax.axhline(y=15, color='#fbbf24', linestyle='--', alpha=0.5, label='Hedef: +15%')
ax.axhline(y=0, color='white', linewidth=0.5)
ax.set_title('Toplam F1 İyileşme (%)', color=COLORS['text'], fontweight='bold', fontsize=13)
ax.set_ylabel('%', color=COLORS['text'])
ax.legend(facecolor=COLORS['bg'], edgecolor=COLORS['grid'])
ax.set_facecolor(COLORS['bg'])
ax.tick_params(colors=COLORS['text'])
ax.set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=8)
ax.grid(True, alpha=0.1, axis='y')
for bar, val in zip(bars, total_imps):
    y = bar.get_height() + (2 if val >= 0 else -5)
    ax.text(bar.get_x() + bar.get_width()/2., y,
            f'{val:+.1f}%', ha='center', color=COLORS['text'], fontweight='bold', fontsize=11)

# --- Satır 2: Confusion Matrix (en iyi model) ---
for exp_i, (title, preds, cmap, color) in enumerate([
    ('Ham Veri', r['pred_raw'], 'Greys', '#475569'),
    ('+ Feature Engineering', r['pred_fe'], 'Greens', COLORS['seed']),
    ('+ FE + GAN', r['pred_fegan'], 'YlOrRd', COLORS['improved'])
]):
    ax = fig.add_subplot(gs[1, exp_i], facecolor=COLORS['bg'])
    cm = confusion_matrix(y_test, preds)
    im = ax.imshow(cm, cmap=cmap, aspect='auto')
    names_short = ['NOR', 'SPK', 'DRF', 'DRP', 'FRZ', 'NSE']
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(names_short, color=COLORS['text'], fontsize=8)
    ax.set_yticklabels(names_short, color=COLORS['text'], fontsize=8)
    ax.set_xlabel('Tahmin', color=COLORS['text'])
    ax.set_ylabel('Gerçek', color=COLORS['text'])
    ax.set_title(f'{best}\n{title}', color=color, fontweight='bold', fontsize=11)
    for i in range(6):
        for j in range(6):
            txt_color = 'white' if cm[i,j] > cm.max()*0.5 else 'black'
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', color=txt_color, fontweight='bold', fontsize=10)

# --- Satır 3: Sınıf bazlı F1 + Feature importance ---
# Sınıf bazlı F1
ax = fig.add_subplot(gs[2, :2], facecolor=COLORS['bg'])
f1_raw = f1_score(y_test, r['pred_raw'], average=None)
f1_fe = f1_score(y_test, r['pred_fe'], average=None)
f1_fegan = f1_score(y_test, r['pred_fegan'], average=None)

x_pos = np.arange(6)
width = 0.25
ax.bar(x_pos - width, f1_raw, width, color='#475569', label='Ham Veri', edgecolor='white', linewidth=0.5)
ax.bar(x_pos, f1_fe, width, color=COLORS['seed'], label='+ FE', edgecolor='white', linewidth=0.5)
ax.bar(x_pos + width, f1_fegan, width, color=COLORS['improved'], label='+ FE + GAN', edgecolor='white', linewidth=0.5)

class_labels = [LABEL_NAMES[i] for i in range(6)]
ax.set_xticks(x_pos)
ax.set_xticklabels(class_labels, fontsize=10)
ax.set_title(f'Sınıf Bazlı F1 Score ({best})', color=COLORS['text'], fontweight='bold', fontsize=13)
ax.set_ylabel('F1 Score', color=COLORS['text'])
ax.legend(facecolor=COLORS['bg'], edgecolor=COLORS['grid'], fontsize=10)
ax.set_facecolor(COLORS['bg'])
ax.tick_params(colors=COLORS['text'])
ax.grid(True, alpha=0.1, axis='y')
ax.set_ylim(0, 1.1)

# Feature importance (RF veya GB)
ax = fig.add_subplot(gs[2, 2], facecolor=COLORS['bg'])
try:
    # En iyi modelin feature importance'ı (RF veya GB ise)
    fi_model = clone(models_config['Random Forest'])
    fi_model.fit(X_aug_scaled, y_aug)
    importances = fi_model.feature_importances_
    top_idx = np.argsort(importances)[-10:]
    top_names = [FEATURE_NAMES[i] for i in top_idx]
    top_vals = importances[top_idx]
    
    colors_fi = plt.cm.viridis(np.linspace(0.3, 0.9, 10))
    ax.barh(range(10), top_vals, color=colors_fi, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(10))
    ax.set_yticklabels(top_names, fontsize=8, color=COLORS['text'])
    ax.set_title('Top 10 Özellik (Feature Importance)', color=COLORS['text'], fontweight='bold', fontsize=11)
    ax.set_xlabel('Önem', color=COLORS['text'])
except:
    ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', color=COLORS['text'])
ax.set_facecolor(COLORS['bg'])
ax.tick_params(colors=COLORS['text'])
ax.grid(True, alpha=0.1, axis='x')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f'{OUTPUT_DIR}/09_final_results.png', dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("  ✅ 09_final_results.png")

# ============================================================
# 7. SONUÇ KAYDET
# ============================================================
summary = {
    'experiment': 'Feature Engineering + GAN Augmentation',
    'train_cities': ['Milano', 'Roma', 'Torino'],
    'test_city': 'Napoli',
    'train_size': {'seed': len(X_train_raw), 'gan': len(X_gan_raw), 'total': len(X_aug_scaled)},
    'test_size': len(X_test_raw),
    'n_features': N_FEATURES,
    'feature_names': FEATURE_NAMES,
    'models': {}
}
for m, r in all_results.items():
    summary['models'][m] = {
        'raw_f1': round(r['raw_f1'], 4), 'fe_f1': round(r['fe_f1'], 4), 
        'fegan_f1': round(r['fegan_f1'], 4),
        'total_improvement_pct': round(r['imp_total'], 2)
    }

with open(f'{OUTPUT_DIR}/final_results.json', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# ============================================================
# ÖZET
# ============================================================
print("\n" + "=" * 60)
print("🎉 FAZ 4 TAMAMLANDI - FİNAL SONUÇLAR")
print("=" * 60)

print(f"\n  📊 3 Aşamalı Pipeline Sonuçları:")
print(f"     {'Model':25s} | {'Ham':>6s} | {'+ FE':>6s} | {'+ GAN':>6s} | {'Toplam':>8s}")
print(f"     {'-'*25} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*8}")
for m, r in all_results.items():
    print(f"     {m:25s} | {r['raw_f1']:>6.2f} | {r['fe_f1']:>6.2f} | {r['fegan_f1']:>6.2f} | {r['imp_total']:>+7.1f}%")

best_total = max(all_results.values(), key=lambda x: x['fegan_f1'])
print(f"\n  🏆 En iyi F1: {max(r['fegan_f1'] for r in all_results.values()):.4f}")
print(f"  📈 En büyük iyileşme: {max(r['imp_total'] for r in all_results.values()):+.1f}%")
