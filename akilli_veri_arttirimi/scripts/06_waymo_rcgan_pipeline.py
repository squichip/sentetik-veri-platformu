"""
============================================================
 Akıllı Veri Üretimi - Faz 6: Waymo RCGAN Pipeline
============================================================
 
 Bu script:
 1. Waymo seed verisinden feature extraction yapar
 2. RCGAN üretimi veriyi de feature extraction'dan geçirir
 3. Seed-only vs Seed+RCGAN augmented karşılaştırması yapar
 4. Gradient Boosting, Random Forest, MLP ile sınıflandırma
 5. Sonuçları görselleştirir
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
import os, warnings, json
warnings.filterwarnings('ignore')

OUTPUT_DIR = "outputs"
np.random.seed(42)

plt.style.use('dark_background')
COLORS = {
    'bg': '#0f172a', 'text': '#f1f5f9', 'grid': '#1e293b',
    'normal': '#10b981', 'spike': '#ef4444', 'drift': '#f59e0b',
    'dropout': '#8b5cf6', 'freeze': '#3b82f6', 'noise': '#06b6d4',
    'seed': '#10b981', 'augmented': '#f472b6', 'accent': '#fbbf24'
}

LABEL_MAP = {'normal': 0, 'spike': 1, 'drift': 2, 'dropout': 3, 'freeze': 4, 'noise': 5}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}

print("=" * 60)
print("🚀 FAZ 6: WAYMO RCGAN PİPELİNE")
print("   Feature Engineering + Sınıflandırma + Augmentasyon Testi")
print("=" * 60)

# ============================================================
# 1. VERİ YÜKLEME
# ============================================================
print("\n📂 Veriler yükleniyor...")
seed_df = pd.read_csv('waymo_seed_MASSIVE.csv')
gen_df = pd.read_csv(f'{OUTPUT_DIR}/waymo_rcgan_generated.csv')

print(f"  📌 Waymo Seed: {seed_df.shape[0]:,} örnek, {seed_df.shape[1]} sütun")
print(f"  📌 RCGAN Generated: {gen_df.shape[0]:,} örnek, {gen_df.shape[1]} sütun")
print(f"  📌 Seed Label Dağılımı:")
for label, count in seed_df['label'].value_counts().items():
    print(f"      {label:10s}: {count}")

# ============================================================
# 2. FEATURE ENGINEERING (5-Kanallı Waymo İçin)
# ============================================================
print("\n🔧 Feature Engineering başlıyor...")

SEQ_LEN = 20

def extract_waymo_features(row):
    """5 kanallı Waymo yörüngesinden 50+ özellik çıkarır"""
    x = np.array([row[f'x({i+1})'] for i in range(SEQ_LEN)])
    y = np.array([row[f'y({i+1})'] for i in range(SEQ_LEN)])
    speed = np.array([row[f'speed({i+1})'] for i in range(SEQ_LEN)])
    vx = np.array([row[f'vx({i+1})'] for i in range(SEQ_LEN)])
    vy = np.array([row[f'vy({i+1})'] for i in range(SEQ_LEN)])
    
    features = {}
    
    # === KOORDİNAT ÖZELLİKLERİ ===
    dx = np.diff(x)
    dy = np.diff(y)
    step_distances = np.sqrt(dx**2 + dy**2)
    
    features['total_distance'] = np.sum(step_distances)
    features['mean_step'] = np.mean(step_distances)
    features['std_step'] = np.std(step_distances)
    features['max_step'] = np.max(step_distances)
    features['min_step'] = np.min(step_distances)
    features['step_range'] = features['max_step'] - features['min_step']
    
    # Düzlük (Straightness)
    endpoint_dist = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
    features['straightness'] = endpoint_dist / (features['total_distance'] + 1e-8)
    
    # Açısal değişimler (Turning)
    angles = np.arctan2(dy, dx)
    angle_changes = np.abs(np.diff(angles))
    angle_changes = np.minimum(angle_changes, 2*np.pi - angle_changes)
    features['mean_turning'] = np.mean(angle_changes)
    features['max_turning'] = np.max(angle_changes)
    features['total_turning'] = np.sum(angle_changes)
    
    # === HIZ ÖZELLİKLERİ ===
    features['mean_speed'] = np.mean(speed)
    features['std_speed'] = np.std(speed)
    features['max_speed'] = np.max(speed)
    features['min_speed'] = np.min(speed)
    features['speed_range'] = features['max_speed'] - features['min_speed']
    
    # Hız değişimi (ivme)
    speed_diff = np.diff(speed)
    features['mean_acceleration'] = np.mean(speed_diff)
    features['std_acceleration'] = np.std(speed_diff)
    features['max_acceleration'] = np.max(speed_diff)
    features['max_deceleration'] = np.min(speed_diff)
    
    # Jerk (ivme değişimi)
    jerk = np.diff(speed_diff)
    features['mean_jerk'] = np.mean(np.abs(jerk))
    features['max_jerk'] = np.max(np.abs(jerk))
    
    # === HIZLANMA BİLEŞENLERİ (vx, vy) ===
    features['mean_vx'] = np.mean(vx)
    features['std_vx'] = np.std(vx)
    features['mean_vy'] = np.mean(vy)
    features['std_vy'] = np.std(vy)
    
    vx_diff = np.diff(vx)
    vy_diff = np.diff(vy)
    features['mean_ax'] = np.mean(vx_diff)
    features['std_ax'] = np.std(vx_diff)
    features['mean_ay'] = np.mean(vy_diff)
    features['std_ay'] = np.std(vy_diff)
    
    # === FREEZE / DROPOUT TESPİT ÖZELLİKLERİ ===
    zero_speed_ratio = np.sum(speed < 0.01) / len(speed)
    features['zero_speed_ratio'] = zero_speed_ratio
    
    # Ardışık aynı konum sayısı
    same_pos = np.sum((np.abs(dx) < 1e-6) & (np.abs(dy) < 1e-6))
    features['freeze_count'] = same_pos
    
    # En uzun freeze süresi
    max_freeze = 0
    current_freeze = 0
    for i in range(len(dx)):
        if abs(dx[i]) < 1e-6 and abs(dy[i]) < 1e-6:
            current_freeze += 1
            max_freeze = max(max_freeze, current_freeze)
        else:
            current_freeze = 0
    features['max_freeze_duration'] = max_freeze
    
    # === SPİKE TESPİT ÖZELLİKLERİ ===
    step_mean = np.mean(step_distances)
    step_std = np.std(step_distances)
    spikes = np.sum(step_distances > step_mean + 3 * step_std)
    features['spike_count'] = spikes
    
    # === ENTROPİ ÖZELLİKLERİ ===
    speed_hist, _ = np.histogram(speed, bins=10, density=True)
    speed_hist = speed_hist[speed_hist > 0]
    features['speed_entropy'] = entropy(speed_hist)
    
    step_hist, _ = np.histogram(step_distances, bins=10, density=True)
    step_hist = step_hist[step_hist > 0]
    features['step_entropy'] = entropy(step_hist)
    
    # === İSTATİSTİKSEL MOMENTLER ===
    features['x_range'] = np.ptp(x)
    features['y_range'] = np.ptp(y)
    features['xy_correlation'] = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else 0
    
    # Otokorelasyon (zaman bağımlılığı)
    if len(speed) > 1 and np.std(speed) > 0:
        speed_norm = (speed - np.mean(speed)) / (np.std(speed) + 1e-8)
        features['speed_autocorr'] = np.correlate(speed_norm, speed_norm, mode='full')[len(speed_norm)] / len(speed_norm)
    else:
        features['speed_autocorr'] = 0
    
    return features

def build_feature_matrix(df):
    """DataFrame'den feature matrisi oluştur"""
    all_features = []
    for idx, row in df.iterrows():
        feat = extract_waymo_features(row)
        all_features.append(feat)
    
    feature_df = pd.DataFrame(all_features)
    return feature_df

print("  🔄 Seed verisi için feature extraction...")
seed_features = build_feature_matrix(seed_df)
seed_labels = seed_df['label'].map(LABEL_MAP).values
print(f"     ✅ {seed_features.shape[0]} örnek, {seed_features.shape[1]} özellik")

print("  🔄 RCGAN verisi için feature extraction...")
gen_features = build_feature_matrix(gen_df)
gen_labels = gen_df['label'].map(LABEL_MAP).values
print(f"     ✅ {gen_features.shape[0]} örnek, {gen_features.shape[1]} özellik")

# NaN/Inf temizle
seed_features = seed_features.replace([np.inf, -np.inf], np.nan).fillna(0)
gen_features = gen_features.replace([np.inf, -np.inf], np.nan).fillna(0)

# ============================================================
# 3. SINIFLANDIRMA PİPELİNE
# ============================================================
print("\n" + "=" * 60)
print("🤖 SINIFLANDIRMA DENEYLERİ")
print("=" * 60)

# Train/Test Split (Seed verisinden)
X_train_seed, X_test, y_train_seed, y_test = train_test_split(
    seed_features, seed_labels, test_size=0.25, random_state=42, stratify=seed_labels
)

# Augmented train set: Seed train + RCGAN generated
X_train_aug = pd.concat([X_train_seed, gen_features], ignore_index=True)
y_train_aug = np.concatenate([y_train_seed, gen_labels])

scaler_seed = StandardScaler()
X_train_seed_scaled = scaler_seed.fit_transform(X_train_seed)
X_test_seed_scaled = scaler_seed.transform(X_test)

scaler_aug = StandardScaler()
X_train_aug_scaled = scaler_aug.fit_transform(X_train_aug)
X_test_aug_scaled = scaler_aug.transform(X_test)

models = {
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    'MLP Neural Net': MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42, early_stopping=True),
}

results = {}

for name, model in models.items():
    print(f"\n  📊 {name}:")
    
    # Seed Only
    m1 = type(model)(**model.get_params())
    m1.fit(X_train_seed_scaled, y_train_seed)
    y_pred_seed = m1.predict(X_test_seed_scaled)
    f1_seed = f1_score(y_test, y_pred_seed, average='weighted')
    
    # Seed + RCGAN Augmented
    m2 = type(model)(**model.get_params())
    m2.fit(X_train_aug_scaled, y_train_aug)
    y_pred_aug = m2.predict(X_test_aug_scaled)
    f1_aug = f1_score(y_test, y_pred_aug, average='weighted')
    
    improvement = ((f1_aug - f1_seed) / f1_seed) * 100
    
    print(f"     Seed Only     F1: {f1_seed:.4f}")
    print(f"     Seed + RCGAN  F1: {f1_aug:.4f}")
    print(f"     Değişim:         {'+' if improvement > 0 else ''}{improvement:.1f}%")
    
    results[name] = {
        'seed_f1': float(f1_seed),
        'augmented_f1': float(f1_aug),
        'improvement': float(improvement)
    }

# En iyi modelin detaylı raporu
best_model_name = max(results, key=lambda k: results[k]['augmented_f1'])
print(f"\n  🏆 En İyi Model: {best_model_name}")
print(f"     Augmented F1: {results[best_model_name]['augmented_f1']:.4f}")

# En iyi modeli yeniden eğit ve classification report al
best_model = models[best_model_name]
best_m = type(best_model)(**best_model.get_params())
best_m.fit(X_train_aug_scaled, y_train_aug)
y_pred_best = best_m.predict(X_test_aug_scaled)

print("\n" + "=" * 60)
print(f"📋 {best_model_name} (Augmented) Detaylı Rapor:")
print("=" * 60)
print(classification_report(y_test, y_pred_best, target_names=list(LABEL_MAP.keys())))

# ============================================================
# 4. GÖRSELLEŞTİRME
# ============================================================
print("\n🎨 Görselleştirmeler oluşturuluyor...")

fig = plt.figure(figsize=(28, 20), facecolor=COLORS['bg'])
fig.suptitle('WAYMO RCGAN PİPELİNE SONUÇLARI', fontsize=24, color=COLORS['text'], fontweight='bold', y=0.98)
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# 1. F1 Score Karşılaştırması
ax1 = fig.add_subplot(gs[0, 0], facecolor=COLORS['bg'])
model_names = list(results.keys())
seed_scores = [results[m]['seed_f1'] for m in model_names]
aug_scores = [results[m]['augmented_f1'] for m in model_names]
x_pos = np.arange(len(model_names))
width = 0.35
bars1 = ax1.bar(x_pos - width/2, seed_scores, width, label='Seed Only', color=COLORS['seed'], alpha=0.8)
bars2 = ax1.bar(x_pos + width/2, aug_scores, width, label='Seed + RCGAN', color=COLORS['augmented'], alpha=0.8)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=9, color=COLORS['text'])
ax1.set_ylabel('F1 Score (Weighted)', color=COLORS['text'])
ax1.set_title('Model Performans Karşılaştırması', color=COLORS['accent'], fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.set_ylim(0, 1.1)
for bar in bars1: ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8, color=COLORS['text'])
for bar in bars2: ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8, color=COLORS['text'])
ax1.grid(alpha=0.1)

# 2. İyileşme Yüzdeleri
ax2 = fig.add_subplot(gs[0, 1], facecolor=COLORS['bg'])
improvements = [results[m]['improvement'] for m in model_names]
colors_imp = [COLORS['seed'] if imp >= 0 else COLORS['spike'] for imp in improvements]
ax2.barh(model_names, improvements, color=colors_imp, alpha=0.8)
ax2.axvline(x=0, color=COLORS['text'], linestyle='--', alpha=0.3)
for i, (imp, name) in enumerate(zip(improvements, model_names)):
    ax2.text(imp + (0.5 if imp >= 0 else -0.5), i, f'{imp:+.1f}%', va='center', ha='left' if imp >= 0 else 'right', fontsize=11, color=COLORS['text'], fontweight='bold')
ax2.set_title('RCGAN Augmentasyon Etkisi (%)', color=COLORS['accent'], fontsize=14, fontweight='bold')
ax2.set_xlabel('F1 Score Değişimi (%)', color=COLORS['text'])
ax2.grid(alpha=0.1)

# 3. Confusion Matrix
ax3 = fig.add_subplot(gs[0, 2], facecolor=COLORS['bg'])
cm = confusion_matrix(y_test, y_pred_best)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
im = ax3.imshow(cm_normalized, interpolation='nearest', cmap='magma', aspect='auto')
label_names = list(LABEL_MAP.keys())
ax3.set_xticks(range(len(label_names)))
ax3.set_yticks(range(len(label_names)))
ax3.set_xticklabels(label_names, rotation=45, ha='right', fontsize=9, color=COLORS['text'])
ax3.set_yticklabels(label_names, fontsize=9, color=COLORS['text'])
for i in range(len(label_names)):
    for j in range(len(label_names)):
        text_color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
        ax3.text(j, i, f'{cm_normalized[i,j]:.2f}', ha='center', va='center', color=text_color, fontsize=9)
ax3.set_title(f'Confusion Matrix ({best_model_name})', color=COLORS['accent'], fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

# 4. Sınıf Bazlı F1 Skorları
ax4 = fig.add_subplot(gs[1, 0], facecolor=COLORS['bg'])
from sklearn.metrics import f1_score as f1_per_class
f1_per = f1_score(y_test, y_pred_best, average=None)
class_colors = [COLORS.get(LABEL_NAMES[i], '#94a3b8') for i in range(len(f1_per))]
bars = ax4.bar(label_names, f1_per, color=class_colors, alpha=0.85)
for bar, val in zip(bars, f1_per):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=10, color=COLORS['text'], fontweight='bold')
ax4.set_ylim(0, 1.15)
ax4.set_title('Sınıf Bazlı F1 Skorları (Augmented)', color=COLORS['accent'], fontsize=14, fontweight='bold')
ax4.set_ylabel('F1 Score', color=COLORS['text'])
ax4.grid(alpha=0.1)

# 5. Veri Dağılımı
ax5 = fig.add_subplot(gs[1, 1], facecolor=COLORS['bg'])
seed_counts = seed_df['label'].value_counts()
gen_counts = gen_df['label'].value_counts()
x_pos = np.arange(len(label_names))
ax5.bar(x_pos - 0.2, [seed_counts.get(n, 0) for n in label_names], 0.4, label='Waymo Seed', color=COLORS['seed'], alpha=0.8)
ax5.bar(x_pos + 0.2, [gen_counts.get(n, 0) for n in label_names], 0.4, label='RCGAN Üretimi', color=COLORS['augmented'], alpha=0.8)
ax5.set_xticks(x_pos)
ax5.set_xticklabels(label_names, fontsize=9, color=COLORS['text'])
ax5.set_title('Veri Kaynağı Dağılımı', color=COLORS['accent'], fontsize=14, fontweight='bold')
ax5.set_ylabel('Örnek Sayısı', color=COLORS['text'])
ax5.legend(fontsize=10)
ax5.grid(alpha=0.1)

# 6. Feature Importance (Top 15)
ax6 = fig.add_subplot(gs[1, 2], facecolor=COLORS['bg'])
if best_model_name == 'Gradient Boosting' or best_model_name == 'Random Forest':
    importances = best_m.feature_importances_
    feat_names = list(seed_features.columns)
    sorted_idx = np.argsort(importances)[-15:]
    ax6.barh([feat_names[i] for i in sorted_idx], importances[sorted_idx], color=COLORS['accent'], alpha=0.8)
    ax6.set_title('En Önemli 15 Özellik', color=COLORS['accent'], fontsize=14, fontweight='bold')
    ax6.set_xlabel('Önem Skoru', color=COLORS['text'])
else:
    ax6.text(0.5, 0.5, 'MLP modeli\niçin feature importance\nhesaplanamaz', ha='center', va='center', fontsize=14, color=COLORS['text'])
    ax6.set_title('Feature Importance', color=COLORS['accent'], fontsize=14, fontweight='bold')
ax6.grid(alpha=0.1)

plt.savefig(f'{OUTPUT_DIR}/12_waymo_rcgan_pipeline.png', dpi=150, facecolor=COLORS['bg'], bbox_inches='tight')
print(f"  ✅ Görselleştirme kaydedildi: {OUTPUT_DIR}/12_waymo_rcgan_pipeline.png")

# ============================================================
# 5. SONUÇ KAYDETME
# ============================================================
final_results = {
    'pipeline': 'Waymo RCGAN',
    'seed_samples': int(len(seed_df)),
    'rcgan_samples': int(len(gen_df)),
    'total_features': int(seed_features.shape[1]),
    'models': results,
    'best_model': best_model_name,
    'best_f1': float(results[best_model_name]['augmented_f1']),
}

with open(f'{OUTPUT_DIR}/waymo_rcgan_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print("\n" + "=" * 60)
print("🎉 WAYMO RCGAN PİPELİNE TAMAMLANDI!")
print("=" * 60)
print(f"\n📊 Özet:")
print(f"   Seed Veri:     {len(seed_df):,} yörünge (Waymo gerçek veri)")
print(f"   RCGAN Üretimi: {len(gen_df):,} yörünge (LSTM-GAN sentetik)")
print(f"   Toplam:        {len(seed_df) + len(gen_df):,} yörünge")
print(f"   Özellik Sayısı:{seed_features.shape[1]}")
print(f"\n🏆 En İyi Model:  {best_model_name}")
print(f"   Seed F1:       {results[best_model_name]['seed_f1']:.4f}")
print(f"   Augmented F1:  {results[best_model_name]['augmented_f1']:.4f}")
print(f"   Değişim:       {results[best_model_name]['improvement']:+.1f}%")
