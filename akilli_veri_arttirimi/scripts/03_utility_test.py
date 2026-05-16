"""
============================================================
 Akıllı Veri Üretimi - Faz 3: Utility (Fayda) Testi
============================================================
 
 Amaç: GAN ile üretilen sentetik verinin anomali tespit 
 modelinin performansını gerçekten artırıp artırmadığını
 kanıtlamak.
 
 Deney Tasarımı:
   1. Test seti ayır (hiç dokunulmayacak)
   2. Model A: Sadece seed veriyle eğit
   3. Model B: Seed + GAN veriyle eğit
   4. İkisini de AYNI test seti üzerinde karşılaştır
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, f1_score, recall_score, 
                              precision_score, confusion_matrix, precision_recall_curve,
                              auc)
from sklearn.preprocessing import LabelEncoder
import os, warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "outputs"
np.random.seed(42)

plt.style.use('dark_background')
COLORS = {
    'seed': '#10b981', 'gan': '#f472b6', 'bg': '#0f172a', 
    'text': '#f1f5f9', 'grid': '#1e293b',
    'normal': '#10b981', 'spike': '#ef4444', 'drift': '#f59e0b',
    'dropout': '#8b5cf6', 'freeze': '#3b82f6', 'noise': '#06b6d4'
}

print("=" * 60)
print("🧪 UTİLİTY (FAYDA) TESTİ")
print("=" * 60)
print("Soru: GAN verisi modelin performansını artırıyor mu?")

# ============================================================
# 1. VERİ HAZIRLAMA
# ============================================================
print("\n📂 Veriler yükleniyor...")

# FCD'den taze test verisi üret (GAN'ın hiç görmediği veri)
fcd = pd.read_csv("data/fcd-italy/naples_fcd_prep.csv")  # Napoli = farklı şehir!
x_cols = [c for c in fcd.columns if c.startswith('x(')]
y_cols = [c for c in fcd.columns if c.startswith('y(')]

clean = fcd[~fcd[x_cols + y_cols].isna().any(axis=1)]

def make_feat(row):
    x = np.array([row[c] for c in x_cols])
    y = np.array([row[c] for c in y_cols])
    return np.concatenate([x, y])

# Anomali fonksiyonları
def inject_spike(x, y):
    xa, ya = x.copy(), y.copy()
    for _ in range(np.random.randint(1, 4)):
        idx = np.random.randint(0, len(x))
        xa[idx] += np.random.choice([-1, 1]) * np.random.uniform(0.2, 0.5)
        ya[idx] += np.random.choice([-1, 1]) * np.random.uniform(0.2, 0.5)
    return np.concatenate([xa, ya])

def inject_drift(x, y):
    xa, ya = x.copy(), y.copy()
    start = np.random.randint(5, 12)
    rate = np.random.uniform(0.01, 0.035)
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
    factor = np.random.uniform(0.03, 0.08)
    for i in range(start, len(x)):
        xa[i] += np.random.normal(0, factor * (1 + (i - start) * 0.2))
        ya[i] += np.random.normal(0, factor * (1 + (i - start) * 0.2))
    return np.concatenate([xa, ya])

def inject_dropout_interp(x, y):
    xa, ya = x.copy(), y.copy()
    n = int(len(x) * np.random.uniform(0.2, 0.35))
    start = np.random.randint(3, len(x) - n - 2)
    for i in range(start, start + n):
        xa[i] = xa[start - 1]; ya[i] = ya[start - 1]
    xa[start + n] += np.random.uniform(-0.15, 0.15)
    ya[start + n] += np.random.uniform(-0.15, 0.15)
    return np.concatenate([xa, ya])

LABEL_MAP = {'normal': 0, 'spike': 1, 'drift': 2, 'dropout': 3, 'freeze': 4, 'noise': 5}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}
generators = {1: inject_spike, 2: inject_drift, 3: inject_dropout_interp, 4: inject_freeze, 5: inject_noise}

# --- TEST SETİ: Napoli verisinden (GAN'ın hiç görmediği) ---
N_TEST = 50  # Her sınıf için
test_features = []
test_labels = []

test_indices = np.random.choice(len(clean), N_TEST * 6, replace=False)

# Normal
for i in range(N_TEST):
    test_features.append(make_feat(clean.iloc[test_indices[i]]))
    test_labels.append(0)

# Anomaliler
for cid, gen_func in generators.items():
    for i in range(N_TEST):
        idx = test_indices[N_TEST + (cid-1)*N_TEST + i]
        row = clean.iloc[idx]
        x = np.array([row[c] for c in x_cols])
        y = np.array([row[c] for c in y_cols])
        test_features.append(gen_func(x, y))
        test_labels.append(cid)

X_test = np.array(test_features, dtype=np.float32)
y_test = np.array(test_labels)

print(f"  ✅ Test seti: {len(X_test)} örnek (Napoli - GAN'ın görmediği şehir!)")

# --- SEED EĞİTİM SETİ: Milano'dan (Faz 1'deki gibi) ---
fcd_milan = pd.read_csv("data/fcd-italy/milan_fcd_prep.csv")
clean_milan = fcd_milan[~fcd_milan[x_cols + y_cols].isna().any(axis=1)]

N_SEED = 80  # Her sınıf için
seed_features = []
seed_labels = []

seed_indices = np.random.choice(len(clean_milan), N_SEED * 6, replace=False)

for i in range(N_SEED):
    seed_features.append(make_feat(clean_milan.iloc[seed_indices[i]]))
    seed_labels.append(0)

for cid, gen_func in generators.items():
    for i in range(N_SEED):
        idx = seed_indices[N_SEED + (cid-1)*N_SEED + i]
        row = clean_milan.iloc[idx]
        x = np.array([row[c] for c in x_cols])
        y = np.array([row[c] for c in y_cols])
        seed_features.append(gen_func(x, y))
        seed_labels.append(cid)

X_seed = np.array(seed_features, dtype=np.float32)
y_seed = np.array(seed_labels)

print(f"  ✅ Seed eğitim seti: {len(X_seed)} örnek (Milano)")

# --- GAN VERİSİ ---
gan_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'gan_generated_dataset.csv'))
X_gan = gan_df.drop('label', axis=1).values.astype(np.float32)
y_gan = gan_df['label'].map(LABEL_MAP).values

print(f"  ✅ GAN verisi: {len(X_gan)} örnek")

# Birleşik set
X_augmented = np.vstack([X_seed, X_gan])
y_augmented = np.concatenate([y_seed, y_gan])

print(f"  ✅ Birleşik set: {len(X_augmented)} örnek (seed + GAN)")

# ============================================================
# 2. MODEL EĞİTİMİ & KARŞILAŞTIRMA
# ============================================================
print("\n" + "=" * 60)
print("🏋️ MODEL EĞİTİMİ & KARŞILAŞTIRMA")
print("=" * 60)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
    'Neural Network (MLP)': MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500,
                                           early_stopping=True, random_state=42)
}

results = {}

for model_name, model_template in models.items():
    print(f"\n  📊 {model_name}:")
    
    # --- Model A: Sadece Seed ---
    from sklearn.base import clone
    model_a = clone(model_template)
    model_a.fit(X_seed, y_seed)
    y_pred_a = model_a.predict(X_test)
    
    f1_a = f1_score(y_test, y_pred_a, average='macro')
    recall_a = recall_score(y_test, y_pred_a, average='macro')
    precision_a = precision_score(y_test, y_pred_a, average='macro')
    
    # --- Model B: Seed + GAN ---
    model_b = clone(model_template)
    model_b.fit(X_augmented, y_augmented)
    y_pred_b = model_b.predict(X_test)
    
    f1_b = f1_score(y_test, y_pred_b, average='macro')
    recall_b = recall_score(y_test, y_pred_b, average='macro')
    precision_b = precision_score(y_test, y_pred_b, average='macro')
    
    # İyileşme
    f1_improvement = ((f1_b - f1_a) / (f1_a + 1e-8)) * 100
    recall_improvement = ((recall_b - recall_a) / (recall_a + 1e-8)) * 100
    
    results[model_name] = {
        'seed_f1': f1_a, 'seed_recall': recall_a, 'seed_precision': precision_a,
        'aug_f1': f1_b, 'aug_recall': recall_b, 'aug_precision': precision_b,
        'f1_improvement': f1_improvement, 'recall_improvement': recall_improvement,
        'y_pred_seed': y_pred_a, 'y_pred_aug': y_pred_b
    }
    
    status_f1 = "✅" if f1_b > f1_a else "❌"
    status_recall = "✅" if recall_b > recall_a else "❌"
    
    print(f"     {'':15s} | {'Sadece Seed':>12s} | {'Seed + GAN':>12s} | {'İyileşme':>10s}")
    print(f"     {'-'*15} | {'-'*12} | {'-'*12} | {'-'*10}")
    print(f"     {'F1 Score':15s} | {f1_a:>12.4f} | {f1_b:>12.4f} | {status_f1} {f1_improvement:>+.1f}%")
    print(f"     {'Recall':15s} | {recall_a:>12.4f} | {recall_b:>12.4f} | {status_recall} {recall_improvement:>+.1f}%")
    print(f"     {'Precision':15s} | {precision_a:>12.4f} | {precision_b:>12.4f} |")

# ============================================================
# 3. SINIF BAZLI ANALİZ (En iyi model için)
# ============================================================
print("\n" + "=" * 60)
print("📋 SINIF BAZLI DETAYLI ANALİZ")
print("=" * 60)

# En iyi modeli bul
best_model = max(results.keys(), key=lambda k: results[k]['f1_improvement'])
print(f"\n  🏆 En iyi iyileşme: {best_model}")

print(f"\n  --- Sadece Seed ile ---")
print(classification_report(y_test, results[best_model]['y_pred_seed'], 
                            target_names=[LABEL_NAMES[i] for i in range(6)]))

print(f"  --- Seed + GAN ile ---")
print(classification_report(y_test, results[best_model]['y_pred_aug'],
                            target_names=[LABEL_NAMES[i] for i in range(6)]))

# ============================================================
# 4. GÖRSELLEŞTİRME
# ============================================================
print("\n🎨 Sonuç grafikleri oluşturuluyor...")

# A) Ana karşılaştırma grafiği
fig = plt.figure(figsize=(22, 14), facecolor=COLORS['bg'])
fig.suptitle('Utility Testi: GAN Verisi Modeli İyileştiriyor mu?', 
             fontsize=18, fontweight='bold', color=COLORS['text'], y=0.99)
fig.text(0.5, 0.96, 'Test Seti: Napoli verisi (GAN\'ın hiç görmediği farklı şehir)',
         fontsize=11, color='#94a3b8', ha='center')

gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# F1 Score karşılaştırması
ax = fig.add_subplot(gs[0, 0], facecolor=COLORS['bg'])
model_names = list(results.keys())
seed_f1s = [results[m]['seed_f1'] for m in model_names]
aug_f1s = [results[m]['aug_f1'] for m in model_names]

x_pos = np.arange(len(model_names))
width = 0.35
bars1 = ax.bar(x_pos - width/2, seed_f1s, width, color=COLORS['seed'], label='Sadece Seed', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x_pos + width/2, aug_f1s, width, color=COLORS['gan'], label='Seed + GAN', edgecolor='white', linewidth=0.5)
ax.set_title('F1 Score Karşılaştırması', color=COLORS['text'], fontweight='bold', fontsize=13)
ax.set_ylabel('F1 Score', color=COLORS['text'])
ax.set_xticks(x_pos)
ax.set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=8)
ax.legend(facecolor=COLORS['bg'], edgecolor=COLORS['grid'])
ax.set_facecolor(COLORS['bg'])
ax.tick_params(colors=COLORS['text'])
ax.grid(True, alpha=0.1, axis='y')
ax.set_ylim(0, 1.1)

for bar, val in zip(bars1, seed_f1s):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{val:.2f}', ha='center', va='bottom', color=COLORS['text'], fontsize=8, fontweight='bold')
for bar, val in zip(bars2, aug_f1s):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{val:.2f}', ha='center', va='bottom', color=COLORS['text'], fontsize=8, fontweight='bold')

# Recall karşılaştırması
ax = fig.add_subplot(gs[0, 1], facecolor=COLORS['bg'])
seed_recalls = [results[m]['seed_recall'] for m in model_names]
aug_recalls = [results[m]['aug_recall'] for m in model_names]

bars1 = ax.bar(x_pos - width/2, seed_recalls, width, color=COLORS['seed'], label='Sadece Seed', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x_pos + width/2, aug_recalls, width, color=COLORS['gan'], label='Seed + GAN', edgecolor='white', linewidth=0.5)
ax.set_title('Recall (Anomali Yakalama) Karşılaştırması', color=COLORS['text'], fontweight='bold', fontsize=13)
ax.set_ylabel('Recall', color=COLORS['text'])
ax.set_xticks(x_pos)
ax.set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=8)
ax.legend(facecolor=COLORS['bg'], edgecolor=COLORS['grid'])
ax.set_facecolor(COLORS['bg'])
ax.tick_params(colors=COLORS['text'])
ax.grid(True, alpha=0.1, axis='y')
ax.set_ylim(0, 1.1)

for bar, val in zip(bars1, seed_recalls):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{val:.2f}', ha='center', va='bottom', color=COLORS['text'], fontsize=8, fontweight='bold')
for bar, val in zip(bars2, aug_recalls):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{val:.2f}', ha='center', va='bottom', color=COLORS['text'], fontsize=8, fontweight='bold')

# İyileşme yüzdesi
ax = fig.add_subplot(gs[0, 2], facecolor=COLORS['bg'])
improvements = [results[m]['f1_improvement'] for m in model_names]
colors_bar = [COLORS['seed'] if imp >= 0 else COLORS['spike'] for imp in improvements]
bars = ax.bar(model_names, improvements, color=colors_bar, edgecolor='white', linewidth=0.5)
ax.axhline(y=0, color='white', linewidth=0.5)
ax.axhline(y=15, color='#fbbf24', linestyle='--', alpha=0.5, label='Hedef: +15%')
ax.set_title('F1 Score İyileşme (%)', color=COLORS['text'], fontweight='bold', fontsize=13)
ax.set_ylabel('İyileşme %', color=COLORS['text'])
ax.set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=8)
ax.legend(facecolor=COLORS['bg'], edgecolor=COLORS['grid'])
ax.set_facecolor(COLORS['bg'])
ax.tick_params(colors=COLORS['text'])
ax.grid(True, alpha=0.1, axis='y')

for bar, val in zip(bars, improvements):
    y_pos = bar.get_height() + (1 if val >= 0 else -3)
    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
            f'{val:+.1f}%', ha='center', va='bottom', color=COLORS['text'], fontweight='bold', fontsize=10)

# Confusion Matrix - Seed
ax = fig.add_subplot(gs[1, 0], facecolor=COLORS['bg'])
cm_seed = confusion_matrix(y_test, results[best_model]['y_pred_seed'])
class_names_short = ['NOR', 'SPK', 'DRF', 'DRP', 'FRZ', 'NSE']
im = ax.imshow(cm_seed, cmap='Greens', aspect='auto')
ax.set_title(f'Confusion Matrix - Sadece Seed', color=COLORS['seed'], fontweight='bold')
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(class_names_short, color=COLORS['text'], fontsize=9)
ax.set_yticklabels(class_names_short, color=COLORS['text'], fontsize=9)
ax.set_xlabel('Tahmin', color=COLORS['text'])
ax.set_ylabel('Gerçek', color=COLORS['text'])
for i in range(6):
    for j in range(6):
        color = 'white' if cm_seed[i, j] > cm_seed.max() * 0.5 else 'black'
        ax.text(j, i, str(cm_seed[i, j]), ha='center', va='center', color=color, fontweight='bold')

# Confusion Matrix - Augmented
ax = fig.add_subplot(gs[1, 1], facecolor=COLORS['bg'])
cm_aug = confusion_matrix(y_test, results[best_model]['y_pred_aug'])
im = ax.imshow(cm_aug, cmap='RdPu', aspect='auto')
ax.set_title(f'Confusion Matrix - Seed + GAN', color=COLORS['gan'], fontweight='bold')
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(class_names_short, color=COLORS['text'], fontsize=9)
ax.set_yticklabels(class_names_short, color=COLORS['text'], fontsize=9)
ax.set_xlabel('Tahmin', color=COLORS['text'])
ax.set_ylabel('Gerçek', color=COLORS['text'])
for i in range(6):
    for j in range(6):
        color = 'white' if cm_aug[i, j] > cm_aug.max() * 0.5 else 'black'
        ax.text(j, i, str(cm_aug[i, j]), ha='center', va='center', color=color, fontweight='bold')

# Sınıf bazlı F1 karşılaştırması
ax = fig.add_subplot(gs[1, 2], facecolor=COLORS['bg'])

# Her sınıf için ayrı F1
from sklearn.metrics import f1_score as f1_per_class
f1_seed_per = f1_per_class(y_test, results[best_model]['y_pred_seed'], average=None)
f1_aug_per = f1_per_class(y_test, results[best_model]['y_pred_aug'], average=None)

class_names_full = [LABEL_NAMES[i] for i in range(6)]
x_pos = np.arange(6)
width = 0.35
bars1 = ax.bar(x_pos - width/2, f1_seed_per, width, color=COLORS['seed'], label='Sadece Seed', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x_pos + width/2, f1_aug_per, width, color=COLORS['gan'], label='Seed + GAN', edgecolor='white', linewidth=0.5)
ax.set_title(f'Sınıf Bazlı F1 ({best_model})', color=COLORS['text'], fontweight='bold', fontsize=12)
ax.set_ylabel('F1 Score', color=COLORS['text'])
ax.set_xticks(x_pos)
ax.set_xticklabels(class_names_full, fontsize=8, rotation=15)
ax.legend(facecolor=COLORS['bg'], edgecolor=COLORS['grid'], fontsize=8)
ax.set_facecolor(COLORS['bg'])
ax.tick_params(colors=COLORS['text'])
ax.grid(True, alpha=0.1, axis='y')
ax.set_ylim(0, 1.2)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(f'{OUTPUT_DIR}/08_utility_test_results.png', dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("  ✅ 08_utility_test_results.png")

# ============================================================
# 5. ÖZET RAPOR
# ============================================================
print("\n" + "=" * 60)
print("📊 SONUÇ RAPORU")
print("=" * 60)

print(f"""
  ╔══════════════════════════════════════════════════════════╗
  ║              UTİLİTY TESTİ SONUÇLARI                   ║
  ╠══════════════════════════════════════════════════════════╣
  ║                                                        ║
  ║  Test Seti: Napoli (GAN'ın hiç görmediği şehir)        ║
  ║  Test Boyutu: {len(X_test)} örnek (6 sınıf × {N_TEST} örnek)           ║
  ║                                                        ║""")

for model_name, res in results.items():
    status = "✅ İYİLEŞTİ" if res['f1_improvement'] > 0 else "❌ KÖTÜLEŞ"
    print(f"  ║  {model_name:25s}                         ║")
    print(f"  ║    Seed F1: {res['seed_f1']:.4f} → Augmented F1: {res['aug_f1']:.4f}  ║")
    print(f"  ║    İyileşme: {res['f1_improvement']:+.1f}% {status:20s}    ║")
    print(f"  ║                                                        ║")

# Ortalama iyileşme
avg_improvement = np.mean([r['f1_improvement'] for r in results.values()])
avg_recall_imp = np.mean([r['recall_improvement'] for r in results.values()])

print(f"""  ╠══════════════════════════════════════════════════════════╣
  ║  ORTALAMA İYİLEŞME:                                   ║
  ║    F1 Score:  {avg_improvement:+.1f}%                                   ║
  ║    Recall:    {avg_recall_imp:+.1f}%                                   ║
  ╚══════════════════════════════════════════════════════════╝
""")

# Sonuç kaydet
summary = {
    'test_city': 'Napoli',
    'test_size': len(X_test),
    'seed_size': len(X_seed),
    'gan_size': len(X_gan),
    'models': {}
}
for m, r in results.items():
    summary['models'][m] = {
        'seed_f1': round(r['seed_f1'], 4),
        'aug_f1': round(r['aug_f1'], 4),
        'f1_improvement_pct': round(r['f1_improvement'], 2),
        'seed_recall': round(r['seed_recall'], 4),
        'aug_recall': round(r['aug_recall'], 4)
    }

import json
with open(f'{OUTPUT_DIR}/utility_test_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("  💾 utility_test_results.json kaydedildi")

print("\n🎉 FAZ 3 TAMAMLANDI!")
