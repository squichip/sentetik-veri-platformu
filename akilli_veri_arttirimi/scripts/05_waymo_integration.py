"""
============================================================
 Akıllı Veri Üretimi - Faz 5: Waymo Entegrasyonu
 Çoklu Veri Kaynağı + Sensör Füzyon Anomali Tespiti
============================================================
 
 Bu script:
 1. Waymo Motion Dataset'ten gerçek otonom araç yörüngelerini çıkarır
 2. Waymo'nun zengin sensör verisine (x,y,z,speed,velocity,yaw) anomali inject eder
 3. FCD + Waymo birleşik pipeline çalıştırır
 4. Multi-source karşılaştırma yapar
============================================================
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, classification_report, confusion_matrix
from sklearn.base import clone
import os, warnings, json
warnings.filterwarnings('ignore')

OUTPUT_DIR = "outputs"
np.random.seed(42)

plt.style.use('dark_background')
COLORS = {
    'fcd': '#10b981', 'waymo': '#f59e0b', 'combined': '#8b5cf6',
    'bg': '#0f172a', 'text': '#f1f5f9', 'grid': '#1e293b',
    'normal': '#10b981', 'spike': '#ef4444', 'drift': '#f59e0b',
    'dropout': '#8b5cf6', 'freeze': '#3b82f6', 'noise': '#06b6d4',
    'generated': '#f472b6'
}

LABEL_MAP = {'normal': 0, 'spike': 1, 'drift': 2, 'dropout': 3, 'freeze': 4, 'noise': 5}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}

print("=" * 60)
print("🚗 FAZ 5: WAYMO ENTEGRASYONU + MULTI-SOURCE PİPELİNE")
print("=" * 60)

# ============================================================
# 1. WAYMO VERİSİNİ PARSE ET
# ============================================================
print("\n📡 Waymo Motion Dataset parsing...")

WAYMO_FILE = 'data/waymo/validation_tfexample.tfrecord-00000-of-00150'

# Feature description for Waymo Motion Dataset
features_description = {
    # Geçmiş (10 adım, 100ms aralıklı = 1 saniye geçmiş)
    'state/past/x': tf.io.FixedLenFeature([1280], tf.float32, default_value=[0.0]*1280),
    'state/past/y': tf.io.FixedLenFeature([1280], tf.float32, default_value=[0.0]*1280),
    'state/past/speed': tf.io.FixedLenFeature([1280], tf.float32, default_value=[0.0]*1280),
    'state/past/velocity_x': tf.io.FixedLenFeature([1280], tf.float32, default_value=[0.0]*1280),
    'state/past/velocity_y': tf.io.FixedLenFeature([1280], tf.float32, default_value=[0.0]*1280),
    'state/past/vel_yaw': tf.io.FixedLenFeature([1280], tf.float32, default_value=[0.0]*1280),
    'state/past/valid': tf.io.FixedLenFeature([1280], tf.int64, default_value=[0]*1280),
    
    # Şu an (1 adım)
    'state/current/x': tf.io.FixedLenFeature([128], tf.float32, default_value=[0.0]*128),
    'state/current/y': tf.io.FixedLenFeature([128], tf.float32, default_value=[0.0]*128),
    'state/current/speed': tf.io.FixedLenFeature([128], tf.float32, default_value=[0.0]*128),
    'state/current/velocity_x': tf.io.FixedLenFeature([128], tf.float32, default_value=[0.0]*128),
    'state/current/velocity_y': tf.io.FixedLenFeature([128], tf.float32, default_value=[0.0]*128),
    'state/current/vel_yaw': tf.io.FixedLenFeature([128], tf.float32, default_value=[0.0]*128),
    'state/current/valid': tf.io.FixedLenFeature([128], tf.int64, default_value=[0]*128),
    
    # Gelecek (80 adım = 8 saniye gelecek)
    'state/future/x': tf.io.FixedLenFeature([10240], tf.float32, default_value=[0.0]*10240),
    'state/future/y': tf.io.FixedLenFeature([10240], tf.float32, default_value=[0.0]*10240),
    'state/future/speed': tf.io.FixedLenFeature([10240], tf.float32, default_value=[0.0]*10240),
    'state/future/velocity_x': tf.io.FixedLenFeature([10240], tf.float32, default_value=[0.0]*10240),
    'state/future/velocity_y': tf.io.FixedLenFeature([10240], tf.float32, default_value=[0.0]*10240),
    'state/future/vel_yaw': tf.io.FixedLenFeature([10240], tf.float32, default_value=[0.0]*10240),
    'state/future/valid': tf.io.FixedLenFeature([10240], tf.int64, default_value=[0]*10240),
    
    # Meta
    'state/type': tf.io.FixedLenFeature([128], tf.float32, default_value=[0.0]*128),
    'state/is_sdc': tf.io.FixedLenFeature([128], tf.int64, default_value=[0]*128),
    'state/id': tf.io.FixedLenFeature([128], tf.float32, default_value=[0.0]*128),
}

N_AGENTS = 128  # Her senaryoda max 128 ajan
N_PAST = 10     # 10 geçmiş adım
N_FUTURE = 80   # 80 gelecek adım
N_STEPS = N_PAST + 1 + N_FUTURE  # 91 toplam adım

dataset = tf.data.TFRecordDataset(WAYMO_FILE, compression_type='')

waymo_trajectories = []
waymo_speeds = []
waymo_meta = []
n_scenarios = 0

print("  Senaryolar parse ediliyor...")

try:
  for raw_record in dataset:
    try:
        example = tf.io.parse_single_example(raw_record, features_description)
    except Exception as e:
        print(f"    ⚠️ Parse hatası atlandı")
        continue
    
    # Her senaryodaki ajanları işle
    for agent_idx in range(N_AGENTS):
        # Geçerliliği kontrol et
        past_valid = example['state/past/valid'].numpy().reshape(N_AGENTS, N_PAST)
        current_valid = example['state/current/valid'].numpy().reshape(N_AGENTS, 1)
        future_valid = example['state/future/valid'].numpy().reshape(N_AGENTS, N_FUTURE)
        
        all_valid = np.concatenate([past_valid[agent_idx], current_valid[agent_idx], future_valid[agent_idx]])
        
        # En az 20 geçerli adım olan ajanları al
        if np.sum(all_valid) < 20:
            continue
        
        # Koordinatları birleştir: past + current + future
        past_x = example['state/past/x'].numpy().reshape(N_AGENTS, N_PAST)[agent_idx]
        cur_x = example['state/current/x'].numpy().reshape(N_AGENTS, 1)[agent_idx]
        fut_x = example['state/future/x'].numpy().reshape(N_AGENTS, N_FUTURE)[agent_idx]
        
        past_y = example['state/past/y'].numpy().reshape(N_AGENTS, N_PAST)[agent_idx]
        cur_y = example['state/current/y'].numpy().reshape(N_AGENTS, 1)[agent_idx]
        fut_y = example['state/future/y'].numpy().reshape(N_AGENTS, N_FUTURE)[agent_idx]
        
        past_speed = example['state/past/speed'].numpy().reshape(N_AGENTS, N_PAST)[agent_idx]
        cur_speed = example['state/current/speed'].numpy().reshape(N_AGENTS, 1)[agent_idx]
        fut_speed = example['state/future/speed'].numpy().reshape(N_AGENTS, N_FUTURE)[agent_idx]
        
        past_vx = example['state/past/velocity_x'].numpy().reshape(N_AGENTS, N_PAST)[agent_idx]
        cur_vx = example['state/current/velocity_x'].numpy().reshape(N_AGENTS, 1)[agent_idx]
        fut_vx = example['state/future/velocity_x'].numpy().reshape(N_AGENTS, N_FUTURE)[agent_idx]
        
        past_vy = example['state/past/velocity_y'].numpy().reshape(N_AGENTS, N_PAST)[agent_idx]
        cur_vy = example['state/current/velocity_y'].numpy().reshape(N_AGENTS, 1)[agent_idx]
        fut_vy = example['state/future/velocity_y'].numpy().reshape(N_AGENTS, N_FUTURE)[agent_idx]
        
        full_x = np.concatenate([past_x, cur_x, fut_x])
        full_y = np.concatenate([past_y, cur_y, fut_y])
        full_speed = np.concatenate([past_speed, cur_speed, fut_speed])
        full_vx = np.concatenate([past_vx, cur_vx, fut_vx])
        full_vy = np.concatenate([past_vy, cur_vy, fut_vy])
        full_valid = all_valid
        
        # 20 adımlık pencerelere böl (FCD ile uyumlu)
        # 20 adımlık pencerelere böl (FCD ile uyumlu)
        for start in range(0, len(full_x) - 20 + 1, 10):
            window_valid = full_valid[start:start+20]
            if np.sum(window_valid) < 18:
                continue
                
            x_w = full_x[start:start+20]
            y_w = full_y[start:start+20]
            speed_w = full_speed[start:start+20]
            vx_w = full_vx[start:start+20]
            vy_w = full_vy[start:start+20]
            
            # 🚨 1. KALKAN: Işık Hızına Çıkan Araçları Çöpe At (Max 45 m/s)
            if np.max(speed_w) > 45.0 or np.min(speed_w) < 0.0:
                continue
                
            # 🚨 2. KALKAN: Işınlanan / GPS Hatası Olan Araçları Çöpe At (Bir adımda max 5 metre)
            dx, dy = np.diff(x_w), np.diff(y_w)
            step_dist = np.sqrt(dx**2 + dy**2)
            if len(step_dist) > 0 and np.max(step_dist) > 5.0:
                continue

            # Normalize: Merkezle
            x_w = x_w - x_w[0]
            y_w = y_w - y_w[0]
            
            # 🚨 3. KALKAN: Merkeze rağmen absürt koordinatlara fırlayanları çöpe at
            if np.max(np.abs(x_w)) > 300.0 or np.max(np.abs(y_w)) > 300.0:
                continue
            
            # Sıfır yörüngeler (duran araçlar) filtrele
            if np.max(np.abs(x_w)) < 0.5 and np.max(np.abs(y_w)) < 0.5:
                continue
            
            # Yörünge vektörü: x(20) + y(20) + speed(20) + vx(20) + vy(20) = 100 boyut
            traj = np.concatenate([x_w, y_w, speed_w, vx_w, vy_w])
            waymo_trajectories.append(traj)
    
    n_scenarios += 1
    if n_scenarios % 20 == 0:
        print(f"    {n_scenarios} senaryo işlendi, {len(waymo_trajectories)} yörünge çıkarıldı")
    
    # İlk 100 senaryo yeterli (binlerce yörünge üretir)
    if n_scenarios >= 100:
        break
except Exception as e:
    print(f"    ⚠️ TFRecord okuma hatası, {n_scenarios} senaryo ile devam ediliyor: {type(e).__name__}")

waymo_data = np.array(waymo_trajectories, dtype=np.float32)
print(f"\n  ✅ Waymo: {len(waymo_data)} yörünge çıkarıldı ({n_scenarios} senaryodan)")
print(f"  📐 Boyut: {waymo_data.shape} (20 x + 20 y + 20 speed + 20 vx + 20 vy)")

# --- STEP 2: SAVE CLEANED WAYMO SEED ---
print("\n💾 Temizlenmiş Waymo Seed kaydediliyor...")
cols = [f'x({i+1})' for i in range(20)] + [f'y({i+1})' for i in range(20)] + \
       [f'speed({i+1})' for i in range(20)] + [f'vx({i+1})' for i in range(20)] + \
       [f'vy({i+1})' for i in range(20)]

df_waymo = pd.DataFrame(waymo_data, columns=cols)
df_waymo['label'] = 'normal'
df_waymo.to_csv('waymo_seed_MASSIVE.csv', index=False)
print(f"  ✨ BAŞARILI: waymo_seed_MASSIVE.csv (Boyut: {df_waymo.shape}) üretildi!")

# ============================================================
# 2. WAYMO FEATURE ENGINEERING 
# ============================================================
print("\n🔧 Waymo Feature Engineering...")

def extract_waymo_features(traj):
    """100 boyutlu waymo verisinden özellik çıkar"""
    x = traj[0:20]
    y = traj[20:40]
    speed = traj[40:60]
    vx = traj[60:80]
    vy = traj[80:100]
    
    features = {}
    
    # Konum bazlı
    dx = np.diff(x)
    dy = np.diff(y)
    pos_speed = np.sqrt(dx**2 + dy**2)
    
    features['pos_speed_mean'] = np.mean(pos_speed)
    features['pos_speed_std'] = np.std(pos_speed)
    features['pos_speed_max'] = np.max(pos_speed)
    features['pos_speed_cv'] = np.std(pos_speed) / (np.mean(pos_speed) + 1e-8)
    
    # Sensör hız verisi
    features['sensor_speed_mean'] = np.mean(speed)
    features['sensor_speed_std'] = np.std(speed)
    features['sensor_speed_max'] = np.max(speed)
    features['sensor_speed_range'] = np.max(speed) - np.min(speed)
    
    # Hız tutarsızlığı (konum->hız vs sensör hızı)
    speed_diff = np.abs(pos_speed - speed[1:])  # 19 değer
    features['speed_inconsistency'] = np.mean(speed_diff)
    features['speed_inconsistency_max'] = np.max(speed_diff)
    
    # İvme
    accel = np.diff(speed)
    features['accel_mean'] = np.mean(accel)
    features['accel_std'] = np.std(accel)
    features['accel_max'] = np.max(np.abs(accel))
    
    # Jerk
    jerk = np.diff(accel)
    features['jerk_mean'] = np.mean(np.abs(jerk))
    features['jerk_max'] = np.max(np.abs(jerk))
    
    # Velocity vektör analizi
    v_mag = np.sqrt(vx**2 + vy**2)
    features['v_mag_mean'] = np.mean(v_mag)
    features['v_mag_std'] = np.std(v_mag)
    
    # Hız yönü değişimi
    v_angle = np.arctan2(vy, vx + 1e-8)
    angle_change = np.abs(np.diff(v_angle))
    features['direction_change_mean'] = np.mean(angle_change)
    features['direction_change_max'] = np.max(angle_change)
    features['n_sharp_turns'] = np.sum(angle_change > np.pi / 4)
    
    # Spike tespiti
    q1, q3 = np.percentile(pos_speed, [25, 75])
    iqr = q3 - q1
    features['n_speed_outliers'] = np.sum(pos_speed > q3 + 1.5 * iqr)
    features['max_speed_zscore'] = (np.max(pos_speed) - np.mean(pos_speed)) / (np.std(pos_speed) + 1e-8)
    features['jump_ratio'] = np.max(pos_speed) / (np.median(pos_speed) + 1e-8)
    
    # Drift tespiti
    dist_from_start = np.sqrt(x**2 + y**2)
    t = np.arange(20)
    corr = np.corrcoef(t, dist_from_start)[0, 1]
    features['drift_trend'] = corr if not np.isnan(corr) else 0
    
    # Freeze tespiti
    near_zero = np.sum(pos_speed < 0.01)
    features['n_zero_speed'] = near_zero
    features['zero_speed_ratio'] = near_zero / len(pos_speed)
    max_freeze = 0
    cur_freeze = 0
    for s in pos_speed:
        if s < 0.01:
            cur_freeze += 1
            max_freeze = max(max_freeze, cur_freeze)
        else:
            cur_freeze = 0
    features['max_freeze_length'] = max_freeze
    
    # Noise tespiti
    dir_changes_x = np.sum(np.diff(np.sign(dx)) != 0)
    dir_changes_y = np.sum(np.diff(np.sign(dy)) != 0)
    features['direction_changes'] = dir_changes_x + dir_changes_y
    
    # Genel
    features['total_distance'] = np.sum(pos_speed)
    features['net_displacement'] = np.sqrt(x[-1]**2 + y[-1]**2)
    features['path_efficiency'] = features['net_displacement'] / (features['total_distance'] + 1e-8)
    
    # Sensör füzyon tutarsızlığı
    sensor_pos_ratio = np.mean(speed) / (np.mean(pos_speed) + 1e-8)
    features['sensor_fusion_ratio'] = sensor_pos_ratio
    
    return features

# Test
test_feat = extract_waymo_features(waymo_data[0])
WAYMO_FEATURES = list(test_feat.keys())
print(f"  ✅ {len(WAYMO_FEATURES)} özellik per yörünge")

# ============================================================
# 3. WAYMO ANOMALİ INJECTION
# ============================================================
print("\n⚡ Waymo Anomali Injection...")

def waymo_inject_spike(traj):
    t = traj.copy()
    for _ in range(np.random.randint(1, 4)):
        idx = np.random.randint(0, 20)
        # Konum spike
        t[idx] += np.random.choice([-1,1]) * np.random.uniform(5, 20)
        t[20+idx] += np.random.choice([-1,1]) * np.random.uniform(5, 20)
        # Hız spike
        t[40+idx] *= np.random.uniform(3, 8)
    return t

def waymo_inject_drift(traj):
    t = traj.copy()
    start = np.random.randint(5, 12)
    rate = np.random.uniform(0.3, 1.0)
    angle = np.random.uniform(0, 2*np.pi)
    for i in range(start, 20):
        t[i] += (i-start) * rate * np.cos(angle)
        t[20+i] += (i-start) * rate * np.sin(angle)
    return t

def waymo_inject_freeze(traj):
    t = traj.copy()
    start = np.random.randint(4, 12)
    dur = np.random.randint(4, 10)
    for i in range(start, min(start+dur, 20)):
        t[i] = t[start]       # x freeze
        t[20+i] = t[20+start] # y freeze
        t[40+i] = 0           # speed = 0
        t[60+i] = 0           # vx = 0
        t[80+i] = 0           # vy = 0
    return t

def waymo_inject_noise(traj):
    t = traj.copy()
    start = np.random.randint(3, 8)
    for i in range(start, 20):
        factor = 0.5 + (i-start) * 0.3
        t[i] += np.random.normal(0, factor)
        t[20+i] += np.random.normal(0, factor)
        t[40+i] += np.random.normal(0, factor * 0.5)
    return t

def waymo_inject_dropout(traj):
    t = traj.copy()
    n = int(20 * np.random.uniform(0.2, 0.4))
    start = np.random.randint(3, 20 - n - 2)
    for i in range(start, start+n):
        t[i] = t[start-1]
        t[20+i] = t[20+start-1]
        t[40+i] = 0
        t[60+i] = 0
        t[80+i] = 0
    t[start+n] += np.random.uniform(-3, 3)
    t[20+start+n] += np.random.uniform(-3, 3)
    return t

waymo_generators = {1: waymo_inject_spike, 2: waymo_inject_drift, 
                    3: waymo_inject_dropout, 4: waymo_inject_freeze, 5: waymo_inject_noise}

# Veri oluştur
N_PER = 200
waymo_all = []
waymo_labels = []

# Normal
indices = np.random.choice(len(waymo_data), N_PER, replace=False)
for idx in indices:
    waymo_all.append(waymo_data[idx])
    waymo_labels.append(0)

# Anomaliler
pool_indices = np.random.choice(len(waymo_data), N_PER * 5, replace=True)
for cid, gen_func in waymo_generators.items():
    for i in range(N_PER):
        idx = pool_indices[(cid-1)*N_PER + i]
        waymo_all.append(gen_func(waymo_data[idx]))
        waymo_labels.append(cid)

X_waymo_raw = np.array(waymo_all, dtype=np.float32)
y_waymo = np.array(waymo_labels)
print(f"  ✅ {len(X_waymo_raw)} Waymo yörünge ({N_PER} per sınıf)")

# Feature extraction
waymo_feat_list = []
for row in X_waymo_raw:
    feat = extract_waymo_features(row)
    waymo_feat_list.append(list(feat.values()))
X_waymo_feat = np.nan_to_num(np.array(waymo_feat_list, dtype=np.float32), nan=0, posinf=1e6, neginf=-1e6)
print(f"  ✅ Features: {X_waymo_feat.shape}")

# ============================================================
# 4. MULTI-SOURCE DENEY
# ============================================================
print("\n" + "=" * 60)
print("🧪 MULTI-SOURCE DENEY")
print("=" * 60)
print("  Deney 1: Sadece FCD (İtalya)")
print("  Deney 2: Sadece Waymo")
print("  Deney 3: FCD + Waymo birleşik")

# FCD verisi hazırla
# FCD verisi inline olarak hazırlanıyor

# FCD yükle
x_cols = None
fcd_cities = {
    'Milano': 'data/fcd-italy/milan_fcd_prep.csv',
    'Roma': 'data/fcd-italy/rome_fcd_prep.csv',
    'Torino': 'data/fcd-italy/turin_fcd_prep.csv',
}

all_fcd = []
for city, path in fcd_cities.items():
    df = pd.read_csv(path)
    if x_cols is None:
        x_cols = [c for c in df.columns if c.startswith('x(')]
        y_cols = [c for c in df.columns if c.startswith('y(')]
    clean = df[~df[x_cols + y_cols].isna().any(axis=1)]
    all_fcd.append(clean)

fcd_pool = pd.concat(all_fcd).sample(frac=1, random_state=42).reset_index(drop=True)

# FCD Feature engineering (aynı Waymo özellik kümesini kullan)
def extract_fcd_as_waymo_features(row):
    """FCD verisinden Waymo uyumlu özellikler çıkar"""
    x = np.array([row[c] for c in x_cols])
    y = np.array([row[c] for c in y_cols])
    
    # Speed, vx, vy hesapla (FCD'de yok, biz hesaplayalım)
    dx = np.diff(x)
    dy = np.diff(y)
    speed = np.sqrt(dx**2 + dy**2)
    speed = np.concatenate([[speed[0]], speed])  # 20 adıma tamamla
    vx = np.concatenate([[dx[0]], dx])
    vy = np.concatenate([[dy[0]], dy])
    
    traj = np.concatenate([x, y, speed, vx, vy])  # 100 boyut
    return traj

# FCD anomali injection
def fcd_inject_spike(x, y):
    xa, ya = x.copy(), y.copy()
    for _ in range(np.random.randint(1, 4)):
        idx = np.random.randint(0, len(x))
        xa[idx] += np.random.choice([-1,1]) * np.random.uniform(0.2, 0.6)
        ya[idx] += np.random.choice([-1,1]) * np.random.uniform(0.2, 0.6)
    return xa, ya

def fcd_inject_drift(x, y):
    xa, ya = x.copy(), y.copy()
    start = np.random.randint(5, 12)
    rate = np.random.uniform(0.01, 0.04)
    angle = np.random.uniform(0, 2*np.pi)
    for i in range(start, len(x)):
        xa[i] += (i-start) * rate * np.cos(angle)
        ya[i] += (i-start) * rate * np.sin(angle)
    return xa, ya

def fcd_inject_freeze(x, y):
    xa, ya = x.copy(), y.copy()
    start = np.random.randint(4, 12)
    dur = np.random.randint(4, 10)
    for i in range(start, min(start+dur, len(x))):
        xa[i] = xa[start]; ya[i] = ya[start]
    return xa, ya

def fcd_inject_noise(x, y):
    xa, ya = x.copy(), y.copy()
    start = np.random.randint(3, 8)
    f = np.random.uniform(0.03, 0.1)
    for i in range(start, len(x)):
        xa[i] += np.random.normal(0, f*(1+(i-start)*0.25))
        ya[i] += np.random.normal(0, f*(1+(i-start)*0.25))
    return xa, ya

def fcd_inject_dropout(x, y):
    xa, ya = x.copy(), y.copy()
    n = int(len(x) * np.random.uniform(0.2, 0.4))
    start = np.random.randint(3, len(x) - n - 2)
    for i in range(start, start+n):
        xa[i] = xa[start-1]; ya[i] = ya[start-1]
    return xa, ya

fcd_generators = {1: fcd_inject_spike, 2: fcd_inject_drift, 
                  3: fcd_inject_dropout, 4: fcd_inject_freeze, 5: fcd_inject_noise}

# FCD veri oluştur
N_FCD = 200
fcd_all_raw = []
fcd_labels = []

for i in range(N_FCD):
    row = fcd_pool.iloc[i]
    traj = extract_fcd_as_waymo_features(row)
    fcd_all_raw.append(traj)
    fcd_labels.append(0)

for cid, gen_func in fcd_generators.items():
    for i in range(N_FCD):
        row = fcd_pool.iloc[N_FCD + (cid-1)*N_FCD + i]
        x = np.array([row[c] for c in x_cols])
        y = np.array([row[c] for c in y_cols])
        xa, ya = gen_func(x, y)
        
        dx = np.diff(xa)
        dy = np.diff(ya)
        speed = np.sqrt(dx**2 + dy**2)
        speed = np.concatenate([[speed[0]], speed])
        vx = np.concatenate([[dx[0]], dx])
        vy = np.concatenate([[dy[0]], dy])
        traj = np.concatenate([xa, ya, speed, vx, vy])
        fcd_all_raw.append(traj)
        fcd_labels.append(cid)

X_fcd_raw = np.array(fcd_all_raw, dtype=np.float32)
y_fcd = np.array(fcd_labels)

# FCD feature extraction
fcd_feat_list = []
for row in X_fcd_raw:
    feat = extract_waymo_features(row)  # Aynı fonksiyon!
    fcd_feat_list.append(list(feat.values()))
X_fcd_feat = np.nan_to_num(np.array(fcd_feat_list, dtype=np.float32), nan=0, posinf=1e6, neginf=-1e6)

print(f"\n  📊 FCD:   {X_fcd_feat.shape}")
print(f"  📊 Waymo: {X_waymo_feat.shape}")

# Train/test split
from sklearn.model_selection import train_test_split

# FCD: %70 train, %30 test
X_fcd_train, X_fcd_test, y_fcd_train, y_fcd_test = train_test_split(
    X_fcd_feat, y_fcd, test_size=0.3, random_state=42, stratify=y_fcd)

# Waymo: %70 train, %30 test
X_waymo_train, X_waymo_test, y_waymo_train, y_waymo_test = train_test_split(
    X_waymo_feat, y_waymo, test_size=0.3, random_state=42, stratify=y_waymo)

# Combined
X_combined_train = np.vstack([X_fcd_train, X_waymo_train])
y_combined_train = np.concatenate([y_fcd_train, y_waymo_train])
X_combined_test = np.vstack([X_fcd_test, X_waymo_test])
y_combined_test = np.concatenate([y_fcd_test, y_waymo_test])

# Scale
scaler_fcd = StandardScaler().fit(X_fcd_train)
scaler_waymo = StandardScaler().fit(X_waymo_train)
scaler_combined = StandardScaler().fit(X_combined_train)

X_fcd_train_s = scaler_fcd.transform(X_fcd_train)
X_fcd_test_s = scaler_fcd.transform(X_fcd_test)
X_waymo_train_s = scaler_waymo.transform(X_waymo_train)
X_waymo_test_s = scaler_waymo.transform(X_waymo_test)
X_combined_train_s = scaler_combined.transform(X_combined_train)
X_combined_test_s = scaler_combined.transform(X_combined_test)

print(f"\n  📊 FCD Train: {X_fcd_train_s.shape} | Test: {X_fcd_test_s.shape}")
print(f"  📊 Waymo Train: {X_waymo_train_s.shape} | Test: {X_waymo_test_s.shape}")
print(f"  📊 Combined Train: {X_combined_train_s.shape} | Test: {X_combined_test_s.shape}")

# Modeller
model_template = GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42)

experiments = {}

# Deney 1: FCD only
print("\n  🇮🇹 Deney 1: Sadece FCD...")
m1 = clone(model_template)
m1.fit(X_fcd_train_s, y_fcd_train)
pred_fcd = m1.predict(X_fcd_test_s)
f1_fcd = f1_score(y_fcd_test, pred_fcd, average='macro')
r_fcd = recall_score(y_fcd_test, pred_fcd, average='macro')
experiments['FCD (İtalya)'] = {'f1': f1_fcd, 'recall': r_fcd, 'pred': pred_fcd, 'y_true': y_fcd_test}
print(f"     F1: {f1_fcd:.4f} | Recall: {r_fcd:.4f}")

# Deney 2: Waymo only
print("  🚗 Deney 2: Sadece Waymo...")
m2 = clone(model_template)
m2.fit(X_waymo_train_s, y_waymo_train)
pred_waymo = m2.predict(X_waymo_test_s)
f1_waymo = f1_score(y_waymo_test, pred_waymo, average='macro')
r_waymo = recall_score(y_waymo_test, pred_waymo, average='macro')
experiments['Waymo'] = {'f1': f1_waymo, 'recall': r_waymo, 'pred': pred_waymo, 'y_true': y_waymo_test}
print(f"     F1: {f1_waymo:.4f} | Recall: {r_waymo:.4f}")

# Deney 3: Combined
print("  🌍 Deney 3: FCD + Waymo Birleşik...")
m3 = clone(model_template)
m3.fit(X_combined_train_s, y_combined_train)
pred_combined = m3.predict(X_combined_test_s)
f1_combined = f1_score(y_combined_test, pred_combined, average='macro')
r_combined = recall_score(y_combined_test, pred_combined, average='macro')
experiments['FCD + Waymo'] = {'f1': f1_combined, 'recall': r_combined, 'pred': pred_combined, 'y_true': y_combined_test}
print(f"     F1: {f1_combined:.4f} | Recall: {r_combined:.4f}")

# Cross-domain test: Waymo'da eğit, FCD'de test et
print("\n  🔄 Deney 4: Cross-Domain (Waymo'da eğit → FCD'de test)...")
X_fcd_test_cross = scaler_waymo.transform(X_fcd_test)
pred_cross = m2.predict(X_fcd_test_cross)
f1_cross = f1_score(y_fcd_test, pred_cross, average='macro')
r_cross = recall_score(y_fcd_test, pred_cross, average='macro')
experiments['Cross-Domain'] = {'f1': f1_cross, 'recall': r_cross, 'pred': pred_cross, 'y_true': y_fcd_test}
print(f"     F1: {f1_cross:.4f} | Recall: {r_cross:.4f}")

# ============================================================
# 5. GÖRSELLEŞTİRME
# ============================================================
print("\n🎨 Multi-source grafikleri oluşturuluyor...")

fig = plt.figure(figsize=(24, 20), facecolor=COLORS['bg'])
fig.suptitle('Multi-Source Anomali Tespiti: FCD + Waymo', 
             fontsize=20, fontweight='bold', color=COLORS['text'], y=0.99)

gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

# Satır 1: Waymo yörünge örnekleri
ax = fig.add_subplot(gs[0, 0], facecolor=COLORS['bg'])
for i in range(20):
    ax.plot(waymo_data[i, :20], waymo_data[i, 20:40], alpha=0.5, linewidth=1, color=COLORS['normal'])
ax.set_title('Waymo Normal Yörüngeler', color=COLORS['waymo'], fontweight='bold')
ax.set_xlabel('X (metre)', color=COLORS['text'])
ax.set_ylabel('Y (metre)', color=COLORS['text'])
ax.tick_params(colors=COLORS['text'])
ax.grid(True, alpha=0.1)

# Waymo anomali örnekleri
for plot_idx, (cid, name, color) in enumerate([(1,'Spike','spike'), (2,'Drift','drift'), (4,'Freeze','freeze')]):
    ax = fig.add_subplot(gs[0, 1+plot_idx], facecolor=COLORS['bg'])
    mask = y_waymo == cid
    samples = X_waymo_raw[mask]
    for i in range(min(15, len(samples))):
        ax.plot(samples[i, :20], samples[i, 20:40], alpha=0.5, linewidth=1, color=COLORS[color])
    ax.set_title(f'Waymo {name}', color=COLORS[color], fontweight='bold')
    ax.tick_params(colors=COLORS['text'])
    ax.grid(True, alpha=0.1)

# Satır 2: F1 + Recall karşılaştırma
ax = fig.add_subplot(gs[1, :2], facecolor=COLORS['bg'])
exp_names = list(experiments.keys())
exp_f1s = [experiments[e]['f1'] for e in exp_names]
exp_recalls = [experiments[e]['recall'] for e in exp_names]

x_pos = np.arange(len(exp_names))
width = 0.35
b1 = ax.bar(x_pos - width/2, exp_f1s, width, color=[COLORS['fcd'], COLORS['waymo'], COLORS['combined'], '#94a3b8'], 
            label='F1 Score', edgecolor='white', linewidth=0.5)
b2 = ax.bar(x_pos + width/2, exp_recalls, width, color=[COLORS['fcd'], COLORS['waymo'], COLORS['combined'], '#94a3b8'],
            alpha=0.6, label='Recall', edgecolor='white', linewidth=0.5)

for bars, vals in [(b1, exp_f1s), (b2, exp_recalls)]:
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', color=COLORS['text'], fontsize=10, fontweight='bold')

ax.set_title('Multi-Source F1 Score & Recall', color=COLORS['text'], fontweight='bold', fontsize=14)
ax.set_ylabel('Skor', color=COLORS['text'])
ax.set_xticks(x_pos)
ax.set_xticklabels(exp_names, fontsize=10)
ax.legend(facecolor=COLORS['bg'], edgecolor=COLORS['grid'])
ax.set_facecolor(COLORS['bg'])
ax.tick_params(colors=COLORS['text'])
ax.grid(True, alpha=0.1, axis='y')
ax.set_ylim(0, 1.1)

# Confusion matrix - Combined
ax = fig.add_subplot(gs[1, 2:], facecolor=COLORS['bg'])
cm = confusion_matrix(y_combined_test, pred_combined)
names_short = ['NOR', 'SPK', 'DRF', 'DRP', 'FRZ', 'NSE']
im = ax.imshow(cm, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(names_short, color=COLORS['text'])
ax.set_yticklabels(names_short, color=COLORS['text'])
ax.set_xlabel('Tahmin', color=COLORS['text'])
ax.set_ylabel('Gerçek', color=COLORS['text'])
ax.set_title('FCD + Waymo Birleşik Confusion Matrix', color=COLORS['combined'], fontweight='bold')
for i in range(6):
    for j in range(6):
        c = 'white' if cm[i,j] > cm.max()*0.5 else 'black'
        ax.text(j, i, str(cm[i,j]), ha='center', va='center', color=c, fontweight='bold')

# Satır 3: Sınıf bazlı + veri kaynağı karşılaştırma
ax = fig.add_subplot(gs[2, :2], facecolor=COLORS['bg'])
f1_fcd_per = f1_score(y_fcd_test, pred_fcd, average=None)
f1_waymo_per = f1_score(y_waymo_test, pred_waymo, average=None)
f1_combined_per = f1_score(y_combined_test, pred_combined, average=None)

x_pos = np.arange(6)
width = 0.25
ax.bar(x_pos - width, f1_fcd_per, width, color=COLORS['fcd'], label='FCD', edgecolor='white', linewidth=0.5)
ax.bar(x_pos, f1_waymo_per, width, color=COLORS['waymo'], label='Waymo', edgecolor='white', linewidth=0.5)
ax.bar(x_pos + width, f1_combined_per, width, color=COLORS['combined'], label='Birleşik', edgecolor='white', linewidth=0.5)

class_labels = [LABEL_NAMES[i] for i in range(6)]
ax.set_xticks(x_pos)
ax.set_xticklabels(class_labels, fontsize=10)
ax.set_title('Sınıf Bazlı F1: Veri Kaynağı Karşılaştırması', color=COLORS['text'], fontweight='bold')
ax.set_ylabel('F1 Score', color=COLORS['text'])
ax.legend(facecolor=COLORS['bg'], edgecolor=COLORS['grid'])
ax.set_facecolor(COLORS['bg'])
ax.tick_params(colors=COLORS['text'])
ax.grid(True, alpha=0.1, axis='y')
ax.set_ylim(0, 1.1)

# Veri özeti tablosu
ax = fig.add_subplot(gs[2, 2:], facecolor=COLORS['bg'])
ax.axis('off')
summary_text = f"""
╔═══════════════════════════════════════════╗
║     📊 MULTI-SOURCE PIPELINE ÖZET        ║
╠═══════════════════════════════════════════╣
║                                           ║
║  Veri Kaynakları:                         ║
║    🇮🇹 FCD İtalya: 4 şehir, ~500K yörünge ║
║    🚗 Waymo: {len(waymo_data):,} otonom araç yörüngesi  ║
║                                           ║
║  Eğitim Verisi:                           ║
║    FCD:     {len(X_fcd_train):,} örnek                  ║
║    Waymo:   {len(X_waymo_train):,} örnek                  ║
║    Birleşik: {len(X_combined_train):,} örnek                ║
║                                           ║
║  Sonuçlar (F1 Score):                     ║
║    FCD tek:       {f1_fcd:.4f}                   ║
║    Waymo tek:     {f1_waymo:.4f}                   ║
║    Birleşik:      {f1_combined:.4f}                   ║
║    Cross-Domain:  {f1_cross:.4f}                   ║
║                                           ║
╚═══════════════════════════════════════════╝
"""
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', color=COLORS['text'], fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor=COLORS['grid'], alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f'{OUTPUT_DIR}/10_multi_source_results.png', dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("  ✅ 10_multi_source_results.png")

# ============================================================
# 6. DETAYLI RAPOR ÇIKTISI
# ============================================================
print("\n" + "=" * 60)
print("🏆 FAZ 5 TAMAMLANDI - MULTI-SOURCE SONUÇLAR")
print("=" * 60)

print(f"""
  📡 Waymo Entegrasyonu Başarılı!
  
  Waymo Motion Dataset:
    • {n_scenarios} senaryo parse edildi
    • {len(waymo_data):,} otonom araç yörüngesi çıkarıldı
    • 5 kanal: x, y, speed, velocity_x, velocity_y
    • 100ms çözünürlük (FCD'den 600x daha detaylı!)
  
  Deney Sonuçları (Gradient Boosting):
  ┌───────────────────┬────────┬────────┐
  │ Veri Kaynağı      │ F1     │ Recall │
  ├───────────────────┼────────┼────────┤
  │ FCD (İtalya)      │ {f1_fcd:.4f} │ {r_fcd:.4f} │
  │ Waymo             │ {f1_waymo:.4f} │ {r_waymo:.4f} │
  │ FCD + Waymo       │ {f1_combined:.4f} │ {r_combined:.4f} │
  │ Cross-Domain      │ {f1_cross:.4f} │ {r_cross:.4f} │
  └───────────────────┴────────┴────────┘
""")

# JSON kaydet
result_json = {
    'waymo_scenarios': n_scenarios,
    'waymo_trajectories': len(waymo_data),
    'experiments': {k: {'f1': round(v['f1'], 4), 'recall': round(v['recall'], 4)} 
                   for k, v in experiments.items()}
}
with open(f'{OUTPUT_DIR}/multi_source_results.json', 'w') as f:
    json.dump(result_json, f, indent=2)
print("  💾 multi_source_results.json kaydedildi")
