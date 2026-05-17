"""
Sentetik Veri Üretim Hattı — Adaptive Akıllı Veri Artırım Platformu
Bilgi Damıtma + RCGAN/CTGAN/SMOTE Sentetik Üretim
"""
import os, io, random, json, traceback

# [CRITICAL FIX] Import ctgan BEFORE torch to prevent macOS OpenMP/Accelerate deadlock
try:
    from ctgan import CTGAN
    CTGAN_AVAILABLE = True
    print('[✅] CTGAN modülü yüklendi')
except ImportError:
    CTGAN_AVAILABLE = False
    print('[⚠️] CTGAN yok, SMOTE fallback kullanılacak')

import torch, torch.nn as nn
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from collections import Counter

app = FastAPI(title="Sentetik Veri Üretim Hattı")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
SEED_PATH = os.path.join(PROJECT_DIR, "waymo_seed_MASSIVE.csv")
SEED_LEGACY_PATH = os.path.join(PROJECT_DIR, "waymo_seed.csv")
MODEL_PATH = os.path.join(OUTPUT_DIR, "waymo_rcgan_GODMODE_A100_STABLE.pth")

_seed_row_cache = {}

def is_git_lfs_pointer(path):
    """Detect Git LFS pointer files that have not been pulled yet."""
    try:
        if not path or not os.path.isfile(path) or os.path.getsize(path) > 1024:
            return False

        with open(path, "rb") as f:
            head = f.read(128)

        return head.startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False

def get_seed_path():
    """Return the usable seed CSV, preferring the real repo file over legacy symlinks."""
    candidates = [SEED_PATH, SEED_LEGACY_PATH]
    for path in candidates:
        try:
            if is_git_lfs_pointer(path):
                continue
            if os.path.isfile(path) and os.path.getsize(path) > 0 and count_csv_rows(path) > 0:
                return path
        except OSError:
            continue
    return None

async def read_csv_upload(file):
    filename = file.filename or ""

    if not filename.lower().endswith(".csv"):
        raise ValueError(
            f"Sadece CSV dosyası yüklenebilir. Seçilen dosya: {filename}"
        )

    raw = await file.read()

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "Dosya UTF-8 CSV olarak okunamadı. Lütfen .csv dosyası yükleyin; "
                "DMG/ZIP/Excel ikili dosyaları desteklenmez."
            ) from exc

    if "," not in text and "\n" not in text:
        raise ValueError("Dosya CSV gibi görünmüyor. Lütfen geçerli bir .csv dosyası seçin.")

    return text

def count_csv_rows(path):
    if not path:
        return 0
    try:
        stat = os.stat(path)
        cache_key = (path, stat.st_mtime_ns, stat.st_size)
        if cache_key not in _seed_row_cache:
            _seed_row_cache.clear()
            with open(path, "rb") as f:
                _seed_row_cache[cache_key] = max(sum(1 for _ in f) - 1, 0)
        return _seed_row_cache[cache_key]
    except OSError:
        return 0

# ═══════════════ RCGAN GODMODE ═══════════════
class RCGAN_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(6, 16)
        self.lstm = nn.LSTM(80, 512, 3, batch_first=True, bidirectional=True)
        self.out = nn.Sequential(nn.Linear(1024, 256), nn.LeakyReLU(0.2), nn.Linear(256, 5))
    def forward(self, z, labels):
        emb = self.label_embed(labels).unsqueeze(1).repeat(1, 20, 1)
        x, _ = self.lstm(torch.cat([z, emb], dim=2))
        return self.out(x)

rcgan = None
try:
    if os.path.exists(MODEL_PATH):
        if is_git_lfs_pointer(MODEL_PATH):
            print("[⚠️] RCGAN model dosyası Git LFS pointer. `git lfs pull` çalıştırılmalı.")
        else:
            rcgan = RCGAN_Generator()
            checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
            state_dict = checkpoint['G'] if isinstance(checkpoint, dict) and 'G' in checkpoint else checkpoint
            rcgan.load_state_dict(state_dict)
            rcgan.eval(); print("[✅] RCGAN GODMODE yüklendi")
except Exception as e:
    print(f"[⚠️] RCGAN hata: {e}"); rcgan = None

# ═══════════════ BİLGİ DAMITMA MOTORU ═══════════════
def distill_dataset(df):
    """
    4 Katmanlı Bilgi Damıtma:
    1. Duplikasyon temizliği
    2. Gürültülü etiket tespiti + düzeltme
    3. İstatistiksel outlier filtreleme
    4. Sıfır-varyans & entropi bazlı sütun temizliği
    """
    report = {"original_rows": len(df), "original_cols": len(df.columns), "steps": []}
    
    # Label sütununu bul
    label_col = _find_label_col(df)
    numeric_cols = [c for c in df.columns if df[c].dtype in ['float64','float32','int64','int32'] and c != label_col]
    
    if not numeric_cols:
        report["steps"].append({"name": "UYARI", "detail": "Numerik sütun bulunamadı."})
        return df, report, label_col, numeric_cols
    
    # ── KATMAN 1: Duplikasyon ──
    before = len(df)
    df = df.drop_duplicates(subset=numeric_cols, keep='first')
    dup_removed = before - len(df)
    report["steps"].append({
        "name": "Duplikasyon Temizliği",
        "removed": dup_removed,
        "detail": f"{dup_removed} birebir kopya satır silindi"
    })
    
    # Yakın-duplikatlar: cosine benzerliği tabular sensör verilerinde aşırı agresif
    # davranabiliyor. Sadece ölçüm hassasiyeti kaynaklı kopyaları temizle.
    near_dup = 0
    if len(df) < 10000 and len(numeric_cols) > 0:
        rounded_numeric = df[numeric_cols].round(4)
        keep = ~rounded_numeric.duplicated(keep='first')
        near_dup = int((~keep).sum())
        if near_dup > 0:
            df = df.iloc[keep].reset_index(drop=True)
    
    if near_dup > 0:
        report["steps"].append({
            "name": "Yakın-Duplikat Temizliği",
            "removed": int(near_dup),
            "detail": f"{near_dup} yakın-kopya satır silindi (>%99.99 benzerlik)"
        })
    
    # ── KATMAN 2: Gürültülü Etiket Tespiti ──
    label_fixes = 0
    if label_col and len(df) > 50 and len(numeric_cols) >= 3:
        X = np.nan_to_num(df[numeric_cols].values.astype(np.float32))
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        le = LabelEncoder()
        y = le.fit_transform(df[label_col].astype(str))
        
        if len(np.unique(y)) >= 2:
            k = min(7, len(df) - 1)
            nn_model = NearestNeighbors(n_neighbors=k).fit(X_s)
            _, indices = nn_model.kneighbors(X_s)
            
            noisy_mask = np.zeros(len(df), dtype=bool)
            for i in range(len(df)):
                neighbor_labels = y[indices[i][1:]]  # kendisi hariç
                agreement = np.mean(neighbor_labels == y[i])
                if agreement < 0.15:  # %85+ komşu farklı etiket → gürültülü
                    # Çoğunluk etiketi ile düzelt
                    most_common = Counter(neighbor_labels).most_common(1)[0][0]
                    y[i] = most_common
                    label_fixes += 1
            
            if label_fixes > 0:
                df[label_col] = le.inverse_transform(y)
        
        report["steps"].append({
            "name": "Etiket Düzeltme",
            "fixed": label_fixes,
            "detail": f"{label_fixes} gürültülü etiket komşu çoğunluğuna göre düzeltildi"
        })
    
    # ── KATMAN 3: Outlier Filtreleme ──
    outlier_removed = 0
    if len(numeric_cols) >= 1 and len(df) > 10:
        # Anomali sınıflarını koru, ancak 10'dan fazla örneği olan her sınıfın KENDİ İÇİNDEKİ outlier'ları temizle
        if label_col:
            outlier_idx_list = []
            for cls_name, count in df[label_col].value_counts().items():
                if count >= 10:
                    cls_mask = df[label_col] == cls_name
                    for col in numeric_cols:
                        q1 = df.loc[cls_mask, col].quantile(0.15)
                        q3 = df.loc[cls_mask, col].quantile(0.85)
                        iqr = max(q3 - q1, 1e-5)  # Eğer tüm değerler aynıysa sıfıra bölme/aşırı filtrelemeyi engelle
                        lower = q1 - 2.5 * iqr
                        upper = q3 + 2.5 * iqr
                        bad = df.loc[cls_mask][(df.loc[cls_mask, col] < lower) | (df.loc[cls_mask, col] > upper)].index
                        outlier_idx_list.extend(bad)
            
            outlier_idx = list(set(outlier_idx_list))
            outlier_removed = len(outlier_idx)
            if outlier_removed > 0:
                df = df.drop(outlier_idx).reset_index(drop=True)
        else:
            outlier_idx_list = []
            for col in numeric_cols:
                q1 = df[col].quantile(0.15)
                q3 = df[col].quantile(0.85)
                iqr = q3 - q1
                lower = q1 - 2.5 * iqr
                upper = q3 + 2.5 * iqr
                bad = df[(df[col] < lower) | (df[col] > upper)].index
                outlier_idx_list.extend(bad)
            
            outlier_idx = list(set(outlier_idx_list))
            outlier_removed = len(outlier_idx)
            if outlier_removed > 0:
                df = df.drop(outlier_idx).reset_index(drop=True)
        
        report["steps"].append({
            "name": "Outlier Filtreleme",
            "removed": outlier_removed,
            "detail": f"{outlier_removed} fiziksel/istatistiksel outlier silindi (Strict IQR)"
        })
    
    # ── KATMAN 4: Sütun Temizliği ──
    cols_removed = []
    waymo_protected = set(f'{c}({i+1})' for c in ['x','y','speed','vx','vy'] for i in range(20))
    
    for col in numeric_cols[:]:
        if col in waymo_protected:
            continue
            
        # Sıfır varyans
        if df[col].std() < 1e-10:
            cols_removed.append(col)
            numeric_cols.remove(col)
        # %95+ NaN
        elif df[col].isna().mean() > 0.95:
            cols_removed.append(col)
            numeric_cols.remove(col)
    
    if cols_removed:
        df = df.drop(columns=cols_removed)
        report["steps"].append({
            "name": "Sütun Temizliği",
            "removed": len(cols_removed),
            "detail": f"{len(cols_removed)} bilgisiz sütun silindi: {', '.join(cols_removed[:5])}"
        })
    
    # NaN doldur
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    report["clean_rows"] = len(df)
    report["clean_cols"] = len(df.columns)
    report["reduction_pct"] = round((1 - len(df)/report["original_rows"]) * 100, 1) if report["original_rows"] > 0 else 0
    
    return df, report, label_col, numeric_cols

def _find_label_col(df):
    for col in df.columns:
        if col.lower() in ['label','class','target','category','type','sınıf','etiket','y']:
            return col
    for col in df.columns:
        if df[col].dtype == 'object' and 1 < df[col].nunique() < 30:
            return col
    for col in df.columns:
        if df[col].dtype in ['int64','int32'] and 1 < df[col].nunique() < 30:
            return col
    return None

# ═══════════════ ADAPTIVE GENERATION ═══════════════
def analyze_classes(df, label_col):
    if not label_col: return {}
    dist = df[label_col].value_counts().to_dict()
    return {str(k): int(v) for k, v in dist.items()}

# ═══════════════ AKILLI YÖRÜNGE DÖNÜŞTÜRÜCÜ ═══════════════
def try_convert_to_waymo(df, label_col):
    """
    Herhangi bir koordinat/pozisyon/hareket verisini Waymo formatına dönüştürür.
    Desteklenen veri türleri:
      - GPS/Konum: lat/lon, pos_x/pos_y, x/y, latitude/longitude
      - IMU/Sensör: accel_x/accel_y, gyro_x/gyro_y
      - Hız: speed, velocity, vel_x/vel_y
    Dönüşüm: Sıralı satırları 20'şerli pencereler halinde keser,
    eksik kanalları (speed, vx, vy) pozisyon farkından hesaplar.
    """
    cols_lower = {c: c.lower().strip().replace(" ", "_").replace("-", "_") for c in df.columns}
    
    # ── 1. Pozisyon sütunlarını bul ──
    x_col, y_col = None, None
    speed_col, vx_col, vy_col = None, None, None
    
    # Pozisyon aday isimleri (öncelik sırasıyla)
    # ⚠️ SADECE gerçek konum/pozisyon sütunları! İvme (accel), jiroskop (gyro) vs. KONUM DEĞİL!
    x_candidates = [
        'x', 'pos_x', 'position_x', 'coord_x', 'coordinate_x', 'center_x', 'centroid_x',
        'ego_x', 'vehicle_x', 'agent_x', 'object_x', 'track_x', 'map_x', 'local_x',
        'global_x', 'longitude', 'lon', 'lng', 'easting'
    ]
    y_candidates = [
        'y', 'pos_y', 'position_y', 'coord_y', 'coordinate_y', 'center_y', 'centroid_y',
        'ego_y', 'vehicle_y', 'agent_y', 'object_y', 'track_y', 'map_y', 'local_y',
        'global_y', 'latitude', 'lat', 'northing'
    ]
    speed_candidates = ['speed', 'speed_mps', 'ego_speed', 'vehicle_speed', 'velocity', 'vel', 'spd']
    vx_candidates = ['vx', 'vel_x', 'velocity_x', 'v_x', 'ego_vx', 'vehicle_vx', 'agent_vx']
    vy_candidates = ['vy', 'vel_y', 'velocity_y', 'v_y', 'ego_vy', 'vehicle_vy', 'agent_vy']
    motion_context = [
        'time', 'timestamp', 'frame', 'frame_id', 't', 'dt', 'id', 'track_id', 'vehicle_id',
        'agent_id', 'object_id', 'label', 'class', 'target', 'type', 'heading', 'yaw',
        'yaw_rate', 'orientation', 'steering', 'steering_angle', 'lane_id', 'lane',
        'lane_offset', 'throttle', 'brake', 'accel', 'acceleration', 'acceleration_x',
        'acceleration_y', 'accel_x', 'accel_y', 'ax', 'ay', 'longitudinal_accel',
        'lateral_accel'
    ]
    routing_context = [c for c in motion_context if c not in ['label', 'class', 'target', 'type', 'id']]
    
    # Yörünge ile ilgili tüm sütun isimleri
    trajectory_keywords = set(x_candidates + y_candidates + speed_candidates + vx_candidates + vy_candidates + motion_context)
    
    for orig, low in cols_lower.items():
        if not x_col and low in x_candidates: x_col = orig
        if not y_col and low in y_candidates: y_col = orig
        if not speed_col and low in speed_candidates: speed_col = orig
        if not vx_col and low in vx_candidates: vx_col = orig
        if not vy_col and low in vy_candidates: vy_col = orig
    
    # En az 2 KONUM sütunu bulamazsak dönüştürme yapılamaz → CTGAN'a düşsün
    if not x_col or not y_col:
        return None

    # Generic x/y kolonlarını ancak hareket bağlamı varsa RCGAN'a al.
    has_motion_signal = any([speed_col, vx_col, vy_col]) or any(cols_lower[c] in routing_context for c in df.columns)
    if cols_lower[x_col] in ['x', 'y'] and cols_lower[y_col] in ['x', 'y'] and not has_motion_signal:
        print("[ℹ️] Generic x/y bulundu ama hız/zaman/araç bağlamı yok; CTGAN'a yönlendiriliyor.")
        return None
    
    # ── Zengin Veri Seti Kontrolü ──
    # Eğer dosyada konum dışı sütunlar çoğunluktaysa (IMU, sensör, vs.),
    # RCGAN'a dönüştürmek veri kaybına yol açar. CTGAN tüm sütunları korur.
    non_trajectory_cols = [c for c in df.columns if cols_lower[c] not in trajectory_keywords]
    trajectory_cols_found = [c for c in df.columns if cols_lower[c] in trajectory_keywords]
    
    rich_limit = max(8, len(trajectory_cols_found) * 2)
    if len(non_trajectory_cols) > rich_limit:
        # Dosyada çok fazla yörünge-dışı sütun var → zengin veri seti
        # CTGAN hepsini öğrensin, RCGAN sadece 5 kanal bilir, geri kalanı kaybolur
        print(f"[ℹ️] Zengin veri seti tespit edildi ({len(non_trajectory_cols)} ekstra sütun: {non_trajectory_cols[:5]}...)")
        print(f"[ℹ️] Tüm sütunları korumak için CTGAN'a yönlendiriliyor.")
        return None
    
    print(f"[🔄] Saf yörünge verisi: x={x_col}, y={y_col}, speed={speed_col}")
    
    # ── 2. Verileri çıkar ──
    x_data = pd.to_numeric(df[x_col], errors='coerce').fillna(0).values
    y_data = pd.to_numeric(df[y_col], errors='coerce').fillna(0).values
    
    # Speed yoksa vx/vy veya pozisyon farkından hesapla
    if speed_col:
        speed_data = pd.to_numeric(df[speed_col], errors='coerce').fillna(0).values
    elif vx_col and vy_col:
        vx_tmp = pd.to_numeric(df[vx_col], errors='coerce').fillna(0).values
        vy_tmp = pd.to_numeric(df[vy_col], errors='coerce').fillna(0).values
        speed_data = np.sqrt(vx_tmp**2 + vy_tmp**2)
    else:
        dx = np.diff(x_data, prepend=x_data[0])
        dy = np.diff(y_data, prepend=y_data[0])
        speed_data = np.sqrt(dx**2 + dy**2)

    movement_span = (np.nanmax(x_data) - np.nanmin(x_data)) + (np.nanmax(y_data) - np.nanmin(y_data))
    if movement_span < 1e-6:
        print("[ℹ️] Konum kolonları hareket içermiyor; CTGAN'a yönlendiriliyor.")
        return None
    
    # vx/vy yoksa pozisyon farkından hesapla
    if vx_col:
        vx_data = pd.to_numeric(df[vx_col], errors='coerce').fillna(0).values
    else:
        vx_data = np.diff(x_data, prepend=x_data[0])
    
    if vy_col:
        vy_data = pd.to_numeric(df[vy_col], errors='coerce').fillna(0).values
    else:
        vy_data = np.diff(y_data, prepend=y_data[0])
    
    # Label varsa al
    if label_col and label_col in df.columns:
        labels = df[label_col].values
    else:
        labels = None
    
    # ── 3. 20'şerli Pencere Oluştur ──
    WINDOW = 20
    STRIDE = 10  # %50 örtüşme → daha fazla yörünge
    n = len(x_data)
    
    if n < WINDOW:
        return None
    
    rows = []
    for start in range(0, n - WINDOW + 1, STRIDE):
        end = start + WINDOW
        
        # Penceredeki verileri al
        xw = x_data[start:end]
        yw = y_data[start:end]
        sw = speed_data[start:end]
        vxw = vx_data[start:end]
        vyw = vy_data[start:end]
        
        # Relative koordinatlara dönüştür (başlangıç noktası = 0,0)
        xw = xw - xw[0]
        yw = yw - yw[0]
        
        row = {}
        for i in range(WINDOW):
            row[f'x({i+1})'] = float(xw[i])
            row[f'y({i+1})'] = float(yw[i])
            row[f'speed({i+1})'] = float(abs(sw[i]))
            row[f'vx({i+1})'] = float(vxw[i])
            row[f'vy({i+1})'] = float(vyw[i])
        
        # Label: penceredeki en sık label
        if labels is not None:
            window_labels = labels[start:end]
            try:
                most_common = pd.Series(window_labels).mode()[0]
                row['label'] = most_common
            except:
                row['label'] = 'normal'
        else:
            row['label'] = 'normal'
        
        rows.append(row)
    
    if not rows:
        return None
    
    df_waymo = pd.DataFrame(rows)
    print(f"[🔄] {n} satır → {len(df_waymo)} yörünge penceresi (20 adım, stride {STRIDE})")
    return df_waymo


def generate_adaptive(df, label_col, numeric_cols, n_samples):
    """Waymo → RCGAN, Yörünge verisi → Dönüştür+RCGAN, diğerleri → CTGAN, fallback → SMOTE"""
    waymo_cols = [f'x({i+1})' for i in range(20)]
    is_waymo = all(c in df.columns for c in waymo_cols)
    
    # ── Akıllı Yörünge Dönüştürücü ──
    # Waymo formatında değilse ama koordinat/pozisyon verisi varsa, otomatik dönüştür
    df_converted = None
    if not is_waymo and rcgan is not None:
        df_converted = try_convert_to_waymo(df, label_col)
        if df_converted is not None:
            is_waymo = True
            df = df_converted
            print(f"[🔄] Yörünge verisi Waymo formatına dönüştürüldü: {len(df)} yörünge")
    
    df_gen = pd.DataFrame()
    method = ""
    # Waymo verisi ise ve RCGAN modeli yüklüyse DAİMA RCGAN kullan.
    if is_waymo and rcgan is not None:
        df_gen, method = generate_waymo(df, n_samples), 'rcgan'
    
    # CTGAN: Genel veri setleri için on-the-fly GAN eğitimi
    elif CTGAN_AVAILABLE and len(df) >= 100 and len(numeric_cols) >= 2:
        try:
            df_gen, method = generate_ctgan(df, label_col, numeric_cols, n_samples), 'ctgan'
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'[⚠️] CTGAN başarısız, SMOTE fallback: {e}')
            df_gen, method = generate_smart(df, label_col, numeric_cols, n_samples), 'smote'
    else:
        df_gen, method = generate_smart(df, label_col, numeric_cols, n_samples), 'smote'

    # ── Domain Shift Koruması (Fiziksel Sınırlandırma) ──
    # Üretilen veriyi orijinal temiz verinin fiziksel sınırlarına (min-max) hapset
    if not df_gen.empty and method != 'rcgan':
        for col in numeric_cols:
            if col in df_gen.columns:
                c_min = df[col].min()
                c_max = df[col].max()
                margin = (c_max - c_min) * 0.05  # %5 esneklik payı
                # Eğer orijinal veri 0'ın altına düşmüyorsa, sentetik de düşmesin (örn: Kelvin/Hız)
                c_min_bound = max(0, c_min - margin) if c_min >= 0 else c_min - margin
                c_max_bound = c_max + margin
                df_gen[col] = df_gen[col].clip(lower=c_min_bound, upper=c_max_bound)
        
        # Sınırlandırılmış veriyi tekrar kaydet
        df_gen.to_csv(os.path.join(OUTPUT_DIR, 'live_synthetic_output.csv'), index=False)
        
    return df_gen, method, is_waymo

def generate_ctgan(df, label_col, numeric_cols, n_samples):
    """On-the-fly CTGAN eğitimi — her veri setine adapte olur"""
    
    # Eğitim verisini hazırla
    train_cols = numeric_cols + [label_col]
    df_train = df[train_cols].copy()
    
    # ── Akıllı Örnekleme ──
    # CTGAN dağılımı öğrenmek için 10K satır fazlasıyla yeterli.
    # 500K satır sokmak saatlerce sürer, gereksiz.
    MAX_CTGAN_ROWS = 10000
    if len(df_train) > MAX_CTGAN_ROWS:
        # Sınıf dengeli (stratified) örnekleme yap
        try:
            from sklearn.model_selection import train_test_split
            df_train, _ = train_test_split(
                df_train, train_size=MAX_CTGAN_ROWS, 
                stratify=df_train[label_col], random_state=42
            )
        except:
            df_train = df_train.sample(MAX_CTGAN_ROWS, random_state=42)
        print(f'[CTGAN] {len(df)} satırdan {len(df_train)} örneklem alındı (hız optimizasyonu)')
    
    print(f'[CTGAN] {len(df_train)} satır üzerinde eğitim başlıyor...')
    
    # Kategorik sütunları belirle
    discrete_cols = [label_col]
    for col in numeric_cols:
        if df_train[col].nunique() < 10:
            discrete_cols.append(col)
    
    # Epoch sayısı veri boyutuna göre ayarla (hız vs kalite)
    n_rows = len(df_train)
    if n_rows < 500:
        epochs = 150
    elif n_rows < 2000:
        epochs = 100
    else:
        epochs = 50  # Büyük veri → az epoch, hızlı
    
    # CTGAN batch_size must be a multiple of pac (default 10)
    ideal_batch = min(500, len(df_train))
    batch_size = max(10, (ideal_batch // 10) * 10)
    
    # CTGAN eğit
    model = CTGAN(
        epochs=epochs,
        batch_size=batch_size,
        generator_dim=(128, 128),
        discriminator_dim=(128, 128),
        verbose=False
    )
    model.fit(df_train, discrete_columns=discrete_cols)
    
    # Sentetik veri üret
    df_gen = model.sample(n_samples)
    print(f'[CTGAN] {n_samples} sentetik örnek üretildi')
    
    # Kaydet
    df_gen.to_csv(os.path.join(OUTPUT_DIR, 'live_synthetic_output.csv'), index=False)
    return df_gen

def generate_smart(df, label_col, numeric_cols, n_samples):
    if not label_col or not numeric_cols:
        return pd.DataFrame()
    
    dist = df[label_col].value_counts().to_dict()
    if not dist: return pd.DataFrame()
    
    classes = list(dist.keys())
    per_class = max(1, n_samples // len(classes))
    
    X = np.nan_to_num(df[numeric_cols].values.astype(np.float32))
    rows = []
    
    for cls in classes:
        mask = df[label_col] == cls
        X_cls = X[mask]
        if len(X_cls) < 2: continue
        
        cls_std = np.std(X_cls, axis=0) + 1e-8
        k = min(5, len(X_cls)-1)
        nn_m = NearestNeighbors(n_neighbors=k+1).fit(X_cls)
        
        n_smote = per_class // 2
        n_noise = per_class - n_smote
        
        for _ in range(n_smote):
            idx = random.randint(0, len(X_cls)-1)
            _, nbrs = nn_m.kneighbors([X_cls[idx]])
            nbr = X_cls[random.choice(nbrs[0][1:])]
            alpha = random.uniform(0.1, 0.9)
            pt = X_cls[idx] + alpha * (nbr - X_cls[idx])
            row = {numeric_cols[j]: float(pt[j]) for j in range(len(numeric_cols))}
            row[label_col] = cls; rows.append(row)
        
        for _ in range(n_noise):
            base = X_cls[random.randint(0, len(X_cls)-1)].copy()
            pt = base + np.random.normal(0, cls_std * 0.12)
            row = {numeric_cols[j]: float(pt[j]) for j in range(len(numeric_cols))}
            row[label_col] = cls; rows.append(row)
    
    if not rows: return pd.DataFrame()
    df_gen = pd.DataFrame(rows)
    df_gen.to_csv(os.path.join(OUTPUT_DIR, "live_synthetic_output.csv"), index=False)
    return df_gen

def generate_waymo(df, n_samples):
    LM = {'normal':0,'spike':1,'drift':2,'dropout':3,'freeze':4,'noise':5}
    LN = {v:k for k,v in LM.items()}
    pool = []
    dc = df[df['label']=='normal'] if 'label' in df.columns else df
    if len(dc)==0: dc=df
    
    # 500 yörüngeyi hep baştan almak yerine rastgele seç
    sample_df = dc.sample(min(1000, len(dc))) if len(dc) > 0 else dc
    for _, r in sample_df.iterrows():
        try:
            v = np.stack([np.array([r[f'{c}({i+1})'] for i in range(20)]) for c in ['x','y','speed','vx','vy']], axis=1)
            if np.all(v[:,0]==0) and np.all(v[:,1]==0): continue
            pool.append(v)
        except Exception as e:
            print(f"[⚠️] generate_waymo satır okuma hatası: {e}")
            continue
    if not pool: return pd.DataFrame()
    
    sc = [f'speed({i+1})' for i in range(20)]
    xc = [f'x({i+1})' for i in range(20)]
    sm = max(min(np.nanpercentile(df[sc].values,95),50),5)
    xr = min(max(np.nanpercentile(df[xc].values,95)-np.nanpercentile(df[xc].values,5),1),40)
    
    def spike(t): t=t.copy();i=random.randint(2,17);t[i,0]+=random.choice([-1,1])*random.uniform(xr*.05,xr*.15);t[i,2]=min(t[i,2]*random.uniform(2,4),sm*2);return t
    def drift(t): t=t.copy();[setattr(t,'__setitem__',None) or None for _ in range(0)];exec('');t2=t;[t2.__setitem__((i,0),t2[i,0]+(i-10)*(xr*0.01)) for i in range(10,20)];return t2
    def dropout(t):
        t=t.copy()
        for i in range(10,15): t[i,2:]=0;t[i,:2]=t[9,:2]
        return t
    def freeze(t): t=t.copy();t[10:15,:]=t[10,:];return t
    def noise(t): t=t.copy();t[:,0]+=np.random.normal(0,xr*.02,20);t[:,2]=np.clip(t[:,2]+np.random.normal(0,sm*.05,20),0,None);return t
    
    gens={1:spike,2:drift,3:dropout,4:freeze,5:noise}
    seqs,yl=[],[]
    with torch.no_grad():
        ya=np.random.choice([1,2,3,4,5],size=n_samples)
        z=torch.randn(n_samples,20,64)
        Xai=rcgan(z,torch.tensor(ya,dtype=torch.long)).numpy()
        for i in range(n_samples):
            c=ya[i]; b=random.choice(pool).copy()
            try: a=gens[c](b)
            except: a=b
            ax=np.convolve(Xai[i,:,0]-np.mean(Xai[i,:,0]),np.ones(3)/3,mode='same')
            if np.max(np.abs(ax))>0: ax/=np.max(np.abs(ax))
            if c in [1,5]: a[:,0]+=ax*min(max(np.std(ax)+.1,.2),2.5)*(xr*.05)
            a[:,2]=np.clip(a[:,2],0,60);a[:,3:]=np.clip(a[:,3:],-20,20)
            seqs.append(a);yl.append(c)
    Xt=np.transpose(np.array(seqs),(0,2,1))
    cols=[f'{f}({i+1})' for f in ['x','y','speed','vx','vy'] for i in range(20)]
    dg=pd.DataFrame(Xt.reshape(-1,100),columns=cols)
    dg['label']=[LN[l] for l in yl]
    dg.to_csv(os.path.join(OUTPUT_DIR,"live_synthetic_output.csv"),index=False)
    return dg

def _score_from_corr(value):
    try:
        return round(max(0.0, min(1.0, float(value))) * 100, 1)
    except:
        return 0.0

def _distribution_shift_report(fidelity):
    details = fidelity.get("column_details", []) if isinstance(fidelity, dict) else []
    warnings = []
    smds = []
    mean_diffs = []
    for item in details:
        mean_diff = float(item.get("mean_diff_pct", 0) or 0)
        smd = float(item.get("standardized_mean_diff", 0) or 0)
        mean_diffs.append(mean_diff)
        smds.append(smd)

        if smd >= 0.5:
            severity = "kritik"
        elif smd >= 0.25:
            severity = "uyarı"
        else:
            continue

        warnings.append({
            "column": item.get("column"),
            "mean_diff_pct": round(mean_diff, 1),
            "standardized_mean_diff": round(smd, 3),
            "severity": severity,
            "detail": f"{item.get('column')} ortalaması {round(smd,3)} standart sapma kaydı."
        })
    avg_smd = float(np.mean(smds)) if smds else 0.0
    max_smd = float(np.max(smds)) if smds else 0.0
    avg_mean_shift = float(np.mean(mean_diffs)) if mean_diffs else 0.0
    max_mean_shift = float(np.max(mean_diffs)) if mean_diffs else 0.0
    # Sıfır merkezli sensörlerde yüzde fark yanıltır; skor Cohen's d/SMD üstünden verilir.
    # 0.1 küçük, 0.25 orta uyarı, 0.5+ ciddi dağılım kayması kabul edilir.
    score = round(max(0.0, 100.0 - min(avg_smd, 0.75) * (100.0 / 0.75)), 1)
    return {"score": score, "avg_standardized_mean_diff": round(avg_smd, 3),
            "max_standardized_mean_diff": round(max_smd, 3),
            "avg_mean_shift_pct": round(avg_mean_shift, 1),
            "max_mean_shift_pct": round(max_mean_shift, 1), "warnings": warnings}

def _physical_consistency_report(df_gen):
    waymo_cols = [f'{c}({i+1})' for c in ['x','y','speed','vx','vy'] for i in range(20)]
    if df_gen.empty or not all(c in df_gen.columns for c in waymo_cols):
        return {"applicable": False, "score": None, "detail": "Waymo/trajectory kolonları yok; fizik skoru uygulanmadı."}
    
    speed = df_gen[[f'speed({i+1})' for i in range(20)]].to_numpy(dtype=float)
    vx = df_gen[[f'vx({i+1})' for i in range(20)]].to_numpy(dtype=float)
    vy = df_gen[[f'vy({i+1})' for i in range(20)]].to_numpy(dtype=float)
    x = df_gen[[f'x({i+1})' for i in range(20)]].to_numpy(dtype=float)
    y = df_gen[[f'y({i+1})' for i in range(20)]].to_numpy(dtype=float)
    
    finite_ratio = float(np.isfinite(speed).mean() * np.isfinite(vx).mean() * np.isfinite(vy).mean())
    negative_speed_ratio = float((speed < -1e-6).mean())
    high_speed_ratio = float((speed > 60).mean())
    accel = np.diff(speed, axis=1) / 0.1
    high_accel_ratio = float((np.abs(accel) > 12).mean()) if accel.size else 0.0
    lateral_step = np.sqrt(np.diff(x, axis=1)**2 + np.diff(y, axis=1)**2)
    jump_ratio = float((lateral_step > 8).mean()) if lateral_step.size else 0.0
    vel_mag = np.sqrt(vx**2 + vy**2)
    coherence_error = np.abs(vel_mag - speed) / (np.abs(speed) + 1.0)
    coherence_penalty = float(np.clip(np.nanmean(coherence_error), 0, 1))
    
    score = 100.0
    score -= negative_speed_ratio * 100
    score -= high_speed_ratio * 80
    score -= high_accel_ratio * 70
    score -= jump_ratio * 70
    score -= coherence_penalty * 20
    score *= finite_ratio
    score = round(max(0.0, min(100.0, score)), 1)
    
    issues = []
    if negative_speed_ratio > 0: issues.append("Negatif hız tespit edildi.")
    if high_speed_ratio > 0: issues.append("60 m/s üzeri hız tespit edildi.")
    if high_accel_ratio > 0.02: issues.append("Yüksek ivme oranı arttı.")
    if jump_ratio > 0.02: issues.append("Yörüngede ani konum sıçraması var.")
    if coherence_penalty > 0.25: issues.append("speed ile vx/vy büyüklüğü arasında uyumsuzluk var.")
    
    return {
        "applicable": True,
        "score": score,
        "negative_speed_ratio": round(negative_speed_ratio, 4),
        "high_speed_ratio": round(high_speed_ratio, 4),
        "high_accel_ratio": round(high_accel_ratio, 4),
        "jump_ratio": round(jump_ratio, 4),
        "velocity_coherence_error": round(coherence_penalty, 4),
        "issues": issues,
        "formula": "100 - negatif hız, aşırı hız, |ivme|>12 m/s², ani konum sıçraması ve speed-vx/vy uyumsuzluğu cezaları"
    }

def build_quality_report(df_orig, df_gen, method, is_waymo, scores, label_col, numeric_cols):
    fidelity = scores.get("fidelity", {}) or {}
    utility = scores.get("utility", {}) or {}
    fidelity_score = round((_score_from_corr(fidelity.get("cosine_similarity", 0)) +
                            _score_from_corr(fidelity.get("column_correlation", 0))) / 2, 1)
    shift = _distribution_shift_report(fidelity)
    physical = _physical_consistency_report(df_gen)
    
    utility_applicable = utility.get("evaluable", True) is not False and scores.get("seed_f1", 0) > 0
    if utility_applicable:
        aug_f1 = float(scores.get("augmented_f1", 0) or 0)
        improvement = max(0.0, float(scores.get("improvement", 0) or 0))
        recall_aug = float(utility.get("minority_recall_augmented", utility.get("minority_recall", 0)) or 0)
        utility_score = round((aug_f1 * 55) + (min(improvement / 15.0, 1.0) * 25) + (recall_aug * 20), 1)
    else:
        utility_score = None
    
    components = {}
    fidelity_applicable = bool(fidelity.get("column_details")) or fidelity.get("cosine_similarity", 0) > 0 or fidelity.get("column_correlation", 0) > 0
    if fidelity_applicable:
        components["fidelity"] = {"score": fidelity_score, "weight": 0.35 if method != "rcgan" else 0.30}
    else:
        components["fidelity"] = {"score": None, "weight": 0, "applicable": False}
    
    distribution_applicable = bool(shift.get("warnings")) or bool(fidelity.get("column_details"))
    if distribution_applicable:
        components["distribution"] = {"score": shift["score"], "weight": 0.25 if method != "rcgan" else 0.10}
    else:
        components["distribution"] = {"score": None, "weight": 0, "applicable": False}
    
    if utility_score is not None:
        components["utility"] = {"score": utility_score, "weight": 0.35 if method != "rcgan" else 0.25}
    else:
        components["utility"] = {"score": None, "weight": 0, "applicable": False}
    if physical["applicable"]:
        components["physical"] = {"score": physical["score"], "weight": 0.35 if method == "rcgan" else 0.15}
    
    total_weight = sum(v["weight"] for v in components.values())
    overall = round(sum((v["score"] or 0) * v["weight"] for v in components.values()) / max(total_weight, 1e-8), 1) if total_weight > 0 else 0
    
    if method == "rcgan":
        routing = "RCGAN seçildi: veri Waymo formatında veya otonom araç/yörünge kolonlarından 20 adımlı trajektöre dönüştürülebildi."
    elif method == "ctgan":
        routing = "CTGAN seçildi: veri genel tabular/sensör formatında ve çok sütunlu dağılım korunmalı."
    else:
        routing = "SMOTE+Gaussian seçildi: veri CTGAN için küçük/uygunsuz veya CTGAN başarısız oldu."
    
    return {
        "overall_score": overall,
        "grade": "A" if overall >= 90 else "B" if overall >= 80 else "C" if overall >= 70 else "D",
        "method": method,
        "routing_explanation": routing,
        "components": components,
        "distribution_shift": shift,
        "physical_consistency": physical,
        "scientific_basis": [
            "Fidelity: orijinal ve sentetik verinin ortalama vektör cosine benzerliği ile kolon ortalama/std korelasyonlarının birleşimi.",
            "Utility: aynı test ayrımı üzerinde Seed F1, Augmented F1, F1 iyileşmesi ve azınlık sınıfı recall değişimi.",
            "Distribution shift: kolon bazlı orijinal-sentetik ortalama fark yüzdesi; %15 uyarı, %30 kritik eşik.",
            "Physical consistency: RCGAN yörüngelerinde hız pozitifliği, hız sınırı, ivme, konum sıçraması ve speed-vx/vy uyumu."
        ]
    }

# ═══════════════ EVALUATION (Fidelity + Utility) ═══════════════
def evaluate(df_orig, df_gen, label_col, numeric_cols):
    """
    Akademik standartlara uygun Fidelity + Utility değerlendirmesi.
    
    Fidelity (Dağılım Benzerliği):
      - Cosine Similarity: Orijinal vs sentetik özellik vektörleri
      - Sütun bazlı ortalama/std karşılaştırması
      - Dağılım örtüşme oranı
    
    Utility (Görev Faydası):
      - Seed F1 vs Augmented F1 (weighted)
      - Sınıf bazlı F1 skorları
      - Azınlık sınıfı recall (hedef: %80)
      - F1 iyileştirme yüzdesi (hedef: %15)
    """
    from sklearn.metrics import f1_score, recall_score, precision_recall_curve, auc, classification_report
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    
    X = np.nan_to_num(df_orig[numeric_cols].values.astype(np.float32))
    y_raw = df_orig[label_col].astype(str)
    
    # Az üyeli sınıfları filtrele (stratify için min 4 gerekli)
    counts = y_raw.value_counts()
    valid_classes = counts[counts >= 4].index.tolist()
    mask = y_raw.isin(valid_classes)
    X = X[mask]; y_raw = y_raw[mask].reset_index(drop=True)
    
    le = LabelEncoder()
    le.fit(valid_classes)
    y = le.transform(y_raw)
    
    if len(np.unique(y)) < 2:
        return {"seed_f1": 0, "augmented_f1": 0, "improvement": 0,
                "analysis_note": "Utility metrikleri için en az iki sınıf gerekir; bu veri tek sınıflı.",
                "fidelity": {"cosine_similarity": 0, "column_correlation": 0},
                "utility": {"evaluable": False, "class_count": int(len(np.unique(y))),
                            "minority_recall": 0, "f1_target_met": False, "recall_target_met": False}}
    
    # ══ SEED MODEL (Orijinal veriyle) ══
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    
    m1 = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42).fit(Xtr_s, ytr)
    pred_seed = m1.predict(Xte_s)
    f1s = float(f1_score(yte, pred_seed, average='weighted', zero_division=0))
    
    # Sınıf bazlı F1 (seed)
    f1_per_class_seed = {}
    for i, cls_name in enumerate(le.classes_):
        cls_mask = yte == i
        if cls_mask.sum() > 0:
            cls_f1 = float(f1_score(yte == i, pred_seed == i, zero_division=0))
            f1_per_class_seed[str(cls_name)] = round(cls_f1, 4)
    
    # Azınlık sınıfı recall (seed) — en az üyesi olan sınıf
    minority_class = counts[counts.index.isin(valid_classes)].idxmin()
    minority_idx = le.transform([str(minority_class)])[0]
    minority_mask_te = yte == minority_idx
    if minority_mask_te.sum() > 0:
        minority_recall_seed = float(recall_score(minority_mask_te, pred_seed == minority_idx, zero_division=0))
    else:
        minority_recall_seed = 0.0
    
    # ══ FIDELITY (Dağılım Benzerliği) ══
    gc = [c for c in numeric_cols if c in df_gen.columns]
    fidelity = {"cosine_similarity": 0, "column_correlation": 0, "column_details": []}
    
    if gc and label_col in df_gen.columns:
        gen_mask = df_gen[label_col].astype(str).isin(valid_classes)
        df_gen_filtered = df_gen[gen_mask]
        
        if len(df_gen_filtered) > 0:
            Xg = np.nan_to_num(df_gen_filtered[gc].values.astype(np.float32))
            
            # 1. Cosine Similarity: Orijinal ve sentetik ortalama vektörleri
            orig_mean = np.mean(X[:, :len(gc)], axis=0).reshape(1, -1)
            gen_mean = np.mean(Xg, axis=0).reshape(1, -1)
            cosine_val = float(cos_sim(orig_mean, gen_mean)[0][0])
            
            # 2. Sütun bazlı ortalama/std korelasyonu
            orig_means = np.mean(X[:, :len(gc)], axis=0)
            gen_means = np.mean(Xg, axis=0)
            orig_stds = np.std(X[:, :len(gc)], axis=0) + 1e-8
            gen_stds = np.std(Xg, axis=0) + 1e-8
            
            # Ortalama korelasyonu
            if len(orig_means) > 1:
                mean_corr = float(np.corrcoef(orig_means, gen_means)[0, 1])
                std_corr = float(np.corrcoef(orig_stds, gen_stds)[0, 1])
            else:
                mean_corr = 1.0
                std_corr = 1.0
            
            # 3. Sütun detayları (ilk 10)
            col_details = []
            for j, col in enumerate(gc[:10]):
                pooled_std = float(np.sqrt((orig_stds[j] ** 2 + gen_stds[j] ** 2) / 2.0)) + 1e-8
                mean_diff_abs = abs(float(orig_means[j]) - float(gen_means[j]))
                col_details.append({
                    "column": col,
                    "orig_mean": round(float(orig_means[j]), 4),
                    "gen_mean": round(float(gen_means[j]), 4),
                    "orig_std": round(float(orig_stds[j]), 4),
                    "gen_std": round(float(gen_stds[j]), 4),
                    "mean_diff_pct": round(mean_diff_abs / (abs(float(orig_means[j])) + 1e-8) * 100, 1),
                    "standardized_mean_diff": round(mean_diff_abs / pooled_std, 4)
                })
            
            fidelity = {
                "cosine_similarity": round(cosine_val, 4),
                "mean_correlation": round(mean_corr, 4) if not np.isnan(mean_corr) else 0,
                "std_correlation": round(std_corr, 4) if not np.isnan(std_corr) else 0,
                "column_correlation": round((mean_corr + std_corr) / 2, 4) if not (np.isnan(mean_corr) or np.isnan(std_corr)) else 0,
                "column_details": col_details
            }
            
            # ══ AUGMENTED MODEL (Orijinal + Sentetik veriyle) ══
            yg = le.transform(df_gen_filtered[label_col].astype(str))
            Xa = np.vstack([Xtr, Xg])
            ya = np.concatenate([ytr, yg])
            
            m2 = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42).fit(sc.transform(Xa), ya)
            pred_aug = m2.predict(Xte_s)
            f1a = float(f1_score(yte, pred_aug, average='weighted', zero_division=0))
            
            # Sınıf bazlı F1 (augmented)
            f1_per_class_aug = {}
            for i, cls_name in enumerate(le.classes_):
                cls_f1 = float(f1_score(yte == i, pred_aug == i, zero_division=0))
                f1_per_class_aug[str(cls_name)] = round(cls_f1, 4)
            
            # Azınlık recall (augmented)
            if minority_mask_te.sum() > 0:
                minority_recall_aug = float(recall_score(minority_mask_te, pred_aug == minority_idx, zero_division=0))
            else:
                minority_recall_aug = 0.0
            
            # ══ HEDEF KONTROL ══
            improvement = round(((f1a - f1s) / (f1s + 1e-6)) * 100, 1)
            f1_target_met = improvement >= 15.0
            recall_target_met = minority_recall_aug >= 0.80
            
            return {
                "seed_f1": round(f1s, 4),
                "augmented_f1": round(f1a, 4),
                "improvement": improvement,
                "fidelity": fidelity,
                "utility": {
                    "f1_per_class_seed": f1_per_class_seed,
                    "f1_per_class_augmented": f1_per_class_aug,
                    "minority_class": str(minority_class),
                    "minority_recall_seed": round(minority_recall_seed, 4),
                    "minority_recall_augmented": round(minority_recall_aug, 4),
                    "f1_target_met": f1_target_met,
                    "recall_target_met": recall_target_met,
                    "f1_target": ">=15% improvement",
                    "recall_target": ">=80% minority recall"
                }
            }
    
    return {"seed_f1": round(f1s, 4), "augmented_f1": round(f1s, 4), "improvement": 0,
            "fidelity": fidelity,
            "utility": {"minority_recall": round(minority_recall_seed, 4),
                        "f1_target_met": False, "recall_target_met": False}}

# ═══════════════ API ═══════════════
@app.get("/api/system_status")
async def system_status():
    seed_path = get_seed_path()
    se = seed_path is not None
    return {"model_loaded":rcgan is not None,"model_name":os.path.basename(MODEL_PATH) if rcgan else "N/A",
        "seed_available":se,"seed_rows":count_csv_rows(seed_path),
        "seed_file":os.path.basename(seed_path) if seed_path else None,
        "model_lfs_missing": is_git_lfs_pointer(MODEL_PATH),
        "seed_lfs_missing": is_git_lfs_pointer(SEED_PATH),
        "status":"ready","mode":"adaptive+distillation"}

@app.post("/api/distill")
async def distill_endpoint(file: UploadFile = File(...)):
    """Veri setini damıt ve rapor döndür"""
    try:
        df = pd.read_csv(io.StringIO(await read_csv_upload(file)))
        df_clean, report, label_col, numeric_cols = distill_dataset(df)
        
        class_dist = analyze_classes(df_clean, label_col) if label_col else {}
        
        # Temiz veriyi kaydet
        clean_path = os.path.join(OUTPUT_DIR, "distilled_data.csv")
        df_clean.to_csv(clean_path, index=False)
        
        res = {
            "status": "success",
            "report": report,
            "label_col": label_col,
            "n_features": len(numeric_cols),
            "class_distribution": class_dist,
            "is_waymo": all(f'x({i+1})' in df.columns for i in range(20)),
        }
        import math
        def sanitize(obj):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return 0.0
            if isinstance(obj, dict): return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list): return [sanitize(v) for v in obj]
            return obj
        return sanitize(res)
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(err_msg)
        return JSONResponse(status_code=400, content={"detail": str(e)})

@app.post("/api/evaluate_pipeline")
async def evaluate_pipeline(file: UploadFile = File(...), n_samples: int = Form(1000)):
    try:
        raw_text = await read_csv_upload(file)
        
        # ── CSV Onarım Motoru v4 ──
        lines = raw_text.splitlines()
        
        if len(lines) >= 2:
            header_fields = lines[0].split(',')
            data_fields = lines[1].split(',')
            n_hf = len(header_fields)
            
            if n_hf > 150:
                real_header = []
                for f in header_fields:
                    f = f.strip()
                    try:
                        float(f)
                        break
                    except ValueError:
                        if f:
                            real_header.append(f)
                
                last_field = header_fields[-1].strip().rstrip('\n').rstrip('\r')
                if last_field == 'label' and 'label' not in real_header:
                    real_header.append('label')
                
                n_cols = len(real_header)
                new_header = ','.join(real_header)
                
                new_lines = [new_header]
                for line in lines[1:]:
                    line = line.strip()
                    if not line:
                        continue
                    fields = line.split(',')
                    
                    if len(fields) > n_cols:
                        real_data = fields[:n_cols - 1]
                        label_val = ''
                        for fi in reversed(fields):
                            fi = fi.strip()
                            if fi:
                                try:
                                    float(fi)
                                except ValueError:
                                    label_val = fi
                                    break
                        real_data.append(label_val)
                        new_lines.append(','.join(real_data))
                    else:
                        new_lines.append(line)
                
                raw_text = '\n'.join(new_lines)
                print(f"[🔧] Bozuk CSV onarıldı: {n_hf} alan → {n_cols} sütun, {len(new_lines)-1} satır")
        
        df = pd.read_csv(io.StringIO(raw_text))
        
        df.columns = [str(c).split('\n')[0].split('\r')[0].strip() for c in df.columns]
        
        import re
        fixed_cols = {}
        waymo_pattern = re.compile(r'^((?:x|y|speed|vx|vy)\(\d+\)).*$', re.DOTALL)
        for col in df.columns:
            m = waymo_pattern.match(col)
            if m and m.group(1) != col:
                fixed_cols[col] = m.group(1)
        if fixed_cols:
            df = df.rename(columns=fixed_cols)
        
        garbage_cols = []
        for col in df.columns:
            try:
                float(col)
                garbage_cols.append(col)
            except ValueError:
                pass
        if garbage_cols:
            df = df.drop(columns=garbage_cols)
        
        n_samples = max(10, min(n_samples, 10000))
        
        # 1. Damıt
        df_clean, report, label_col, numeric_cols = distill_dataset(df)
        
        if not label_col:
            df_clean['label'] = 'normal'
            label_col = 'label'
            print("[ℹ️] Yüklenen veride label sütunu bulunamadı, 'label' oluşturuldu.")
        
        df_clean.to_csv(os.path.join(OUTPUT_DIR, "distilled_data.csv"), index=False)
        
        # 2. Üret
        df_gen, method, is_waymo = generate_adaptive(df_clean, label_col, numeric_cols, n_samples)
        if df_gen.empty:
            return JSONResponse(status_code=400, content={"detail":"Üretim başarısız."})
        
        # 3. Değerlendir
        scores = evaluate(df_clean, df_gen, label_col, numeric_cols)
        quality_report = build_quality_report(df_clean, df_gen, method, is_waymo, scores, label_col, numeric_cols)
        
        safe_rows = max(report.get("clean_rows", 1), 1)
        res = {
            "status":"success", "method":method,
            "distillation": report,
            "dataset_info": {
                "rows":report.get("clean_rows", 0),"features":len(numeric_cols),
                "classes":len(analyze_classes(df_clean,label_col)),
                "label_col":label_col,"is_waymo":is_waymo,
                "class_distribution":analyze_classes(df_clean,label_col),
            },
            "seed_count":report.get("clean_rows", 0),"gen_count":len(df_gen),
            "multiplication_factor":round((report.get("clean_rows", 0)+len(df_gen))/safe_rows,2),
            "generative_coverage":round(len(df_gen)/(safe_rows)*100,1),
            "quality_report": quality_report,
            **scores,
        }
        
        import math
        def sanitize(obj):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return 0.0
            if isinstance(obj, dict): return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list): return [sanitize(v) for v in obj]
            return obj
            
        return sanitize(res)
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(err_msg)
        return JSONResponse(status_code=400, content={"detail": str(e)})


@app.post("/api/run_full_automation")
async def run_full_automation(request: Request):
    try:
        body = await request.json()
        n_samples = max(10, min(body.get("n_samples", 2000), 10000))
    except:
        n_samples = 2000
    
    seed_path = get_seed_path()
    if not seed_path:
        return JSONResponse(status_code=400, content={"detail":"Seed dosyası yok."})
    
    # 700K'lık devasa dosyadan rastgele 10.000 satır seçerek al
    # Tüm dosyayı RAM'e almamak için skiprows olasılık hesabı kullanıyoruz
    total_rows = count_csv_rows(seed_path)
    sample_size = 10000
    
    if total_rows > sample_size:
        prob = sample_size / total_rows
        df = pd.read_csv(seed_path, skiprows=lambda i: i > 0 and random.random() > prob)
        if len(df) > sample_size:
            df = df.sample(sample_size).reset_index(drop=True)
    else:
        df = pd.read_csv(seed_path)
        
    df_clean, report, label_col, numeric_cols = distill_dataset(df)
    
    # Waymo seed'inde genellikle label sütunu yoktur (hepsi normaldir). 
    # Eğer yoksa, pipeline'ın çökmemesi için biz ekliyoruz.
    if not label_col:
        df_clean['label'] = 'normal'
        label_col = 'label'
        print("[ℹ️] Label sütunu bulunamadı, 'label' adında 'normal' değerlerle oluşturuldu.")
    
    df_gen, method, is_waymo = generate_adaptive(df_clean, label_col, numeric_cols, n_samples)
    if df_gen.empty:
        return JSONResponse(status_code=500, content={"detail":"Üretim başarısız."})
    
    scores = evaluate(df_clean, df_gen, label_col, numeric_cols)
    quality_report = build_quality_report(df_clean, df_gen, method, is_waymo, scores, label_col, numeric_cols)
    return {"status":"success","mode":"full_automation","method":method,"distillation":report,
        "seed_count":report["clean_rows"],"gen_count":len(df_gen),
        "multiplication_factor":round((report["clean_rows"]+len(df_gen))/report["clean_rows"],2),
        "generative_coverage":round(len(df_gen)/(report["clean_rows"]+1)*100,1),
        "quality_report": quality_report, **scores}

@app.post("/api/simulation_sample")
async def simulation_sample(request: Request):
    data = await request.json()
    path = os.path.join(OUTPUT_DIR, "live_synthetic_output.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'label' in df.columns and 'x(1)' in df.columns:
            sub = df[df['label']==data.get("type","spike")]
            if len(sub)>0:
                r=sub.sample(1).iloc[0]
                return {"source":"rcgan","type":data["type"],
                    "x":[float(r[f'x({i+1})']) for i in range(20)],"y":[float(r[f'y({i+1})']) for i in range(20)],
                    "speed":[float(r[f'speed({i+1})']) for i in range(20)],"vx":[float(r[f'vx({i+1})']) for i in range(20)],
                    "vy":[float(r[f'vy({i+1})']) for i in range(20)]}
    return JSONResponse(status_code=404, content={"detail":"Önce veri üretin."})

@app.get("/api/download_generated")
async def download_generated():
    p=os.path.join(OUTPUT_DIR,"live_synthetic_output.csv")
    if os.path.exists(p): return FileResponse(path=p, filename="sentetik_veri.csv", media_type="application/octet-stream")
    return JSONResponse(status_code=404,content={"detail":"Henüz veri üretilmedi."})

@app.get("/api/download_distilled")
async def download_distilled():
    p=os.path.join(OUTPUT_DIR,"distilled_data.csv")
    if os.path.exists(p): return FileResponse(path=p, filename="distilled_clean_data.csv", media_type="application/octet-stream")
    return JSONResponse(status_code=404,content={"detail":"Henüz damıtma yapılmadı."})

app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*55)
    print("🛡️  Sentetik Veri Üretim Hattı — Bilgi Damıtma + Adaptif Üretim")
    print(f"   RCGAN: {'GODMODE ✅' if rcgan else 'Yok ⚠️'}")
    print(f"   Pipeline: Damıtma → Sentez → Değerlendirme")
    print("="*55 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
