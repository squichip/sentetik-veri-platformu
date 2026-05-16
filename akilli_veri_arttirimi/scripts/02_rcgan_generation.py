"""
============================================================
 Akıllı Veri Üretimi - RCGAN (Recurrent Conditional GAN)
============================================================
 
 RCGAN mimarisi kullanıyoruz.
 * MPS cihazlarında WGAN-GP (ikinci türev) desteklenmediği 
   ve CPU'da haftalar süreceği için standart GAN (BCE) 
   ile eğiteceğiz.
 * MinMax Normalization (-1, 1) ve Generator Tanh çıkışı 
   kullanarak stabilitesi sağlanmıştır.
============================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os, warnings

warnings.filterwarnings('ignore')

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(42)
torch.manual_seed(42)

# MPS/CUDA/CPU seçimi - Standart GAN olduğu için MPS kullanılabilir
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Device: {device}")

plt.style.use('dark_background')
COLORS = {
    'bg': '#0f172a', 'text': '#f1f5f9', 'grid': '#1e293b',
    'normal': '#10b981', 'spike': '#ef4444', 'drift': '#f59e0b',
    'dropout': '#8b5cf6', 'freeze': '#3b82f6', 'noise': '#06b6d4',
    'gen': '#f472b6', 'real': '#10b981'
}

LABEL_MAP = {'normal': 0, 'spike': 1, 'drift': 2, 'dropout': 3, 'freeze': 4, 'noise': 5}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}
N_CLASSES = 6

print("=" * 60)
print("🧠 RCGAN - Recurrent Conditional GAN")
print("   LSTM tabanlı zaman serisi sentetik veri üretimi")
print("   Eğitim: MPS Hızlandırmalı Standart BCE GAN")
print("=" * 60)

# ============================================================
# 1. VERİ HAZIRLAMA
# ============================================================
fcd_files = {
    'Milano': 'data/fcd-italy/milan_fcd_prep.csv',
    'Roma': 'data/fcd-italy/rome_fcd_prep.csv',
    'Torino': 'data/fcd-italy/turin_fcd_prep.csv',
}

x_cols = None
all_clean = []
for city, path in fcd_files.items():
    df = pd.read_csv(path)
    if x_cols is None:
        x_cols = [c for c in df.columns if c.startswith('x(')]
        y_cols = [c for c in df.columns if c.startswith('y(')]
    clean = df[~df[x_cols + y_cols].isna().any(axis=1)]
    all_clean.append(clean)
pool = pd.concat(all_clean).sample(frac=1, random_state=42).reset_index(drop=True)

SEQ_LEN = len(x_cols)
N_CHANNELS = 2

def inject_spike(x, y):
    xa, ya = x.copy(), y.copy()
    idx = np.random.randint(0, len(x))
    xa[idx] += np.random.choice([-1,1]) * np.random.uniform(0.3, 0.6)
    ya[idx] += np.random.choice([-1,1]) * np.random.uniform(0.3, 0.6)
    return xa, ya

def inject_drift(x, y):
    xa, ya = x.copy(), y.copy()
    start = np.random.randint(5, 12)
    rate = np.random.uniform(0.01, 0.04)
    angle = np.random.uniform(0, 2*np.pi)
    for i in range(start, len(x)):
        xa[i] += (i-start) * rate * np.cos(angle)
        ya[i] += (i-start) * rate * np.sin(angle)
    return xa, ya

def inject_dropout(x, y):
    xa, ya = x.copy(), y.copy()
    n = int(len(x) * np.random.uniform(0.2, 0.4))
    start = np.random.randint(3, len(x) - n - 2)
    for i in range(start, start+n):
        xa[i] = xa[start-1]; ya[i] = ya[start-1]
    xa[start+n] += np.random.uniform(-0.15, 0.15)
    ya[start+n] += np.random.uniform(-0.15, 0.15)
    return xa, ya

def inject_freeze(x, y):
    xa, ya = x.copy(), y.copy()
    start = np.random.randint(4, 12)
    dur = np.random.randint(4, 10)
    for i in range(start, min(start+dur, len(x))):
        xa[i] = xa[start]; ya[i] = ya[start]
    return xa, ya

def inject_noise(x, y):
    xa, ya = x.copy(), y.copy()
    start = np.random.randint(3, 8)
    f = np.random.uniform(0.03, 0.1)
    for i in range(start, len(x)):
        xa[i] += np.random.normal(0, f*(1+(i-start)*0.25))
        ya[i] += np.random.normal(0, f*(1+(i-start)*0.25))
    return xa, ya

generators = {1: inject_spike, 2: inject_drift, 3: inject_dropout, 4: inject_freeze, 5: inject_noise}

N_SEED = 150
seed_sequences, seed_labels = [], []

for i in range(N_SEED):
    row = pool.iloc[i]
    x = np.array([row[c] for c in x_cols])
    y = np.array([row[c] for c in y_cols])
    seq = np.stack([x, y], axis=1)
    seed_sequences.append(seq)
    seed_labels.append(0)

for cid, gen_func in generators.items():
    for i in range(N_SEED):
        row = pool.iloc[N_SEED + (cid-1)*N_SEED + i]
        x = np.array([row[c] for c in x_cols])
        y = np.array([row[c] for c in y_cols])
        xa, ya = gen_func(x, y)
        seq = np.stack([xa, ya], axis=1)
        seed_sequences.append(seq)
        seed_labels.append(cid)

X_seed = np.array(seed_sequences, dtype=np.float32)
y_seed = np.array(seed_labels)

# MinMax (-1 to 1) 
x_min = X_seed.min(axis=(0,1), keepdims=True)
x_max = X_seed.max(axis=(0,1), keepdims=True)
X_seed_norm = 2.0 * (X_seed - x_min) / (x_max - x_min + 1e-8) - 1.0

# ============================================================
# 2. RCGAN MİMARİSİ
# ============================================================
NOISE_DIM = 32
HIDDEN_DIM = 64
N_LSTM_LAYERS = 2
EMBED_DIM = 16

class LSTMGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(N_CLASSES, EMBED_DIM)
        self.lstm = nn.LSTM(NOISE_DIM + EMBED_DIM, HIDDEN_DIM, N_LSTM_LAYERS, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, N_CHANNELS),
            nn.Tanh() # Çıktıyı -1, 1 aralığına sıkıştır.
        )
    def forward(self, noise, labels):
        batch, seq_len, _ = noise.shape
        emb = self.label_embed(labels).unsqueeze(1).expand(-1, seq_len, -1)
        lstm_input = torch.cat([noise, emb], dim=2)
        lstm_out, _ = self.lstm(lstm_input)
        return self.out(lstm_out)

class LSTMDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(N_CLASSES, EMBED_DIM)
        self.lstm = nn.LSTM(N_CHANNELS + EMBED_DIM, HIDDEN_DIM, N_LSTM_LAYERS, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1) # BCEWithLogitsLoss için sigmoid yok
        )
    def forward(self, seq, labels):
        batch, seq_len, _ = seq.shape
        emb = self.label_embed(labels).unsqueeze(1).expand(-1, seq_len, -1)
        lstm_input = torch.cat([seq, emb], dim=2)
        lstm_out, (h_n, _) = self.lstm(lstm_input)
        return self.classifier(h_n[-1])

G = LSTMGenerator().to(device)
D = LSTMDiscriminator().to(device)

print(f"\n🏗️ Mimariler MPS'te ({device}) başarıyla oluşturuldu.")

# ============================================================
# 3. EĞİTİM LOOP (BCE, MPS Destekli)
# ============================================================
EPOCHS = 3500
BATCH_SIZE = 64
LR = 0.0002

opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
criterion = nn.BCEWithLogitsLoss()

dataset_train = DataLoader(TensorDataset(torch.FloatTensor(X_seed_norm), torch.LongTensor(y_seed)), 
                           batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

d_losses, g_losses = [], []
REAL_LABEL, FAKE_LABEL = 0.9, 0.1 # Label Smoothing

print(f"🏋️ RCGAN (BCE kaydıyla) {device} üzerinde eğitiliyor... (Epochs: {EPOCHS})")

for epoch in range(EPOCHS):
    epoch_d, epoch_g = 0, 0
    
    for seqs, labels in dataset_train:
        b_size = seqs.size(0)
        seqs, labels = seqs.to(device), labels.to(device)
        
        # Discriminator Eğit
        opt_D.zero_grad()
        real_out = D(seqs, labels)
        d_loss_real = criterion(real_out, torch.full_like(real_out, REAL_LABEL))
        
        noise = torch.randn(b_size, SEQ_LEN, NOISE_DIM, device=device)
        fake_labels = torch.randint(0, N_CLASSES, (b_size,), device=device)
        fake_seqs = G(noise, fake_labels)
        
        fake_out = D(fake_seqs.detach(), fake_labels)
        d_loss_fake = criterion(fake_out, torch.full_like(fake_out, FAKE_LABEL))
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        opt_D.step()
        epoch_d += d_loss.item()
        
        # Generator Eğit
        opt_G.zero_grad()
        fake_out = D(fake_seqs, fake_labels)
        g_loss = criterion(fake_out, torch.full_like(fake_out, 1.0))
        g_loss.backward()
        opt_G.step()
        epoch_g += g_loss.item()
        
    d_losses.append(epoch_d / len(dataset_train))
    g_losses.append(epoch_g / len(dataset_train))
    
    if (epoch+1) % 500 == 0:
        print(f"  Epoch [{epoch+1:4d}/{EPOCHS}] | D: {d_losses[-1]:.4f} | G: {g_losses[-1]:.4f}")

# ============================================================
# 4. ÜRETİM VE FİDELİTY
# ============================================================
print("\n🔮 Sentetik yörüngeler üretiliyor...")
G.eval()
N_GEN = 500
gen_seqs, gen_labels = [], []

with torch.no_grad():
    for cid in range(N_CLASSES):
        l = torch.full((N_GEN,), cid, dtype=torch.long, device=device)
        n = torch.randn(N_GEN, SEQ_LEN, NOISE_DIM, device=device)
        f_seq = G(n, l).cpu().numpy()
        
        # Denormalize ([-1, 1] -> Orijinal)
        f_seq = (f_seq + 1.0) / 2.0 * (x_max - x_min + 1e-8) + x_min
        gen_seqs.append(f_seq)
        gen_labels.extend([cid] * N_GEN)

X_gen = np.concatenate(gen_seqs, axis=0)
y_gen = np.array(gen_labels)

from sklearn.metrics.pairwise import cosine_similarity
fidelity_results = {}
for cid in range(N_CLASSES):
    rmask, gmask = y_seed == cid, y_gen == cid
    rflat = X_seed[rmask].reshape(-1, SEQ_LEN * N_CHANNELS)
    gflat = X_gen[gmask].reshape(-1, SEQ_LEN * N_CHANNELS)
    
    cos_sim = cosine_similarity(rflat[:50], gflat[:50]).mean()
    mae = np.mean(np.abs(rflat[:50] - gflat[:50]))
    
    fidelity_results[LABEL_NAMES[cid]] = {'cosine': float(cos_sim), 'mae': float(mae)}
    status = "✅" if cos_sim > 0.80 else "⚠️"
    print(f"  {status} {LABEL_NAMES[cid]:10s} | Cosine: {cos_sim:.4f} | MAE: {mae:.4f}")

# Kaydet
def export_dataset(X, y, fname):
    df = pd.DataFrame(X.reshape(-1, SEQ_LEN * 2), columns=[f'x({i+1})' for i in range(SEQ_LEN)] + [f'y({i+1})' for i in range(SEQ_LEN)])
    df['label'] = [LABEL_NAMES[l] for l in y]
    df.to_csv(f'{OUTPUT_DIR}/{fname}', index=False)

export_dataset(X_gen, y_gen, 'rcgan_generated_dataset.csv')
export_dataset(np.concatenate((X_seed, X_gen), axis=0), np.concatenate((y_seed, y_gen)), 'rcgan_augmented_dataset.csv')
torch.save({'G': G.state_dict(), 'D': D.state_dict()}, f'{OUTPUT_DIR}/rcgan_model.pth')

# Görselleştirme
fig = plt.figure(figsize=(24, 18), facecolor=COLORS['bg'])
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

ax = fig.add_subplot(gs[0, 0], facecolor=COLORS['bg'])
ax.plot(d_losses, color='#ef4444', alpha=0.5, label='D Loss')
ax.plot(g_losses, color='#10b981', alpha=0.5, label='G Loss')
ax.legend()
ax.set_title("Eğitim Loss (BCE)", color=COLORS['text'])
ax.grid(alpha=0.1)

ax = fig.add_subplot(gs[0, 1], facecolor=COLORS['bg'])
classes = list(fidelity_results.keys())
ax.bar(classes, [fidelity_results[c]['cosine'] for c in classes], color=[COLORS.get(c, '#94a3b8') for c in classes])
ax.axhline(0.80, color='#fbbf24', linestyle='--')
ax.set_ylim(0, 1.1)
ax.set_title("Cosine Similarity", color=COLORS['text'])

for plot_idx, class_id in enumerate(range(6)):
    row, col = (1, plot_idx) if plot_idx < 3 else (2, plot_idx - 3)
    ax = fig.add_subplot(gs[row, col], facecolor=COLORS['bg'])
    color = COLORS.get(LABEL_NAMES[class_id], '#94a3b8')
    rs = X_seed[y_seed == class_id]
    fo = X_gen[y_gen == class_id]
    for i in range(10): ax.plot(rs[i,:,0], rs[i,:,1], color=color, alpha=0.4)
    for i in range(10): ax.plot(fo[i,:,0], fo[i,:,1], color=COLORS['gen'], alpha=0.6, linestyle='--')
    ax.set_title(LABEL_NAMES[class_id].upper(), color=color)

plt.savefig(f'{OUTPUT_DIR}/11_rcgan_results.png', dpi=150, facecolor=COLORS['bg'])
print("\n🎉 Tüm işlemler tamamlandı!")
