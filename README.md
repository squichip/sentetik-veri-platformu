# Sentetik Veri Platformu

Bu repo iki ana uygulamayi tek proje altinda toplar:

1. **Goruntu Robustness Pipeline**
   Clean kamera frame'lerinden RCGAN ile bozulmus goruntu uretir, EDSR ile upscale eder, ardindan YOLO ve SegFormer ile robustness analizi yapar.

2. **Akilli Veri Artirimi**
   CSV/tabular/yorunge verisini damitir; veri tipine gore RCGAN, CTGAN veya SMOTE ile sentetik veri uretir ve utility/fidelity raporlari olusturur.

Kokteki `main_launcher.py` iki uygulama icin ana secim ekranidir.

## Proje Yapisi

```text
projects/
├── main_launcher.py                  # Ana secim ekrani
├── PROJE_NOTLARI.md                  # Gelistirme notlari ve proje analizi
├── rcgan_qt_gui_app_v1/              # Goruntu RCGAN Qt arayuzu
├── detector/                         # YOLO, SegFormer ve EDSR adimlari
├── akilli_veri_arttirimi/            # CSV/tabular sentetik veri platformu
├── clean/                            # Secilen clean goruntu kopyalari
├── outputs/                          # Uretilen gorsel ciktilar
└── results/                          # Metrikler, grafikler ve raporlar
```

## Git LFS

Bu projede buyuk model ve veri dosyalari Git LFS ile tutulur. Repoyu klonlamadan once Git LFS kurulu olmalidir.

LFS ile takip edilen dosya tipleri:

```text
*.pt
*.pth
*.pb
akilli_veri_arttirimi/waymo_seed_MASSIVE.csv
```

macOS:

```bash
brew install git-lfs
git lfs install
```

Windows:

```powershell
git lfs install
```

Git LFS kurulu degilse `.pt`, `.pth`, `.pb` veya buyuk `.csv` dosyalari gercek icerik yerine kucuk pointer dosyasi olarak iner.

## Sifirdan Kurulum

### 1. Repoyu klonla

```bash
git clone https://github.com/squichip/my-project.git
cd my-project
git lfs pull
```

### 2. Python surumu

Onerilen surumlar:

```text
Python 3.10, 3.11 veya 3.12
```

Projede iki farkli sanal ortam kullanmak daha sagliklidir:

- `rcgan_qt_gui_app_v1/qtvenv`: goruntu arayuzu ve detector pipeline'i
- `akilli_veri_arttirimi/otonom_env`: CSV/tabular veri artirimi arayuzu

## Goruntu Robustness Pipeline Kurulumu

### 1. Sanal ortam olustur

```bash
cd rcgan_qt_gui_app_v1
python -m venv qtvenv
source qtvenv/bin/activate
python -m pip install --upgrade pip
```

Windows:

```powershell
cd rcgan_qt_gui_app_v1
python -m venv qtvenv
qtvenv\Scripts\activate
python -m pip install --upgrade pip
```

### 2. Python paketlerini yukle

```bash
pip install -r requirements_qt.txt
pip install opencv-python matplotlib pandas tqdm ultralytics transformers
```

Notlar:

- `torch` ve `torchvision` sistemine gore CPU veya GPU paketi olarak kurulabilir.
- Apple Silicon Mac'te PyTorch MPS otomatik kullanilabilir.
- YOLO ilk calismada gerekirse ek model agirliklarini indirebilir.
- SegFormer modeli Hugging Face uzerinden cekildigi icin ilk calistirmada internet gerekebilir.

### 3. Gerekli buyuk dosyalari kontrol et

Bu dosyalar Git LFS ile gelmelidir:

```text
rcgan_qt_gui_app_v1/checkpoint_epoch_29.pt
detector/EDSR_x4.pb
detector/yolov8n.pt
```

Eger dosyalar cok kucuk gorunuyorsa:

```bash
git lfs pull
```

### 4. Uygulamayi calistir

`rcgan_qt_gui_app_v1` klasorundeyken:

```bash
python qt_gui_app_updated.py
```

Kok ana ekrandan calistirmak icin:

```bash
cd ..
python main_launcher.py
```

## Goruntu Pipeline Akisi

Arayuzde clean frame'ler secilir. Pipeline sirayla:

1. Secilen clean goruntuleri `clean/` klasorune kopyalar.
2. RCGAN ile hatali goruntuleri `outputs/gan_generated/` altina uretir.
3. EDSR ile upscale edip `outputs/gan_upscaled/` altina yazar.
4. Detector veri setini `detector/robustness_dataset/` altinda hazirlar.
5. YOLO sonuclarini `results/yolo/` ve `outputs/yolo_comparisons/` altina yazar.
6. Segmentasyon sonuclarini `results/segmentation/`, `outputs/segmentation_outputs/` ve `outputs/segmentation_comparisons/` altina yazar.

Arayuzde tek hata tipi veya 3 hata tipi icin ayri seviyeler secilebilir:

```text
blur: low / medium / high
occlusion: low / medium / high
brightness: low / medium / high
```

## Akilli Veri Artirimi Kurulumu

### 1. Sanal ortam olustur

```bash
cd akilli_veri_arttirimi
python -m venv otonom_env
source otonom_env/bin/activate
python -m pip install --upgrade pip
```

Windows:

```powershell
cd akilli_veri_arttirimi
python -m venv otonom_env
otonom_env\Scripts\activate
python -m pip install --upgrade pip
```

### 2. Paketleri yukle

```bash
pip install -r requirements.txt
```

### 3. Gerekli buyuk dosyalari kontrol et

Bu dosyalar Git LFS ile gelmelidir:

```text
akilli_veri_arttirimi/waymo_seed_MASSIVE.csv
akilli_veri_arttirimi/outputs/waymo_rcgan_GODMODE_A100_STABLE.pth
```

Eger dosyalar pointer olarak geldiyse:

```bash
git lfs pull
```

### 4. Uygulamayi calistir

```bash
python main.py
```

Bu komut FastAPI backend'i baslatir ve masaustu webview penceresini acar.

Sadece web sunucusunu acmak istersen:

```bash
python backend/server.py
```

Sonra tarayicida:

```text
http://127.0.0.1:8000
```

## Ana Launcher Ile Calistirma

Kok klasorde, goruntu ortamini aktif ederek:

```bash
cd /path/to/my-project
source rcgan_qt_gui_app_v1/qtvenv/bin/activate
python main_launcher.py
```

Windows:

```powershell
cd C:\path\to\my-project
rcgan_qt_gui_app_v1\qtvenv\Scripts\activate
python main_launcher.py
```

Ana ekran iki secenek sunar:

- **Goruntu Modelini Ac**
- **Veri Artirimi Modelini Ac**

Veri artirimi icin `akilli_veri_arttirimi/otonom_env` varsa launcher onu kullanir.

## Sik Karsilasilan Hatalar

### LFS dosyalari kucuk gorunuyor

```bash
git lfs install
git lfs pull
```

### `ModuleNotFoundError` aliyorum

Ilgili sanal ortami aktif ettiginden emin ol:

```bash
source rcgan_qt_gui_app_v1/qtvenv/bin/activate
```

veya:

```bash
source akilli_veri_arttirimi/otonom_env/bin/activate
```

Sonra requirements dosyasini tekrar yukle.

### Port 8000 kullanimda

Daha once acik kalmis `python main.py` veya `backend/server.py` surecini kapat.

### macOS `AVFFrameReceiver` uyarisi

`av` ve `opencv-python` paketlerinin icindeki ffmpeg kutuphaneleri ayni Objective-C siniflarini yuklediginde gorulebilir. Genellikle uyari seviyesindedir; crash olursa ayni ortamda `av` paketini kaldirmak veya temiz sanal ortam kurmak denenebilir.

### Hugging Face token uyarisi

SegFormer modeli indirirken token yoksa hiz limiti uyarisi gorulebilir. Zorunlu degildir; cok sik indirme yapiliyorsa `HF_TOKEN` ayarlanabilir.

## Gelistirme Notlari

- Detayli proje notlari icin `PROJE_NOTLARI.md` dosyasina bak.
- Sanal ortamlar ve cache dosyalari git'e alinmaz.
- Buyuk model/veri dosyalari Git LFS ile takip edilir.
- Ciktilar `outputs/` ve `results/` altinda duzenli tutulur.
