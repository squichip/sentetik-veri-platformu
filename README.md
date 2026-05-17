<h1 align="center">
  🚀 Sentetik Veri & Robustness Platformu
</h1>

<p align="center">
  <strong>Görüntü işleme ve yapısal/tabular veri setleri için uçtan uca sentetik veri üretimi, yapay zeka dayanıklılık (robustness) testi ve veri artırımı ekosistemi.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-orange" alt="PyTorch">
  <img src="https://img.shields.io/badge/YOLOv8-Object%20Detection-yellow" alt="YOLOv8">
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688" alt="FastAPI">
  <img src="https://img.shields.io/badge/Qt-Desktop%20UI-41CD52" alt="Qt">
</p>

---

## 🌟 Proje Vizyonu ve Kapsamı

Bu devasa depo (repository), makine öğrenmesi modellerinin karşılaştığı veri kıtlığı ve çevresel faktörlere karşı kırılganlık sorunlarını çözmek için tasarlanmış **iki ana endüstriyel çözümü** tek bir çatı altında toplar:

1. **📷 Görüntü Robustness Pipeline:** Temiz (clean) kamera görüntülerini alır, **RCGAN** mimarisi ile sentetik olarak bozar (bulanıklık, parlaklık değişimi vb.), **EDSR** ile çözünürlüğünü artırır ve en sonunda **YOLO / SegFormer** modellerinin bu zorlu şartlar altındaki performansını (robustness) analiz eder.
2. **📊 Akıllı Veri Artırımı:** Standart tabular CSV verilerini veya özel **Waymo Yörünge (Trajectory)** verilerini analiz eder; veri yapısına en uygun olan modeli (**RCGAN, CTGAN veya SMOTE**) otomatik seçerek otonom sürüş sistemleri ve sensör ağları için yüksek kaliteli sentetik veri üretir.

Merkezi `main_launcher.py` dosyası, her iki uygulamanın da tek bir tıklama ile yönetilebileceği ana komuta merkezidir.

---

## 📸 Platform Arayüzleri ve Görseller

### 1. Akıllı Veri Artırımı & Kontrol Paneli
<p align="center">
  <img src="docs/images/synthetic_data_dashboard.png" alt="Synthetic Data Dashboard" width="90%">
</p>

### 2. Bilgisayarlı Görü Dayanıklılık (Robustness) Test Akışı
<p align="center">
  <img src="docs/images/robustness_cv_pipeline.png" alt="Computer Vision Robustness Pipeline" width="90%">
</p>

---

## 🏗️ Sistem Mimarisi ve İş Akışı

Aşağıdaki şema, platformun iki kola ayrılan ana iş akışını özetlemektedir:

```mermaid
graph TD
    A[<b>main_launcher.py</b><br>Ana Komuta Merkezi] -->|Görüntü İşleme| B[RCGAN Qt Masaüstü Arayüzü]
    A -->|Sensör/Tabular Veri| C[Akıllı Veri Artırımı Web Arayüzü]
    
    %% Görüntü Pipeline
    B --> D[Clean Görüntüler]
    D --> E[RCGAN ile Sentetik Bozulma<br><i>Blur, Occlusion, vb.</i>]
    E --> F[EDSR ile Upscale]
    F --> G[YOLO & SegFormer<br>Robustness Analizi]
    
    %% Tabular Pipeline
    C --> H[CSV / Sensör / Yörünge Verisi]
    H --> I{Adaptive Algoritma Motoru}
    I -->|Tabular Veri| J[CTGAN]
    I -->|Waymo Yörünge| K[RCGAN]
    I -->|Küçük Veri| L[SMOTE + Gaussian]
    J & K & L --> M[Akademik Kalite Kontrolü<br><i>Fidelity & Utility Skorlaması</i>]
```

---

## 🔍 Detaylı Sistem Analizi

### 1. Görüntü Robustness Pipeline
Bilgisayarlı görü (CV) modellerinin gerçek hayat şartlarındaki (sis, aşırı yağış, sensör bozulmaları) başarısını test etmek amacıyla tasarlanmış zincirleme bir işlem mimarisidir.

* **Adım 1: RCGAN ile Yapay Bozulma (Synthesizing Perturbations):** RCGAN (Recurrent Conditional GAN) modeli eğitilerek, temiz kamera görüntülerine gerçekçi sis (blur), cisim engellemeleri (occlusion) ve ışık dalgalanmaları (brightness) sentetik olarak enjekte edilir.
* **Adım 2: EDSR ile Süper Çözünürlük (Super Resolution):** Üretilen sentetik ve bozuk görüntüler, **EDSR (Enhanced Deep Residual Single Image Super-Resolution)** modeliyle 4 kat büyütülür (upscale). Bu adım, pikselsel bozulmaları düzelterek nesne algılama başarısını artırmak için kritik öneme sahiptir.
* **Adım 3: YOLOv8 ile Nesne Algılama (Object Detection) Analizi:** Bozulmuş ve ardından upscale edilmiş görüntüler üzerindeki araç, yaya ve trafik işaretleri YOLOv8 ile taranır. Temiz görüntü ile bozulmuş görüntü arasındaki Güven (Confidence Score) ve Tespit Oranı (Recall) kaybı ölçülür.
* **Adım 4: SegFormer ile Semantik Segmentasyon (Semantic Segmentation):** Piksel düzeyinde sınıflandırma yapılarak (yol, kaldırım, gökyüzü, araçlar), derinlik ve mekansal farkındalığın yapay koşullarda ne kadar saptığı ölçülür.

### 2. Akıllı Veri Artırımı (Smart Data Augmentation Engine)
Sistem, yüklenen veri setinin yapısını inceleyen bir **Otomatik Sınıflandırma ve Model Belirleme Motoru** içerir:

* **CTGAN (Conditional Tabular GAN):** Eğer yüklenen veri seti standart tabular / sayısal sensör verisi ise ve yeterli boyuttaysa (100 satır üstü), CTGAN devreye girer. CTGAN, sürekli ve kategorik sütunların olasılık dağılımlarını koruyarak sıfırdan yüksek kaliteli sentetik satırlar üretir.
* **RCGAN (Recurrent Conditional GAN):** Veri setinde yörünge (trajectory) bilgileri, zaman serileri veya ardışık otonom sürüş koordinatları (`x(1)...x(20)`, `speed`, `vx`, `vy`) bulunursa, model otomatik olarak RCGAN motoruna yönlendirilir. RCGAN, zaman serisindeki adımların birbirleriyle olan korelasyonunu korur.
* **SMOTE + Gaussian Fallback:** Veri boyutu 100 satırın altındaysa, derin öğrenme tabanlı GAN modelleri ezberleme (overfitting) yapacağı için, sistem otomatik olarak SMOTE ve Gaussian gürültü ekleme algoritmasına geçiş yapar.

#### 📊 Akademik Değerlendirme (Fidelity & Utility) Metrikleri
Üretilen verinin kalitesi sadece görsel değil, matematiksel ve istatistiksel testlerden geçirilir:
* **Fidelity (Benzerlik) Analizi:**
  * *Cosine Similarity:* Orijinal ve sentetik veri kümelerinin ortalama vektörleri arasındaki kosinüs benzerliği ölçülür.
  * *Correlation Comparison:* Sütunların kendi aralarındaki korelasyon matrisleri karşılaştırılarak, sentetik verinin orijinal verideki ilişkileri koruma başarısı test edilir.
* **Utility (Kullanılabilirlik/Yararlılık) Analizi:**
  * Orijinal veri üzerinde bir **Gradient Boosting Classifier** eğitilir (`Seed Model`) ve test edilir.
  * Ardından, orijinal + sentetik verinin birleşimi üzerinde aynı parametrelerle yeni bir model eğitilir (`Augmented Model`).
  * İki model aynı bağımsız test verisinde yarıştırılır. Sentetik verinin model başarısını (özellikle veri seti dengesiz ise azınlık sınıfı recall değerini) ne kadar artırdığı raporlanır.

---

## 📁 Proje Yapısı (Directory Structure)

```text
📦 projects/
 ┣ 📂 akilli_veri_arttirimi/   # Tabular/Yörünge veri artırımı (CTGAN/RCGAN) platformu
 ┃ ┣ 📂 backend/               # FastAPI sunucusu ve veri işleme motorları
 ┃ ┣ 📂 docs/                  # Akıllı Veri Artırımı özel dokümantasyonu
 ┃ ┣ 📂 otonom_env/            # Veri artırımı özel Python sanal ortamı (venv)
 ┃ ┗ 📜 requirements.txt       # Veri artırımı kütüphaneleri (PyTorch, CTGAN, vb.)
 ┣ 📂 detector/                # YOLO, SegFormer ve EDSR yapay zeka modelleri
 ┣ 📂 rcgan_qt_gui_app_v1/     # Görüntü Robustness için Qt tabanlı kullanıcı arayüzü
 ┃ ┣ 📂 qtvenv/                # Görüntü işleme özel Python sanal ortamı (venv)
 ┃ ┗ 📜 requirements_qt.txt    # Görüntü işleme kütüphaneleri (Qt, Ultralytics, vb.)
 ┣ 📂 docs/                    # Genel proje dokümantasyonu ve görseller
 ┃ ┗ 📂 images/                # Dashboard ve Infografik görselleri
 ┣ 📂 clean/                   # Referans alınan temiz kamera görüntüleri
 ┣ 📂 outputs/                 # Üretilen sentetik ve bozulmuş görüntüler
 ┣ 📂 results/                 # Metrikler, tespit haritaları ve kapsamlı performans raporları
 ┣ 📜 main_launcher.py         # Tüm sistemi başlatan ana kontrol ekranı
 ┗ 📜 PROJE_NOTLARI.md         # Kapsamlı geliştirme ve mühendislik notları
```

---

## ⚙️ Git LFS (Büyük Dosya Yönetimi)

Bu projede devasa boyutlu Derin Öğrenme modelleri ve dev veri setleri **Git LFS** ile barındırılmaktadır. Repoyu klonlamadan önce Git LFS'in kurulu olması **ZORUNLUDUR**.

Takip edilen LFS uzantıları: `*.pt`, `*.pth`, `*.pb`, `waymo_seed_MASSIVE.csv`

**macOS Kurulumu:**
```bash
brew install git-lfs
git lfs install
```

**Windows Kurulumu:**
```powershell
git lfs install
```

*(Eğer modeller eksik inerse veya 1 KB boyutunda görünürse, depo dizinindeyken `git lfs pull` komutunu çalıştırın.)*

---

## 🚀 Başlangıç ve Sıfırdan Kurulum

### 1. Repoyu Klonlama

```bash
git clone https://github.com/squichip/sentetik-veri-platformu.git
cd sentetik-veri-platformu
git lfs pull
```

### 2. Sanal Ortamlar (Virtual Environments)
Sistem çakışmalarını önlemek için projede iki farklı sanal ortam (venv) kullanılmaktadır:
* `qtvenv`: Görüntü arayüzü ve bilgisayarlı görü (CV) modelleri için.
* `otonom_env`: Veri artırımı ve istatistiksel modeller için.

---

## 🛠️ Kurulum Talimatları

<details open>
<summary><b>1️⃣ Görüntü Robustness Pipeline Kurulumu</b></summary>
<br>

**1. Sanal Ortam Oluştur:**
```bash
cd rcgan_qt_gui_app_v1
python -m venv qtvenv
source qtvenv/bin/activate
python -m pip install --upgrade pip
```
*(Windows için: `qtvenv\Scripts\activate`)*

**2. Gereksinimleri Yükle:**
```bash
pip install -r requirements_qt.txt
pip install opencv-python matplotlib pandas tqdm ultralytics transformers
```
*Not: YOLO ve SegFormer modelleri, ilk çalıştırmada Hugging Face ve Ultralytics üzerinden gerekli model ağırlıklarını indirecektir.*

**3. Uygulamayı Başlat:**
```bash
cd ..
python main_launcher.py
```
</details>

<details open>
<summary><b>2️⃣ Akıllı Veri Artırımı Kurulumu (ÖNEMLİ)</b></summary>
<br>

**1. Sanal Ortam Oluştur:**
```bash
cd akilli_veri_arttirimi
python -m venv otonom_env
source otonom_env/bin/activate
python -m pip install --upgrade pip
```
*(Windows için: `otonom_env\Scripts\activate`)*

**2. Gereksinimleri Yükle:**
```bash
pip install -r requirements.txt
```

**3. LFS Modellerini Kontrol Et:**
Şu büyük dosyaların tam boyutuyla indiğinden emin olun (Gerekirse `git lfs pull` yapın):
* `akilli_veri_arttirimi/waymo_seed_MASSIVE.csv`
* `akilli_veri_arttirimi/outputs/waymo_rcgan_GODMODE_A100_STABLE.pth`

**4. Uygulamayı Başlat:**
```bash
python main.py
```
Veya doğrudan kök klasörden ana launcher ile başlatabilirsiniz:
```bash
cd /path/to/sentetik-veri-platformu
source rcgan_qt_gui_app_v1/qtvenv/bin/activate
python main_launcher.py
```
*(Launcher, arka planda otonom_env'yi bularak sunucuyu doğru ortamda başlatır.)*
</details>

---

## 📡 Özel Mühendislik Çözümü: WebKit Timeout Aşımı
FastAPI backend mimarisinde, çok derin CTGAN eğitimleri veya büyük veri setlerinin Gradient Boosting ile değerlendirilmesi 1 dakikadan uzun sürebilmektedir. macOS WebKit (`pywebview` masaüstü motoru) varsayılan olarak 60. saniyede ağ isteklerini zaman aşımına uğratıp bağlantıyı kesmekteydi.

**Nasıl Çözdük?**
Arayüzdeki standart `fetch` API'sini tamamen kaldırıp, yerine **XMLHttpRequest (XHR)** mimarisine geçiş yaptık ve arayüze özel olarak **300.000 ms (5 dakika) zaman aşımı** tanımladık:
```javascript
const xhr = new XMLHttpRequest();
xhr.open("POST", API + '/api/evaluate_pipeline', true);
xhr.timeout = 300000; // 5 dakikalık genişletilmiş WebKit desteği
```
Bu sayede en ağır veri setleri dahi arayüz bağlantısı kopmadan en üst kalitede eğitilip değerlendirilebilmektedir.

---

## 💡 Sık Karşılaşılan Sorunlar (Troubleshooting)

| Sorun | Çözüm Yöntemi |
|---|---|
| **Modeller veya CSV'ler Çalışmıyor (1 KB Görünüyor)** | Git LFS kurulmamış. `git lfs install` ve ardından `git lfs pull` komutunu çalıştırın. |
| **`ModuleNotFoundError` Hatası Alıyorum** | Yanlış sanal ortamdasınız. Görüntü modülü için `qtvenv`, Veri modülü için `otonom_env`'yi aktif edin (`source bin/activate`). |
| **Port 8000 veya 8001 Kullanımda Hatası** | Arka planda açık kalmış `python main.py` veya `server.py` sürecini terminalden sonlandırın (Ctrl+C). |
| **`Load failed` veya Zaman Aşımı Hatası (Veri Artırımı)** | Aşırı büyük veri setlerinde (CTGAN ile) sistem uzun sürebilir. *Not: Altyapı artık 5 dakikalık genişletilmiş WebKit XHR timeout desteğiyle çalışmaktadır.* |
| **macOS `AVFFrameReceiver` Uyarısı** | `av` ve `opencv-python` kütüphanelerinin C++ çakışmasından kaynaklanan zararsız bir uyarıdır. |

---

## 👨‍💻 Geliştirme ve Mimari Kararlar

Projenin altında yatan teorik yaklaşımlar, yaşanan darboğazlar ve alınan mimari kararlar (örneğin; neden Time-Series için Min-Max yerine fiziksel limitasyon uygulandığı, macOS kilitlenmelerini aşmak için FastAPI Streaming/XHR çözümlerinin nasıl uygulandığı) hakkında kapsamlı okuma yapmak için kök dizindeki **`PROJE_NOTLARI.md`** belgesini inceleyebilirsiniz.

---
<p align="center">
  <i>Squichip tarafından yüksek endüstriyel standartlarla tasarlanmıştır.</i>
</p>
