# 🚀 Akıllı Sentetik Veri Artırım Platformu - Kurulum Rehberi

Bu rehber, projenin hem **macOS** hem de **Windows** sistemlerde masaüstü uygulaması olarak kurulup çalıştırılmasını sağlamak için hazırlanmıştır.

> [!WARNING]
> **ÖNEMLİ (Git LFS):** Projede yer alan büyük veri seti (`waymo_seed_MASSIVE.csv`) ve RCGAN model dosyası standart GitHub limitlerini aştığı için **Git LFS (Large File Storage)** ile tutulur. Projeyi klonlamadan önce sisteminizde Git LFS kurulu olmalıdır.

> [!NOTE]
> Önerilen Python sürümü: **Python 3.11**. Proje masaüstü pencereyi `main.py` ile açar; `backend/server.py` yalnızca geliştirme/opsiyonel web sunucusu olarak çalıştırılmalıdır.

---

## 🍎 macOS İçin Kurulum Adımları

### 1. Ön Koşullar ve Git LFS Kurulumu
Eğer sisteminizde Homebrew yüklüyse terminali açıp şu komutları sırasıyla çalıştırın:
```bash
# Git LFS yükle ve etkinleştir
brew install git-lfs
git lfs install
```

### 2. Projeyi Klonlama
Git LFS aktif edildikten sonra projeyi bilgisayarınıza indirin:
```bash
git clone https://github.com/aliturhan0/akilli_veri_arttirimi.git
cd akilli_veri_arttirimi
git lfs pull
```

### 3. Sanal Ortam (Virtual Environment) Kurulumu
Sisteminizdeki Python paketleriyle çakışmaması için projenin kendi izole ortamını oluşturun:
```bash
# Sanal ortamı oluştur
python3 -m venv otonom_env

# Sanal ortamı aktif et
source otonom_env/bin/activate
```
*(Terminal satırının başında `(otonom_env)` yazısını görmelisiniz.)*

### 4. Gerekli Kütüphanelerin Yüklenmesi
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Sistemi Başlatma
```bash
python main.py
```
Komut çalışınca uygulama yerel FastAPI sunucusunu arka planda başlatır ve masaüstü penceresini otomatik açar.

> Geliştirme için yalnızca web sunucusunu açmak isterseniz `python backend/server.py` komutunu kullanabilir, ardından **http://127.0.0.1:8000** adresinden arayüze erişebilirsiniz.

---

## 🪟 Windows İçin Kurulum Adımları

### 1. Ön Koşullar ve Git LFS Kurulumu
Windows için Git LFS eklentisini indirip kurmamız gerekiyor:
1. [Git LFS Resmi Sitesine (git-lfs.github.com)](https://git-lfs.github.com/) gidin ve indirip kurun.
2. Kurulum bittikten sonra **Komut İstemcisi (CMD)** veya **PowerShell**'i açın ve şu komutu yazın:
```cmd
git lfs install
```

### 2. Projeyi Klonlama
CMD veya PowerShell üzerinden projenin inmesini istediğiniz klasöre gidip klonlayın:
```cmd
git clone https://github.com/aliturhan0/akilli_veri_arttirimi.git
cd akilli_veri_arttirimi
git lfs pull
```

### 3. Sanal Ortam (Virtual Environment) Kurulumu
```cmd
# Sanal ortamı oluştur
python -m venv otonom_env

# Sanal ortamı aktif et
otonom_env\Scripts\activate
```
*(Komut satırının başında `(otonom_env)` yazısını görmelisiniz.)*

### 4. Gerekli Kütüphanelerin Yüklenmesi
```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Sistemi Başlatma
```cmd
python main.py
```
Komut çalışınca uygulama yerel FastAPI sunucusunu arka planda başlatır ve masaüstü penceresini otomatik açar.

> Geliştirme için yalnızca web sunucusunu açmak isterseniz `python backend\server.py` komutunu kullanabilir, ardından **http://127.0.0.1:8000** adresinden arayüze erişebilirsiniz.

---

## ❓ Olası Hatalar ve Çözümleri

* **Hata:** Projeyi klonladım ama `.csv` veya `.pth` dosyaları 1-2 KB boyutunda görünüyor.
  * **Çözüm:** Bilgisayarınızda Git LFS kurulu değil veya aktif edilmemiş. `git lfs install` yaptıktan sonra proje klasörünün içinde `git lfs pull` komutunu çalıştırarak büyük dosyaların orijinal hallerini çekebilirsiniz.
* **Hata:** `ModuleNotFoundError: No module named 'fastapi'` (veya benzeri)
  * **Çözüm:** Sanal ortamı (otonom_env) aktif etmeyi unutmuş olabilirsiniz. Adım 3'teki aktivasyon komutunu tekrar çalıştırın ve kütüphaneleri kurduğunuzdan emin olun.
* **Hata:** Uygulama penceresi açılmıyor ama terminalde sunucu çalışıyor gibi görünüyor.
  * **Çözüm:** `pywebview` kurulumu eksik olabilir. Sanal ortam aktifken `pip install -r requirements.txt` komutunu tekrar çalıştırın. Geçici olarak `python backend/server.py` ile web modunu açıp tarayıcıdan test edebilirsiniz.
* **Hata:** `Address already in use` veya port 8000 kullanımda hatası alıyorum.
  * **Çözüm:** Daha önce açık kalan uygulama penceresini/terminalini kapatın ve `python main.py` komutunu yeniden çalıştırın.
