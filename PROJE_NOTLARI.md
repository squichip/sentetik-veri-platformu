# Proje Notlari

Bu depo, RCGAN tabanli sentetik kamera bozulmasi ureten bir masaustu uygulamasi ve bu uretimlerin algilama/segmentasyon modelleri uzerindeki etkisini olcen deney scriptlerinden olusuyor.

## Ana Ekran / Uygulama Secici

Proje kokune `main_launcher.py` eklendi. Bu dosya tek bir ana ekran acar ve iki uygulama secenegi sunar:

- `Goruntu Robustness Pipeline`: `rcgan_qt_gui_app_v1/qt_gui_app_updated.py` uygulamasini ayri surec olarak acar.
- `Akilli Veri Artirimi`: `akilli_veri_arttirimi/main.py` uygulamasini ayri surec olarak acar.

Ana ekran calistirma:

```bash
cd /Users/ozcan/Desktop/projects
source rcgan_qt_gui_app_v1/qtvenv/bin/activate
python main_launcher.py
```

Not: `Akilli Veri Artirimi` uygulamasi kendi agir bagimliliklerine sahip. `akilli_veri_arttirimi/otonom_env/bin/python` varsa launcher onu kullanir; yoksa mevcut Python ile calistirmayi dener. Eksik paket hatasi alinirsa `akilli_veri_arttirimi` icin ayri sanal ortam kurulmalidir.

Launcher guncellendi: `Akilli Veri Artirimi` acilmadan once gerekli Python modullerini kontrol eder. Eksik paket varsa artik sessizce beklemez; log alaninda ve popup mesajinda hangi modullerin eksik oldugunu ve kurulacak komutlari gosterir.

Bu uygulama icin onerilen kurulum:

```bash
cd /Users/ozcan/Desktop/projects/akilli_veri_arttirimi
python3 -m venv otonom_env
source otonom_env/bin/activate
pip install -r requirements.txt
```

## Ana Parcalar

### `rcgan_qt_gui_app_v1/`

RCGAN ConvLSTM modelini kullanarak ardışık clean kamera frame'lerinden bozulmus goruntu uretir.

- `qt_gui_app_updated.py`: PySide6 tabanli masaustu arayuz. Checkpoint, input frame listesi, hata tipi ve siddet secilir; tek kosul veya tum 9 kosul icin zaman serisi uretimi baslatilir.
- `generate.py`: CLI ve GUI tarafinin kullandigi uretim fonksiyonlari. `blur`, `occlusion`, `brightness` hata tiplerini ve `low`, `medium`, `high` siddetlerini destekler.
- `model.py`: `RecurrentGenerator` mimarisi. Encoder + ConvLSTM + decoder yapisi ile onceki ve mevcut frame'i kosul haritasi ile isler.
- `checkpoint_epoch_29.pt`: GUI'nin varsayilan olarak bekledigi egitilmis model agirligi.
- `input/`: Ornek clean zaman serisi goruntuleri.
- `outputs_qt_sequence/`: Daha once uretilmis ornek zaman serisi ciktilari.
- `requirements_qt.txt`: Temel bagimliliklar: `torch`, `torchvision`, `pillow`, `numpy`, `PySide6`.

GUI calistirma:

```bash
cd /Users/ozcan/Desktop/projects/rcgan_qt_gui_app_v1
source qtvenv/bin/activate
python qt_gui_app_updated.py
```

CLI ile ornek zaman serisi uretimi:

```bash
python generate.py \
  --checkpoint checkpoint_epoch_29.pt \
  --input_folder input \
  --out outputs_qt_sequence \
  --fault brightness \
  --severity high
```

Not: Model tek fotografla calismaz. En az iki ardışık clean frame gerekir. Ilk frame icin cikti uretilmez; 2. frame'den itibaren `prev + curr` ciftleriyle cikti uretilir.

### `detector/`

Uretilen sentetik bozulmalarin bilgisayarli goru modellerine etkisini olcer.

- `robustness_dataset/clean`: Clean referans goruntuler.
- `robustness_dataset/generated/{blur_high,brightness_high,occlusion_high}`: Uretilmis bozulmus goruntuler.
- `semantic_segmentation_robustness.py`: SegFormer Cityscapes modeli ile clean ve generated goruntulerin segmentasyon tahminlerini karsilastirir.
- `yolo_robustness_evaluation.py`: YOLOv8n ile nesne sayisi ve confidence degisimini olcer.
- `upscale_generated_images.py`: Generated goruntuleri EDSR super-resolution modeli ile 1600x900'e buyutur.
- `segmentation_visual_comparison.py`: Clean/generated goruntu ve segmentasyonlarini 2x2 akademik gorsel olarak yan yana koyar.
- `plot_academic_results.py`: Segmentasyon metrikleri icin ortalama/std iceren akademik grafikler uretir.
- `results/`: Segmentasyon metrik CSV'leri ve grafikler.
- `yolo_results/`: YOLO metrik CSV'leri ve grafikler.
- `EDSR_x4.pb`, `yolov8n.pt`: Detector tarafinda kullanilan model dosyalari.

## Mevcut Sonuc Ozeti

Segmentasyon ozeti (`detector/results/robustness_summary.csv`):

- `blur_high`: prediction IoU yaklasik `0.525`, robustness drop yaklasik `0.475`.
- `brightness_high`: prediction IoU yaklasik `0.542`, robustness drop yaklasik `0.458`.
- `occlusion_high`: prediction IoU yaklasik `0.366`, robustness drop yaklasik `0.634`.

YOLO ozeti (`detector/yolo_results/yolo_summary.csv`):

- Clean ortalama nesne sayisi: `11.928`.
- `blur_high`: generated nesne sayisi `9.224`, detection drop `2.704`.
- `brightness_high`: generated nesne sayisi `7.576`, detection drop `4.352`.
- `occlusion_high`: generated nesne sayisi `7.888`, detection drop `4.040`.

Bu sonuclara gore mevcut deneylerde segmentasyon icin en yikici kosul `occlusion_high`, YOLO nesne sayisi dususu icin ise `brightness_high` gorunuyor.

## Dikkat Edilecekler

- Klasorlerde `.DS_Store`, `__pycache__`, uretilmis PNG'ler ve model agirliklari var; gelistirme yaparken bunlari gereksiz yere degistirmemek iyi olur.
- `detector` scriptleri goreli path kullaniyor. Bu scriptleri calistirirken calisma dizini `detector/` olmali.
- SegFormer modeli Hugging Face'ten yukleniyor; ilk calistirmada internet/model cache gerekebilir.
- YOLO scripti lokal `yolov8n.pt` dosyasini bekliyor.
- Git durum komutu bu klasorde cok yavas yanit verebiliyor; buyuk model/goruntu dosyalari buna sebep olabilir.

## Muhtemel Gelistirme Yollari

- GUI'ye ilerleme cubugu, iptal butonu ve cikti klasoru/condition bazli daha duzenli kayit eklenebilir.
- `generate.py` icin daha net hata mesajlari ve checkpoint/device secimi iyilestirilebilir.
- `detector` scriptleri ortak config dosyasina baglanabilir; ayni path ve condition listeleri tekrar ediyor.
- Deney sonuclari tek HTML/PDF rapora veya dashboard'a donusturulebilir.
- Uretim ciktilari otomatik olarak `detector/robustness_dataset/generated/...` yapisina aktarilabilir.

## Hedef Uygulama

Asil hedef, RCGAN uretim tarafi ile detector analiz tarafini tek bir arayuz uygulamasinda birlestirmek.

Kullanici arayuzden clean veri seti/frame klasoru sececek. Uygulama once RCGAN ile secilen hata tipleri ve siddetleri icin 256x256 bozulmus goruntuleri uretecek. Sonra bu ciktilari detector tarafinin bekledigi veri seti yapisina koyacak. Ardindan EDSR super-resolution modeli ile generated goruntuler 1600x900 kalitesine yukseltilecek. YOLO ve segmentasyon analizleri ham 256x256 generated goruntulerle degil, bu upscaled goruntulerle calistirilacak.

Beklenen iki calisma modu:

- Tam otomatik mod: Kullanici ayarlari yapar ve `Tum Pipeline'i Calistir` ile uretimden rapora kadar her sey sirayla calisir.
- Adim adim mod: Kullanici `Sadece Uret`, `Upscale Uygula`, `YOLO Degerlendir`, `Segmentasyon Degerlendir`, `Gorsel Karsilastirma Uret`, `Grafikleri/Raporu Uret` gibi adimlari ayri ayri calistirabilir.

Hedef pipeline:

1. Clean veri seti veya frame klasoru secilir.
2. RCGAN checkpoint secilir.
3. Hata tipleri ve siddetleri secilir.
4. RCGAN bozulmus goruntuleri uretir.
5. Clean ve generated goruntuler detector veri seti yapisina yerlestirilir.
6. Generated goruntuler EDSR ile upscale edilir. Varsayilan hedef boyut `1600x900`.
7. Detector adimlari `generated_upscaled` klasorundeki goruntuler uzerinden calisir.
8. YOLO robustness degerlendirmesi calisir.
9. Semantic segmentation robustness degerlendirmesi calisir.
10. Segmentasyon gorsel karsilastirmalari uretilir.
11. Akademik grafikler ve CSV ozetleri uretilir.
12. Arayuzde log, ilerleme durumu, sonuc dosyalari ve temel metrik ozeti gosterilir.

Ilk surum icin odak:

- Mevcut PySide6 GUI uzerinden ilerlemek.
- Once uretim ciktilarini detector klasor yapisina otomatik yerlestiren kucuk bir kopru katmani yazmak.
- RCGAN 256x256 ciktilarini EDSR upscale adimina baglamak ve analizlerde sabit olarak upscaled ciktilari kullanmak.
- Sonra detector scriptlerini GUI'den sirali calistirilabilir hale getirmek.
- En sonda arayuze pipeline butonlari, ilerleme bilgisi ve sonuc ozet paneli eklemek.

## Yapilan Ilk Kucuk Adim

`rcgan_qt_gui_app_v1/pipeline_bridge.py` eklendi.

Bu modul, RCGAN tarafinda uretilen dosyalari detector tarafinin bekledigi yapıya tasimak icin hazirlandi:

- Clean frame'leri `detector/robustness_dataset/clean` altina kopyalar.
- Generated dosyalarin adindaki `blur_high`, `brightness_medium`, `occlusion_low` gibi kosul son eklerini algilar.
- Generated dosyalari `detector/robustness_dataset/generated/<condition>` klasorlerine dagitir.
- Kopyalanan ve kosulu anlasilamayan dosyalari raporlayan bir sonuc sozlugu dondurur.

Bu henuz GUI'ye baglanmadi. Bir sonraki adimda GUI icine `Detector veri setini hazirla` veya pipeline icinde otomatik calisan bir asama olarak eklenebilir.

Test sirasinda `pipeline_bridge.py` hafifletildi: artik `generate.py` import etmiyor, bu yuzden sadece dosya kopyalama/kosul algilama icin torch/torchvision yuklemeye calismiyor.

## Detector Fonksiyonlastirma Durumu

`detector/yolo_robustness_evaluation.py` ilk asama olarak fonksiyonlastirildi.

Yeni ana fonksiyon:

```python
run_yolo_evaluation(
    dataset_root="robustness_dataset",
    results_dir="yolo_results",
    yolo_model_path="yolov8n.pt",
    generated_dirs=None,
    log_callback=None,
)
```

Bu degisiklikten sonra dosya import edilince otomatik analiz baslamaz. Terminalden dogrudan calistirilirsa eski davranisa yakin sekilde YOLO analizini calistirmaya devam eder:

```bash
cd /Users/ozcan/Desktop/projects/detector
python yolo_robustness_evaluation.py
```

GUI tarafinda ileride `log_callback=self.log_box.append` benzeri bir callback ile calistirilabilir.

YOLO analizinin upscaled generated dosyalarla calismasi icin `default_upscaled_generated_dirs(...)` yardimcisi eklendi. GUI/pipeline tarafinda YOLO cagirilirken `generated_dirs=default_upscaled_generated_dirs(dataset_root)` verilmelidir.

`detector/semantic_segmentation_robustness.py` ikinci asama olarak fonksiyonlastirildi.

Yeni ana fonksiyon:

```python
run_segmentation_evaluation(
    dataset_root="robustness_dataset",
    output_dir="segmentation_outputs",
    results_dir="results",
    model_name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    generated_dirs=None,
    device=None,
    log_callback=None,
)
```

Bu degisiklikten sonra SegFormer modeli dosya import edilirken yuklenmez. Model yalnizca `run_segmentation_evaluation(...)` cagrildiginda yuklenir. Terminalden dogrudan calistirilirsa eski davranisa yakin sekilde segmentasyon robustness analizini calistirir:

```bash
cd /Users/ozcan/Desktop/projects/detector
python semantic_segmentation_robustness.py
```

Segmentasyon analizinin upscaled generated dosyalarla calismasi icin `default_upscaled_generated_dirs(...)` yardimcisi eklendi. GUI/pipeline tarafinda segmentasyon cagirilirken `generated_dirs=default_upscaled_generated_dirs(dataset_root)` verilmelidir.

`detector/upscale_generated_images.py` EDSR upscale asamasi icin fonksiyonlastirildi.

Yeni ana fonksiyon:

```python
run_upscale_generated_images(
    dataset_root="robustness_dataset",
    generated_dirs=None,
    upscaled_root=None,
    model_path="EDSR_x4.pb",
    target_width=1600,
    target_height=900,
    log_callback=None,
)
```

Bu asama RCGAN'in 256x256 generated ciktilarini `robustness_dataset/generated_upscaled/<condition>` altina 1600x900 olarak kaydeder. Pipeline'da RCGAN uretiminden ve detector veri seti hazirligindan sonra, YOLO/segmentasyon analizinden once calistirilir. YOLO ve segmentasyon analizlerinde hedef kaynak sabit olarak `generated_upscaled` olmalidir.

## GUI Pipeline Ilk Surum

`rcgan_qt_gui_app_v1/qt_gui_app_updated.py` icine pipeline kontrolleri eklendi.

Kullaniciya donen ciktilar iki ana klasorde toplanacak:

- `/Users/ozcan/Desktop/projects/clean`: Arayuzden secilen clean veri setinin proje icindeki sabit kopyasi.
- `/Users/ozcan/Desktop/projects/outputs`: Gorsel ciktilar.
- `/Users/ozcan/Desktop/projects/results`: Sayisal metrikler, CSV dosyalari ve grafikler.

Planlanan merkezi cikti yapisi:

```text
outputs/
  gan_generated/              # RCGAN'in 256x256 generated goruntuleri
  gan_upscaled/               # EDSR ile 1600x900'e buyutulen generated goruntuler
  segmentation_outputs/       # Clean ve hatali goruntuler icin segmentasyon overlay ciktilari
  segmentation_comparisons/   # Clean/hata segmentasyon karsilastirma gorselleri
  yolo_comparisons/           # Clean/hata YOLO bbox karsilastirma gorselleri

results/
  yolo/                       # YOLO metrik CSV'leri ve grafikler
  segmentation/               # Segmentasyon metrik CSV'leri ve grafikler
```

`detector/robustness_dataset` klasoru pipeline icinde ara veri seti/staging alani olarak kalir. Kullanici tarafindan incelenecek ana ciktilar `outputs` ve `results` altinda olur.

Yeni alan:

- `Detector`: detector klasor yolu. Varsayilan olarak `/Users/ozcan/Desktop/projects/detector` kullanilir.

Yeni butonlar:

- `Pipeline: Secilen Kosul`: Secili hata tipi + siddet icin GAN uretimi, detector veri seti hazirlama, EDSR upscale, YOLO ve segmentasyon analizlerini sirayla calistirir.
- `Pipeline: 3 Hata Tipi`: `blur`, `occlusion`, `brightness` icin arayuzde secilen birer siddetle ayni tam pipeline'i calistirir.
- `Detector Veri Setini Hazirla`: Mevcut GAN ciktilarini detector veri setine kopyalar.
- `Upscale Uygula`: `robustness_dataset/generated` altindaki goruntuleri EDSR ile `generated_upscaled` altina yazar.
- `YOLO Degerlendir`: Upcaled generated goruntuler uzerinden YOLO analizini calistirir.
- `Segmentasyon Degerlendir`: Upcaled generated goruntuler uzerinden SegFormer analizini calistirir.
- `Outputs Klasorunu Ac`: Merkezi gorsel cikti klasorunu acar.
- `Results Klasorunu Ac`: Merkezi metrik/istatistik klasorunu acar.

Arayuzde artik `Tüm 9 Koşul` ana akis olarak kullanilmiyor. Onun yerine her hata tipi icin bir seviye seciliyor:

- `Blur seviyesi`
- `Occlusion seviyesi`
- `Brightness seviyesi`

Bu mod toplam 3 kosul uretir; ornek olarak `blur_low`, `occlusion_high`, `brightness_medium`.

Arayuz guncellendi: onizleme sag tarafta kalir, alt kisimda `Ciktilar` ve `Log` sekmeleri bulunur. Stil daha temiz kart/panel yapisina alindi.

Tam pipeline artik su sirayla calisir:

1. Secilen clean veri seti `clean/` altina kopyalanir ve pipeline bundan sonra bu kopyayi kullanir.
2. RCGAN ciktilari `outputs/gan_generated` altina yazilir.
3. Clean ve generated goruntuler detector staging yapisina kopyalanir.
4. EDSR upscale ciktilari `outputs/gan_upscaled` altina yazilir.
5. YOLO analizleri `outputs/gan_upscaled` uzerinden calisir; metrikler `results/yolo` altina yazilir.
6. Segmentasyon analizleri `outputs/gan_upscaled` uzerinden calisir; overlay gorseller `outputs/segmentation_outputs`, metrikler `results/segmentation` altina yazilir.
7. YOLO clean/hata karsilastirma gorselleri `outputs/yolo_comparisons` altina yazilir.
8. Segmentasyon clean/hata karsilastirma gorselleri `outputs/segmentation_comparisons` altina yazilir.

GUI calistirma komutu:

```bash
cd /Users/ozcan/Desktop/projects/rcgan_qt_gui_app_v1
source qtvenv/bin/activate
python qt_gui_app_updated.py
```

## Ek Klasor Analizi: `akilli_veri_arttirimi/`

Proje kokunde ikinci bir uygulama olarak `akilli_veri_arttirimi/` klasoru bulunuyor. Bu klasor, mevcut goruntu tabanli RCGAN + detector pipeline'indan farkli olarak CSV/tabular veri uzerinden sentetik veri artirimi yapan ayri bir platform.

### Genel Amac

`akilli_veri_arttirimi`, otonom arac veya sensor verilerini CSV olarak alip temizleyen, uygun sentetik veri uretim motorunu secen ve uretilen verinin faydasini/kalitesini olcen bir sistem.

Ana fikir:

1. CSV veri yuklenir.
2. Veri 4 katmanli bilgi damitmadan gecer.
3. Veri tipine gore uretim yontemi secilir:
   - Waymo formatindaki zaman serisi verisi icin RCGAN.
   - Genel tabular/sensor verisi icin CTGAN.
   - Kucuk veya problemli veri icin SMOTE + Gaussian noise.
4. Uretilen sentetik veri baseline modelle karsilastirilir.
5. Fidelity, utility, F1, recall, dagilim kaymasi gibi metrikler raporlanir.

Bu klasor mevcut `rcgan_qt_gui_app_v1/` uygulamasiyla dogrudan ayni kodu kullanmiyor. Oradaki RCGAN goruntu uretirken, buradaki RCGAN Waymo/yolculuk/yörünge sekanslari gibi tabular zaman serisi uretiyor.

### Calistirma Sekli

Masaustu uygulamasi:

```bash
cd /Users/ozcan/Desktop/projects/akilli_veri_arttirimi
python main.py
```

`main.py`, FastAPI backend'i arka planda baslatir ve `pywebview` ile masaustu pencere acar.

Sadece web backend:

```bash
cd /Users/ozcan/Desktop/projects/akilli_veri_arttirimi
python backend/server.py
```

Sonra tarayicidan:

```text
http://127.0.0.1:8000
```

### Ana Dosyalar

- `main.py`: Desktop wrapper. Uvicorn/FastAPI sunucusunu arka thread'de calistirir, pywebview penceresi acar. Ayrica uretilen ve temizlenmis CSV'leri kaydetmek icin desktop file dialog API'si saglar.
- `backend/server.py`: Asil uygulama mantigi. FastAPI endpointleri, bilgi damitma, RCGAN/CTGAN/SMOTE uretim motorlari, kalite/utility degerlendirmeleri burada.
- `backend/index.html`, `backend/style.css`, `backend/script.js`: Web arayuzu.
- `requirements.txt`: FastAPI, pywebview, torch, ctgan, tensorflow, pandas, scikit-learn gibi bagimliliklar.
- `KURULUM_REHBERI.md`: macOS/Windows kurulum ve Git LFS notlari.
- `README.md`: Platform hedefi, mimari, ekran goruntuleri, endpointler ve akademik referanslar.
- `RCGAN_TEST_VERISI.csv`: Waymo/RCGAN formatina benzeyen kucuk test CSV'si.
- `waymo_seed_MASSIVE.csv`: Git LFS pointer gibi gorunuyor; asil buyuk dosya icin `git lfs pull` gerekebilir.
- `outputs/waymo_rcgan_GODMODE_A100_STABLE.pth`: Tabular/yörünge RCGAN agirligi.
- `outputs/waymo_normalization_min.npy`, `outputs/waymo_normalization_max.npy`: RCGAN uretimi icin normalizasyon dosyalari.
- `outputs/multi_source_results.json`, `outputs/10_multi_source_results.png`: Daha once calistirilmis deney sonuc/figurleri.

### Backend Akisi

`backend/server.py` icinde dikkat ceken ana fonksiyonlar:

- `distill_dataset(df)`: 4 katmanli bilgi damitma yapar.
  - Duplikasyon temizligi.
  - KNN tabanli gurultulu etiket duzeltme.
  - IQR/outlier filtreleme.
  - Sifir varyans ve yuksek NaN oranli sutun temizligi.
- `try_convert_to_waymo(df, label_col)`: Koordinat/hiz/zaman baglami olan veriyi 20 adimli Waymo benzeri pencere formatina cevirmeye calisir.
- `generate_adaptive(...)`: Uygun sentetik veri motorunu secer.
  - Waymo format + model varsa `generate_waymo`.
  - Genel tabular veri icin `generate_ctgan`.
  - Fallback icin `generate_smart`.
- `generate_waymo(df, n_samples)`: RCGAN ile Waymo formatli sentetik yörünge/anomali uretir.
- `generate_ctgan(...)`: CTGAN'i veri seti uzerinde on-the-fly egitir.
- `generate_smart(...)`: SMOTE benzeri interpolasyon + Gaussian noise fallback.
- `evaluate(...)` ve kalite yardimcilari: Uretilen verinin fidelity/utility metriklerini hesaplar.

API endpointleri:

- `GET /api/system_status`
- `POST /api/distill`
- `POST /api/evaluate_pipeline`
- `POST /api/run_full_automation`
- `POST /api/simulation_sample`
- `GET /api/download_generated`
- `GET /api/download_distilled`

### Scripts Klasoru

`scripts/` altindaki dosyalar arastirma/deney fazlari gibi duruyor:

- `01_data_exploration.py`: FCD Italy verisi icin veri kesfi, anomali injection ve gorsellestirme.
- `02_rcgan_generation.py`: LSTM tabanli RCGAN egitimi/uretimi.
- `03_utility_test.py`: Seed-only vs Seed+GAN utility karsilastirmasi.
- `04_feature_engineering.py`: Yörüngelerden hiz, ivme, sarsinti, egrilik gibi ozellikler cikarip model iyilestirme.
- `05_waymo_integration.py`: Waymo Motion Dataset parsing ve multi-source pipeline.
- `06_waymo_rcgan_pipeline.py`: Waymo seed + RCGAN generated verisiyle feature engineering ve siniflandirma karsilastirmasi.

Bu scriptler genellikle `outputs/` klasorune grafik/JSON/model sonuc dosyalari yazar.

### Mevcut Durum ve Riskler

- Bu klasor kendi `.git` deposuna sahip. Ana `/Users/ozcan/Desktop/projects` deposu icinde nested git repo gibi duruyor. Commit/versiyonlama yaparken dikkat edilmeli.
- `waymo_seed_MASSIVE.csv` dosyasi lokal ortamda Git LFS pointer olarak gorunuyor olabilir. Dosyanin gercek 705 MB icerigi icin `git lfs pull` gerekebilir.
- Bu makinede `git lfs` komutu su an bulunmuyor. `waymo_seed_MASSIVE.csv` ve `outputs/waymo_rcgan_GODMODE_A100_STABLE.pth` 134 baytlik Git LFS pointer olarak gorundu. Bu nedenle RCGAN GODMODE modeli yuklenemez; uygulama CTGAN/SMOTE tarafiyla acilabilir. Full RCGAN modu icin once Git LFS kurulup `git lfs pull` yapilmali.
- `requirements.txt` cok agir: `torch`, `tensorflow`, `ctgan`, `pywebview`, `fastapi` beraber geliyor. Mevcut `qtvenv` ile karistirmamak iyi olur; ayri sanal ortam kullanilmali.
- `backend/server.py`, import sirasina ozellikle dikkat ediyor: CTGAN'i torch'tan once import ederek macOS OpenMP/Accelerate deadlock riskini azaltmaya calisiyor.
- Bu proje tabular/yörünge sentetik veri uretimi icin; mevcut goruntu RCGAN + detector pipeline'ina dogrudan baglanacak bir modul degil. Fakat konsept olarak ikisi de "sentetik veri uretimi + kalite/fayda degerlendirme" amacina hizmet ediyor.

### Mevcut Goruntu Pipeline ile Iliski

Mevcut ana calisma hattimiz:

```text
clean goruntu dataset -> RCGAN image corruption -> EDSR upscale -> YOLO/SegFormer robustness analysis
```

`akilli_veri_arttirimi` hattinin konusu:

```text
CSV/tabular/yörünge dataset -> bilgi damitma -> RCGAN/CTGAN/SMOTE sentetik veri -> ML utility/fidelity analysis
```

Bu yuzden su an icin iki klasor ayri uygulama olarak dusunulmeli. Ileride tek bir akademik sunum/rapor icinde "goruntu bozulma robustluk pipeline'i" ve "tabular sentetik veri artirim platformu" olarak iki bolum halinde birlestirilebilir.
