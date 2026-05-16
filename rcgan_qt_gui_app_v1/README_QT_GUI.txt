RCGAN Qt GUI - Tarayıcısız Masaüstü Pencere
==========================================

Bu paket Tkinter kullanmaz. Bu yüzden Mac'teki "No module named _tkinter" hatasını vermez.
Terminalden tek komutla pencere açılır.

Klasör içeriği:
- qt_gui_app.py              -> Qt/PySide6 arayüz dosyası
- model.py                   -> RecurrentGenerator model mimarisi
- generate.py                -> üretim yardımcı fonksiyonları
- requirements_qt.txt        -> gerekli Python paketleri

Kurulum:
1) Bu zip dosyasını Masaüstüne aç.
2) checkpoint_epoch_29.pt dosyasını bu klasörün içine koy.
3) Terminalde klasöre gir:

   cd ~/Desktop/rcgan_qt_gui_app_v1

4) Sanal ortam oluştur:

   python3 -m venv qtvenv
   source qtvenv/bin/activate

5) Paketleri kur:

   pip install -r requirements_qt.txt

6) Arayüzü çalıştır:

   python qt_gui_app_updated.py

Kullanım:
- Checkpoint olarak checkpoint_epoch_29.pt seç.
- Prev Frame: önceki clean kareyi seç.
- Curr Frame: mevcut clean kareyi seç.
- Hata tipi: blur / occlusion / brightness seç.
- Derece: low / medium / high seç.
- "Seçilen Koşulu Üret" veya "Tüm 9 Koşulu Üret" butonuna bas.

Not:
Bu model tek fotoğrafla değil, iki ardışık clean frame ile çalışır.
Prev = önceki kare
Curr = üretilecek bozulmuş görüntünün referans karesi
