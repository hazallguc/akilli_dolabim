# 👗 Akıllı Dolabım

### Yapay Zekâ Destekli Dijital Gardırop ve Kombin Öneri Sistemi

---

## 📌 Proje Hakkında

Akıllı Dolabım, kullanıcıların kıyafet fotoğraflarını yükleyerek dijital gardırop oluşturmalarını ve yapay zekâ destekli kombin önerileri almalarını amaçlayan bir Python projesidir.

## 🎯 Projenin Amaçları

- Kıyafetleri dijital ortamda saklamak
- Yapay zekâ ile kıyafet türlerini tanımak
- Kıyafet renklerini analiz etmek
- Hava durumuna uygun kombinler önermek
- Gardırop yönetimini kolaylaştırmak

## ✨ Proje Özellikleri

### 👚 Dijital Gardırop
Kıyafet fotoğrafları eklenebilir, görüntülenebilir, düzenlenebilir ve silinebilir.

### 🤖 Yapay Zekâ ile Kıyafet Tanıma
CLIP modeli kullanılarak kıyafet görsellerinin kategorilerinin belirlenmesi hedeflenir.

### 🎨 Renk Algılama
Görüntü işleme yöntemleriyle kıyafet renkleri analiz edilir. Pembe-siyah gibi iki renkli ürünler de desteklenebilir.

### 🌦️ Hava Durumuna Göre Kombin
Open-Meteo verilerinden yararlanılarak günlük, spor ve şık kombin önerileri oluşturulur.

## 🛠️ Kullanılan Teknolojiler

| Teknoloji | Kullanım Alanı |
| --- | --- |
| Python | Uygulama geliştirme |
| Streamlit | Web arayüzü |
| Transformers | Yapay zekâ modelleri |
| CLIP | Görsel sınıflandırma |
| SegFormer | Görüntü segmentasyonu |
| Pillow | Görüntü işleme |
| Open-Meteo | Hava durumu verileri |
| JSON | Gardırop verileri |

## 📁 Proje Dosya Yapısı

```text
akilli_dolabim/
├── app.py
├── requirements.txt
└── README.md
```

## ⚙️ Kurulum ve Çalıştırma

### 1. Depoyu indirin

```bash
git clone https://github.com/hazallguc/akilli_dolabim.git
```

### 2. Proje klasörüne girin

```bash
cd akilli_dolabim
```

### 3. Kütüphaneleri yükleyin

```bash
pip install -r requirements.txt
```

### 4. Uygulamayı çalıştırın

```bash
streamlit run app.py
```

## 🚀 Geliştirme Hedefleri

- Kıyafet tanıma doğruluğunu artırmak
- Daha fazla kıyafet kategorisi eklemek
- Kullanıcıya özel gardırop oluşturmak
- Güvenli ve kalıcı veri saklama sağlamak
- Kombin önerilerini geliştirmek

## 👩‍💻 Geliştirici

**Hazal Güç**

[GitHub Profilim](https://github.com/hazallguc)

[Akıllı Dolabım Proje Deposu](https://github.com/hazallguc/akilli_dolabim)

---

*Akıllı Dolabım — Yapay zekâ ile gardırobunu dijitalleştir, kombinlerini kolaylaştır.*
