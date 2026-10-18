# 👗 Akıllı Dolabım – Yapay Zekâ Destekli Kombin Öneri Sistemi

## 📌 Proje Hakkında

**Akıllı Dolabım**, kullanıcıların kıyafetlerini fotoğraf yükleyerek dijital ortamda saklamalarını ve yapay zekâ desteğiyle kombin önerileri almalarını sağlayan bir uygulamadır.

Projenin amacı, kıyafet seçimini kolaylaştırmak, gardırop düzenini dijitalleştirmek ve hava durumuna uygun kombinler oluşturmaya yardımcı olmaktır.

## 🎯 Projenin Amaçları

- Kıyafetleri fotoğraf yükleyerek dijital gardıroba eklemek
- Yapay zekâ ile kıyafet türlerini tanımak
- Kıyafet renklerini tespit etmek
- Hava durumuna uygun kombin önerileri sunmak
- Kıyafetleri kategorilere ayırarak düzenli şekilde görüntülemek

## ✨ Proje Özellikleri

### 👚 Dijital Gardırop
Kullanıcılar kıyafet fotoğraflarını yükleyebilir, kıyafetlerini görüntüleyebilir, düzenleyebilir ve silebilir.

### 🤖 Yapay Zekâ ile Kıyafet Tanıma
Görüntü işleme ve yapay zekâ modelleri kullanılarak yüklenen fotoğraflardaki kıyafetlerin türleri belirlenmeye çalışılır.

### 🎨 Renk Algılama
Kıyafetlerin renkleri analiz edilir. İki renkli ürünler için pembe-siyah gibi özel renk kategorileri de kullanılabilir.

### 🌦️ Hava Durumuna Göre Kombin
Hava durumu bilgileri dikkate alınarak günlük, spor ve şık kombin önerileri oluşturulur.

### 🗂️ Kıyafet Yönetimi
Kıyafetler kategorilerine göre görüntülenebilir ve gardırop kayıtları düzenlenebilir.

## 🛠️ Kullanılan Teknolojiler

| Teknoloji | Kullanım Amacı |
|---|---|
| Python | Uygulama geliştirme |
| Streamlit | Web arayüzü |
| Hugging Face Transformers | Yapay zekâ modelleri |
| CLIP | Görsel ve metin tabanlı kıyafet sınıflandırma |
| SegFormer | Görüntü segmentasyonu |
| Pillow (PIL) | Görselleri işleme |
| Open-Meteo | Hava durumu verileri |
| JSON | Gardırop verilerini saklama |

## 📁 Proje Dosya Yapısı

```text
akilli_dolabim/
│
├── app.py
├── requirements.txt
└── README.md
```

## ⚙️ Kurulum ve Çalıştırma

### 1. Depoyu bilgisayarınıza indirin

```bash
git clone https://github.com/hazallguc/akilli_dolabim.git
```

### 2. Proje klasörüne girin

```bash
cd akilli_dolabim
```

### 3. Gerekli kütüphaneleri yükleyin

```bash
pip install -r requirements.txt
```

### 4. Uygulamayı çalıştırın

```bash
streamlit run app.py
```

Uygulama çalıştırıldıktan sonra tarayıcı üzerinden açılabilir.

## 🧠 Yapay Zekâ Modelleri

Projede kullanılan modeller, kıyafet görsellerini analiz etmek ve kıyafet özelliklerini belirlemek amacıyla tasarlanmıştır.

- **CLIP:** Görselleri metin açıklamalarıyla karşılaştırarak sınıflandırma yapılmasına yardımcı olur.
- **SegFormer:** Görüntüdeki kıyafet bölgelerinin ayrıştırılmasında kullanılır.

Model sonuçları görselin kalitesine, ışık koşullarına ve kıyafetin özelliklerine göre değişebilir.

## 🚀 Projenin Geliştirilme Amacı

Akıllı Dolabım, yapay zekâ ve görüntü işleme teknolojilerini günlük yaşamda kullanılabilecek bir uygulamaya dönüştür
