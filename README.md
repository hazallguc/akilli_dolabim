# 👗 Akıllı Dolabım

### Yapay Zekâ Destekli Dijital Gardırop ve Kombin Öneri Sistemi

---

## 📌 Proje Hakkında

**Akıllı Dolabım**, kullanıcıların kıyafet fotoğraflarını yükleyerek dijital bir gardırop oluşturmasını ve yapay zekâ destekli kombin önerileri almasını amaçlayan bir Python projesidir.

Uygulama; görüntü işleme, yapay zekâ modelleri ve hava durumu verilerinden yararlanarak kıyafetleri tanımayı, renklerini analiz etmeyi ve farklı tarzlara uygun kombinler oluşturmayı hedefler.

## 🎯 Projenin Amacı

Projenin temel amacı, günlük kıyafet seçimini kolaylaştırmak ve yapay zekâ teknolojilerini günlük yaşamda kullanılabilecek bir uygulamaya dönüştürmektir.

- Kıyafetleri dijital ortamda saklamak
- Kıyafet türlerini yapay zekâ yardımıyla belirlemek
- Kıyafet renklerini analiz etmek
- Hava durumuna uygun kombinler önermek
- Gardırop yönetimini kolaylaştırmak

## ✨ Proje Özellikleri

### 👚 Dijital Gardırop

- Fotoğraf yükleyerek kıyafet ekleme
- Kıyafetleri galeri görünümünde görüntüleme
- Kıyafet bilgilerini düzenleme ve kayıtları silme
- Kıyafetleri türlerine göre sınıflandırma

### 🤖 Yapay Zekâ ile Kıyafet Tanıma

Hugging Face Transformers kütüphanesi aracılığıyla kullanılan CLIP modeli, kıyafet görsellerini belirlenen kategorilerle karşılaştırarak sınıflandırma yapılmasına yardımcı olur.

### 🎨 Renk Algılama

Görüntü işleme yöntemleriyle kıyafet renkleri analiz edilir. Pembe-siyah gibi iki renkli ürünlerin daha doğru tanımlanması hedeflenir.

### 🌦️ Hava Durumuna Göre Kombin Önerileri

Open-Meteo hava durumu verileri kullanılarak hava koşullarına uygun kıyafet ve kombin önerileri oluşturulması amaçlanır.

### 👗 Farklı Kombin Tarzları

- Günlük
- Spor
- Şık

## 🛠️ Kullanılan Teknolojiler

| Teknoloji | Kullanım Alanı |
| --- | --- |
| Python | Uygulama geliştirme |
| Streamlit | Web arayüzü |
| Hugging Face Transformers | Yapay zekâ modelleri |
| CLIP | Görsel sınıflandırma |
| SegFormer | Görüntü segmentasyonu |
| Pillow (PIL) | Görüntü işleme |
| Open-Meteo | Hava durumu verileri |
| JSON | Gardırop kayıtlarının saklanması |

## 📁 Proje Dosya Yapısı

```text
akilli_dolabim/
├── app.py
├── requirements.txt
└── README.md
```

### Dosyaların Açıklamaları

- **app.py:** Uygulamanın arayüzünü ve temel işlevlerini içerir.
- **requirements.txt:** Projenin çalışması için gereken Python kütüphanelerini listeler.
- **README.md:** Projenin açıklamasını ve kurulum adımlarını içerir.

## ⚙️ Kurulum ve Çalıştırma

Projeyi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

### 1. Depoyu indirin

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

Komut çalıştırıldıktan sonra uygulama, Streamlit tarafından sağlanan yerel web adresinde açılır.

## 🧠 Yapay Zekâ Modelleri

Projede kullanılan temel modeller şunlardır:

- **CLIP — `openai/clip-vit-base-patch32`:** Görselleri metin açıklamalarıyla karşılaştırarak sınıflandırma yapılmasına yardımcı olur.
- **SegFormer — `mattmdjaga/segformer_b2_clothes`:** Görüntüdeki kıyafet bölgelerinin belirlenmesi ve ayrıştırılması için kullanılır.

Model sonuçları fotoğrafın kalitesine, ışık koşullarına ve kıyafetin görünürlüğüne göre değişebilir.

## 🚀 Geliştirme Hedefleri

- Kıyafet tanıma ve renk algılama doğruluğunu artırmak
- Yeni kıyafet kategorileri eklemek
- Kullanıcıya özel gardırop kayıtları oluşturmak
- Verileri güvenli ve kalıcı biçimde saklamak
- Kombin önerilerini kişiselleştirmek
- Uygulamayı çevrim içi erişime açmak

## 👩‍💻 Geliştirici

**Hazal Güç**

GitHub profili: [hazallguc](https://github.com/hazallguc)

Proje deposu: [Akıllı Dolabım](https://github.com/hazallguc/akilli_dolabim)

---

*Akıllı Dolabım — Yapay zekâ ile gardırobunu dijitalleştir, kombinlerini kolaylaştır.*
