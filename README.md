# 👗 Akıllı Dolabım
### Yapay Zekâ Destekli Dijital Gardırop ve Kombin Öneri Sistemi

---

## 📌 Proje Hakkında

**Akıllı Dolabım**, kullanıcıların kıyafetlerini fotoğraf yükleyerek dijital gardıroplarını oluşturmalarını ve yapay zekâ destekli kombin önerileri almalarını amaçlayan bir Python projesidir.

Uygulama; görüntü işleme, yapay zekâ modelleri ve hava durumu verilerinden yararlanarak kıyafetleri tanımayı, renklerini analiz etmeyi ve farklı tarzlara uygun kombinler sunmayı hedefler.

## 🎯 Projenin Amacı

Bu projenin temel amacı, günlük kıyafet seçimini kolaylaştırmak ve yapay zekâ teknolojilerini günlük yaşamda kullanılabilecek bir uygulamaya dönüştürmektir.

- Kıyafetleri dijital ortamda saklamak
- Kıyafet türlerini yapay zekâ ile belirlemek
- Kıyafet renklerini analiz etmek
- Hava durumuna uygun kombinler oluşturmak
- Gardırop yönetimini kolaylaştırmak

## ✨ Proje Özellikleri

### 👚 1. Dijital Gardırop

- Fotoğraf yükleyerek kıyafet ekleme
- Kıyafetleri galeri görünümünde inceleme
- Kıyafet bilgilerini düzenleme ve kayıtları silme
- Kıyafetleri türlerine göre sınıflandırma

### 🤖 2. Yapay Zekâ ile Kıyafet Tanıma

Projede Hugging Face Transformers kütüphanesi üzerinden kullanılan **CLIP** modeliyle görsellerin kıyafet kategorileriyle eşleştirilmesi amaçlanır.

### 🎨 3. Renk Algılama

Görüntü işleme yöntemleriyle kıyafetlerin renkleri analiz edilir. Pembe-siyah gibi iki renkli ürünlerin de daha doğru tanımlanması hedeflenir.

### 🌦️ 4. Hava Durumuna Göre Kombin Önerileri

Open-Meteo üzerinden alınan hava durumu verileri kullanılarak koşullara uygun kıyafet ve kombin önerileri sunulması amaçlanır.

### 👗 5. Farklı Kombin Tarzları

- Günlük
- Spor
- Şık

## 🛠️ Kullanılan Teknolojiler

| Teknoloji | Kullanım Alanı |
|:--|:--|
| Python
