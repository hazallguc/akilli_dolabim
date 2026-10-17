
import streamlit as st
import json
import os
import base64
import io
import colorsys
from itertools import combinations

import numpy as np
from PIL import Image
from transformers import pipeline

# --------------------------------------------------
# SAYFA AYARLARI
# --------------------------------------------------

st.set_page_config(
    page_title="Akıllı Dolabım",
    page_icon="👗",
    layout="wide"
)

st.title("👗 Akıllı Dolabım")
st.write(
    "Kıyafetlerini dolabına ekle, renklerini keşfet "
    "ve yapay zekâ destekli kombinler oluştur."
)

DOSYA = "dolap.json"

KIYAFET_TURLERI = [
    "Üst giyim",
    "Alt giyim",
    "Elbise",
    "Ayakkabı",
    "Dış giyim",
    "Çanta",
    "Aksesuar"
]

# --------------------------------------------------
# YAPAY ZEKÂ MODELLERİ
# --------------------------------------------------

@st.cache_resource(show_spinner="Yapay zekâ modelleri yükleniyor...")
def modelleri_yukle():
    tur_modeli = pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32"
    )

    bolge_modeli = pipeline(
        "image-segmentation",
        model="mattmdjaga/segformer_b2_clothes"
    )

    return tur_modeli, bolge_modeli


# --------------------------------------------------
# DOLAP VERİLERİ
# --------------------------------------------------

def dolabi_yukle():
    if not os.path.exists(DOSYA):
        return []

    try:
        with open(DOSYA, "r", encoding="utf-8") as dosya:
            veriler = json.load(dosya)

        return veriler if isinstance(veriler, list) else []

    except (json.JSONDecodeError, OSError):
        st.error(
            "Dolap verileri okunamadı. "
            "dolap.json dosyasını kontrol et."
        )
        return []


def dolabi_kaydet(veriler):
    with open(DOSYA, "w", encoding="utf-8") as dosya:
        json.dump(
            veriler,
            dosya,
            ensure_ascii=False,
            indent=2
        )


def resim_base64(resim):
    tampon = io.BytesIO()
    resim.save(tampon, format="JPEG")
    return base64.b64encode(tampon.getvalue()).decode("utf-8")


def base64_resim_ac(veri):
    ham_veri = base64.b64decode(veri)
    return Image.open(io.BytesIO(ham_veri)).convert("RGB")


# --------------------------------------------------
# KIYAFET TÜRÜ TESPİTİ
# --------------------------------------------------

def kiyafet_turu_bul(resim, tur_modeli):
    etiketler = [
        "a photograph of a T-shirt or shirt",
        "a photograph of trousers or jeans",
        "a photograph of a skirt",
        "a photograph of a dress",
        "a photograph of shoes or sneakers",
        "a photograph of a coat or jacket",
        "a photograph of a handbag or backpack",
        "a photograph of an accessory such as a hat or scarf"
    ]

    sonuc = tur_modeli(
        resim,
        candidate_labels=etiketler
    )

    etiket = sonuc[0]["label"].lower()

    if "shoes" in etiket or "sneakers" in etiket:
        return "Ayakkabı"
    if "trousers" in etiket or "jeans" in etiket:
        return "Alt giyim"
    if "skirt" in etiket:
        return "Alt giyim"
    if "dress" in etiket:
        return "Elbise"
    if "coat" in etiket or "jacket" in etiket:
        return "Dış giyim"
    if "handbag" in etiket or "backpack" in etiket:
        return "Çanta"
    if "accessory" in etiket:
        return "Aksesuar"

    return "Üst giyim"


# --------------------------------------------------
# GÖRÜNTÜDEN RENK ÖRNEKLERİ ALMA
# --------------------------------------------------

def hsv_pikselleri(resim, maske=None, maksimum=12000):
    """
    Resmi HSV renk uzayına dönüştürür.
    Maske verilirse yalnızca maskenin içindeki pikselleri kullanır.
    """

    rgb = np.asarray(resim.convert("RGB").resize((120, 120)))
    pikseller = rgb.reshape(-1, 3).astype(np.float32) / 255.0

    if maske is not None:
        maske_resmi = Image.fromarray(
            (maske.astype(np.uint8) * 255)
        ).resize((120, 120), Image.Resampling.NEAREST)

        maske_dizisi = np.asarray(maske_resmi).reshape(-1) > 0

        if np.any(maske_dizisi):
            pikseller = pikseller[maske_dizisi]

    if len(pikseller) == 0:
        pikseller = rgb.reshape(-1, 3).astype(np.float32) / 255.0

    if len(pikseller) > maksimum:
        secilenler = np.linspace(
            0, len(pikseller) - 1, maksimum
        ).astype(int)
        pikseller = pikseller[secilenler]

    hsv = np.array([
        colorsys.rgb_to_hsv(float(r), float(g), float(b))
        for r, g, b in pikseller
    ])

    return hsv


# --------------------------------------------------
# RENK SINIFLANDIRMA
# --------------------------------------------------

def pembe_siyah_kontrol(hsv):
    """
    Pembe-siyah kararında kırmızı pikselleri ayrıca sayar.
    Kırmızı baskınsa ayakkabıyı pembe-siyah diye etiketlemez.
    """

    ton = hsv[:, 0]
    doygunluk = hsv[:, 1]
    parlaklik = hsv[:, 2]

    # Pembe: kırmızı ton aralığından daha dar bir aralık.
    pembe = (
        (ton >= 0.86) &
        (ton < 0.95) &
        (doygunluk >= 0.28) &
        (parlaklik >= 0.35)
    )

    # Kırmızı ton, HSV çemberinin iki ucunda bulunur.
    kirmizi = (
        ((ton < 0.04) | (ton >= 0.97)) &
        (doygunluk >= 0.28) &
        (parlaklik >= 0.25)
    )

    siyah = parlaklik < 0.20

    pembe_sayisi = int(np.sum(pembe))
    kirmizi_sayisi = int(np.sum(kirmizi))
    siyah_sayisi = int(np.sum(siyah))

    renkli_sayisi = int(
        np.sum((doygunluk >= 0.20) & (parlaklik >= 0.20))
    )

    # Görüntüde yeterli sayıda piksel yoksa karar verme.
    if renkli_sayisi == 0:
        return False

    pembe_orani = pembe_sayisi / renkli_sayisi
    siyah_orani = siyah_sayisi / len(hsv)
    kirmizi_orani = kirmizi_sayisi / renkli_sayisi

    # Kırmızı belirginse pembe-siyah sınıfını engelle.
    if kirmizi_orani >= 0.20:
        return False

    # İki renk de gerçekten görünür olmalı.
    if (
        pembe_sayisi >= 20
        and siyah_sayisi >= 20
        and pembe_orani >= 0.15
        and siyah_orani >= 0.10
        and pembe_sayisi > kirmizi_sayisi * 1.5
    ):
        return True

    return False


def piksel_renk_sinifi(r, g, b):
    h, s, v = colorsys.rgb_to_hsv(
        r / 255.0,
        g / 255.0,
        b / 255.0
    )

    if v < 0.20:
        return "Siyah"

    if s < 0.18:
        if v > 0.90:
            return "Beyaz"
        if v < 0.48:
            return "Gri"
        return "Gri"

    if h < 0.04 or h >= 0.97:
        return "Kırmızı"

    if h < 0.10:
        return "Turuncu"

    if h < 0.17:
        return "Sarı"

    if h < 0.43:
        return "Yeşil"

    if h < 0.56:
        return "Turkuaz"

    if h < 0.72:
        return "Mavi"

    if h < 0.82:
        return "Mor"

    if h < 0.95:
        return "Pembe"

    return "Kırmızı"


def renk_bul(resim, bolge_modeli, kiyafet_turu):
    """
    Önce kıyafetin bölgesini bulmaya çalışır.
    Ayakkabıda ayakkabı maskesini kullanır.
    Sonra pembe-siyah ve baskın renk kontrollerini yapar.
    """

    analiz_resmi = resim.convert("RGB")
    maske = None

    try:
        bolgeler = bolge_modeli(analiz_resmi)

        if kiyafet_turu == "Ayakkabı":
            ayakkabi_maskeleri = []

            for bolge in bolgeler:
                etiket = bolge.get("label", "").lower()

                if "shoe" in etiket:
                    maske_resmi = bolge["mask"].convert("L")
                    maske_dizisi = np.asarray(maske_resmi) > 127

                    if np.sum(maske_dizisi) > 20:
                        ayakkabi_maskeleri.append(maske_resmi)

            if ayakkabi_maskeleri:
                birlesik = np.zeros(
                    (analiz_resmi.height, analiz_resmi.width),
                    dtype=bool
                )

                for maske_resmi in ayakkabi_maskeleri:
                    maske_resmi = maske_resmi.resize(
                        analiz_resmi.size,
                        Image.Resampling.NEAREST
                    )
                    birlesik |= np.asarray(maske_resmi) > 127

                maske = birlesik

    except Exception:
        # Bölge tespiti başarısız olursa tüm fotoğrafı analiz et.
        maske = None

    hsv = hsv_pikselleri(analiz_resmi, maske)

    if kiyafet_turu == "Ayakkabı" and pembe_siyah_kontrol(hsv):
        return "Pembe-Siyah"

    # Çok açık/koyu ve renksiz piksellerin baskın rengi yanıltmasını azalt.
    renkli = hsv[
        (hsv[:, 1] >= 0.18) &
        (hsv[:, 2] >= 0.20)
    ]

    if len(renkli) > 0:
        hsv_kullan = renkli
    else:
        hsv_kullan = hsv

    siniflar = []

    for h, s, v in hsv_kullan:
        # HSV'den RGB'ye dönüşüm
        r, g, b = colorsys.hsv_to_rgb(float(h), float(s), float(v))

        siniflar.append(
            piksel_renk_sinifi(
                int(r * 255),
                int(g * 255),
                int(b * 255)
            )
        )

    if not siniflar:
        return "Bilinmiyor"

    sayim = {}
    for sinif in siniflar:
        sayim[sinif] = sayim.get(sinif, 0) + 1

    return max(sayim, key=sayim.get)


# --------------------------------------------------
# KOMBİN ÖNERİLERİ
# --------------------------------------------------

def kombin_icin_renk_uyumu(renk1, renk2):
    temel_uyumlar = {
        "Siyah": ["Beyaz", "Gri", "Kırmızı", "Pembe", "Mavi",
                  "Yeşil", "Sarı", "Mor", "Turuncu", "Siyah"],
        "Beyaz": ["Siyah", "Gri", "Mavi", "Pembe", "Kırmızı",
                  "Yeşil", "Mor", "Turuncu", "Sarı", "Beyaz"],
        "Gri": ["Siyah", "Beyaz", "Pembe", "Kırmızı", "Mavi",
                "Mor", "Yeşil", "Gri"],
        "Pembe": ["Siyah", "Beyaz", "Gri", "Mavi", "Mor",
                  "Pembe", "Kırmızı"],
        "Kırmızı": ["Siyah", "Beyaz", "Gri", "Mavi", "Kırmızı"],
        "Mavi": ["Beyaz", "Siyah", "Gri", "Pembe", "Mavi",
                 "Turuncu", "Sarı"],
        "Yeşil": ["Beyaz", "Siyah", "Gri", "Sarı", "Yeşil"],
        "Sarı": ["Siyah", "Beyaz", "Mavi", "Gri", "Sarı"],
        "Mor": ["Beyaz", "Siyah", "Gri", "Pembe", "Mor"],
        "Turuncu": ["Siyah", "Beyaz", "Mavi", "Gri", "Turuncu"],
        "Turkuaz": ["Beyaz", "Siyah", "Gri", "Mavi", "Turkuaz"],
        "Pembe-Siyah": ["Beyaz", "Gri", "Siyah", "Pembe", "Mavi"],
        "Bilinmiyor": ["Siyah", "Beyaz", "Gri"]
    }

    if renk1 == "Pembe-Siyah":
        return renk2 in temel_uyumlar["Pembe-Siyah"]

    if renk2 == "Pembe-Siyah":
        return renk1 in temel_uyumlar["Pembe-Siyah"]

    return renk2 in temel_uyumlar.get(renk1, [])


def kombin_aciklamasi(parcalar, ortam):
    renkler = [p["renk"] for p in parcalar]
    turler = [p["tur"] for p in parcalar]

    if ortam == "Günlük":
        stil = "Rahat, kullanışlı ve günlük hayata uygun"
        ortam_metni = "okul, arkadaş buluşması veya günlük işler"
    elif ortam == "Şık":
        stil = "Dengeli, özenli ve daha şık"
        ortam_metni = "akşam buluşması veya özel bir etkinlik"
    else:
        stil = "Konforlu ve sportif"
        ortam_metni = "yürüyüş, gezi veya hareketli bir gün"

    if len(set(renkler)) == 1:
        renk_aciklama = (
            f"{renkler[0]} tonları bir araya gelerek "
            "bütünlüklü bir görünüm oluşturuyor."
        )
    else:
        renk_aciklama = (
            f"{', '.join(renkler)} renkleri bir arada kullanılıyor. "
            "Nötr renkler, diğer renklerin dengeli görünmesine yardımcı olur."
        )

    ayakkabi_var = "Ayakkabı" in turler

    ayakkabi_notu = (
        "Ayakkabı, kombinin alt kısmını tamamlıyor."
        if ayakkabi_var
        else "Ayakkabı ekleyerek kombini tamamlayabilirsin."
    )

    return {
        "stil": stil,
        "ortam": ortam_metni,
        "renk": renk_aciklama,
        "ayakkabi": ayakkabi_notu,
        "aksesuar": (
            "Sade bir çanta ve minimal takılar görünümü tamamlayabilir."
        ),
        "alternatif": (
            "Daha farklı bir görünüm için parçalardan birini "
            "siyah, beyaz veya gri gibi nötr bir renkle değiştirebilirsin."
        )
    }


def kombin_oner(dolap, ortam, adet=3):
    ustler = [
        x for x in dolap
        if x["tur"] == "Üst giyim"
    ]

    altlar = [
        x for x in dolap
        if x["tur"] == "Alt giyim"
    ]

    elbiseler = [
        x for x in dolap
        if x["tur"] == "Elbise"
    ]

    ayakkabilar = [
        x for x in dolap
        if x["tur"] == "Ayakkabı"
    ]

    kombinler = []

    # Üst + alt + ayakkabı
    for ust in ustler:
        for alt in altlar:
            if not kombin_icin_renk_uyumu(ust["renk"], alt["renk"]):
                continue

            if ayakkabilar:
                for ayakkabi in ayakkabilar:
                    if (
                        kombin_icin_renk_uyumu(
                            ust["renk"], ayakkabi["renk"]
                        )
                        or kombin_icin_renk_uyumu(
                            alt["renk"], ayakkabi["renk"]
                        )
                    ):
                        parcalar = [ust, alt, ayakkabi]
                        kombinler.append({
                            "parcalar": parcalar,
                            "aciklama": kombin_aciklamasi(
                                parcalar, ortam
                            )
                        })
            else:
                parcalar = [ust, alt]
                kombinler.append({
                    "parcalar": parcalar,
                    "aciklama": kombin_aciklamasi(parcalar, ortam)
                })

    # Elbise + ayakkabı
    for elbise in elbiseler:
        if ayakkabilar:
            for ayakkabi in ayakkabilar:
                if kombin_icin_renk_uyumu(
                    elbise["renk"], ayakkabi["renk"]
                ):
                    parcalar = [elbise, ayakkabi]
                    kombinler.append({
                        "parcalar": parcalar,
                        "aciklama": kombin_aciklamasi(
                            parcalar, ortam
                        )
                    })
        else:
            parcalar = [elbise]
            kombinler.append({
                "parcalar": parcalar,
                "aciklama": kombin_aciklamasi(parcalar, ortam)
            })

    # Aynı parçaları tekrar tekrar göstermemek için benzersizleştir.
    benzersiz = []
    gorulen = set()

    for kombin in kombinler:
        anahtar = tuple(
            sorted(p["dosya_adi"] for p in kombin["parcalar"])
        )
        if anahtar not in gorulen:
            gorulen.add(anahtar)
            benzersiz.append(kombin)

    return benzersiz[:adet]


# --------------------------------------------------
# MODELİ BAŞLAT
# --------------------------------------------------

with st.spinner("Akıllı Dolabım hazırlanıyor..."):
    try:
        tur_modeli, bolge_modeli = modelleri_yukle()
    except Exception as hata:
        st.error(
            "Yapay zekâ modelleri yüklenemedi. "
            "İnternet bağlantısını ve requirements.txt dosyasını kontrol et."
        )
        st.code(str(hata))
        st.stop()

dolap = dolabi_yukle()

# --------------------------------------------------
# YAN MENÜ
# --------------------------------------------------

st.sidebar.header("📌 Menü")

sayfa = st.sidebar.radio(
    "Ne yapmak istiyorsun?",
    [
        "Kıyafet Ekle",
        "Dolabım",
        "Kombin Öner"
    ]
)

# --------------------------------------------------
# KIYAFET EKLE
# --------------------------------------------------

if sayfa == "Kıyafet Ekle":
    st.header("📸 Dolabına kıyafet ekle")

    yuklenen = st.file_uploader(
        "Kıyafet fotoğrafını seç",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if yuklenen is not None:
        resim = Image.open(yuklenen).convert("RGB")

        st.image(
            resim,
            caption="Yüklediğin fotoğraf",
            width=300
        )

        tur_secimi = st.selectbox(
            "Kıyafet türü",
            ["Yapay zekâ belirlesin"] + KIYAFET_TURLERI
        )

        if st.button("✨ Kıyafeti analiz et ve dolaba ekle"):
            try:
                with st.spinner("Kıyafet analiz ediliyor..."):
                    if tur_secimi == "Yapay zekâ belirlesin":
                        tur = kiyafet_turu_bul(resim, tur_modeli)
                    else:
                        tur = tur_secimi

                    renk = renk_bul(
                        resim,
                        bolge_modeli,
                        tur
                    )

                    kayit = {
                        "dosya_adi": yuklenen.name,
                        "tur": tur,
                        "renk": renk,
                        "resim": resim_base64(resim)
                    }

                    dolap.append(kayit)
                    dolabi_kaydet(dolap)

                st.success("Kıyafet dolabına eklendi!")
                st.write(f"**Kıyafet türü:** {tur}")
                st.write(f"**Algılanan renk:** {renk}")

                if tur == "Ayakkabı" and renk == "Pembe-Siyah":
                    st.caption(
                        "İki renkli ayakkabı kontrolü uygulandı. "
                        "Sonuç fotoğrafın ışığına ve renklerine göre değişebilir."
                    )

                st.rerun()

            except Exception as hata:
                st.error("Kıyafet analiz edilirken bir hata oluştu.")
                st.code(str(hata))


# --------------------------------------------------
# DOLABIM
# --------------------------------------------------

elif sayfa == "Dolabım":
    st.header("🧥 Dolabımdaki kıyafetler")

    if not dolap:
        st.info(
            "Dolabın henüz boş. Kıyafet Ekle bölümünden "
            "ilk fotoğrafını yükleyebilirsin."
        )
    else:
        filtre = st.selectbox(
            "Kıyafet türüne göre filtrele",
            ["Tümü"] + KIYAFET_TURLERI
        )

        goruntulenecek = dolap

        if filtre != "Tümü":
            goruntulenecek = [
                x for x in dolap if x["tur"] == filtre
            ]

        sutun_sayisi = 3

        for baslangic in range(
            0, len(goruntulenecek), sutun_sayisi
        ):
            sutunlar = st.columns(sutun_sayisi)

            for i, sutun in enumerate(sutunlar):
                sira = baslangic + i

                if sira >= len(goruntulenecek):
                    continue

                parca = goruntulenecek[sira]

                with sutun:
                    try:
                        st.image(
                            base64_resim_ac(parca["resim"]),
                            use_container_width=True
                        )
                    except Exception:
                        st.warning("Fotoğraf görüntülenemedi.")

                    st.markdown(f"**{parca['dosya_adi']}**")
                    st.write(f"Tür: {parca['tur']}")
                    st.write(f"Renk: {parca['renk']}")

                    if st.button(
                        "🗑️ Sil",
                        key=f"sil_{sira}_{parca['dosya_adi']}"
                    ):
                        dolap.remove(parca)
                        dolabi_kaydet(dolap)
                        st.rerun()


# --------------------------------------------------
# KOMBİN ÖNER
# --------------------------------------------------

elif sayfa == "Kombin Öner":
    st.header("✨ Sana özel kombinler")

    if not dolap:
        st.info("Önce dolabına birkaç kıyafet eklemelisin.")
    else:
        ortam = st.selectbox(
            "Kombini hangi ortam için istiyorsun?",
            ["Günlük", "Şık", "Sportif"]
        )

        adet = st.slider(
            "Kaç kombin gösterilsin?",
            min_value=1,
            max_value=10,
            value=3
        )

        if st.button("👗 Kombinlerimi oluştur"):
            kombinler = kombin_oner(dolap, ortam, adet)

            if not kombinler:
                st.warning(
                    "Henüz uygun kombin oluşturulamadı. "
                    "Dolabına üst giyim, alt giyim veya elbise "
                    "ve mümkünse ayakkabı eklemeyi dene."
                )
            else:
                for sira, kombin in enumerate(kombinler, start=1):
                    st.divider()
                    st.subheader(f"💖 Kombin {sira}")

                    parcalar = kombin["parcalar"]
                    aciklama = kombin["aciklama"]

                    sutunlar = st.columns(len(parcalar))

                    for sutun, parca in zip(sutunlar, parcalar):
                        with sutun:
                            try:
                                st.image(
                                    base64_resim_ac(parca["resim"]),
                                    use_container_width=True
                                )
                            except Exception:
                                st.warning(
                                    "Bu parçanın fotoğrafı açılamadı."
                                )

                            st.markdown(f"**{parca['tur']}**")
                            st.write(f"Renk: {parca['renk']}")
                            st.caption(parca["dosya_adi"])

                    st.markdown("#### 🪄 Kombin açıklaması")
                    st.write(aciklama["stil"] + " bir kombin.")

                    st.markdown("**🎯 Nerede giyebilirsin?**")
                    st.write(
                        "Bu kombin " + aciklama["ortam"] + " için uygun olabilir."
                    )

                    st.markdown("**🎨 Renk uyumu**")
                    st.write(aciklama["renk"])

                    st.markdown("**👟 Ayakkabı**")
                    st.write(aciklama["ayakkabi"])

                    st.markdown("**👜 Aksesuar önerisi**")
                    st.write(aciklama["aksesuar"])

                    st.markdown("**🔄 Alternatif öneri**")
                    st.write(aciklama["alternatif"])

st.sidebar.divider()
st.sidebar.caption("Akıllı Dolabım • Yapay zekâ destekli kombin asistanı")
