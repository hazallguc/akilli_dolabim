
import streamlit as st
from PIL import Image
from transformers import pipeline
import io
import json
import os
import base64
import colorsys
import numpy as np

# --------------------------------------------------
# SAYFA AYARLARI
# --------------------------------------------------

st.set_page_config(
    page_title="Akıllı Dolabım",
    page_icon="👗",
    layout="wide"
)

st.title("👗 Akıllı Dolabım")
st.write("Yapay zekâ ile kıyafetlerini tanı, dolabını düzenle ve kombin önerileri al.")

DOLAP_DOSYASI = "dolap.json"


# --------------------------------------------------
# DOLABI KAYDETME VE YÜKLEME
# --------------------------------------------------

def dolabi_kaydet(dolap):
    try:
        with open(DOLAP_DOSYASI, "w", encoding="utf-8") as dosya:
            json.dump(dolap, dosya, ensure_ascii=False)
    except Exception as hata:
        st.warning(f"Dolap kaydedilemedi: {hata}")


def dolabi_yukle():
    if os.path.exists(DOLAP_DOSYASI):
        try:
            with open(DOLAP_DOSYASI, "r", encoding="utf-8") as dosya:
                return json.load(dosya)
        except Exception:
            return []
    return []


def resmi_base64_yap(resim):
    tampon = io.BytesIO()
    resim.save(tampon, format="JPEG")
    return base64.b64encode(tampon.getvalue()).decode("utf-8")


def base64_resim_ac(veri):
    return Image.open(
        io.BytesIO(base64.b64decode(veri))
    ).convert("RGB")


# --------------------------------------------------
# YAPAY ZEKÂ MODELLERİ
# --------------------------------------------------

@st.cache_resource
def modelleri_yukle():
    kiyafet_modeli = pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32"
    )

    segmentasyon_modeli = pipeline(
        "image-segmentation",
        model="mattmdjaga/segformer_b2_clothes"
    )

    return kiyafet_modeli, segmentasyon_modeli


KIYAFETLER = [
    "a T-shirt",
    "a shirt",
    "a sweater",
    "a jacket",
    "a coat",
    "a dress",
    "a skirt",
    "a pair of trousers",
    "a pair of jeans",
    "a pair of shorts",
    "a pair of shoes",
    "a pair of sneakers",
    "a handbag",
    "a hat",
    "a scarf"
]

KIYAFET_CEVIRI = {
    "a T-shirt": "Tişört",
    "a shirt": "Gömlek",
    "a sweater": "Kazak",
    "a jacket": "Ceket",
    "a coat": "Mont / Kaban",
    "a dress": "Elbise",
    "a skirt": "Etek",
    "a pair of trousers": "Pantolon",
    "a pair of jeans": "Kot Pantolon",
    "a pair of shorts": "Şort",
    "a pair of shoes": "Ayakkabı",
    "a pair of sneakers": "Spor Ayakkabı",
    "a handbag": "Çanta",
    "a hat": "Şapka",
    "a scarf": "Atkı"
}


# --------------------------------------------------
# RENK ALGILAMA
# --------------------------------------------------

def piksellerin_renklerini_say(pikseller):
    sayac = {
        "Siyah": 0,
        "Beyaz": 0,
        "Gri": 0,
        "Pembe": 0,
        "Kırmızı": 0,
        "Turuncu": 0,
        "Sarı": 0,
        "Yeşil": 0,
        "Mavi": 0,
        "Lacivert": 0,
        "Mor": 0,
        "Kahverengi": 0
    }

    for piksel in pikseller:
        r, g, b = [float(x) / 255 for x in piksel[:3]]
        ton, doygunluk, parlaklik = colorsys.rgb_to_hsv(r, g, b)

        # Siyah, gri ve beyazı ayırt et
        if parlaklik < 0.20:
            sayac["Siyah"] += 1
            continue

        if doygunluk < 0.16:
            if parlaklik > 0.82:
                sayac["Beyaz"] += 1
            elif parlaklik < 0.48:
                sayac["Gri"] += 1
            else:
                sayac["Gri"] += 1
            continue

        # Pembe ve mor tonları
        if 0.82 <= ton < 0.96:
            if parlaklik < 0.30:
                sayac["Kahverengi"] += 1
            else:
                sayac["Pembe"] += 1
            continue

        if 0.75 <= ton < 0.82:
            sayac["Mor"] += 1
            continue

        # Kırmızı
        if ton < 0.04 or ton >= 0.96:
            sayac["Kırmızı"] += 1
            continue

        # Kahverengi ve turuncu
        if ton < 0.10:
            if parlaklik < 0.60:
                sayac["Kahverengi"] += 1
            else:
                sayac["Turuncu"] += 1
            continue

        # Sarı ve kahverengi
        if ton < 0.17:
            if parlaklik < 0.65:
                sayac["Kahverengi"] += 1
            else:
                sayac["Sarı"] += 1
            continue

        if ton < 0.20:
            if parlaklik < 0.55:
                sayac["Kahverengi"] += 1
            else:
                sayac["Sarı"] += 1
            continue

        if ton < 0.43:
            sayac["Yeşil"] += 1
        elif ton < 0.55:
            sayac["Mavi"] += 1
        elif ton < 0.72:
            if parlaklik < 0.55:
                sayac["Lacivert"] += 1
            else:
                sayac["Mavi"] += 1
        else:
            sayac["Mor"] += 1

    return sayac


def renk_bul(resim, segmentasyon_modeli=None):
    """
    Kıyafetin renklerini bulmaya çalışır.
    Segmentasyon başarılı olursa kıyafet maskesini kullanır.
    """

    resim = resim.convert("RGB")
    renk_pikselleri = None

    if segmentasyon_modeli is not None:
        try:
            sonuc = segmentasyon_modeli(resim)
            maskeler = []

            for parca in sonuc:
                etiket = str(parca.get("label", "")).lower()

                # Arka plan ve yüz gibi bölgeleri kullanma
                if any(
                    kelime in etiket
                    for kelime in [
                        "background", "face", "hair",
                        "skin", "left-leg", "right-leg",
                        "left-arm", "right-arm"
                    ]
                ):
                    continue

                maske = parca.get("mask")

                if maske is not None:
                    if not isinstance(maske, Image.Image):
                        maske = Image.fromarray(np.array(maske))

                    maske = maske.convert("L").resize(resim.size)
                    maske_dizisi = np.array(maske) > 127

                    if maske_dizisi.any():
                        maskeler.append(maske_dizisi)

            if maskeler:
                birlesik_maske = np.logical_or.reduce(maskeler)
                resim_dizisi = np.array(resim)
                renk_pikselleri = resim_dizisi[birlesik_maske]

        except Exception:
            renk_pikselleri = None

    # Segmentasyon kullanılamazsa resmin orta bölümünü incele
    if renk_pikselleri is None or len(renk_pikselleri) == 0:
        resim_dizisi = np.array(resim)
        h, w, _ = resim_dizisi.shape

        orta_bolge = resim_dizisi[
            int(h * 0.15):int(h * 0.90),
            int(w * 0.15):int(w * 0.90)
        ]

        renk_pikselleri = orta_bolge.reshape(-1, 3)

    # Çok fazla pikseli inceleyerek uygulamayı yavaşlatma
    if len(renk_pikselleri) > 20000:
        adim = max(1, len(renk_pikselleri) // 20000)
        renk_pikselleri = renk_pikselleri[::adim]

    sayac = piksellerin_renklerini_say(renk_pikselleri)
    toplam = sum(sayac.values())

    if toplam == 0:
        return "Bilinmiyor"

    # İki renkli pembe-siyah ayakkabı için özel kontrol
    pembe_orani = sayac["Pembe"] / toplam
    siyah_orani = sayac["Siyah"] / toplam

    if pembe_orani >= 0.10 and siyah_orani >= 0.10:
        return "Pembe-Siyah"

    # Arka plan veya küçük ayrıntılar sonucu yanıltmasın
    sirali = sorted(
        sayac.items(),
        key=lambda x: x[1],
        reverse=True
    )

    birinci_renk, birinci_sayi = sirali[0]

    # Siyah ve pembe birlikteyse renk adını koru
    if (
        birinci_renk == "Siyah"
        and pembe_orani >= 0.08
    ):
        return "Pembe-Siyah"

    if (
        birinci_renk == "Pembe"
        and siyah_orani >= 0.08
    ):
        return "Pembe-Siyah"

    return birinci_renk


# --------------------------------------------------
# KOMBİN ÖNERİSİ
# --------------------------------------------------

def kombin_oner(dolap):
    if len(dolap) < 2:
        return [
            "Kombin önerisi için dolabına en az iki parça ekle."
        ]

    oneriler = []

    ustler = [
        x for x in dolap
        if x["tur"] in [
            "Tişört", "Gömlek", "Kazak", "Ceket"
        ]
    ]

    altlar = [
        x for x in dolap
        if x["tur"] in [
            "Pantolon", "Kot Pantolon", "Etek", "Şort"
        ]
    ]

    ayakkabilar = [
        x for x in dolap
        if "Ayakkabı" in x["tur"]
    ]

    uyumlu_renkler = {
        "Siyah": ["Beyaz", "Gri", "Pembe", "Kırmızı", "Mavi"],
        "Beyaz": ["Siyah", "Mavi", "Lacivert", "Pembe", "Kahverengi"],
        "Gri": ["Siyah", "Beyaz", "Pembe", "Mavi", "Kırmızı"],
        "Pembe": ["Siyah", "Beyaz", "Gri", "Lacivert"],
        "Pembe-Siyah": ["Beyaz", "Gri", "Siyah", "Kot Pantolon"],
        "Lacivert": ["Beyaz", "Gri", "Pembe", "Kahverengi"],
        "Mavi": ["Beyaz", "Gri", "Siyah", "Kahverengi"],
        "Kahverengi": ["Beyaz", "Bej", "Krem", "Lacivert"],
        "Bej": ["Beyaz", "Kahverengi", "Siyah", "Lacivert"],
        "Kırmızı": ["Siyah", "Beyaz", "Gri", "Lacivert"],
        "Yeşil": ["Beyaz", "Siyah", "Bej", "Kahverengi"],
        "Mor": ["Siyah", "Beyaz", "Gri"],
        "Sarı": ["Siyah", "Beyaz", "Lacivert"],
        "Turuncu": ["Siyah", "Beyaz", "Kot Pantolon"]
    }

    if ustler and altlar:
        for ust in ustler:
            for alt in altlar:
                ust_rengi = ust["renk"]
                alt_rengi = alt["renk"]

                if (
                    ust_rengi == alt_rengi
                    or alt_rengi in uyumlu_renkler.get(ust_rengi, [])
                    or ust_rengi in uyumlu_renkler.get(alt_rengi, [])
                ):
                    oneriler.append(
                        f"👕 {ust['tur']} ({ust_rengi}) + "
                        f"👖 {alt['tur']} ({alt_rengi})"
                    )

    if ayakkabilar and oneriler:
        ayakkabi = ayakkabilar[0]
        oneriler[0] += (
            f" + 👟 Ayakkabı ({ayakkabi['renk']})"
        )

    if not oneriler:
        oneriler.append(
            "Nötr renkleri (siyah, beyaz ve gri) "
            "birlikte kullanmayı deneyebilirsin."
        )

    return oneriler[:10]


# --------------------------------------------------
# OTURUM VE DOLAP
# --------------------------------------------------

if "dolap" not in st.session_state:
    st.session_state["dolap"] = dolabi_yukle()

dolap = st.session_state["dolap"]

st.subheader("📸 Dolabına kıyafet ekle")

yuklenenler = st.file_uploader(
    "Kıyafet veya ayakkabı fotoğraflarını seç",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="kiyafet_yukleyici"
)

if st.button("✨ Fotoğrafları analiz et ve dolaba ekle"):
    if not yuklenenler:
        st.warning("Önce en az bir fotoğraf yükle.")
    else:
        with st.spinner(
            "Yapay zekâ modelleri yükleniyor ve fotoğraflar analiz ediliyor..."
        ):
            try:
                kiyafet_modeli, segmentasyon_modeli = modelleri_yukle()

                eklenen = 0
                atlanan = 0

                for dosya in yuklenenler:
                    try:
                        resim = Image.open(dosya).convert("RGB")

                        # Aynı isimdeki dosyaları tekrar ekleme
                        if any(
                            parca.get("dosya_adi") == dosya.name
                            for parca in dolap
                        ):
                            atlanan += 1
                            continue

                        tahminler = kiyafet_modeli(
                            resim,
                            candidate_labels=KIYAFETLER
                        )

                        tahmin = tahminler[0]["label"]
                        tur = KIYAFET_CEVIRI.get(
                            tahmin,
                            "Diğer"
                        )

                        renk = renk_bul(
                            resim,
                            segmentasyon_modeli
                        )

                        yeni_parca = {
                            "dosya_adi": dosya.name,
                            "tur": tur,
                            "renk": renk,
                            "resim": resmi_base64_yap(resim)
                        }

                        dolap.append(yeni_parca)
                        eklenen += 1

                    except Exception as hata:
                        st.warning(
                            f"{dosya.name} analiz edilemedi: {hata}"
                        )

                st.session_state["dolap"] = dolap
                dolabi_kaydet(dolap)

                if eklenen:
                    st.success(
                        f"{eklenen} fotoğraf dolabına eklendi!"
                    )

                if atlanan:
                    st.info(
                        f"{atlanan} fotoğraf aynı dosya adıyla "
                        "dolapta bulunduğu için eklenmedi."
                    )

            except Exception as hata:
                st.error(
                    "Yapay zekâ modelleri yüklenemedi. "
                    f"Hata ayrıntısı: {hata}"
                )


# --------------------------------------------------
# DOLABI GÖSTER
# --------------------------------------------------

st.divider()
st.subheader("🧥 Benim Dolabım")

dolap = st.session_state["dolap"]

if not dolap:
    st.info("Dolabın henüz boş. Fotoğraf yükleyerek başlayabilirsin.")
else:
    st.write(f"Dolabında toplam **{len(dolap)} parça** var.")

    sutunlar = st.columns(3)

    for i, parca in enumerate(dolap):
        with sutunlar[i % 3]:
            try:
                resim = base64_resim_ac(parca["resim"])
                st.image(resim, use_container_width=True)
            except Exception:
                st.warning("Bu fotoğraf açılamadı.")

            st.write(f"**Tür:** {parca['tur']}")
            st.write(f"**Renk:** {parca['renk']}")
            st.caption(parca.get("dosya_adi", ""))

            if st.button(
                "🗑️ Dolabımdan kaldır",
                key=f"sil_{i}"
            ):
                st.session_state["dolap"].pop(i)
                dolabi_kaydet(st.session_state["dolap"])
                st.rerun()


# --------------------------------------------------
# KOMBİN ÖNERİLERİ
# --------------------------------------------------

st.divider()
st.subheader("💡 Sana Özel Kombin Önerileri")

dolap = st.session_state["dolap"]

if dolap:
    oneriler = kombin_oner(dolap)

    for oner in oneriler:
        st.write(f"- {oner}")

    st.caption(
        "Kombin önerileri kıyafet türü ve renk uyumuna göre "
        "oluşturulur. Her öneri moda uzmanı değerlendirmesi değildir."
    )
else:
    st.info("Kombin önerileri için önce dolabına kıyafet ekle.")


# --------------------------------------------------
# BİLGİ
# --------------------------------------------------

st.divider()
st.caption(
    "Not: dolap.json dosyası dolap verilerini yerel dosya olarak saklar. "
    "Streamlit Cloud yeniden başlatıldığında veya yeniden dağıtımda "
    "bu dosyanın kalıcılığı garanti edilmez."
)
