
import streamlit as st
from PIL import Image
from transformers import pipeline
import io
import json
import os
import base64

st.set_page_config(
    page_title="Akıllı Dolabım",
    page_icon="👗",
    layout="wide"
)

st.title("👗 Akıllı Dolabım")
st.subheader("Yapay zekâ destekli kombin asistanın")
st.write("Kıyafetlerini yükle, analiz et ve kombin önerilerini keşfet!")

DOLAP_DOSYASI = "dolap.json"


# -----------------------------
# KIYAFETLERİ KAYDET VE YÜKLE
# -----------------------------

def dolabi_kaydet(dolap):
    kayitlar = []

    for kiyafet in dolap:
        tampon = io.BytesIO()
        kiyafet["resim"].convert("RGB").save(
            tampon, format="JPEG", quality=85
        )

        kayitlar.append({
            "isim": kiyafet["isim"],
            "tur": kiyafet["tur"],
            "renk": kiyafet["renk"],
            "guven": float(kiyafet["guven"]),
            "resim": base64.b64encode(
                tampon.getvalue()
            ).decode("utf-8")
        })

    with open(DOLAP_DOSYASI, "w", encoding="utf-8") as dosya:
        json.dump(kayitlar, dosya, ensure_ascii=False)


def dolabi_yukle():
    if not os.path.exists(DOLAP_DOSYASI):
        return []

    try:
        with open(DOLAP_DOSYASI, "r", encoding="utf-8") as dosya:
            kayitlar = json.load(dosya)

        for kiyafet in kayitlar:
            resim_bytes = base64.b64decode(kiyafet["resim"])
            kiyafet["resim"] = Image.open(
                io.BytesIO(resim_bytes)
            ).convert("RGB")

        return kayitlar

    except Exception as hata:
        st.warning(f"Kayıtlı dolap yüklenemedi: {hata}")
        return []


# -----------------------------
# YAPAY ZEKÂ MODELLERİ
# -----------------------------

@st.cache_resource
def segmentasyon_modeli_yukle():
    return pipeline(
        "image-segmentation",
        model="mattmdjaga/segformer_b2_clothes",
        device=-1
    )


@st.cache_resource
def model_yukle():
    return pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32",
        device=-1
    )


# -----------------------------
# KIYAFET TÜRLERİ
# -----------------------------

KIYAFETLER = [
    "a photo of a t-shirt",
    "a photo of a shirt",
    "a photo of a sweater",
    "a photo of a jacket",
    "a photo of a coat",
    "a photo of trousers",
    "a photo of jeans",
    "a photo of a skirt",
    "a photo of a dress",
    "a photo of shorts",
    "a photo of sneakers",
    "a photo of shoes"
]

TUR_CEVIRI = {
    "a photo of a t-shirt": "Tişört",
    "a photo of a shirt": "Gömlek",
    "a photo of a sweater": "Kazak",
    "a photo of a jacket": "Ceket",
    "a photo of a coat": "Mont / Kaban",
    "a photo of trousers": "Pantolon",
    "a photo of jeans": "Kot pantolon",
    "a photo of a skirt": "Etek",
    "a photo of a dress": "Elbise",
    "a photo of a shorts": "Şort",
    "a photo of shorts": "Şort",
    "a photo of sneakers": "Spor ayakkabı",
    "a photo of shoes": "Ayakkabı"
}


# -----------------------------
# KIYAFET RENGİNİ BUL
# -----------------------------

def renk_bul(resim):
    import numpy as np
    import colorsys

    resim = resim.convert("RGB")
    bolgeler = segmentasyon_modeli_yukle()(resim)

    kiyafet_siniflari = {
        "Upper-clothes", "Skirt", "Pants", "Dress",
        "Belt", "Left-shoe", "Right-shoe", "Bag", "Scarf"
    }

    pikseller = np.array(resim)
    maskeler = []

    for bolge in bolgeler:
        if bolge["label"] not in kiyafet_siniflari:
            continue

        maske = bolge["mask"].convert("L").resize(resim.size)
        maskeler.append(np.array(maske) > 127)

    if not maskeler:
        return "Belirlenemedi"

    maske = np.logical_or.reduce(maskeler)
    rgb = pikseller[maske]

    if len(rgb) == 0:
        return "Belirlenemedi"

    rgb = rgb[::max(1, len(rgb) // 10000)]

    hsv = np.array([
        colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        for r, g, b in rgb
    ])

    tonlar = hsv[:, 0]
    doygunluklar = hsv[:, 1]
    parlakliklar = hsv[:, 2]

    if np.median(parlakliklar) < 0.22:
        return "Siyah"

    renkliler = (
        (doygunluklar > 0.20) &
        (parlakliklar > 0.18)
    )

    if np.mean(renkliler) < 0.15:
        return "Beyaz" if np.median(parlakliklar) > 0.82 else "Gri"

    secilen_tonlar = tonlar[renkliler]
    secilen_s = doygunluklar[renkliler]
    secilen_v = parlakliklar[renkliler]

    ton = float(np.median(secilen_tonlar))
    sat = float(np.median(secilen_s))
    val = float(np.median(secilen_v))

    if sat < 0.28:
        if val > 0.82:
            return "Beyaz"
        if val < 0.35:
            return "Siyah"
        return "Gri"

    if 0.18 <= ton < 0.43:
        return "Yeşil"
    if 0.43 <= ton < 0.55:
        return "Mavi"
    if 0.55 <= ton < 0.72:
        return "Lacivert" if val < 0.55 else "Mavi"
    if ton < 0.04 or ton >= 0.96:
        return "Kırmızı"
    if ton < 0.10:
        return "Turuncu"
    if ton < 0.17:
        return "Sarı"
    if ton < 0.82:
        return "Mor"
    if ton < 0.94:
        return "Pembe"

    return "Kırmızı"


# -----------------------------
# DOLABI YÜKLE
# -----------------------------

if "dolap" not in st.session_state:
    st.session_state["dolap"] = dolabi_yukle()


# -----------------------------
# FOTOĞRAF YÜKLE VE ANALİZ ET
# -----------------------------

st.divider()
st.header("📸 Dijital dolabını oluştur")

fotograflar = st.file_uploader(
    "Kıyafet fotoğraflarını seç",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    key="kiyafet_fotograf_yukleyici"
)

if fotograflar:
    if len(fotograflar) > 12:
        st.warning("İlk sürümde en fazla 12 fotoğraf seçmeni öneriyorum.")

    if st.button("🤖 Kıyafetlerimi yapay zekâyla analiz et"):
        try:
            with st.spinner("Fotoğraflar analiz ediliyor..."):
                model = model_yukle()
                dolap = st.session_state["dolap"]

                mevcut_isimler = {
                    p["isim"] for p in dolap
                }

                eklenen = 0
                atlanan = 0

                for fotograf in fotograflar[:12]:
                    if fotograf.name in mevcut_isimler:
                        atlanan += 1
                        continue

                    resim = Image.open(
                        io.BytesIO(fotograf.getvalue())
                    ).convert("RGB")

                    tahminler = model(
                        resim,
                        candidate_labels=KIYAFETLER
                    )

                    en_iyi = tahminler[0]

                    dolap.append({
                        "isim": fotograf.name,
                        "resim": resim,
                        "tur": TUR_CEVIRI[en_iyi["label"]],
                        "renk": renk_bul(resim),
                        "guven": float(en_iyi["score"])
                    })

                    mevcut_isimler.add(fotograf.name)
                    eklenen += 1

            st.session_state["dolap"] = dolap
            dolabi_kaydet(dolap)

            st.success(f"{eklenen} yeni kıyafet eklendi.")

            if atlanan:
                st.info(
                    f"{atlanan} fotoğraf aynı dosya adına sahip "
                    "olduğu için tekrar eklenmedi."
                )

        except Exception as hata:
            st.error("Analiz sırasında bir hata oluştu.")
            st.caption(str(hata))


# -----------------------------
# DOLAPTAKİ KIYAFETLER VE SİLME
# -----------------------------

dolap = st.session_state["dolap"]

if dolap:
    st.divider()
    st.header("🧥 Dolabımdaki kıyafetler")
    st.write(f"Toplam kıyafet: **{len(dolap)}**")

    sutunlar = st.columns(3)

    for i, parca in enumerate(dolap):
        with sutunlar[i % 3]:
            st.image(parca["resim"], use_container_width=True)
            st.markdown(f"**{parca['isim']}**")
            st.write(f"Tür: {parca['tur']}")
            st.write(f"Renk: {parca['renk']}")
            st.caption(
                f"Model tahmin puanı: %{parca['guven'] * 100:.1f}"
            )

            if st.button(
                "🗑️ Dolabımdan kaldır",
                key=f"sil_{i}"
            ):
                st.session_state["dolap"].pop(i)
                dolabi_kaydet(st.session_state["dolap"])
                st.rerun()

    # -----------------------------
    # KOMBİN DANIŞMANI
    # -----------------------------

    st.divider()
    st.header("👗 Akıllı Kombin Danışmanın")

    ustler = [
        p for p in dolap
        if p["tur"] in [
            "Tişört", "Gömlek", "Kazak", "Ceket", "Mont / Kaban"
        ]
    ]

    altlar = [
        p for p in dolap
        if p["tur"] in [
            "Pantolon", "Kot pantolon", "Etek", "Şort"
        ]
    ]

    elbiseler = [
        p for p in dolap if p["tur"] == "Elbise"
    ]

    ayakkabilar = [
        p for p in dolap
        if p["tur"] in ["Ayakkabı", "Spor ayakkabı"]
    ]

    def kombin_aciklamasi(ust, alt):
        ust_rengi = ust["renk"]
        alt_rengi = alt["renk"]

        if ust_rengi == alt_rengi:
            renk_yorumu = (
                f"{ust_rengi} üst ve {alt_rengi} alt benzer tonlarda "
                "bütünlüklü bir görünüm oluşturur."
            )
        else:
            renk_yorumu = (
                f"{ust_rengi} üst ile {alt_rengi} alt kontrast "
                "oluşturabilir. Diğer parçaları sade tutabilirsin."
            )

        takilar = (
            "Altın renkli takılar sıcak bir görünüm; gümüş renkli "
            "takılar daha soğuk ve sade bir görünüm verebilir."
        )

        ayakkabi = (
            "Beyaz veya siyah ayakkabı sade bir seçimdir. "
            "Bej daha yumuşak, kırmızı daha dikkat çekici olabilir."
        )

        canta = (
            "Siyah çanta klasik, bej çanta yumuşak, beyaz çanta "
            "ise ferah bir görünüm sağlayabilir."
        )

        alternatifler = [
            r for r in ["Beyaz", "Siyah", "Krem", "Gri", "Pembe", "Mavi"]
            if r != ust_rengi
        ][:4]

        return (
            f"**Renk uyumu:** {renk_yorumu}\n\n"
            f"**Takı önerisi:** {takilar}\n\n"
            f"**Ayakkabı önerisi:** {ayakkabi}\n\n"
            f"**Çanta önerisi:** {canta}\n\n"
            f"**Alternatif üst renkleri:** {', '.join(alternatifler)}\n\n"
            "**Stil önerisi:** Günlük kullanımda spor ayakkabı; "
            "daha şık bir ortamda zarif takılar ve klasik ayakkabı "
            "tercih edebilirsin. Bu öneriler renk uyumuna dayalıdır."
        )

    kombin_sayisi = 0

    if ustler and altlar:
        for ust in ustler:
            for alt in altlar:
                if ust["isim"] == alt["isim"]:
                    continue

                kombin_sayisi += 1

                with st.expander(
                    f"✨ Kombin {kombin_sayisi}: "
                    f"{ust['renk']} {ust['tur']} + "
                    f"{alt['renk']} {alt['tur']}",
                    expanded=(kombin_sayisi == 1)
                ):
                    kolonlar = st.columns(2)

                    with kolonlar[0]:
                        st.image(ust["resim"], use_container_width=True)
                        st.caption(f"{ust['tur']} — {ust['renk']}")

                    with kolonlar[1]:
                        st.image(alt["resim"], use_container_width=True)
                        st.caption(f"{alt['tur']} — {alt['renk']}")

                    st.markdown(kombin_aciklamasi(ust, alt))

                    if ayakkabilar:
                        st.info(
                            "Dolabındaki ayakkabı: "
                            f"{ayakkabilar[0]['renk']} "
                            f"{ayakkabilar[0]['tur']}."
                        )

                if kombin_sayisi >= 5:
                    break

            if kombin_sayisi >= 5:
                break

    if kombin_sayisi == 0 and elbiseler:
        for elbise in elbiseler[:3]:
            st.subheader(
                f"{elbise['renk']} elbise için öneriler"
            )
            st.image(elbise["resim"], width=240)
            st.markdown(
                "Siyah veya bej çanta deneyebilirsin. Altın renkli "
                "takılar sıcak, gümüş renkli takılar daha soğuk "
                "bir görünüm verebilir. Beyaz, siyah veya bej "
                "ayakkabılar farklı stillere uyarlanabilir."
            )

    if kombin_sayisi == 0 and not elbiseler:
        st.info(
            "Kombin önerileri için en az bir üst ve bir alt parça "
            "veya bir elbise yükle."
        )

else:
    st.info(
        "Dolabın boş. Fotoğraf yükleyerek kıyafet ekleyebilirsin."
    )

st.divider()
st.caption("Akıllı Dolabım | CLIP tabanlı ilk yapay zekâ prototipi")
