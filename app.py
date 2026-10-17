
import streamlit as st
from PIL import Image
from transformers import pipeline
from collections import Counter
import io

st.set_page_config(
    page_title="Akıllı Dolabım",
    page_icon="👗",
    layout="wide"
)

st.title("👗 Akıllı Dolabım")
st.subheader("Yapay zekâ destekli kombin asistanın")
st.write("Kıyafetlerini yükle, analiz et ve kombin önerilerini keşfet!")

@st.cache_resource
def segmentasyon_modeli_yukle():
    from transformers import pipeline
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
    "a photo of shorts": "Şort",
    "a photo of sneakers": "Spor ayakkabı",
    "a photo of shoes": "Ayakkabı"
}

RENKLER = {
    "Siyah": (0, 0, 0),
    "Beyaz": (255, 255, 255),
    "Gri": (128, 128, 128),
    "Kırmızı": (220, 30, 30),
    "Mavi": (35, 90, 190),
    "Lacivert": (20, 35, 90),
    "Yeşil": (30, 130, 60),
    "Sarı": (240, 210, 40),
    "Pembe": (240, 140, 180),
    "Mor": (130, 70, 160),
    "Kahverengi": (120, 75, 40),
    "Bej": (210, 190, 150),
    "Turuncu": (235, 120, 30)
}




def renk_bul(resim):
    import numpy as np
    import colorsys

    resim = resim.convert("RGB")

    model = segmentasyon_modeli_yukle()
    bolgeler = model(resim)

    kiyafet_siniflari = {
        "Upper-clothes",
        "Skirt",
        "Pants",
        "Dress",
        "Belt",
        "Left-shoe",
        "Right-shoe",
        "Bag",
        "Scarf"
    }

    pikseller = np.array(resim)
    secilen_maskeler = []

    for bolge in bolgeler:
        if bolge["label"] not in kiyafet_siniflari:
            continue

        maske = bolge["mask"].convert("L").resize(resim.size)
        maske = np.array(maske) > 127
        secilen_maskeler.append(maske)

    if not secilen_maskeler:
        return "Belirlenemedi"

    maske = np.logical_or.reduce(secilen_maskeler)

    # Beyaz ve çok açık arka plan piksellerini dışarıda bırakmak
    # için değil, yalnızca modelin kıyafet maskesini kullanıyoruz.
    rgb_pikseller = pikseller[maske]

    if len(rgb_pikseller) == 0:
        return "Belirlenemedi"

    # Çok fazla piksel varsa rastgele olmayan aralıklı örnekleme.
    rgb_pikseller = rgb_pikseller[::max(1, len(rgb_pikseller) // 5000)]

    hsv_pikseller = [
        colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        for r, g, b in rgb_pikseller
    ]

    # Çok koyu pikselleri siyah; düşük doygunluklu olanları
    # gri/beyaz olarak değerlendir.
    ortalama_v = np.median([hsv[2] for hsv in hsv_pikseller])
    ortalama_s = np.median([hsv[1] for hsv in hsv_pikseller])

    if ortalama_v < 0.16:
        return "Siyah"

    if ortalama_s < 0.12:
        if ortalama_v > 0.82:
            return "Beyaz"
        if ortalama_v < 0.35:
            return "Gri"
        return "Gri"

    # Doygun renklerde en sık görülen tonu bul.
    renkli = [
        hsv for hsv in hsv_pikseller
        if hsv[1] >= 0.12 and hsv[2] >= 0.16
    ]

    if not renkli:
        return "Gri"

    tonlar = [hsv[0] for hsv in renkli]
    ton = float(np.median(tonlar))
    doygunluk = float(np.median([hsv[1] for hsv in renkli]))
    parlaklik = float(np.median([hsv[2] for hsv in renkli]))

    if ton < 0.04 or ton >= 0.96:
        return "Kırmızı"
    if ton < 0.10:
        return "Turuncu"
    if ton < 0.17:
        return "Sarı"
    if ton < 0.45:
        return "Yeşil"
    if ton < 0.57:
        return "Mavi"
    if ton < 0.72:
        return "Lacivert" if parlaklik < 0.55 else "Mavi"
    if ton < 0.82:
        return "Mor"
    if ton < 0.94:
        return "Pembe"

    return "Mavi"



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
        st.warning("İlk sürümde en fazla 12 fotoğraf yüklemeni öneriyorum.")

    if st.button("🤖 Kıyafetlerimi yapay zekâyla analiz et"):
        try:
            with st.spinner("Yapay zekâ modeli yükleniyor ve fotoğraflar analiz ediliyor..."):
                model = model_yukle()
                dolap = []

                for fotograf in fotograflar[:12]:
                    resim = Image.open(
                        io.BytesIO(fotograf.getvalue())
                    ).convert("RGB")

                    tahminler = model(
                        resim,
                        candidate_labels=KIYAFETLER
                    )

                    en_iyi = tahminler[0]
                    tur = TUR_CEVIRI[en_iyi["label"]]
                    renk = renk_bul(resim)

                    dolap.append({
                        "isim": fotograf.name,
                        "resim": resim,
                        "tur": tur,
                        "renk": renk,
                        "guven": en_iyi["score"]
                    })

            st.session_state["dolap"] = dolap
            st.success("Analiz tamamlandı!")

        except Exception as hata:
            st.error("Model çalışırken bir sorun oluştu.")
            st.caption(str(hata))

if "dolap" in st.session_state:
    dolap = st.session_state["dolap"]

    st.divider()
    st.header("🧥 Yapay zekâ kıyafet analizleri")

    sutunlar = st.columns(3)

    for i, parca in enumerate(dolap):
        with sutunlar[i % 3]:
            st.image(
                parca["resim"],
                use_container_width=True
            )
            st.markdown(f"**{parca['isim']}**")
            st.write(f"Tür: {parca['tur']}")
            st.write(f"Baskın renk: {parca['renk']}")
            st.caption(
                f"Model tahmin puanı: %{parca['guven'] * 100:.1f}"
            )

    st.divider()
    st.header("👗 Kombin önerileri")

    ustler = [
        p for p in dolap
        if p["tur"] in [
            "Tişört", "Gömlek", "Kazak",
            "Ceket", "Mont / Kaban"
        ]
    ]

    altlar = [
        p for p in dolap
        if p["tur"] in [
            "Pantolon", "Kot pantolon",
            "Etek", "Şort"
        ]
    ]

    elbiseler = [p for p in dolap if p["tur"] == "Elbise"]
    ayakkabilar = [
        p for p in dolap
        if p["tur"] in ["Ayakkabı", "Spor ayakkabı"]
    ]

    kombin_sayisi = 0

    for ust in ustler:
        for alt in altlar:
            if ust["isim"] == alt["isim"]:
                continue

            if ust["renk"] == alt["renk"]:
                aciklama = (
                    "Aynı renk tonları sade bir görünüm oluşturabilir. "
                    "Farklı aksesuarlarla kombini hareketlendirebilirsin."
                )
            else:
                aciklama = (
                    f"{ust['renk']} üst ve {alt['renk']} alt parça, "
                    "birbirinden ayrılan renkleriyle dengeli bir "
                    "kombin alternatifi olabilir."
                )

            st.subheader(
                f"Kombin {kombin_sayisi + 1}: "
                f"{ust['tur']} + {alt['tur']}"
            )

            kolonlar = st.columns([1, 1, 2])

            with kolonlar[0]:
                st.image(ust["resim"], use_container_width=True)
                st.caption(f"{ust['tur']} — {ust['renk']}")

            with kolonlar[1]:
                st.image(alt["resim"], use_container_width=True)
                st.caption(f"{alt['tur']} — {alt['renk']}")

            with kolonlar[2]:
                st.write(aciklama)

                if ayakkabilar:
                    st.write(
                        "Tamamlayıcı parça: "
                        + ayakkabilar[0]["tur"]
                        + f" ({ayakkabilar[0]['renk']})"
                    )

            kombin_sayisi += 1

            if kombin_sayisi >= 5:
                break

        if kombin_sayisi >= 5:
            break

    if kombin_sayisi == 0 and elbiseler:
        for elbise in elbiseler[:3]:
            st.image(elbise["resim"], width=220)
            st.write(
                f"{elbise['renk']} {elbise['tur']} "
                "tek parça kombin olarak kullanılabilir."
            )
            if ayakkabilar:
                st.write(
                    f"Yanına {ayakkabilar[0]['renk']} "
                    f"{ayakkabilar[0]['tur']} ekleyebilirsin."
                )
            kombin_sayisi += 1

    if kombin_sayisi == 0:
        st.info(
            "Kombin oluşturmak için en az bir üst ve bir alt "
            "parça veya bir elbise yükle."
        )

st.divider()
st.caption(
    "Akıllı Dolabım | CLIP tabanlı ilk yapay zekâ prototipi"
)

