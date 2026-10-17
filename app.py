
import streamlit as st
from PIL import Image
from transformers import pipeline
from collections import Counter
import io
import json
import os

st.set_page_config(
    page_title="Akıllı Dolabım",
    page_icon="👗",
    layout="wide"
)

st.title("👗 Akıllı Dolabım")
import base64

DOLAP_DOSYASI = "dolap.json"


def dolabi_kaydet(dolap):
    kayitlar = []

    for kiyafet in dolap:
        tampon = io.BytesIO()
        kiyafet["resim"].save(tampon, format="JPEG")

        kayitlar.append({
            "isim": kiyafet["isim"],
            "tur": kiyafet["tur"],
            "renk": kiyafet["renk"],
            "guven": float(kiyafet["guven"]),
            "resim": base64.b64encode(tampon.getvalue()).decode("utf-8")
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

    except Exception:
        return []
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

    # Çok koyu pikselleri siyah olarak değerlendir.
    ortalama_v = np.median([hsv[2] for hsv in hsv_pikseller])
    ortalama_s = np.median([hsv[1] for hsv in hsv_pikseller])

    if ortalama_v < 0.25:
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
         dolabi_kaydet(dolap)
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
    st.header("👗 Akıllı Kombin Danışmanın")
    st.write(
        "Dolabındaki kıyafetleri tamamlayan renkleri, "
        "aksesuarları ve alternatif stilleri keşfet!"
    )

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

    elbiseler = [
        p for p in dolap if p["tur"] == "Elbise"
    ]

    ayakkabilar = [
        p for p in dolap
        if p["tur"] in ["Ayakkabı", "Spor ayakkabı"]
    ]

    canta_renkleri = ["Siyah", "Bej", "Beyaz", "Kahverengi"]
    takilar = ["Altın renkli takılar", "Gümüş renkli takılar"]
    ayakkabi_alternatifleri = [
        "Beyaz", "Siyah", "Kırmızı", "Bej"
    ]
    ust_alternatifleri = [
        "Beyaz", "Siyah", "Krem", "Gri", "Pembe", "Mavi"
    ]

    def kombin_aciklamasi(ust, alt):
        ust_rengi = ust["renk"]
        alt_rengi = alt["renk"]

        if ust_rengi == alt_rengi:
            renk_yorumu = (
                f"{ust_rengi} üst ve {alt_rengi} alt parça "
                "birbirine yakın, bütünlüklü bir görünüm oluşturur. "
                "Farklı dokular ve aksesuarlarla kombine hareket katabilirsin."
            )
        else:
            renk_yorumu = (
                f"{ust_rengi} üst ile {alt_rengi} alt parçanın "
                "kontrastını kullanabilirsin. Renklerden biri daha "
                "belirginse diğer parçaları sade tutmak görünümü dengeler."
            )

        if ust_rengi in ["Siyah", "Beyaz", "Gri", "Bej", "Lacivert"]:
            takı_yorumu = (
                "Altın renkli küpe, ince kolye veya bileklik "
                "kombine sıcaklık katabilir. Daha soğuk ve sade "
                "bir görünüm için gümüş renkli takıları da deneyebilirsin."
            )
        else:
            takı_yorumu = (
                "Altın renkli küçük halka küpeler veya ince bir kolye "
                "kombine sıcaklık katabilir. Daha minimal bir görünüm "
                "istersen tek bir aksesuar seçebilirsin."
            )

        if alt_rengi in ["Siyah", "Beyaz", "Gri", "Bej", "Lacivert"]:
            ayakkabi_yorumu = (
                "Kırmızı ayakkabı kombine canlı bir vurgu ekleyebilir. "
                "Daha sade bir alternatif olarak beyaz, siyah veya "
                "bej ayakkabı tercih edebilirsin."
            )
        else:
            ayakkabi_yorumu = (
                "Beyaz veya siyah ayakkabı renk dengesini koruyabilir. "
                "Daha iddialı bir görünüm için kırmızı ayakkabıyı "
                "deneyebilir, ancak diğer aksesuarları sade tutabilirsin."
            )

        canta = (
            "Bej çanta yumuşak bir geçiş sağlar; siyah çanta daha "
            "belirgin ve klasik bir görünüm verir. Beyaz çanta ise "
            "özellikle açık renkli kombinlere ferahlık katar."
        )

        alternatifler = [
            renk for renk in ust_alternatifleri
            if renk != ust_rengi
        ][:4]

        ust_onerisi = (
            f"Üst parçanın yerine {', '.join(alternatifler)} "
            "renklerinde bir tişört de deneyebilirsin. "
            "Beyaz daha ferah, siyah daha güçlü, krem daha yumuşak "
            "ve pastel tonlar daha hafif bir görünüm verir."
        )

        stil = (
            "Günlük kullanımda spor ayakkabı ve sade bir çanta; "
            "daha şık bir ortamda ise zarif takılar ve daha klasik "
            "bir ayakkabı tercih edebilirsin. Ceket eklemek kombini "
            "daha düzenli gösterir."
        )

        return (
            f"**Renk uyumu:** {renk_yorumu}\n\n"
            f"**Takı önerisi:** {takı_yorumu}\n\n"
            f"**Ayakkabı seçenekleri:** {ayakkabi_yorumu}\n\n"
            f"**Çanta önerisi:** {canta}\n\n"
            f"**Alternatif tişörtler:** {ust_onerisi}\n\n"
            f"**Stil önerisi:** {stil}\n\n"
            "**Küçük ipucu:** Bu öneriler renk uyumuna dayalı "
            "alternatiflerdir. Kumaş, desen, mevsim ve kişisel "
            "zevk de son görünümü etkiler."
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
                        st.image(
                            ust["resim"],
                            use_container_width=True
                        )
                        st.caption(
                            f"{ust['tur']} — {ust['renk']}"
                        )

                    with kolonlar[1]:
                        st.image(
                            alt["resim"],
                            use_container_width=True
                        )
                        st.caption(
                            f"{alt['tur']} — {alt['renk']}"
                        )

                    st.markdown(kombin_aciklamasi(ust, alt))

                    if ayakkabilar:
                        st.info(
                            "Dolabındaki ayakkabı: "
                            f"{ayakkabilar[0]['renk']} "
                            f"{ayakkabilar[0]['tur']}. "
                            "Bu parçayı da kombinle deneyebilirsin."
                        )
                    else:
                        st.caption(
                            "Dolabında ayakkabı fotoğrafı bulunmuyor. "
                            "Önerilen ayakkabı renkleri örnektir."
                        )

                if kombin_sayisi >= 5:
                    break

            if kombin_sayisi >= 5:
                break

    if kombin_sayisi == 0 and elbiseler:
        for elbise in elbiseler[:3]:
            st.subheader(
                f"{elbise['renk']} {elbise['tur']} için öneriler"
            )
            st.image(elbise["resim"], width=240)

            st.markdown(
                f"**Renk uyumu:** {elbise['renk']} elbiseni "
                "tamamlayacak aksesuarları elbisenin tonu ve "
                "kullanacağın ortama göre seçebilirsin.\n\n"
                "**Takı:** Altın renkli takılar sıcak bir görünüm; "
                "gümüş renkli takılar daha soğuk bir görünüm verebilir.\n\n"
                "**Ayakkabı:** Siyah, bej veya beyaz ayakkabıları "
                "deneyebilirsin. Kırmızı ayakkabı ise daha dikkat "
                "çekici bir renk vurgusu oluşturabilir.\n\n"
                "**Çanta:** Siyah veya bej çanta kolay eşleşen "
                "seçeneklerdir.\n\n"
                "**Alternatif:** Elbisenin üzerine uygun bir ceket "
                "ekleyerek görünümü günlük ya da daha şık bir tarza "
                "uyarlayabilirsin."
            )

            if ayakkabilar:
                st.write(
                    "Dolabındaki ayakkabı: "
                    f"{ayakkabilar[0]['renk']} "
                    f"{ayakkabilar[0]['tur']}"
                )

            kombin_sayisi += 1

    if kombin_sayisi == 0:
        st.info(
            "Ayrıntılı kombin önerileri için en az bir üst ve "
            "bir alt parça veya bir elbise yükleyip analiz et."
        )

st.divider()
st.caption(
    "Akıllı Dolabım | CLIP tabanlı ilk yapay zekâ prototipi"
)

