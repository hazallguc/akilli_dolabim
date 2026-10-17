
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
    # Görselin orta bölümünü alarak kenarlardaki
    # arka plan etkisini azalt.
    resim = resim.convert("RGB")
    genislik, yukseklik = resim.size

    sol = int(genislik * 0.20)
    ust = int(yukseklik * 0.20)
    sag = int(genislik * 0.80)
    alt = int(yukseklik * 0.80)

    kirpilan = resim.crop((sol, ust, sag, alt))
    kirpilan.thumbnail((100, 100))

    pikseller = list(kirpilan.getdata())

    # Beyaza yakın, düşük renk farklılığına sahip
    # pikselleri arka plan olabilecekleri için çıkar.
    filtrelenmis = [
        (r, g, b)
        for r, g, b in pikseller
        if not (
            min(r, g, b) > 205
            and max(r, g, b) - min(r, g, b) < 25
        )
    ]

    # Görselin çoğu beyazsa boş listeyle kalma.
    if len(filtrelenmis) < len(pikseller) * 0.20:
        filtrelenmis = pikseller

    # Benzer RGB değerlerini gruplandır.
    gruplar = Counter(
        (r // 32, g // 32, b // 32)
        for r, g, b in filtrelenmis
    )

    baskin_grup = max(gruplar, key=gruplar.get)
    baskin_rgb = tuple(v * 32 + 16 for v in baskin_grup)

    # Bulunan rengi en yakın renk adına eşleştir.
    return min(
        RENKLER,
        key=lambda ad: sum(
            (RENKLER[ad][i] - baskin_rgb[i]) ** 2
            for i in range(3)
        )
    )

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

