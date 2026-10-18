
import streamlit as st
import json
import os
import base64
import io
import random
import colorsys
import requests

from PIL import Image
from collections import Counter
from itertools import product

st.set_page_config(page_title="Akıllı Dolabım", page_icon="👗", layout="wide")

st.title("👗 Akıllı Dolabım")
st.caption("Yapay zekâ destekli kıyafet dolabın ve hava durumuna uygun kombin önerilerin.")

DOSYA = "dolap.json"

ALT_TURLER = {
    "Üst giyim": [
        "Tişört", "Bluz", "Gömlek", "Crop top", "Sweatshirt",
        "Hoodie", "Kazak", "Spor üst", "Body", "Diğer üst giyim"
    ],
    "Alt giyim": [
        "Jean", "Kumaş pantolon", "Eşofman altı", "Tayt",
        "Spor tayt", "Etek", "Şort", "Spor şort", "Diğer alt giyim"
    ],
    "Elbise": [
        "Günlük elbise", "Şık elbise", "Abiye",
        "Spor elbise", "Diğer elbise"
    ],
    "Dış giyim": [
        "Blazer ceket", "Günlük ceket", "Deri ceket", "Kot ceket",
        "Şişme mont", "Mont", "Kaban", "Hırka", "Spor ceket",
        "Diğer dış giyim"
    ],
    "Ayakkabı": [
        "Spor ayakkabı", "Sneaker", "Topuklu ayakkabı", "Babet",
        "Loafer", "Bot", "Sandalet", "Klasik ayakkabı", "Diğer ayakkabı"
    ],
    "Çanta": [
        "Günlük çanta", "Şık çanta", "Omuz çantası",
        "El çantası", "Sırt çantası", "Spor çanta", "Diğer çanta"
    ],
    "Aksesuar": [
        "Takı", "Kemer", "Şapka", "Atkı", "Gözlük", "Diğer aksesuar"
    ],
    "Diğer": ["Diğer"]
}

TURLER = list(ALT_TURLER.keys())

RENKLER = [
    "Siyah", "Beyaz", "Gri", "Kırmızı", "Bordo", "Pembe",
    "Pembe-Siyah", "Mavi", "Lacivert", "Yeşil", "Sarı",
    "Turuncu", "Mor", "Kahverengi", "Bej", "Krem",
    "Çok renkli", "Bilinmiyor"
]

ALT_TUR_KATEGORI = {
    alt_tur: tur
    for tur, alt_turler in ALT_TURLER.items()
    for alt_tur in alt_turler
}

MODEL_ETIKETLERI = {
    "Üst giyim": [
        "a t-shirt", "a blouse", "a shirt", "a crop top",
        "a sweatshirt", "a hoodie", "a sweater", "a bodysuit"
    ],
    "Alt giyim": [
        "jeans", "dress pants", "sweatpants", "leggings",
        "a skirt", "shorts"
    ],
    "Elbise": [
        "a casual dress", "an elegant dress", "an evening gown",
        "a sports dress"
    ],
    "Dış giyim": [
        "a blazer", "a jacket", "a leather jacket", "a denim jacket",
        "a puffer jacket", "a coat", "a cardigan"
    ],
    "Ayakkabı": [
        "sneakers", "high heels", "flats", "loafers",
        "boots", "sandals", "dress shoes"
    ],
    "Çanta": [
        "a handbag", "a shoulder bag", "a backpack", "a sports bag"
    ],
    "Aksesuar": [
        "jewelry", "a belt", "a hat", "a scarf", "glasses"
    ],
    "Diğer": ["a clothing item", "an object"]
}


# -----------------------------
# DOSYA İŞLEMLERİ
# -----------------------------

def dolabi_yukle():
    if not os.path.exists(DOSYA):
        return []

    try:
        with open(DOSYA, "r", encoding="utf-8") as f:
            veri = json.load(f)

        if isinstance(veri, dict):
            veri = veri.get("dolap", [])

        if not isinstance(veri, list):
            return []

        for kayit in veri:
            kayit.setdefault("tur", "Diğer")
            kayit.setdefault("alt_tur", "Diğer")
            kayit.setdefault("renk", "Bilinmiyor")
            kayit.setdefault("resim", "")
            kayit.setdefault("isim", kayit.get("alt_tur", "Kıyafet"))

        return veri

    except (json.JSONDecodeError, OSError):
        st.error("dolap.json dosyası okunamadı. Dosyanın bozulmadığından emin ol.")
        return []


def dolabi_kaydet(kayitlar):
    if os.path.exists(DOSYA):
        try:
            with open(DOSYA, "r", encoding="utf-8") as f:
                eski = f.read()
            with open(DOSYA + ".bak", "w", encoding="utf-8") as f:
                f.write(eski)
        except OSError:
            pass

    with open(DOSYA, "w", encoding="utf-8") as f:
        json.dump(kayitlar, f, ensure_ascii=False, indent=2)


def fotografi_koda_cevir(image):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def fotografi_ac(kayit):
    try:
        veri = kayit.get("resim", "")
        if not veri:
            return None
        return Image.open(io.BytesIO(base64.b64decode(veri))).convert("RGB")
    except Exception:
        return None


# -----------------------------
# YAPAY ZEKÂ İLE KIYAFET TANIMA
# -----------------------------

@st.cache_resource(show_spinner="Yapay zekâ modeli yükleniyor...")
def model_yukle():
    from transformers import pipeline
    return pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32"
    )


def yapay_zeka_tahmin(image):
    try:
        model = model_yukle()

        etiketler = []
        etiket_kategorileri = []

        for kategori, ingilizce_etiketler in MODEL_ETIKETLERI.items():
            for etiket in ingilizce_etiketler:
                etiketler.append(etiket)
                etiket_kategorileri.append(kategori)

        sonuclar = model(image.convert("RGB"), candidate_labels=etiketler)
        en_iyi = sonuclar[0]

        kategori = etiket_kategorileri[etiketler.index(en_iyi["label"])]
        guven = float(en_iyi["score"])

        alt_tur = ALT_TURLER[kategori][0]

        return kategori, alt_tur, guven

    except Exception as e:
        st.warning("Kıyafet tanıma modeli çalıştırılamadı. Kategoriyi elle seçebilirsin.")
        return "Diğer", "Diğer", 0.0


# -----------------------------
# KIYAFET RENGİ TAHMİNİ
# -----------------------------

def piksel_rengi(rgb):
    r, g, b = [x / 255 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h *= 360

    # Siyah, beyaz ve griyi doygunluk üzerinden ayrı tanı
    if v < 0.19:
        return "Siyah"

    if s < 0.14:
        if v > 0.88:
            return "Beyaz"
        if v > 0.66:
            return "Gri"
        return "Gri"

    if h < 10 or h >= 350:
        return "Kırmızı" if v < 0.65 else "Kırmızı"
    if h < 20:
        return "Turuncu" if v > 0.45 else "Kahverengi"
    if h < 38:
        if v > 0.78 and s < 0.42:
            return "Bej"
        return "Kahverengi" if v < 0.62 else "Sarı"
    if h < 52:
        return "Sarı"
    if h < 78:
        return "Yeşil"
    if h < 165:
        return "Yeşil"
    if h < 205:
        return "Mavi"
    if h < 250:
        return "Mavi" if v > 0.35 else "Lacivert"
    if h < 285:
        return "Mor"
    if h < 320:
        return "Pembe" if v > 0.45 else "Mor"
    return "Pembe"


@st.cache_resource(show_spinner=False)
def segmentasyon_modeli_yukle():
    from transformers import pipeline
    return pipeline(
        "image-segmentation",
        model="mattmdjaga/segformer_b2_clothes"
    )


def kiyafet_maskesi_olustur(image):
    """
    Kıyafet segmentasyon modeliyle kıyafet piksellerini bulmaya çalışır.
    Model başarısız olursa None döndürür; renk tespitinde orta bölge kullanılır.
    """
    try:
        model = segmentasyon_modeli_yukle()
        sonuc = model(image.convert("RGB"))

        kiyafet_etiketleri = {
            "upper-clothes", "skirt", "pants", "dress",
            "left-shoe", "right-shoe", "belt", "bag", "scarf"
        }

        maskeler = []

        for parca in sonuc:
            etiket = str(parca.get("label", "")).lower().replace("_", "-")

            if etiket not in kiyafet_etiketleri:
                continue

            maske = parca.get("mask")
            if maske is None:
                continue

            if not isinstance(maske, Image.Image):
                continue

            maske = maske.convert("L").resize(image.size)
            maskeler.append(maske)

        if not maskeler:
            return None

        toplam = Image.new("L", image.size, 0)

        for maske in maskeler:
            # Her kıyafet maskesini birleştir
            toplam = Image.frombytes(
                "L",
                toplam.size,
                bytes(max(a, b) for a, b in zip(
                    toplam.tobytes(), maske.tobytes()
                ))
            )

        # Maske çok küçükse güvenilir kabul etme
        beyaz_piksel = sum(1 for p in toplam.getdata() if p > 100)
        toplam_piksel = image.width * image.height

        if beyaz_piksel < toplam_piksel * 0.01:
            return None

        return toplam

    except Exception:
        return None


def renk_tahmin_et(image):
    image = image.convert("RGB")
    image.thumbnail((250, 250))

    maske = kiyafet_maskesi_olustur(image)

    pikseller = []

    if maske is not None:
        rgb = image.load()
        mask = maske.load()

        for y in range(image.height):
            for x in range(image.width):
                if mask[x, y] > 100:
                    pikseller.append(rgb[x, y])

    # Model maske üretemezse arka plan etkisini azaltmak için
    # görüntünün orta kısmına odaklan.
    if len(pikseller) < 50:
        sol = int(image.width * 0.20)
        sag = int(image.width * 0.80)
        ust = int(image.height * 0.15)
        alt = int(image.height * 0.85)

        pikseller = [
            image.getpixel((x, y))
            for y in range(ust, max(ust + 1, alt))
            for x in range(sol, max(sol + 1, sag))
        ]

    # Çok benzer piksellerin hesabını azalt
    if len(pikseller) > 12000:
        adim = max(1, len(pikseller) // 12000)
        pikseller = pikseller[::adim]

    renk_sayaci = Counter(piksel_rengi(p) for p in pikseller)
    toplam = sum(renk_sayaci.values())

    if toplam == 0:
        return "Bilinmiyor"

    sirali = renk_sayaci.most_common()

    # Siyah ve pembe belirgin şekilde birlikteyse iki rengi göster.
    siyah_oran = renk_sayaci.get("Siyah", 0) / toplam
    pembe_oran = renk_sayaci.get("Pembe", 0) / toplam

    if siyah_oran >= 0.12 and pembe_oran >= 0.12:
        return "Pembe-Siyah"

    # İki belirgin renk varsa çok renkli olarak göster.
    belirgin = [
        (renk, adet)
        for renk, adet in sirali
        if adet / toplam >= 0.20
    ]

    if len(belirgin) >= 2:
        ilk_iki = {belirgin[0][0], belirgin[1][0]}
        if ilk_iki == {"Pembe", "Siyah"}:
            return "Pembe-Siyah"
        return "Çok renkli"

    return sirali[0][0]


# -----------------------------
# HAVA DURUMU
# -----------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def hava_durumu_getir(sehir):
    """
    Open-Meteo üzerinden şehir koordinatlarını ve güncel hava durumunu alır.
    API anahtarı gerekmez.
    """
    try:
        geocode_url = "https://geocoding-api.open-meteo.com/v1/search"

        geo = requests.get(
            geocode_url,
            params={
                "name": sehir,
                "count": 1,
                "language": "tr",
                "format": "json"
            },
            timeout=10
        )
        geo.raise_for_status()
        geo_data = geo.json()

        if not geo_data.get("results"):
            return None

        yer = geo_data["results"][0]
        enlem = yer["latitude"]
        boylam = yer["longitude"]
        bulunan_sehir = yer.get("name", sehir)

        hava = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": enlem,
                "longitude": boylam,
                "current": "temperature_2m,precipitation,rain",
                "daily": "precipitation_probability_max",
                "forecast_days": 1,
                "timezone": "auto"
            },
            timeout=10
        )
        hava.raise_for_status()
        data = hava.json()

        mevcut = data.get("current", {})
        gunluk = data.get("daily", {})

        sicaklik = mevcut.get("temperature_2m")
        yagis = mevcut.get("precipitation", 0)
        yagis_ihtimali = 0

        ihtimaller = gunluk.get("precipitation_probability_max", [])
        if ihtimaller:
            yagis_ihtimali = ihtimaller[0] or 0

        if sicaklik is None:
            return None

        return {
            "sehir": bulunan_sehir,
            "sicaklik": float(sicaklik),
            "yagis": float(yagis or 0),
            "yagis_ihtimali": int(yagis_ihtimali or 0)
        }

    except Exception:
        return None


def hava_aciklamasi(hava):
    sicaklik = hava["sicaklik"]

    if sicaklik < 5:
        katman = "Çok soğuk: kaban, mont ve sıcak tutan üstler öncelikli."
    elif sicaklik < 12:
        katman = "Soğuk: mont, kaban veya kalın ceket kullan."
    elif sicaklik < 18:
        katman = "Serin: hırka veya hafif ceket iyi bir seçenek."
    elif sicaklik < 25:
        katman = "Ilık: ince üstler ve gerektiğinde hafif bir ceket tercih et."
    else:
        katman = "Sıcak: ince ve nefes alabilen kıyafetleri öne çıkar."

    if hava["yagis_ihtimali"] >= 50 or hava["yagis"] > 0:
        katman += " Yağış ihtimali var; uygun ayakkabı ve dış katman seç."

    return katman


# -----------------------------
# KOMBİN OLUŞTURMA
# -----------------------------

TERCIHLER = {
    "Günlük": {
        "ust": [
            "Tişört", "Bluz", "Sweatshirt", "Gömlek", "Hoodie",
            "Kazak", "Crop top", "Body", "Spor üst"
        ],
        "alt": [
            "Jean", "Kumaş pantolon", "Etek", "Şort",
            "Eşofman altı", "Tayt", "Spor tayt", "Spor şort"
        ],
        "ayakkabi": [
            "Sneaker", "Spor ayakkabı", "Babet", "Loafer", "Bot", "Sandalet"
        ],
        "dis": [
            "Günlük ceket", "Kot ceket", "Deri ceket", "Hırka",
            "Mont", "Şişme mont", "Kaban"
        ],
        "canta": ["Günlük çanta", "Omuz çantası", "Sırt çantası"]
    },
    "Spor": {
        "ust": [
            "Spor üst", "Tişört", "Sweatshirt", "Hoodie", "Crop top", "Kazak"
        ],
        "alt": [
            "Spor tayt", "Tayt", "Eşofman altı", "Spor şort", "Şort", "Jean"
        ],
        "ayakkabi": ["Spor ayakkabı", "Sneaker"],
        "dis": ["Spor ceket", "Günlük ceket", "Hırka"],
        "canta": ["Spor çanta", "Sırt çantası"]
    },
    "Şık": {
        "ust": ["Bluz", "Gömlek", "Body", "Crop top", "Kazak", "Tişört"],
        "alt": ["Kumaş pantolon", "Etek", "Jean"],
        "ayakkabi": [
            "Topuklu ayakkabı", "Sandalet", "Klasik ayakkabı", "Loafer", "Babet"
        ],
        "dis": ["Blazer ceket", "Deri ceket", "Günlük ceket", "Kot ceket", "Hırka"],
        "canta": ["Şık çanta", "El çantası", "Omuz çantası", "Günlük çanta"]
    }
}


def alt_tur_al(kayit):
    return kayit.get("alt_tur", kayit.get("isim", "Diğer"))


def kayitlar_tur(kayitlar, tur):
    return [
        k for k in kayitlar
        if k.get("tur") == tur
    ]


def kombin_olustur(kayitlar, tarz, hava=None, onceki_imza=None, tur_sayisi=0):
    tercihler = TERCIHLER[tarz]

    ustler = [
        k for k in kayitlar
        if alt_tur_al(k) in tercihler["ust"]
    ]
    altlar = [
        k for k in kayitlar
        if alt_tur_al(k) in tercihler["alt"]
    ]
    elbiseler = [
        k for k in kayitlar
        if k.get("tur") == "Elbise"
    ]
    ayakkabilar = [
        k for k in kayitlar
        if alt_tur_al(k) in tercihler["ayakkabi"]
    ]
    dis_giyim = [
        k for k in kayitlar
        if alt_tur_al(k) in tercihler["dis"]
    ]
    cantalar = [
        k for k in kayitlar
        if alt_tur_al(k) in tercihler["canta"]
    ]
    aksesuarlar = kayitlar_tur(kayitlar, "Aksesuar")

    sicaklik = hava["sicaklik"] if hava else None

    # Hava durumuna göre dış giyim seçimini ayarla
    if sicaklik is not None:
        if sicaklik < 10:
            dis_giyim = [
                k for k in kayitlar
                if alt_tur_al(k) in [
                    "Şişme mont", "Mont", "Kaban", "Kazak",
                    "Hırka", "Günlük ceket", "Spor ceket", "Deri ceket"
                ]
            ]
        elif sicaklik < 18:
            dis_giyim = [
                k for k in kayitlar
                if alt_tur_al(k) in [
                    "Hırka", "Günlük ceket", "Kot ceket", "Deri ceket",
                    "Blazer ceket", "Spor ceket", "Mont", "Kaban"
                ]
            ]
        elif sicaklik >= 25:
            dis_giyim = [
                k for k in dis_giyim
                if alt_tur_al(k) not in ["Şişme mont", "Mont", "Kaban"]
            ]

            # Sıcak havada şort ve etekleri de değerlendirmeye al
            altlar = [
                k for k in kayitlar
                if alt_tur_al(k) in tercihler["alt"]
                or alt_tur_al(k) in ["Şort", "Spor şort", "Etek"]
            ]

    if not ustler and not elbiseler:
        return None, ["Dolabında bu tarza uygun bir üst veya elbise yok."]
    if not ayakkabilar:
        return None, ["Bu tarz için uygun ayakkabı bulunamadı. Dolabına ayakkabı ekleyebilirsin."]

    kombinler = []

    if ustler and altlar:
        for ust, alt, ayakkabi in product(ustler, altlar, ayakkabilar):
            kombin = [ust, alt, ayakkabi]

            if dis_giyim and sicaklik is not None and sicaklik < 18:
                kombin.append(random.choice(dis_giyim))
            elif dis_giyim and sicaklik is None and random.random() < 0.4:
                kombin.append(random.choice(dis_giyim))

            if cantalar and random.random() < 0.65:
                kombin.append(random.choice(cantalar))

            if tarz == "Şık" and aksesuarlar and random.random() < 0.5:
                kombin.append(random.choice(aksesuarlar))

            kombinler.append(kombin)

    if elbiseler:
        for elbise, ayakkabi in product(elbiseler, ayakkabilar):
            kombin = [elbise, ayakkabi]

            if dis_giyim and sicaklik is not None and sicaklik < 18:
                kombin.append(random.choice(dis_giyim))

            if cantalar and random.random() < 0.65:
                kombin.append(random.choice(cantalar))

            if tarz == "Şık" and aksesuarlar and random.random() < 0.5:
                kombin.append(random.choice(aksesuarlar))

            kombinler.append(kombin)

    if not kombinler:
        return None, ["Uygun bir kombin oluşturulamadı. Dolabındaki kıyafetleri kontrol et."]

    if onceki_imza:
        farkli = [
            k for k in kombinler
            if tuple(sorted(id(x) for x in k)) != onceki_imza
        ]
        if farkli:
            kombinler = farkli

    kombin = random.choice(kombinler)

    aciklamalar = []
    if hava:
        aciklamalar.append(
            f"Hava: {hava['sehir']} için {hava['sicaklik']:.1f}°C."
        )
        aciklamalar.append(hava_aciklamasi(hava))

    aciklamalar.append(f"Bu kombin dolabındaki kıyafetlerle {tarz.lower()} tarzına göre hazırlandı.")

    return kombin, aciklamalar


def kolaj_olustur(kombin):
    from PIL import ImageDraw

    kart_genislik = 190
    kart_yukseklik = 230
    bosluk = 12

    kartlar = []

    for kayit in kombin:
        resim = fotografi_ac(kayit)

        kart = Image.new("RGB", (kart_genislik, kart_yukseklik), "white")

        if resim:
            resim.thumbnail((kart_genislik - 16, 165))
            x = (kart_genislik - resim.width) // 2
            y = 8 + (165 - resim.height) // 2
            kart.paste(resim, (x, y))

        cizim = ImageDraw.Draw(kart)
        baslik = alt_tur_al(kayit)
        renk = kayit.get("renk", "Bilinmiyor")
        cizim.text((8, 180), baslik[:26], fill="black")
        cizim.text((8, 202), f"Renk: {renk}"[:28], fill="black")
        kartlar.append(kart)

    genislik = len(kartlar) * kart_genislik + (len(kartlar) + 1) * bosluk
    sonuc = Image.new("RGB", (genislik, kart_yukseklik + 2 * bosluk), "#eeeeee")

    for i, kart in enumerate(kartlar):
        sonuc.paste(kart, (bosluk + i * (kart_genislik + bosluk), bosluk))

    return sonuc


def resim_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# -----------------------------
# UYGULAMA ARAYÜZÜ
# -----------------------------

if "dolap" not in st.session_state:
    st.session_state.dolap = dolabi_yukle()

if "son_analiz" not in st.session_state:
    st.session_state.son_analiz = None

if "son_kombin" not in st.session_state:
    st.session_state.son_kombin = None

if "kombin_imza" not in st.session_state:
    st.session_state.kombin_imza = None

with st.sidebar:
    st.header("Menü")
    sayfa = st.radio(
        "Bir bölüm seç",
        ["Kıyafet Ekle", "Dolabım", "Kombin Öner"]
    )

    st.divider()
    st.subheader("🌦️ Hava durumu")
    sehir = st.text_input("Şehir", value="Konya")
    hava = None

    if st.button("Hava durumunu getir"):
        with st.spinner("Hava durumu alınıyor..."):
            hava = hava_durumu_getir(sehir.strip())
            st.session_state.hava = hava

    hava = st.session_state.get("hava")

    if hava:
        st.metric(
            f"{hava['sehir']} sıcaklığı",
            f"{hava['sicaklik']:.1f} °C"
        )
        st.write(f"Yağış ihtimali: %{hava['yagis_ihtimali']}")
    else:
        st.caption("Hava durumunu almak için şehir yazıp düğmeye bas.")


# -----------------------------
# KIYAFET EKLE
# -----------------------------

if sayfa == "Kıyafet Ekle":
    st.header("📸 Kıyafetini dolabına ekle")
    dosya = st.file_uploader(
        "Kıyafet fotoğrafını yükle",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if dosya:
        try:
            image = Image.open(dosya).convert("RGB")
            st.image(image, caption="Yüklediğin fotoğraf", width=300)

            if st.button("Fotoğrafı analiz et", type="primary"):
                with st.spinner("Yapay zekâ kıyafeti ve rengini analiz ediyor..."):
                    tur, alt_tur, guven = yapay_zeka_tahmin(image)
                    renk = renk_tahmin_et(image)

                    st.session_state.son_analiz = {
                        "isim": alt_tur,
                        "resim": fotografi_koda_cevir(image),
                        "tur": tur,
                        "alt_tur": alt_tur,
                        "renk": renk,
                        "guven": guven
                    }

        except Exception as e:
            st.error(f"Fotoğraf açılamadı: {e}")

    analiz = st.session_state.son_analiz

    if analiz:
        st.subheader("Analiz sonucunu kontrol et")

        if analiz.get("guven", 0) > 0:
            st.write(f"Yapay zekâ güven puanı: %{analiz['guven'] * 100:.1f}")

        st.caption("Yanlış tahmin varsa aşağıdaki alanlardan düzeltebilirsin.")

        tur_index = TURLER.index(analiz["tur"]) if analiz["tur"] in TURLER else 0

        tur = st.selectbox("Kıyafet türü", TURLER, index=tur_index)

        alt_turler = ALT_TURLER[tur]
        alt_index = (
            alt_turler.index(analiz["alt_tur"])
            if analiz["alt_tur"] in alt_turler else 0
        )
        alt_tur = st.selectbox("Kıyafet türünün ayrıntısı", alt_turler, index=alt_index)

        renk_index = RENKLER.index(analiz["renk"]) if analiz["renk"] in RENKLER else 0
        renk = st.selectbox("Renk", RENKLER, index=renk_index)

        isim = st.text_input("Kıyafetin adı", value=alt_tur)

        if st.button("Dolabıma kaydet", type="primary"):
            yeni_kayit = {
                "isim": isim.strip() or alt_tur,
                "tur": tur,
                "alt_tur": alt_tur,
                "renk": renk,
                "resim": analiz["resim"]
            }

            st.session_state.dolap.append(yeni_kayit)
            dolabi_kaydet(st.session_state.dolap)
            st.session_state.son_analiz = None
            st.success("Kıyafet dolabına kaydedildi!")
            st.rerun()


# -----------------------------
# DOLABIM
# -----------------------------

elif sayfa == "Dolabım":
    st.header("🧥 Dolabım")

    kayitlar = st.session_state.dolap

    if not kayitlar:
        st.info("Dolabın henüz boş. Kıyafet Ekle bölümünden fotoğraf yükleyebilirsin.")
    else:
        st.write(f"Toplam **{len(kayitlar)}** kıyafetin var.")

        filtre = st.selectbox("Türüne göre filtrele", ["Tümü"] + TURLER)

        for i, kayit in enumerate(kayitlar):
            if filtre != "Tümü" and kayit.get("tur") != filtre:
                continue

            with st.expander(
                f"{kayit.get('isim', 'Kıyafet')} — {kayit.get('renk', 'Bilinmiyor')}"
            ):
                col1, col2 = st.columns([1, 2])

                with col1:
                    resim = fotografi_ac(kayit)
                    if resim:
                        st.image(resim, use_container_width=True)

                with col2:
                    yeni_isim = st.text_input(
                        "Kıyafet adı",
                        value=kayit.get("isim", "Kıyafet"),
                        key=f"isim_{i}"
                    )

                    mevcut_tur = kayit.get("tur", "Diğer")
                    tur_index = TURLER.index(mevcut_tur) if mevcut_tur in TURLER else 0

                    yeni_tur = st.selectbox(
                        "Tür",
                        TURLER,
                        index=tur_index,
                        key=f"tur_{i}"
                    )

                    altlar = ALT_TURLER[yeni_tur]
                    mevcut_alt = kayit.get("alt_tur", altlar[0])
                    alt_index = altlar.index(mevcut_alt) if mevcut_alt in altlar else 0

                    yeni_alt = st.selectbox(
                        "Alt tür",
                        altlar,
                        index=alt_index,
                        key=f"alt_{i}"
                    )

                    mevcut_renk = kayit.get("renk", "Bilinmiyor")
                    renk_index = RENKLER.index(mevcut_renk) if mevcut_renk in RENKLER else 0

                    yeni_renk = st.selectbox(
                        "Renk",
                        RENKLER,
                        index=renk_index,
                        key=f"renk_{i}"
                    )

                    c1, c2 = st.columns(2)

                    with c1:
                        if st.button("Değişiklikleri kaydet", key=f"kaydet_{i}"):
                            kayit["isim"] = yeni_isim
                            kayit["tur"] = yeni_tur
                            kayit["alt_tur"] = yeni_alt
                            kayit["renk"] = yeni_renk
                            dolabi_kaydet(kayitlar)
                            st.success("Değişiklikler kaydedildi.")
                            st.rerun()

                    with c2:
                        if st.button("Kıyafeti sil", key=f"sil_{i}"):
                            st.session_state["silinecek"] = i

                    if st.session_state.get("silinecek") == i:
                        st.warning("Bu kıyafeti silmek istediğine emin misin?")
                        s1, s2 = st.columns(2)

                        with s1:
                            if st.button("Evet, sil", key=f"evet_{i}"):
                                st.session_state.dolap.pop(i)
                                st.session_state.pop("silinecek", None)
                                dolabi_kaydet(st.session_state.dolap)
                                st.rerun()

                        with s2:
                            if st.button("Vazgeç", key=f"vazgec_{i}"):
                                st.session_state.pop("silinecek", None)
                                st.rerun()


# -----------------------------
# KOMBİN ÖNER
# -----------------------------

elif sayfa == "Kombin Öner":
    st.header("✨ Kombin önerileri")

    if not st.session_state.dolap:
        st.info("Önce Kıyafet Ekle bölümünden dolabına birkaç kıyafet eklemelisin.")
    else:
        tarz = st.selectbox("Nasıl bir kombin istiyorsun?", ["Günlük", "Spor", "Şık"])

        hava = st.session_state.get("hava")

        if hava:
            st.info(
                f"🌡️ {hava['sehir']}: {hava['sicaklik']:.1f}°C — "
                f"Yağış ihtimali %{hava['yagis_ihtimali']}"
            )
        else:
            st.warning(
                "Hava durumu henüz alınmadı. Kombin önerisi hava durumu olmadan hazırlanacak. "
                "Sol menüden şehir seçip 'Hava durumunu getir' düğmesine basabilirsin."
            )

        if st.button("Kombinimi oluştur", type="primary"):
            kombin, aciklamalar = kombin_olustur(
                st.session_state.dolap,
                tarz,
                hava=hava
            )
            st.session_state.son_kombin = kombin
            st.session_state.kombin_aciklamalari = aciklamalar
            st.session_state.kombin_imza = (
                tuple(sorted(id(x) for x in kombin)) if kombin else None
            )

        if st.session_state.son_kombin:
            kombin = st.session_state.son_kombin

            st.subheader("Senin kombin önerin")
            kolaj = kolaj_olustur(kombin)
            st.image(kolaj, use_container_width=True)

            st.download_button(
                "Kombin görselini indir",
                data=resim_bytes(kolaj),
                file_name="akilli_dolabim_kombin.png",
                mime="image/png"
            )

            st.subheader("Kombindeki parçalar")

            for kayit in kombin:
                st.write(
                    f"- **{alt_tur_al(kayit)}** — "
                    f"{kayit.get('renk', 'Bilinmiyor')}"
                )

            for aciklama in st.session_state.get("kombin_aciklamalari", []):
                st.caption(aciklama)

            if st.button("Aynı tarzda başka kombin dene"):
                yeni_kombin, aciklamalar = kombin_olustur(
                    st.session_state.dolap,
                    tarz,
                    hava=hava,
                    onceki_imza=st.session_state.kombin_imza
                )

                st.session_state.son_kombin = yeni_kombin
                st.session_state.kombin_aciklamalari = aciklamalar
                st.session_state.kombin_imza = (
                    tuple(sorted(id(x) for x in yeni_kombin))
                    if yeni_kombin else None
                )
                st.rerun()

        elif "kombin_aciklamalari" in st.session_state:
            for aciklama in st.session_state.kombin_aciklamalari:
                st.warning(aciklama)
