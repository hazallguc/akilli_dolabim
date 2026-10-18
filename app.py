import streamlit as st
import json
import os
import base64
import io
import random
import colorsys
import requests
import re

from PIL import Image, ImageDraw
from collections import Counter
from itertools import product

st.set_page_config(page_title="Akıllı Dolabım", page_icon="👗", layout="wide")

st.title("👗 Akıllı Dolabım")
st.caption("Yapay zekâ destekli kişisel kıyafet dolabın.")

DOSYA = "dolap.json"

ALT_TURLER = {
    "Üst giyim": ["Tişört", "Bluz", "Gömlek", "Crop top", "Sweatshirt", "Hoodie", "Kazak", "Spor üst", "Body", "Diğer üst giyim"],
    "Alt giyim": ["Jean", "Kumaş pantolon", "Eşofman altı", "Tayt", "Spor tayt", "Etek", "Şort", "Spor şort", "Diğer alt giyim"],
    "Elbise": ["Günlük elbise", "Şık elbise", "Abiye", "Spor elbise", "Diğer elbise"],
    "Dış giyim": ["Blazer ceket", "Günlük ceket", "Deri ceket", "Kot ceket", "Şişme mont", "Mont", "Kaban", "Hırka", "Spor ceket", "Diğer dış giyim"],
    "Ayakkabı": ["Spor ayakkabı", "Sneaker", "Topuklu ayakkabı", "Babet", "Loafer", "Bot", "Sandalet", "Klasik ayakkabı", "Diğer ayakkabı"],
    "Çanta": ["Günlük çanta", "Şık çanta", "Omuz çantası", "El çantası", "Sırt çantası", "Spor çanta", "Diğer çanta"],
    "Aksesuar": ["Takı", "Kemer", "Şapka", "Atkı", "Gözlük", "Diğer aksesuar"],
    "Diğer": ["Diğer"]
}

TURLER = list(ALT_TURLER.keys())

RENKLER = [
    "Siyah", "Beyaz", "Gri", "Kırmızı", "Bordo", "Pembe",
    "Pembe-Siyah", "Mavi", "Lacivert", "Yeşil", "Sarı",
    "Turuncu", "Mor", "Kahverengi", "Bej", "Krem",
    "Çok renkli", "Bilinmiyor"
]

MODEL_ETIKETLERI = {
    "Üst giyim": ["a t-shirt", "a blouse", "a shirt", "a crop top", "a sweatshirt", "a hoodie", "a sweater", "a bodysuit"],
    "Alt giyim": ["jeans", "dress pants", "sweatpants", "leggings", "a skirt", "shorts"],
    "Elbise": ["a casual dress", "an elegant dress", "an evening gown", "a sports dress"],
    "Dış giyim": ["a blazer", "a jacket", "a leather jacket", "a denim jacket", "a puffer jacket", "a coat", "a cardigan"],
    "Ayakkabı": ["sneakers", "high heels", "flats", "loafers", "boots", "sandals", "dress shoes"],
    "Çanta": ["a handbag", "a shoulder bag", "a backpack", "a sports bag"],
    "Aksesuar": ["jewelry", "a belt", "a hat", "a scarf", "glasses"],
    "Diğer": ["a clothing item", "an object"]
}


# ------------------------------
# İSİM DÜZENLEME
# ------------------------------

def isim_dosya_adi_mi(isim):
    isim = str(isim or "").strip().lower()
    return isim.endswith((".jpg", ".jpeg", ".png", ".webp", ".jfif"))


def dosya_adini_temizle(isim):
    isim = os.path.basename(str(isim or ""))
    isim = os.path.splitext(isim)[0]
    isim = isim.replace("_", " ").replace("-", " ")
    isim = re.sub(r"\s+", " ", isim).strip()
    return isim


def gorunen_kiyafet_ismi(kayit):
    isim = str(kayit.get("isim", "") or "").strip()
    alt_tur = str(kayit.get("alt_tur", "") or "").strip()
    renk = str(kayit.get("renk", "") or "").strip()

    # Eski kayıtlarda isim dosya adıysa, alt türü göster.
    if not isim or isim_dosya_adi_mi(isim):
        isim = alt_tur or kayit.get("tur", "Kıyafet")

    # İsim yalnızca "çanta" gibi genel bir dosya ismiyse,
    # mümkün olduğunda daha açıklayıcı alt türü göster.
    temiz = dosya_adini_temizle(isim).lower()
    if temiz in ["canta", "çanta", "ayakkabi", "ayakkabı", "kiyafet", "kıyafet", "resim", "image", "photo"]:
        isim = alt_tur or kayit.get("tur", "Kıyafet")

    # "Günlük Çanta" zaten renk içermiyorsa rengi de ekle.
    isim_temiz = dosya_adini_temizle(isim).lower()
    if renk and renk != "Bilinmiyor" and renk.lower() not in isim_temiz:
        isim = f"{renk} {isim}"

    return isim.strip().title()


# ------------------------------
# DOLAP KAYITLARI
# ------------------------------

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

            # Eski kayıttaki dosya adını gerçek isim alanına yazma.
            if isim_dosya_adi_mi(kayit.get("isim", "")):
                kayit["isim"] = kayit.get("alt_tur", "Kıyafet")

        return veri

    except (json.JSONDecodeError, OSError):
        st.error("dolap.json okunamadı. Dosyanın yedeğini kontrol et.")
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


def alt_tur_al(kayit):
    return kayit.get("alt_tur", kayit.get("isim", "Diğer"))


def kayitlar_tur(kayitlar, tur):
    return [k for k in kayitlar if k.get("tur") == tur]


# ------------------------------
# YAPAY ZEKÂ KIYAFET TANIMA
# ------------------------------

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
        kategoriler = []

        for kategori, liste in MODEL_ETIKETLERI.items():
            for etiket in liste:
                etiketler.append(etiket)
                kategoriler.append(kategori)

        sonuclar = model(image.convert("RGB"), candidate_labels=etiketler)
        en_iyi = sonuclar[0]
        indeks = etiketler.index(en_iyi["label"])

        return kategoriler[indeks], ALT_TURLER[kategoriler[indeks]][0], float(en_iyi["score"])

    except Exception:
        st.warning("Yapay zekâ modeli çalışmadı. Kıyafet türünü elle seçebilirsin.")
        return "Diğer", "Diğer", 0.0


# ------------------------------
# RENK TESPİTİ
# ------------------------------

def piksel_rengi(rgb):
    r, g, b = [deger / 255 for deger in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h *= 360

    if v < 0.19:
        return "Siyah"

    if s < 0.14:
        return "Beyaz" if v > 0.88 else "Gri"

    if h < 10 or h >= 350:
        return "Kırmızı"
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


@st.cache_resource(show_spinner="Kıyafet renk modeli hazırlanıyor...")
def segmentasyon_modeli_yukle():
    from transformers import pipeline
    return pipeline(
        "image-segmentation",
        model="mattmdjaga/segformer_b2_clothes"
    )


def kiyafet_maskesi_olustur(image):
    try:
        sonuc = segmentasyon_modeli_yukle()(image.convert("RGB"))
        kiyafet_etiketleri = {
            "upper-clothes", "skirt", "pants", "dress",
            "left-shoe", "right-shoe", "belt", "bag", "scarf"
        }
        maskeler = []

        for parca in sonuc:
            etiket = str(parca.get("label", "")).lower().replace("_", "-")
            maske = parca.get("mask")

            if etiket in kiyafet_etiketleri and isinstance(maske, Image.Image):
                maskeler.append(maske.convert("L").resize(image.size))

        if not maskeler:
            return None

        toplam = Image.new("L", image.size, 0)

        for maske in maskeler:
            toplam = Image.frombytes(
                "L", toplam.size,
                bytes(max(a, b) for a, b in zip(toplam.tobytes(), maske.tobytes()))
            )

        if sum(1 for p in toplam.getdata() if p > 100) < image.width * image.height * 0.01:
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

    if len(pikseller) < 50:
        sol, sag = int(image.width * 0.20), int(image.width * 0.80)
        ust, alt = int(image.height * 0.15), int(image.height * 0.85)
        pikseller = [
            image.getpixel((x, y))
            for y in range(ust, max(ust + 1, alt))
            for x in range(sol, max(sol + 1, sag))
        ]

    if len(pikseller) > 12000:
        pikseller = pikseller[::max(1, len(pikseller) // 12000)]

    sayac = Counter(piksel_rengi(p) for p in pikseller)
    toplam = sum(sayac.values())

    if toplam == 0:
        return "Bilinmiyor"

    if sayac.get("Siyah", 0) / toplam >= 0.12 and sayac.get("Pembe", 0) / toplam >= 0.12:
        return "Pembe-Siyah"

    belirgin = [
        renk for renk, adet in sayac.most_common()
        if adet / toplam >= 0.20
    ]

    if len(belirgin) >= 2:
        return "Çok renkli"

    return sayac.most_common(1)[0][0]


# ------------------------------
# HAVA DURUMU
# ------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def hava_durumu_getir(sehir):
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": sehir, "count": 1, "language": "tr", "format": "json"},
            timeout=10
        )
        geo.raise_for_status()
        sonuclar = geo.json().get("results", [])

        if not sonuclar:
            return None

        yer = sonuclar[0]

        hava = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": yer["latitude"],
                "longitude": yer["longitude"],
                "current": "temperature_2m,precipitation,rain",
                "daily": "precipitation_probability_max",
                "forecast_days": 1,
                "timezone": "auto"
            },
            timeout=10
        )
        hava.raise_for_status()
        veri = hava.json()

        mevcut = veri.get("current", {})
        gunluk = veri.get("daily", {})
        sicaklik = mevcut.get("temperature_2m")

        if sicaklik is None:
            return None

        ihtimaller = gunluk.get("precipitation_probability_max", [0])

        return {
            "sehir": yer.get("name", sehir),
            "sicaklik": float(sicaklik),
            "yagis": float(mevcut.get("precipitation", 0) or 0),
            "yagis_ihtimali": int((ihtimaller[0] or 0) if ihtimaller else 0)
        }

    except Exception:
        return None


def hava_aciklamasi(hava):
    sicaklik = hava["sicaklik"]

    if sicaklik < 5:
        aciklama = "Çok soğuk: kaban ve mont gibi sıcak tutan parçalar öncelikli."
    elif sicaklik < 12:
        aciklama = "Soğuk: mont, kaban veya kalın ceket tercih et."
    elif sicaklik < 18:
        aciklama = "Serin: hırka veya hafif ceket iyi bir seçenek."
    elif sicaklik < 25:
        aciklama = "Ilık: ince kıyafetler ve gerektiğinde hafif ceket seç."
    else:
        aciklama = "Sıcak: ince ve nefes alabilen kıyafetleri öne çıkar."

    if hava["yagis_ihtimali"] >= 50 or hava["yagis"] > 0:
        aciklama += " Yağış ihtimalini de dikkate al."

    return aciklama


# ------------------------------
# KOMBİN TERCİHLERİ
# ------------------------------

TERCIHLER = {
    "Günlük": {
        "ust": ["Tişört", "Bluz", "Sweatshirt", "Gömlek", "Hoodie", "Kazak", "Crop top", "Body", "Spor üst"],
        "alt": ["Jean", "Kumaş pantolon", "Etek", "Şort", "Eşofman altı", "Tayt", "Spor tayt", "Spor şort"],
        "ayakkabi": ["Sneaker", "Spor ayakkabı", "Babet", "Loafer", "Bot", "Sandalet"],
        "dis": ["Günlük ceket", "Kot ceket", "Deri ceket", "Hırka", "Mont", "Şişme mont", "Kaban"],
        "canta": ["Günlük çanta", "Omuz çantası", "Sırt çantası"]
    },
    "Spor": {
        "ust": ["Spor üst", "Tişört", "Sweatshirt", "Hoodie", "Crop top", "Kazak"],
        "alt": ["Spor tayt", "Tayt", "Eşofman altı", "Spor şort", "Şort", "Jean"],
        "ayakkabi": ["Spor ayakkabı", "Sneaker"],
        "dis": ["Spor ceket", "Günlük ceket", "Hırka"],
        "canta": ["Spor çanta", "Sırt çantası"]
    },
    "Şık": {
        "ust": ["Bluz", "Gömlek", "Body", "Crop top", "Kazak", "Tişört"],
        "alt": ["Kumaş pantolon", "Etek", "Jean"],
        "ayakkabi": ["Topuklu ayakkabı", "Sandalet", "Klasik ayakkabı", "Loafer", "Babet"],
        "dis": ["Blazer ceket", "Deri ceket", "Günlük ceket", "Kot ceket", "Hırka"],
        "canta": ["Şık çanta", "El çantası", "Omuz çantası", "Günlük çanta"]
    }
}


def kombin_olustur(kayitlar, tarz, hava=None, onceki_imza=None):
    tercih = TERCIHLER[tarz]

    ustler = [k for k in kayitlar if alt_tur_al(k) in tercih["ust"]]
    altlar = [k for k in kayitlar if alt_tur_al(k) in tercih["alt"]]
    elbiseler = [k for k in kayitlar if k.get("tur") == "Elbise"]
    ayakkabilar = [k for k in kayitlar if alt_tur_al(k) in tercih["ayakkabi"]]
    dis_giyim = [k for k in kayitlar if alt_tur_al(k) in tercih["dis"]]
    cantalar = [k for k in kayitlar if alt_tur_al(k) in tercih["canta"]]
    aksesuarlar = kayitlar_tur(kayitlar, "Aksesuar")

    sicaklik = hava["sicaklik"] if hava else None

    if sicaklik is not None:
        if sicaklik < 10:
            dis_giyim = [
                k for k in kayitlar
                if alt_tur_al(k) in ["Şişme mont", "Mont", "Kaban", "Hırka", "Günlük ceket", "Spor ceket", "Deri ceket"]
            ]
        elif sicaklik < 18:
            dis_giyim = [
                k for k in kayitlar
                if alt_tur_al(k) in ["Hırka", "Günlük ceket", "Kot ceket", "Deri ceket", "Blazer ceket", "Spor ceket", "Mont", "Kaban"]
            ]
        elif sicaklik >= 25:
            dis_giyim = [
                k for k in dis_giyim
                if alt_tur_al(k) not in ["Şişme mont", "Mont", "Kaban"]
            ]

    if not ustler and not elbiseler:
        return None, ["Dolabında uygun bir üst veya elbise yok."]

    if not ayakkabilar:
        return None, ["Bu tarz için uygun ayakkabı bulunamadı."]

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
        return None, ["Uygun bir kombin oluşturulamadı."]

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
        aciklamalar.append(f"Hava durumu: {hava['sehir']} — {hava['sicaklik']:.1f}°C.")
        aciklamalar.append(hava_aciklamasi(hava))

    aciklamalar.append(f"Kombin, dolabındaki parçalarla {tarz.lower()} tarzına göre hazırlandı.")
    return kombin, aciklamalar


# ------------------------------
# KOMBİN GÖRSELİ
# ------------------------------

def kolaj_olustur(kombin):
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
        cizim.text((8, 180), alt_tur_al(kayit)[:26], fill="black")
        cizim.text((8, 202), f"Renk: {kayit.get('renk', 'Bilinmiyor')}"[:28], fill="black")
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


# ------------------------------
# UYGULAMA DURUMU VE MENÜ
# ------------------------------

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
    sayfa = st.radio("Bir bölüm seç", ["Kıyafet Ekle", "Dolabım", "Kombin Öner"])

    st.divider()
    st.subheader("🌦️ Hava durumu")
    sehir = st.text_input("Şehir", value="Konya")

    if st.button("Hava durumunu getir"):
        with st.spinner("Hava durumu alınıyor..."):
            st.session_state.hava = hava_durumu_getir(sehir.strip())

    hava = st.session_state.get("hava")

    if hava:
        st.metric(f"{hava['sehir']} sıcaklığı", f"{hava['sicaklik']:.1f} °C")
        st.write(f"Yağış ihtimali: %{hava['yagis_ihtimali']}")
    else:
        st.caption("Şehir yazıp hava durumunu getir düğmesine bas.")


# ------------------------------
# KIYAFET EKLE
# ------------------------------

if sayfa == "Kıyafet Ekle":
    st.header("📸 Kıyafetini dolabına ekle")

    dosya = st.file_uploader("Kıyafet fotoğrafını yükle", type=["jpg", "jpeg", "png", "webp"])

    if dosya:
        try:
            image = Image.open(dosya).convert("RGB")
            st.image(image, caption="Yüklediğin fotoğraf", width=300)

            if st.button("Fotoğrafı analiz et", type="primary"):
                with st.spinner("Yapay zekâ kıyafeti ve rengi analiz ediyor..."):
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

        st.caption("Yanlış tahmin varsa aşağıdan düzeltebilirsin.")

        tur_index = TURLER.index(analiz["tur"]) if analiz["tur"] in TURLER else 0
        tur = st.selectbox("Kıyafet türü", TURLER, index=tur_index)

        altlar = ALT_TURLER[tur]
        alt_index = altlar.index(analiz["alt_tur"]) if analiz["alt_tur"] in altlar else 0
        alt_tur = st.selectbox("Kıyafet alt türü", altlar, index=alt_index)

        renk_index = RENKLER.index(analiz["renk"]) if analiz["renk"] in RENKLER else 0
        renk = st.selectbox("Renk", RENKLER, index=renk_index)

        varsayilan_isim = alt_tur
        isim = st.text_input("Kıyafetin adı", value=varsayilan_isim)

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


# ------------------------------
# DOLABIM - GALERİ GÖRÜNÜMÜ
# ------------------------------

elif sayfa == "Dolabım":
    st.header("🧥 Dolabım")
    kayitlar = st.session_state.dolap

    if not kayitlar:
        st.info("Dolabın henüz boş. Kıyafet Ekle bölümünden fotoğraf yükleyebilirsin.")
    else:
        st.write(f"Toplam **{len(kayitlar)}** kıyafetin var.")

        arama = st.text_input("🔎 Kıyafet ara", placeholder="Örneğin: siyah tişört")
        filtre = st.selectbox("Türüne göre filtrele", ["Tümü"] + TURLER)

        gosterilecekler = []

        for i, kayit in enumerate(kayitlar):
            if filtre != "Tümü" and kayit.get("tur") != filtre:
                continue

            aranacak = (
                kayit.get("isim", "") + " "
                + kayit.get("alt_tur", "") + " "
                + kayit.get("renk", "")
            ).lower()

            if arama.strip().lower() not in aranacak:
                continue

            gosterilecekler.append((i, kayit))

        if not gosterilecekler:
            st.warning("Bu aramaya uygun kıyafet bulunamadı.")
        else:
            sutunlar = st.columns(4)

            for sira, (i, kayit) in enumerate(gosterilecekler):
                with sutunlar[sira % 4]:
                    with st.container(border=True):
                        resim = fotografi_ac(kayit)

                        if resim:
                            st.image(resim, use_container_width=True)
                        else:
                            st.info("Fotoğraf bulunamadı.")

                        # Dosya adını değil, düzgün kıyafet adını göster.
                        st.markdown(f"**{gorunen_kiyafet_ismi(kayit)}**")
                        st.caption(
                            f"{kayit.get('renk', 'Bilinmiyor')} · "
                            f"{kayit.get('tur', 'Diğer')}"
                        )

                        with st.expander("Düzenle / Sil"):
                            yeni_isim = st.text_input(
                                "Kıyafet adı",
                                value=alt_tur_al(kayit),
                                key=f"galeri_isim_{i}"
                            )

                            mevcut_tur = kayit.get("tur", "Diğer")
                            tur_index = TURLER.index(mevcut_tur) if mevcut_tur in TURLER else 0

                            yeni_tur = st.selectbox(
                                "Tür", TURLER,
                                index=tur_index,
                                key=f"galeri_tur_{i}"
                            )

                            altlar = ALT_TURLER[yeni_tur]
                            mevcut_alt = kayit.get("alt_tur", altlar[0])
                            alt_index = altlar.index(mevcut_alt) if mevcut_alt in altlar else 0

                            yeni_alt = st.selectbox(
                                "Alt tür", altlar,
                                index=alt_index,
                                key=f"galeri_alt_{i}"
                            )

                            mevcut_renk = kayit.get("renk", "Bilinmiyor")
                            renk_index = RENKLER.index(mevcut_renk) if mevcut_renk in RENKLER else 0

                            yeni_renk = st.selectbox(
                                "Renk", RENKLER,
                                index=renk_index,
                                key=f"galeri_renk_{i}"
                            )

                            if st.button("💾 Kaydet", key=f"galeri_kaydet_{i}", use_container_width=True):
                                kayit["isim"] = yeni_isim
                                kayit["tur"] = yeni_tur
                                kayit["alt_tur"] = yeni_alt
                                kayit["renk"] = yeni_renk
                                dolabi_kaydet(kayitlar)
                                st.success("Kıyafet güncellendi.")
                                st.rerun()

                            if st.button("🗑️ Sil", key=f"galeri_sil_{i}", use_container_width=True):
                                st.session_state["silinecek"] = i

                            if st.session_state.get("silinecek") == i:
                                st.warning("Bu kıyafeti silmek istediğine emin misin?")
                                evet, vazgec = st.columns(2)

                                with evet:
                                    if st.button("Evet, sil", key=f"galeri_evet_{i}"):
                                        st.session_state.dolap.pop(i)
                                        st.session_state.pop("silinecek", None)
                                        dolabi_kaydet(st.session_state.dolap)
                                        st.rerun()

                                with vazgec:
                                    if st.button("Vazgeç", key=f"galeri_vazgec_{i}"):
                                        st.session_state.pop("silinecek", None)
                                        st.rerun()


# ------------------------------
# KOMBİN ÖNER
# ------------------------------

elif sayfa == "Kombin Öner":
    st.header("✨ Kombin önerileri")

    if not st.session_state.dolap:
        st.info("Önce Kıyafet Ekle bölümünden dolabına kıyafet ekle.")
    else:
        tarz = st.selectbox("Nasıl bir kombin istiyorsun?", ["Günlük", "Spor", "Şık"])
        hava = st.session_state.get("hava")

        if hava:
            st.info(
                f"🌡️ {hava['sehir']}: {hava['sicaklik']:.1f}°C — "
                f"Yağış ihtimali %{hava['yagis_ihtimali']}"
            )
        else:
            st.warning("Hava durumu alınmadı. Sol menüden şehir seçip hava durumunu getir.")

        if st.button("Kombinimi oluştur", type="primary"):
            kombin, aciklamalar = kombin_olustur(st.session_state.dolap, tarz, hava=hava)
            st.session_state.son_kombin = kombin
            st.session_state.kombin_aciklamalari = aciklamalar
            st.session_state.kombin_imza = tuple(sorted(id(x) for x in kombin)) if kombin else None

        kombin = st.session_state.son_kombin

        if kombin:
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
                st.write(f"- **{gorunen_kiyafet_ismi(kayit)}**")

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
                st.session_state.kombin_imza = tuple(sorted(id(x) for x in yeni_kombin)) if yeni_kombin else None
                st.rerun()

        elif "kombin_aciklamalari" in st.session_state:
            for aciklama in st.session_state.kombin_aciklamalari:
                st.warning(aciklama)
