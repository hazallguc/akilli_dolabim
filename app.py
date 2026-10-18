
import streamlit as st
import json
import os
import base64
import io
import colorsys
import random
from datetime import datetime
from collections import Counter
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np

st.set_page_config(
    page_title="Akıllı Dolabım",
    page_icon="👗",
    layout="wide"
)

DOSYA = "dolap.json"
YEDEK = "dolap.json.bak"

TURLER = [
    "Üst giyim", "Alt giyim", "Elbise", "Dış giyim",
    "Ayakkabı", "Çanta", "Aksesuar", "Diğer"
]

ALT_TURLER = {
    "Üst giyim": [
        "Tişört", "Bluz", "Gömlek", "Crop top",
        "Sweatshirt", "Hoodie", "Kazak", "Spor üst",
        "Body", "Diğer üst"
    ],
    "Alt giyim": [
        "Kot pantolon", "Kumaş pantolon", "Eşofman altı",
        "Tayt", "Etek", "Şort", "Spor tayt",
        "Spor şort", "Diğer alt"
    ],
    "Elbise": [
        "Günlük elbise", "Şık elbise", "Abiye",
        "Spor elbise", "Diğer elbise"
    ],
    "Dış giyim": [
        "Blazer ceket", "Günlük ceket", "Deri ceket",
        "Kot ceket", "Hırka", "Mont", "Kaban",
        "Spor ceket", "Diğer dış giyim"
    ],
    "Ayakkabı": [
        "Spor ayakkabı", "Sneaker", "Topuklu ayakkabı",
        "Babet", "Loafer", "Bot", "Sandalet",
        "Klasik ayakkabı", "Diğer ayakkabı"
    ],
    "Çanta": [
        "Günlük çanta", "Şık çanta", "Omuz çantası",
        "El çantası", "Sırt çantası", "Spor çanta",
        "Diğer çanta"
    ],
    "Aksesuar": [
        "Takı", "Kemer", "Şapka", "Atkı",
        "Gözlük", "Diğer aksesuar"
    ],
    "Diğer": ["Diğer"]
}

RENKLER = [
    "Siyah", "Beyaz", "Gri", "Kırmızı", "Pembe", "Mavi",
    "Lacivert", "Yeşil", "Sarı", "Turuncu", "Mor",
    "Kahverengi", "Bej", "Pembe-Siyah", "Siyah-Beyaz",
    "Çok renkli", "Bilinmiyor"
]


# ---------------------------------------------------------
# KAYITLAR
# ---------------------------------------------------------

def dolabi_yukle():
    if not os.path.exists(DOSYA):
        return []

    try:
        with open(DOSYA, "r", encoding="utf-8") as f:
            veri = json.load(f)
        return veri if isinstance(veri, list) else []
    except Exception:
        st.error(
            "dolap.json okunamadı. Dosyanı silme; "
            "önce içeriğini ve yedeğini kontrol et."
        )
        return []


def dolabi_kaydet(veri):
    try:
        if os.path.exists(DOSYA) and not os.path.exists(YEDEK):
            with open(DOSYA, "rb") as kaynak:
                with open(YEDEK, "wb") as yedek:
                    yedek.write(kaynak.read())

        gecici = DOSYA + ".tmp"
        with open(gecici, "w", encoding="utf-8") as f:
            json.dump(veri, f, ensure_ascii=False, indent=2)

        os.replace(gecici, DOSYA)
        return True
    except Exception as e:
        st.error(f"Kaydetme hatası: {e}")
        return False


def tur_al(kayit):
    tur = kayit.get("tur", "Diğer")
    return tur if tur in TURLER else "Diğer"


def alt_tur_al(kayit):
    alt = kayit.get("alt_tur", "")
    tur = tur_al(kayit)
    if alt in ALT_TURLER.get(tur, []):
        return alt
    return ""


def renk_al(kayit):
    return str(kayit.get("renk", "Bilinmiyor") or "Bilinmiyor")


def dosya_adi(kayit):
    return str(kayit.get("dosya_adi", "")).lower()


def fotograf_ac(kayit):
    try:
        veri = base64.b64decode(kayit["resim"])
        return Image.open(io.BytesIO(veri)).convert("RGB")
    except Exception:
        return None


def fotograf_base64(fotograf):
    tampon = io.BytesIO()
    fotograf.save(tampon, format="JPEG", quality=90)
    return base64.b64encode(tampon.getvalue()).decode("utf-8")


# ---------------------------------------------------------
# DAHA AYRINTILI KIYAFET ANALİZİ
# ---------------------------------------------------------

# Model genel kategoriyi ve mümkün olduğunda kıyafetin alt türünü tahmin eder.
# İngilizce açıklamalar CLIP modeli için kullanılır.
MODEL_ETIKETLERI = [
    ("Üst giyim", "Tişört", "a photo of a t-shirt"),
    ("Üst giyim", "Bluz", "a photo of a women's blouse"),
    ("Üst giyim", "Gömlek", "a photo of a button-up shirt"),
    ("Üst giyim", "Crop top", "a photo of a crop top"),
    ("Üst giyim", "Sweatshirt", "a photo of a sweatshirt"),
    ("Üst giyim", "Hoodie", "a photo of a hoodie"),
    ("Üst giyim", "Kazak", "a photo of a knitted sweater"),
    ("Üst giyim", "Spor üst", "a photo of a sports top"),
    ("Üst giyim", "Body", "a photo of a bodysuit"),

    ("Alt giyim", "Kot pantolon", "a photo of jeans"),
    ("Alt giyim", "Kumaş pantolon", "a photo of dress trousers"),
    ("Alt giyim", "Eşofman altı", "a photo of sweatpants"),
    ("Alt giyim", "Tayt", "a photo of leggings"),
    ("Alt giyim", "Spor tayt", "a photo of sports leggings"),
    ("Alt giyim", "Etek", "a photo of a skirt"),
    ("Alt giyim", "Şort", "a photo of shorts"),
    ("Alt giyim", "Spor şort", "a photo of sports shorts"),

    ("Elbise", "Günlük elbise", "a photo of a casual dress"),
    ("Elbise", "Şık elbise", "a photo of an elegant dress"),
    ("Elbise", "Abiye", "a photo of an evening gown"),
    ("Elbise", "Spor elbise", "a photo of a sporty dress"),

    ("Dış giyim", "Blazer ceket", "a photo of a blazer"),
    ("Dış giyim", "Günlük ceket", "a photo of a casual jacket"),
    ("Dış giyim", "Deri ceket", "a photo of a leather jacket"),
    ("Dış giyim", "Kot ceket", "a photo of a denim jacket"),
    ("Dış giyim", "Hırka", "a photo of a cardigan"),
    ("Dış giyim", "Mont", "a photo of a winter coat"),
    ("Dış giyim", "Kaban", "a photo of a long wool coat"),
    ("Dış giyim", "Spor ceket", "a photo of a sports jacket"),

    ("Ayakkabı", "Spor ayakkabı", "a photo of athletic running shoes"),
    ("Ayakkabı", "Sneaker", "a photo of casual sneakers"),
    ("Ayakkabı", "Topuklu ayakkabı", "a photo of high heels"),
    ("Ayakkabı", "Babet", "a photo of ballet flats"),
    ("Ayakkabı", "Loafer", "a photo of loafers"),
    ("Ayakkabı", "Bot", "a photo of boots"),
    ("Ayakkabı", "Sandalet", "a photo of sandals"),
    ("Ayakkabı", "Klasik ayakkabı", "a photo of formal shoes"),

    ("Çanta", "Günlük çanta", "a photo of a casual handbag"),
    ("Çanta", "Şık çanta", "a photo of an elegant evening handbag"),
    ("Çanta", "Omuz çantası", "a photo of a shoulder bag"),
    ("Çanta", "El çantası", "a photo of a handbag"),
    ("Çanta", "Sırt çantası", "a photo of a backpack"),
    ("Çanta", "Spor çanta", "a photo of a gym bag"),

    ("Aksesuar", "Takı", "a photo of jewelry"),
    ("Aksesuar", "Kemer", "a photo of a belt"),
    ("Aksesuar", "Şapka", "a photo of a hat"),
    ("Aksesuar", "Atkı", "a photo of a scarf"),
    ("Aksesuar", "Gözlük", "a photo of fashion glasses"),
]


@st.cache_resource(show_spinner=False)
def clip_modeli():
    try:
        from transformers import pipeline
        return pipeline(
            "zero-shot-image-classification",
            model="openai/clip-vit-base-patch32"
        )
    except Exception:
        return None


def tur_tahmin_et(fotograf):
    model = clip_modeli()

    if model is None:
        return None, None

    try:
        # Modelin arka plan yerine kıyafete odaklanmasına yardımcı olmak için
        # görselin merkez bölgesini de analiz eder.
        resim = ImageOps.exif_transpose(fotograf).convert("RGB")
        w, h = resim.size

        merkez = resim.crop((
            int(w * 0.08),
            int(h * 0.08),
            max(int(w * 0.92), 1),
            max(int(h * 0.92), 1)
        ))

        adaylar = [x[2] for x in MODEL_ETIKETLERI]
        tahminler = model(merkez, candidate_labels=adaylar)

        if not tahminler:
            return None, None

        bulunan = tahminler[0]["label"]

        for tur, alt_tur, etiket in MODEL_ETIKETLERI:
            if etiket == bulunan:
                return tur, alt_tur

    except Exception:
        pass

    return None, None


# ---------------------------------------------------------
# RENK TAHMİNİ
# ---------------------------------------------------------

def piksel_rengi(rgb):
    r, g, b = [int(x) for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)

    if v < 0.19:
        return "Siyah"

    if s < 0.15:
        return "Beyaz" if v > 0.85 else "Gri"

    derece = h * 360

    if derece < 12 or derece >= 345:
        return "Kırmızı"
    if derece < 25:
        return "Turuncu"
    if derece < 43:
        return "Sarı"
    if derece < 80:
        return "Yeşil"
    if derece < 165:
        return "Yeşil"
    if derece < 205:
        return "Mavi"
    if derece < 255:
        return "Lacivert" if v < 0.55 else "Mavi"
    if derece < 290:
        return "Mor"
    if derece < 330:
        return "Pembe"

    return "Kırmızı"


def renk_tahmin_et(fotograf):
    try:
        resim = ImageOps.exif_transpose(fotograf).convert("RGB")
        w, h = resim.size

        # Kenarlardaki arka plan etkisini azalt.
        resim = resim.crop((
            int(w * 0.20),
            int(h * 0.15),
            max(int(w * 0.80), 1),
            max(int(h * 0.85), 1)
        ))

        resim.thumbnail((100, 100))
        pikseller = np.asarray(resim).reshape(-1, 3)

        if len(pikseller) > 3000:
            pikseller = pikseller[::max(1, len(pikseller) // 3000)]

        sayim = Counter(piksel_rengi(p) for p in pikseller)
        sirali = sayim.most_common()

        if not sirali:
            return "Bilinmiyor"

        toplam = sum(sayim.values())
        renk1, adet1 = sirali[0]
        oran1 = adet1 / toplam

        renk2, adet2 = sirali[1] if len(sirali) > 1 else ("", 0)
        oran2 = adet2 / toplam

        if {renk1, renk2} == {"Pembe", "Siyah"} and oran2 >= 0.15:
            return "Pembe-Siyah"

        if {renk1, renk2} == {"Siyah", "Beyaz"} and oran2 >= 0.20:
            return "Siyah-Beyaz"

        if oran2 >= 0.25 and renk1 != renk2:
            birlesik = f"{renk1}-{renk2}"
            if birlesik in RENKLER:
                return birlesik
            return "Çok renkli"

        if oran1 < 0.20:
            return "Çok renkli"

        return renk1

    except Exception:
        return "Bilinmiyor"


# ---------------------------------------------------------
# TARZ PUANLAMA
# ---------------------------------------------------------

NÖTR = {
    "Siyah", "Beyaz", "Gri", "Bej",
    "Kahverengi", "Lacivert"
}


def renk_listesi(renk):
    if not renk or renk == "Bilinmiyor":
        return []

    return [
        x.strip()
        for x in renk.replace("/", "-").split("-")
    ]


def renk_uyum_puani(renk1, renk2):
    r1 = renk_listesi(renk1)
    r2 = renk_listesi(renk2)

    if not r1 or not r2:
        return 1

    if set(r1) & set(r2):
        return 3

    if any(x in NÖTR for x in r1 + r2):
        return 2

    sicak = {"Kırmızı", "Pembe", "Turuncu", "Sarı"}
    soguk = {"Mavi", "Lacivert", "Yeşil", "Mor"}

    if (
        (set(r1) & sicak and set(r2) & soguk)
        or (set(r2) & sicak and set(r1) & soguk)
    ):
        return 1

    return 0


def tarz_puani(kayit, tarz):
    tur = tur_al(kayit)
    alt = alt_tur_al(kayit).lower()
    ad = dosya_adi(kayit)
    metin = alt + " " + ad
    puan = 0

    if tarz == "Günlük":
        if tur in ["Üst giyim", "Alt giyim"]:
            puan += 3
        if tur == "Ayakkabı":
            puan += 2

        if any(k in metin for k in [
            "günlük", "gunluk", "kot", "jean", "tişört",
            "tisort", "sneaker", "babet", "basic", "casual"
        ]):
            puan += 5

        if any(k in metin for k in [
            "abiye", "topuklu", "eşofman", "esofman"
        ]):
            puan -= 3

    elif tarz == "Spor":
        if alt in ["tayt", "spor tayt", "eşofman altı", "spor şort"]:
            puan += 12
        if alt in ["spor üst", "sweatshirt", "hoodie"]:
            puan += 8
        if alt in ["spor ayakkabı", "sneaker", "spor çanta", "spor ceket"]:
            puan += 12

        if tur in ["Üst giyim", "Alt giyim"]:
            puan += 2

        if any(k in metin for k in [
            "spor", "sneaker", "eşofman", "esofman",
            "tayt", "koşu", "kosu", "antrenman",
            "hoodie", "gym", "running"
        ]):
            puan += 6

        if alt in ["topuklu ayakkabı", "şık elbise", "abiye", "blazer ceket"]:
            puan -= 12

        if any(k in metin for k in ["topuklu", "abiye", "klasik"]):
            puan -= 5

    elif tarz == "Şık":
        if alt in ["şık elbise", "abiye"]:
            puan += 18
        elif alt in ["topuklu ayakkabı", "şık çanta", "blazer ceket"]:
            puan += 14
        elif alt in ["kumaş pantolon", "gömlek", "bluz", "etek", "loafer"]:
            puan += 8
        elif alt in ["kot pantolon", "eşofman altı", "spor tayt"]:
            puan -= 8

        if tur == "Elbise":
            puan += 5
        elif tur == "Dış giyim":
            puan += 3
        elif tur in ["Çanta", "Ayakkabı"]:
            puan += 2

        if any(k in metin for k in [
            "blazer", "topuklu", "gömlek", "gomlek",
            "klasik", "loafer", "abiye", "şık", "sik",
            "etek", "kumaş", "kumas", "elegant", "formal"
        ]):
            puan += 6

        if any(k in metin for k in [
            "eşofman", "esofman", "koşu", "kosu", "gym"
        ]):
            puan -= 8

    return puan


def temel_kombin_puani(temel, tarz):
    turler = [tur_al(p) for p in temel]
    altlar = [alt_tur_al(p).lower() for p in temel]

    puan = sum(tarz_puani(p, tarz) for p in temel)

    renkler = [renk_al(p) for p in temel]
    for i in range(len(renkler)):
        for j in range(i + 1, len(renkler)):
            puan += renk_uyum_puani(renkler[i], renkler[j])

    if "Üst giyim" in turler and "Alt giyim" in turler:
        puan += 8

    if tarz == "Şık":
        if "Elbise" in turler:
            puan += 18
        if any(x in ["şık elbise", "abiye"] for x in altlar):
            puan += 10

    elif tarz == "Spor":
        if "Üst giyim" in turler and "Alt giyim" in turler:
            puan += 10
        if any(x in ["tayt", "spor tayt", "eşofman altı"] for x in altlar):
            puan += 14
        if "Elbise" in turler:
            puan -= 12

    elif tarz == "Günlük":
        if "Üst giyim" in turler and "Alt giyim" in turler:
            puan += 10

    return puan


def en_iyi_aday(adaylar, puan_fonksiyonu):
    if not adaylar:
        return None

    puanli = [(puan_fonksiyonu(a), a) for a in adaylar]
    en_yuksek = max(p for p, _ in puanli)

    # Birbirine yakın puanlı adaylar arasından seçim yaparak
    # her defasında aynı fotoğrafın seçilmesini azalt.
    en_iyiler = [
        a for p, a in puanli
        if p >= en_yuksek - 2
    ]

    return random.choice(en_iyiler)


# ---------------------------------------------------------
# KOMBİN OLUŞTURMA
# ---------------------------------------------------------

def kombin_olustur(dolap, tarz):
    gecerli = [
        p for p in dolap
        if isinstance(p, dict) and fotograf_ac(p) is not None
    ]

    ustler = [p for p in gecerli if tur_al(p) == "Üst giyim"]
    altlar = [p for p in gecerli if tur_al(p) == "Alt giyim"]
    elbiseler = [p for p in gecerli if tur_al(p) == "Elbise"]

    dislar = [p for p in gecerli if tur_al(p) == "Dış giyim"]
    ayakkabilar = [p for p in gecerli if tur_al(p) == "Ayakkabı"]
    cantalar = [p for p in gecerli if tur_al(p) == "Çanta"]
    aksesuarlar = [p for p in gecerli if tur_al(p) == "Aksesuar"]

    temel_adaylari = []

    # Üst-alt seçenekleri oluştur.
    for ust in ustler:
        for alt in altlar:
            temel_adaylari.append([ust, alt])

    # Elbiseyi özellikle şık tarzda güçlü bir alternatif yap.
    # Diğer tarzlarda da dolapta üst-alt yoksa kullan.
    if tarz == "Şık" or not (ustler and altlar):
        for elbise in elbiseler:
            temel_adaylari.append([elbise])

    if not temel_adaylari:
        return [], (
            "Tam kombin oluşturabilmem için dolabında üst giyim + "
            "alt giyim veya elbise bulunmalı. Dolabım bölümünden "
            "kıyafet türlerini kontrol edebilirsin."
        )

    temel = en_iyi_aday(
        temel_adaylari,
        lambda aday: temel_kombin_puani(aday, tarz)
    )

    sonuc = list(temel)

    # Tarza uygun tamamlayıcıları önceliklendir.
    # Her kategoriden en fazla bir parça eklenir.
    gruplar = [
        dislar,
        ayakkabilar,
        cantalar,
        aksesuarlar
    ]

    for grup in gruplar:
        adaylar = [p for p in grup if p not in sonuc]

        if not adaylar:
            continue

        def tamamlayici_puani(parca):
            puan = tarz_puani(parca, tarz)

            for mevcut in sonuc:
                puan += renk_uyum_puani(
                    renk_al(parca),
                    renk_al(mevcut)
                )

            return puan

        secilen = en_iyi_aday(adaylar, tamamlayici_puani)

        if secilen is not None:
            sonuc.append(secilen)

    return sonuc, aciklama_olustur(sonuc, tarz, gecerli)


def aciklama_olustur(kombin, tarz, dolap):
    turler = [tur_al(p) for p in kombin]
    altlar = [alt_tur_al(p) for p in kombin]

    renkler = list(dict.fromkeys(
        renk_al(p)
        for p in kombin
        if renk_al(p) != "Bilinmiyor"
    ))

    if tarz == "Şık":
        metin = "**Seçilen tarz: Şık.** Şık parçaları önceliklendirdim."
    elif tarz == "Spor":
        metin = "**Seçilen tarz: Spor.** Spor parçalarını önceliklendirdim."
    else:
        metin = "**Seçilen tarz: Günlük.** Günlük kullanıma uygun parçaları önceliklendirdim."

    if "Üst giyim" in turler and "Alt giyim" in turler:
        metin += " Üst ve alt giyim birlikte seçildi."
    elif "Elbise" in turler:
        metin += " Elbise kombinin ana parçası olarak seçildi."

    secili_altlar = [x for x in altlar if x]
    if secili_altlar:
        metin += " Seçilen alt türler: " + ", ".join(secili_altlar) + "."

    if renkler:
        metin += " Renkler: " + ", ".join(renkler) + "."

    eksikler = []

    if not any(tur_al(p) == "Dış giyim" for p in kombin):
        if any(tur_al(p) == "Dış giyim" for p in dolap):
            eksikler.append("dış giyim seçilmedi")
        else:
            eksikler.append("ceket veya hırka")

    if not any(tur_al(p) == "Ayakkabı" for p in kombin):
        eksikler.append("ayakkabı")

    if not any(tur_al(p) == "Çanta" for p in kombin):
        eksikler.append("çanta")

    if not any(tur_al(p) == "Aksesuar" for p in kombin):
        eksikler.append("takı veya kemer")

    if eksikler:
        metin += (
            "\n\n**Eksik tamamlayıcılar:** "
            + ", ".join(eksikler)
            + ". Bunlar kolajda varmış gibi gösterilmedi."
        )

    # Seçilen tarzda özel bir alt tür yoksa kullanıcıyı bilgilendir.
    if tarz == "Şık":
        if not any(
            alt_tur_al(p) == "Topuklu ayakkabı"
            for p in dolap
        ) and any(tur_al(p) == "Ayakkabı" for p in dolap):
            metin += (
                "\n\nDolabında topuklu ayakkabı kayıtlı değil; "
                "mevcut ayakkabılardan biri seçildi."
            )

    if tarz == "Spor":
        if not any(
            alt_tur_al(p) in ["Tayt", "Spor tayt", "Eşofman altı"]
            for p in dolap
        ) and any(tur_al(p) == "Alt giyim" for p in dolap):
            metin += (
                "\n\nDolabında tayt veya eşofman altı olarak "
                "işaretlenmiş bir kayıt bulunmuyor. Alt giyim "
                "seçimini Dolabım bölümünden kontrol edebilirsin."
            )

    return metin


# ---------------------------------------------------------
# KOLAJ
# ---------------------------------------------------------

def kolaj_olustur(kombin, boyut=1000):
    tuval = Image.new("RGB", (boyut, boyut), "#f5f1ed")
    cizim = ImageDraw.Draw(tuval)

    try:
        baslik_font = ImageFont.truetype("arial.ttf", 32)
        etiket_font = ImageFont.truetype("arial.ttf", 19)
    except Exception:
        baslik_font = ImageFont.load_default()
        etiket_font = ImageFont.load_default()

    cizim.text(
        (boyut // 2, 25),
        "AKILLI DOLABIM",
        fill="#333333",
        font=baslik_font,
        anchor="mt"
    )

    if not kombin:
        return tuval

    kenar = 30
    ust_bosluk = 85
    sutun_sayisi = 2
    satir_sayisi = (len(kombin) + 1) // 2

    hucre_w = (boyut - kenar * 2) // sutun_sayisi
    hucre_h = (
        boyut - ust_bosluk - kenar
    ) // max(1, satir_sayisi)

    for i, parca in enumerate(kombin):
        fotograf = fotograf_ac(parca)
        if fotograf is None:
            continue

        satir = i // 2
        sutun = i % 2

        x = kenar + sutun * hucre_w
        y = ust_bosluk + satir * hucre_h

        cizim.rounded_rectangle(
            (
                x + 8, y + 8,
                x + hucre_w - 8,
                y + hucre_h - 8
            ),
            radius=16,
            fill="white",
            outline="#e2dcd6",
            width=2
        )

        alan_w = hucre_w - 40
        alan_h = hucre_h - 70

        fotograf.thumbnail(
            (alan_w, alan_h),
            Image.Resampling.LANCZOS
        )

        px = x + (hucre_w - fotograf.width) // 2
        py = y + 15 + (alan_h - fotograf.height) // 2

        tuval.paste(fotograf, (px, py))

        alt = alt_tur_al(parca)
        etiket_turu = f"{tur_al(parca)} ({alt})" if alt else tur_al(parca)
        etiket = f"{etiket_turu} • {renk_al(parca)}"

        cizim.text(
            (x + hucre_w // 2, y + hucre_h - 28),
            etiket,
            fill="#333333",
            font=etiket_font,
            anchor="mm"
        )

    return tuval


# ---------------------------------------------------------
# ARAYÜZ
# ---------------------------------------------------------

st.title("👗 Akıllı Dolabım")
st.caption("Kıyafetlerini düzenle, tarzını seç ve kombinini oluştur.")

dolap = dolabi_yukle()

sayfa = st.sidebar.radio(
    "Menü",
    ["Kıyafet Ekle", "Dolabım", "Kombin Öner"]
)


# ---------------------------------------------------------
# KIYAFET EKLE
# ---------------------------------------------------------

if sayfa == "Kıyafet Ekle":
    st.header("📸 Kıyafet ekle")

    yuklenen = st.file_uploader(
        "Kıyafet fotoğrafını seç",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if yuklenen:
        try:
            fotograf = ImageOps.exif_transpose(
                Image.open(yuklenen)
            ).convert("RGB")

            st.image(fotograf, width=300)

            if st.button("🤖 Kıyafeti analiz et", type="primary"):
                with st.spinner(
                    "Kıyafetin türü ve rengi analiz ediliyor..."
                ):
                    tahmin_turu, tahmin_alt = tur_tahmin_et(fotograf)
                    tahmin_renk = renk_tahmin_et(fotograf)

                st.session_state["gecici_resim"] = fotograf_base64(fotograf)
                st.session_state["gecici_dosya"] = yuklenen.name
                st.session_state["gecici_tur"] = tahmin_turu or "Üst giyim"
                st.session_state["gecici_alt"] = tahmin_alt or ""
                st.session_state["gecici_renk"] = tahmin_renk
                st.session_state["analiz_ok"] = True

                # Önceki analizden kalmış seçimleri yenile.
                st.session_state["ekle_tur"] = tahmin_turu or "Üst giyim"
                st.session_state["ekle_alt"] = tahmin_alt or ""
                st.session_state["ekle_renk"] = tahmin_renk

            if st.session_state.get("analiz_ok"):
                st.info(
                    "Yapay zekâ tahmin yaptı. Kaydetmeden önce "
                    "tür ve renk alanlarını kontrol edebilirsin."
                )

                varsayilan_tur = st.session_state.get(
                    "gecici_tur", "Üst giyim"
                )

                tur = st.selectbox(
                    "Kıyafet türü",
                    TURLER,
                    key="ekle_tur",
                    index=(
                        TURLER.index(varsayilan_tur)
                        if varsayilan_tur in TURLER else 0
                    )
                )

                alt_secenekler = ALT_TURLER[tur]
                tahmin_alt = st.session_state.get("gecici_alt", "")

                if st.session_state.get("ekle_alt") not in alt_secenekler:
                    st.session_state["ekle_alt"] = (
                        tahmin_alt if tahmin_alt in alt_secenekler
                        else alt_secenekler[0]
                    )

                alt_tur = st.selectbox(
                    "Kıyafetin daha ayrıntılı türü",
                    alt_secenekler,
                    key="ekle_alt"
                )

                tahmin_renk = st.session_state.get(
                    "gecici_renk", "Bilinmiyor"
                )

                renk_secenekleri = list(RENKLER)
                if tahmin_renk not in renk_secenekleri:
                    renk_secenekleri.insert(0, tahmin_renk)

                if st.session_state.get("ekle_renk") not in renk_secenekleri:
                    st.session_state["ekle_renk"] = tahmin_renk

                renk = st.selectbox(
                    "Renk",
                    renk_secenekleri,
                    key="ekle_renk"
                )

                if st.button("➕ Dolabıma kaydet", type="primary"):
                    dolap.append({
                        "dosya_adi": st.session_state.get(
                            "gecici_dosya", yuklenen.name
                        ),
                        "tur": tur,
                        "alt_tur": alt_tur,
                        "renk": renk,
                        "resim": st.session_state["gecici_resim"],
                        "eklenme_tarihi": datetime.now().isoformat(
                            timespec="seconds"
                        )
                    })

                    if dolabi_kaydet(dolap):
                        st.session_state["analiz_ok"] = False
                        for anahtar in [
                            "gecici_resim", "gecici_dosya",
                            "gecici_tur", "gecici_alt", "gecici_renk"
                        ]:
                            st.session_state.pop(anahtar, None)

                        st.success("Kıyafet kaydedildi!")
                        st.rerun()

        except Exception as e:
            st.error(f"Fotoğraf işlenemedi: {e}")


# ---------------------------------------------------------
# DOLABIM
# ---------------------------------------------------------

elif sayfa == "Dolabım":
    st.header("🧥 Dolabım")

    if not dolap:
        st.info("Dolabın boş. Kıyafet Ekle bölümünden başlayabilirsin.")
    else:
        st.write(f"Toplam kayıt: **{len(dolap)}**")
        st.caption(
            "Eski kıyafetlerin korunur. Alt tür alanı, kayıtlarını "
            "düzenledikçe eklenir."
        )

        filtre = st.selectbox("Tür filtresi", ["Tümü"] + TURLER)

        for i, parca in enumerate(dolap):
            if not isinstance(parca, dict):
                continue

            tur = tur_al(parca)

            if filtre != "Tümü" and tur != filtre:
                continue

            with st.container(border=True):
                sol, sag = st.columns([1, 2])

                with sol:
                    fotograf = fotograf_ac(parca)
                    if fotograf:
                        st.image(fotograf, use_container_width=True)

                with sag:
                    st.write(
                        f"**Dosya:** {parca.get('dosya_adi', 'Kıyafet')}"
                    )

                    eski_tur = tur_al(parca)
                    yeni_tur = st.selectbox(
                        "Tür",
                        TURLER,
                        index=TURLER.index(eski_tur),
                        key=f"tur_{i}"
                    )

                    alt_secenekler = ALT_TURLER[yeni_tur]
                    eski_alt = alt_tur_al(parca)

                    alt_key = f"alt_{i}_{yeni_tur}"
                    if st.session_state.get(alt_key) not in alt_secenekler:
                        st.session_state[alt_key] = (
                            eski_alt if eski_alt in alt_secenekler
                            else alt_secenekler[0]
                        )

                    yeni_alt = st.selectbox(
                        "Ayrıntılı tür",
                        alt_secenekler,
                        key=alt_key
                    )

                    eski_renk = renk_al(parca)
                    renk_secenekleri = list(RENKLER)

                    if eski_renk not in renk_secenekleri:
                        renk_secenekleri.insert(0, eski_renk)

                    yeni_renk = st.selectbox(
                        "Renk",
                        renk_secenekleri,
                        index=renk_secenekleri.index(eski_renk),
                        key=f"renk_{i}"
                    )

                    c1, c2 = st.columns(2)

                    with c1:
                        if st.button("💾 Kaydet", key=f"kaydet_{i}"):
                            dolap[i]["tur"] = yeni_tur
                            dolap[i]["alt_tur"] = yeni_alt
                            dolap[i]["renk"] = yeni_renk

                            if dolabi_kaydet(dolap):
                                st.success("Bilgiler güncellendi.")
                                st.rerun()

                    with c2:
                        if st.button("🗑️ Sil", key=f"sil_{i}"):
                            st.session_state["sil_onay"] = i

                    if st.session_state.get("sil_onay") == i:
                        st.warning("Bu kıyafeti silmek istiyor musun?")
                        a, b = st.columns(2)

                        with a:
                            if st.button("Evet, sil", key=f"evet_{i}"):
                                dolap.pop(i)
                                if dolabi_kaydet(dolap):
                                    st.session_state.pop("sil_onay", None)
                                    st.rerun()

                        with b:
                            if st.button("Vazgeç", key=f"vazgec_{i}"):
                                st.session_state.pop("sil_onay", None)
                                st.rerun()


# ---------------------------------------------------------
# KOMBİN ÖNER
# ---------------------------------------------------------

elif sayfa == "Kombin Öner":
    st.header("✨ Tarzını seç")
    st.write("Seçtiğin tarza göre dolabındaki parçalar puanlanır.")

    c1, c2, c3 = st.columns(3)

    if "secilen_tarz" not in st.session_state:
        st.session_state["secilen_tarz"] = "Günlük"

    with c1:
        if st.button("👟 Günlük", use_container_width=True):
            st.session_state["secilen_tarz"] = "Günlük"
            st.session_state.pop("son_kombin", None)
            st.session_state.pop("son_aciklama", None)

    with c2:
        if st.button("🏃 Spor", use_container_width=True):
            st.session_state["secilen_tarz"] = "Spor"
            st.session_state.pop("son_kombin", None)
            st.session_state.pop("son_aciklama", None)

    with c3:
        if st.button("✨ Şık", use_container_width=True):
            st.session_state["secilen_tarz"] = "Şık"
            st.session_state.pop("son_kombin", None)
            st.session_state.pop("son_aciklama", None)

    tarz = st.session_state["secilen_tarz"]
    st.markdown(f"**Seçilen tarz: {tarz}**")

    if not dolap:
        st.info("Önce dolabına kıyafet eklemelisin.")
    else:
        if st.button("✨ Bu tarza göre kombin oluştur", type="primary"):
            kombin, aciklama = kombin_olustur(dolap, tarz)
            st.session_state["son_kombin"] = kombin
            st.session_state["son_aciklama"] = aciklama
            st.session_state["son_tarz"] = tarz

        kombin = st.session_state.get("son_kombin", [])
        aciklama = st.session_state.get("son_aciklama", "")

        if kombin:
            kolaj = kolaj_olustur(kombin)

            sol, sag = st.columns([1.2, 1])

            with sol:
                st.image(
                    kolaj,
                    caption=(
                        f"{st.session_state.get('son_tarz', tarz)} "
                        "kombin kolajı"
                    ),
                    use_container_width=True
                )

                tampon = io.BytesIO()
                kolaj.save(tampon, format="PNG")

                st.download_button(
                    "⬇️ Kombin görselini indir",
                    data=tampon.getvalue(),
                    file_name="akilli_dolabim_kombin.png",
                    mime="image/png"
                )

            with sag:
                st.subheader("📝 Kombin açıklaması")
                st.markdown(aciklama)

                st.subheader("👚 Kombindeki gerçek parçalar")

                for parca in kombin:
                    alt = alt_tur_al(parca)
                    detay = f" — {alt}" if alt else ""
                    st.write(
                        f"- **{tur_al(parca)}{detay}** — {renk_al(parca)}"
                    )

        elif aciklama:
            st.info(aciklama)


st.sidebar.divider()
st.sidebar.caption("Akıllı Dolabım • Kişisel kombin asistanı")
