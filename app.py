
import streamlit as st
import json
import os
import base64
import io
import random
from itertools import product
from PIL import Image, ImageDraw
from collections import Counter
import colorsys

# =========================================================
# SAYFA AYARLARI
# =========================================================

st.set_page_config(
    page_title="Akıllı Dolabım",
    page_icon="👗",
    layout="wide"
)

DOSYA = "dolap.json"

# =========================================================
# KIYAFET TÜRLERİ
# Eski "tur" alanını koruyoruz; alt_tur yeni detay alanıdır.
# =========================================================

ALT_TURLER = {
    "Üst giyim": [
        "Tişört", "Bluz", "Gömlek", "Crop top",
        "Sweatshirt", "Hoodie", "Kazak", "Spor üst",
        "Body", "Diğer üst giyim"
    ],
    "Alt giyim": [
        "Jean", "Kumaş pantolon", "Eşofman altı",
        "Tayt", "Spor tayt", "Etek", "Şort",
        "Spor şort", "Diğer alt giyim"
    ],
    "Elbise": [
        "Günlük elbise", "Şık elbise", "Abiye",
        "Spor elbise", "Diğer elbise"
    ],
    "Dış giyim": [
        "Blazer ceket", "Günlük ceket", "Deri ceket",
        "Kot ceket", "Şişme mont", "Mont", "Kaban",
        "Hırka", "Spor ceket", "Diğer dış giyim"
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

TURLER = list(ALT_TURLER.keys())

RENKLER = [
    "Siyah", "Beyaz", "Gri", "Kırmızı", "Bordo",
    "Pembe", "Pembe-Siyah", "Mavi", "Lacivert",
    "Yeşil", "Sarı", "Turuncu", "Mor", "Kahverengi",
    "Bej", "Krem", "Çok renkli", "Bilinmiyor"
]

# =========================================================
# YAPAY ZEKÂ MODELİ
# Model yüklenemezse uygulama yine çalışır; kıyafet türü
# elle seçilebilir.
# =========================================================

MODEL_ETIKETLERI = {
    "Tişört": "a t-shirt",
    "Bluz": "a blouse",
    "Gömlek": "a button-up shirt",
    "Crop top": "a crop top",
    "Sweatshirt": "a sweatshirt",
    "Hoodie": "a hoodie",
    "Kazak": "a sweater",
    "Spor üst": "a sports top",
    "Body": "a bodysuit",
    "Jean": "jeans",
    "Kumaş pantolon": "dress trousers",
    "Eşofman altı": "sweatpants",
    "Tayt": "leggings",
    "Spor tayt": "sports leggings",
    "Etek": "a skirt",
    "Şort": "shorts",
    "Spor şort": "sports shorts",
    "Günlük elbise": "a casual dress",
    "Şık elbise": "an elegant dress",
    "Abiye": "an evening gown",
    "Spor elbise": "a sporty dress",
    "Blazer ceket": "a blazer",
    "Günlük ceket": "a casual jacket",
    "Deri ceket": "a leather jacket",
    "Kot ceket": "a denim jacket",
    "Şişme mont": "a puffer jacket",
    "Mont": "a coat or jacket",
    "Kaban": "a long winter coat",
    "Hırka": "a cardigan",
    "Spor ceket": "a sports jacket",
    "Spor ayakkabı": "sports shoes",
    "Sneaker": "sneakers",
    "Topuklu ayakkabı": "high heels",
    "Babet": "ballet flats",
    "Loafer": "loafers",
    "Bot": "boots",
    "Sandalet": "sandals",
    "Klasik ayakkabı": "formal shoes",
    "Günlük çanta": "a casual handbag",
    "Şık çanta": "an elegant handbag",
    "Omuz çantası": "a shoulder bag",
    "El çantası": "a clutch bag",
    "Sırt çantası": "a backpack",
    "Spor çanta": "a sports bag",
    "Takı": "jewelry",
    "Kemer": "a belt",
    "Şapka": "a hat",
    "Atkı": "a scarf",
    "Gözlük": "glasses",
}

ALT_TUR_KATEGORI = {}
for ana_tur, alt_turler in ALT_TURLER.items():
    for alt_tur in alt_turler:
        ALT_TUR_KATEGORI[alt_tur] = ana_tur

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
        etiketler = list(MODEL_ETIKETLERI.values())
        sonuc = model(image, candidate_labels=etiketler)
        if not sonuc:
            return "Diğer", "Diğer", 0.0

        en_iyi = sonuc[0]
        ingilizce_etiket = en_iyi["label"]
        guven = float(en_iyi.get("score", 0))

        ters_esleme = {
            ingilizce: tur
            for tur, ingilizce in MODEL_ETIKETLERI.items()
        }
        alt_tur = ters_esleme.get(ingilizce_etiket, "Diğer")
        ana_tur = ALT_TUR_KATEGORI.get(alt_tur, "Diğer")

        return ana_tur, alt_tur, guven

    except Exception:
        return "Diğer", "Diğer", 0.0

# =========================================================
# RENK TAHMİNİ
# =========================================================

def renk_tahmin_et(image):
    try:
        img = image.convert("RGB")
        img.thumbnail((150, 150))
        pikseller = list(img.getdata())

        renk_sayaci = Counter()
        for r, g, b in pikseller:
            h, s, v = colorsys.rgb_to_hsv(
                r / 255, g / 255, b / 255
            )

            # Arka planın ve gölgelerin etkisini azalt.
            if v < 0.12:
                renk = "Siyah"
            elif s < 0.12 and v > 0.88:
                renk = "Beyaz"
            elif s < 0.16:
                renk = "Gri"
            elif h < 0.04 or h >= 0.96:
                renk = "Kırmızı"
            elif h < 0.09:
                renk = "Turuncu"
            elif h < 0.17:
                renk = "Sarı"
            elif h < 0.45:
                renk = "Yeşil"
            elif h < 0.72:
                renk = "Mavi"
            elif h < 0.82:
                renk = "Mor"
            elif h < 0.96:
                renk = "Pembe"
            else:
                renk = "Kırmızı"

            renk_sayaci[renk] += 1

        en_cok = renk_sayaci.most_common(2)
        if not en_cok:
            return "Bilinmiyor"

        birinci = en_cok[0][0]
        toplam = sum(renk_sayaci.values())
        birinci_oran = en_cok[0][1] / max(toplam, 1)

        # İki baskın renk birbirine yakınsa ikisini birlikte göster.
        if len(en_cok) > 1:
            ikinci, ikinci_sayi = en_cok[1]
            oran = ikinci_sayi / max(toplam, 1)

            if (
                birinci_oran < 0.60
                and oran > 0.20
                and birinci != ikinci
            ):
                ikili = {birinci, ikinci}
                if ikili == {"Pembe", "Siyah"}:
                    return "Pembe-Siyah"
                return f"{birinci}-{ikinci}"

        return birinci

    except Exception:
        return "Bilinmiyor"

# =========================================================
# DOLAP DOSYASI
# Eski kayıtlar ve fotoğraflar korunur.
# =========================================================

def dolabi_yukle():
    if not os.path.exists(DOSYA):
        return []

    try:
        with open(DOSYA, "r", encoding="utf-8") as f:
            veri = json.load(f)

        if isinstance(veri, list):
            kayitlar = veri
        elif isinstance(veri, dict) and isinstance(veri.get("dolap"), list):
            kayitlar = veri["dolap"]
        else:
            return []

        # Eski kayıtların alanlarını silmeden eksikleri tamamla.
        for kayit in kayitlar:
            kayit.setdefault("isim", kayit.get("dosya_adi", "Kıyafet"))
            kayit.setdefault("dosya_adi", kayit.get("isim", "Kıyafet"))
            kayit.setdefault("resim", "")
            kayit.setdefault("tur", "Diğer")
            kayit.setdefault("alt_tur", "")
            kayit.setdefault("renk", "Bilinmiyor")
            kayit.setdefault("guven", 0.0)

        return kayitlar

    except Exception:
        st.error(
            "dolap.json dosyası okunamadı. Dosyanı silme; "
            "dosyanın biçimini kontrol etmek gerekiyor."
        )
        return []

def dolabi_kaydet(dolap):
    # Mevcut dosyanın yedeğini al.
    if os.path.exists(DOSYA):
        try:
            with open(DOSYA, "r", encoding="utf-8") as f:
                eski_icerik = f.read()
            with open(DOSYA + ".bak", "w", encoding="utf-8") as f:
                f.write(eski_icerik)
        except Exception:
            pass

    with open(DOSYA, "w", encoding="utf-8") as f:
        json.dump(dolap, f, ensure_ascii=False, indent=2)

def fotoğrafı_ac(kayit):
    kod = kayit.get("resim", "")
    if not kod:
        return None
    try:
        ham = base64.b64decode(kod)
        return Image.open(io.BytesIO(ham)).convert("RGB")
    except Exception:
        return None

def fotoğrafı_koda_cevir(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def kayit_alt_turu(kayit):
    alt_tur = kayit.get("alt_tur", "")
    ana_tur = kayit.get("tur", "Diğer")

    if alt_tur in ALT_TUR_KATEGORI:
        return alt_tur

    # Eski kayıtlar için, kayıtlı türün altındaki ilk genel seçenek.
    return ALT_TURLER.get(ana_tur, ["Diğer"])[-1]

# =========================================================
# TARZA GÖRE KIYAFET SEÇİMİ
# =========================================================

GUNLUK_UST = [
    "Tişört", "Bluz", "Gömlek", "Sweatshirt",
    "Hoodie", "Kazak", "Crop top"
]
SPOR_UST = ["Spor üst", "Tişört", "Sweatshirt", "Hoodie"]
SIK_UST = ["Bluz", "Gömlek", "Crop top", "Body", "Kazak"]

GUNLUK_ALT = ["Jean", "Kumaş pantolon", "Etek", "Şort"]
SPOR_ALT = ["Tayt", "Spor tayt", "Eşofman altı", "Spor şort"]
SIK_ALT = ["Kumaş pantolon", "Etek", "Jean"]

GUNLUK_AYAKKABI = ["Sneaker", "Spor ayakkabı", "Babet", "Loafer", "Bot"]
SPOR_AYAKKABI = ["Spor ayakkabı", "Sneaker"]
SIK_AYAKKABI = [
    "Topuklu ayakkabı", "Loafer", "Babet",
    "Klasik ayakkabı", "Sandalet"
]

def alt_turleri(kayitlar, ana_tur):
    return [
        k for k in kayitlar
        if k.get("tur") == ana_tur
    ]

def puanla(kayit, oncelikler):
    alt_tur = kayit_alt_turu(kayit)
    try:
        return len(oncelikler) - oncelikler.index(alt_tur)
    except ValueError:
        return 0

def en_uygun(kayitlar, oncelikler):
    adaylar = []
    for kayit in kayitlar:
        puan = puanla(kayit, oncelikler)
        if puan > 0:
            adaylar.append((puan, kayit))

    if not adaylar:
        return None

    adaylar.sort(key=lambda x: x[0], reverse=True)
    en_yuksek = adaylar[0][0]
    esitler = [k for p, k in adaylar if p == en_yuksek]
    return random.choice(esitler)

def uygun_dis_giyim(kayitlar, tarz):
    dis_giyim = alt_turleri(kayitlar, "Dış giyim")

    if tarz == "Şık":
        # Şık kombin için şişme montu otomatik seçme.
        oncelikler = [
            "Blazer ceket", "Deri ceket", "Günlük ceket",
            "Kot ceket", "Hırka"
        ]
    elif tarz == "Spor":
        oncelikler = ["Spor ceket", "Günlük ceket", "Hırka"]
    else:
        oncelikler = [
            "Günlük ceket", "Kot ceket", "Deri ceket",
            "Hırka", "Mont", "Şişme mont", "Kaban"
        ]

    return en_uygun(dis_giyim, oncelikler)

def uygun_ayakkabi(kayitlar, tarz):
    ayakkabilar = alt_turleri(kayitlar, "Ayakkabı")

    if tarz == "Şık":
        oncelikler = SIK_AYAKKABI
    elif tarz == "Spor":
        oncelikler = SPOR_AYAKKABI
    else:
        oncelikler = GUNLUK_AYAKKABI

    return en_uygun(ayakkabilar, oncelikler)

def uygun_canta(kayitlar, tarz):
    cantalar = alt_turleri(kayitlar, "Çanta")

    if tarz == "Şık":
        oncelikler = ["Şık çanta", "El çantası", "Omuz çantası", "Günlük çanta"]
    elif tarz == "Spor":
        oncelikler = ["Spor çanta", "Sırt çantası"]
    else:
        oncelikler = ["Günlük çanta", "Omuz çantası", "Sırt çantası"]

    return en_uygun(cantalar, oncelikler)

def kombin_olustur(kayitlar, tarz):
    kombin = []

    # Ana parçalar
    elbiseler = alt_turleri(kayitlar, "Elbise")
    ustler = alt_turleri(kayitlar, "Üst giyim")
    altlar = alt_turleri(kayitlar, "Alt giyim")

    if tarz == "Şık":
        elbise_oncelikleri = ["Şık elbise", "Abiye", "Günlük elbise"]
        uygun_elbise = en_uygun(elbiseler, elbise_oncelikleri)

        # Şık elbise/abiye varsa elbise tabanlı kombin yap.
        if uygun_elbise and kayit_alt_turu(uygun_elbise) in ["Şık elbise", "Abiye"]:
            kombin.append(uygun_elbise)
        else:
            ust = en_uygun(ustler, SIK_UST)
            alt = en_uygun(altlar, SIK_ALT)

            if ust:
                kombin.append(ust)
            if alt:
                kombin.append(alt)

            if not ust and not alt and uygun_elbise:
                kombin.append(uygun_elbise)

    elif tarz == "Spor":
        spor_elbise = en_uygun(elbiseler, ["Spor elbise"])
        if spor_elbise and not altlar and not ustler:
            kombin.append(spor_elbise)
        else:
            ust = en_uygun(ustler, SPOR_UST)
            alt = en_uygun(altlar, SPOR_ALT)

            if ust:
                kombin.append(ust)
            if alt:
                kombin.append(alt)

            if not ust and not alt and spor_elbise:
                kombin.append(spor_elbise)

    else:
        ust = en_uygun(ustler, GUNLUK_UST)
        alt = en_uygun(altlar, GUNLUK_ALT)
        gunluk_elbise = en_uygun(
            elbiseler,
            ["Günlük elbise", "Spor elbise", "Şık elbise"]
        )

        if ust:
            kombin.append(ust)
        if alt:
            kombin.append(alt)

        if not ust and not alt and gunluk_elbise:
            kombin.append(gunluk_elbise)

    # Ana kıyafet bulunamadıysa kullanıcıya boş kolaj gösterme.
    if not kombin:
        return [], [
            "Bu tarz için yeterli ana kıyafet bulunamadı.",
            "Dolabım bölümünde kıyafet türünü ve alt türünü kontrol et."
        ]

    # Uygun dış giyim, ayakkabı ve çanta.
    dis = uygun_dis_giyim(kayitlar, tarz)
    ayakkabi = uygun_ayakkabi(kayitlar, tarz)
    canta = uygun_canta(kayitlar, tarz)

    if dis:
        kombin.append(dis)
    if ayakkabi:
        kombin.append(ayakkabi)
    if canta:
        kombin.append(canta)

    # Takı/kemer otomatik eklenmez.
    # Şık kombinlerde yalnızca takı eklenebilir; kemer zorunlu değildir.
    if tarz == "Şık":
        takilar = alt_turleri(kayitlar, "Aksesuar")
        taki = en_uygun(takilar, ["Takı"])
        if taki:
            kombin.append(taki)

    aciklamalar = []

    if not ayakkabi:
        if tarz == "Şık":
            aciklamalar.append(
                "Dolabında bu kombine uygun bir topuklu veya şık ayakkabı seçilmedi."
            )
        elif tarz == "Spor":
            aciklamalar.append(
                "Dolabında spor ayakkabı veya sneaker alt türü seçilmiş bir ayakkabı yok."
            )
        else:
            aciklamalar.append("Bu kombin için uygun bir ayakkabı bulunamadı.")

    if not dis:
        if tarz == "Şık":
            aciklamalar.append(
                "Şık dış giyim olarak blazer veya uygun bir ceket bulunamadı; "
                "bu nedenle şişme mont eklenmedi."
            )
        else:
            aciklamalar.append("Dolabında bu tarza uygun dış giyim bulunamadı.")

    if not canta:
        aciklamalar.append("Dolabında bu tarza uygun bir çanta bulunamadı.")

    return kombin, aciklamalar

# =========================================================
# FOTOĞRAF KOLAJI
# =========================================================

def kolaj_olustur(kombin):
    kart_genislik = 250
    kart_yukseklik = 310
    baslik_yukseklik = 55
    bosluk = 18

    resimli_kayitlar = []
    for kayit in kombin:
        img = fotoğrafı_ac(kayit)
        if img is not None:
            resimli_kayitlar.append((kayit, img))

    if not resimli_kayitlar:
        return None

    sutun = min(3, len(resimli_kayitlar))
    satir = (len(resimli_kayitlar) + sutun - 1) // sutun

    genislik = sutun * kart_genislik + (sutun + 1) * bosluk
    yukseklik = (
        baslik_yukseklik
        + satir * kart_yukseklik
        + (satir + 1) * bosluk
    )

    canvas = Image.new("RGB", (genislik, yukseklik), "#f7f4f1")
    draw = ImageDraw.Draw(canvas)
    draw.text((bosluk, 18), "AKILLI DOLABIM • KOMBİNİN", fill="#272323")

    for i, (kayit, img) in enumerate(resimli_kayitlar):
        x = bosluk + (i % sutun) * (kart_genislik + bosluk)
        y = baslik_yukseklik + bosluk + (i // sutun) * (
            kart_yukseklik + bosluk
        )

        draw.rounded_rectangle(
            [x, y, x + kart_genislik, y + kart_yukseklik],
            radius=12,
            fill="white",
            outline="#e2d9d2",
            width=2
        )

        img.thumbnail((kart_genislik - 20, kart_yukseklik - 65))
        ix = x + (kart_genislik - img.width) // 2
        iy = y + 10
        canvas.paste(img, (ix, iy))

        ad = kayit.get("isim", kayit.get("dosya_adi", "Kıyafet"))
        alt_tur = kayit_alt_turu(kayit)
        etiket = alt_tur if alt_tur != "Diğer" else ad
        draw.text(
            (x + 10, y + kart_yukseklik - 42),
            etiket[:32],
            fill="#272323"
        )
        draw.text(
            (x + 10, y + kart_yukseklik - 23),
            kayit.get("renk", "Bilinmiyor"),
            fill="#77706b"
        )

    return canvas

# =========================================================
# ARAYÜZ
# =========================================================

if "dolap" not in st.session_state:
    st.session_state["dolap"] = dolabi_yukle()

dolap = st.session_state["dolap"]

st.title("👗 Akıllı Dolabım")
st.caption("Kıyafetlerini ekle, dolabını düzenle ve tarzına uygun kombin oluştur.")

sayfa = st.sidebar.radio(
    "Menü",
    ["Kıyafet Ekle", "Dolabım", "Kombin Öner"]
)

# ---------------------------------------------------------
# KIYAFET EKLE
# ---------------------------------------------------------

if sayfa == "Kıyafet Ekle":
    st.header("📸 Dolabına kıyafet ekle")
    dosya = st.file_uploader(
        "Kıyafet fotoğrafını yükle",
        type=["jpg", "jpeg", "png", "webp"],
        key="kiyafet_fotograf_yukleyici"
    )

    if dosya:
        try:
            image = Image.open(dosya).convert("RGB")
            st.image(image, caption="Yüklediğin fotoğraf", width=280)

            if st.button("🤖 Fotoğrafı analiz et", type="primary"):
                with st.spinner("Yapay zekâ kıyafeti analiz ediyor..."):
                    tahmin_tur, tahmin_alt, guven = yapay_zeka_tahmin(image)
                    tahmin_renk = renk_tahmin_et(image)

                st.session_state["son_analiz"] = {
                    "isim": dosya.name,
                    "resim": fotoğrafı_koda_cevir(image),
                    "tur": tahmin_tur,
                    "alt_tur": tahmin_alt,
                    "renk": tahmin_renk,
                    "guven": guven
                }

            analiz = st.session_state.get("son_analiz")

            if analiz and analiz.get("isim") == dosya.name:
                st.subheader("Kıyafet bilgilerini kontrol et")

                tur_index = (
                    TURLER.index(analiz["tur"])
                    if analiz["tur"] in TURLER else TURLER.index("Diğer")
                )

                secilen_tur = st.selectbox(
                    "Ana tür",
                    TURLER,
                    index=tur_index,
                    key="yeni_ana_tur"
                )

                uygun_altlar = ALT_TURLER[secilen_tur]
                tahmin_alt = analiz.get("alt_tur", "")
                alt_index = (
                    uygun_altlar.index(tahmin_alt)
                    if tahmin_alt in uygun_altlar else 0
                )

                secilen_alt = st.selectbox(
                    "Kıyafetin tam türü",
                    uygun_altlar,
                    index=alt_index,
                    key="yeni_alt_tur"
                )

                renk_index = (
                    RENKLER.index(analiz["renk"])
                    if analiz["renk"] in RENKLER
                    else RENKLER.index("Bilinmiyor")
                )

                secilen_renk = st.selectbox(
                    "Renk",
                    RENKLER,
                    index=renk_index,
                    key="yeni_renk"
                )

                if st.button("➕ Dolabıma kaydet", type="primary"):
                    yeni_kayit = {
                        "isim": analiz["isim"],
                        "dosya_adi": analiz["isim"],
                        "resim": analiz["resim"],
                        "tur": secilen_tur,
                        "alt_tur": secilen_alt,
                        "renk": secilen_renk,
                        "guven": analiz.get("guven", 0.0)
                    }

                    dolap.append(yeni_kayit)
                    st.session_state["dolap"] = dolap
                    dolabi_kaydet(dolap)
                    st.session_state.pop("son_analiz", None)
                    st.success("Kıyafet dolabına kaydedildi!")
                    st.rerun()

        except Exception as e:
            st.error(f"Fotoğraf açılamadı: {e}")

# ---------------------------------------------------------
# DOLABIM
# ---------------------------------------------------------

elif sayfa == "Dolabım":
    st.header("🧥 Dolabım")

    if not dolap:
        st.info("Dolabında henüz kıyafet yok. Kıyafet Ekle bölümünden başlayabilirsin.")
    else:
        st.write(f"Toplam **{len(dolap)}** kıyafet kayıtlı.")

        filtre = st.selectbox("Türüne göre filtrele", ["Tümü"] + TURLER)

        for i, kayit in enumerate(dolap):
            if filtre != "Tümü" and kayit.get("tur") != filtre:
                continue

            with st.container(border=True):
                sol, sag = st.columns([1, 2])

                with sol:
                    img = fotoğrafı_ac(kayit)
                    if img:
                        st.image(img, use_container_width=True)

                with sag:
                    st.markdown(
                        f"**{kayit.get('isim', kayit.get('dosya_adi', 'Kıyafet'))}**"
                    )

                    mevcut_tur = kayit.get("tur", "Diğer")
                    if mevcut_tur not in TURLER:
                        mevcut_tur = "Diğer"

                    tur_index = TURLER.index(mevcut_tur)
                    yeni_tur = st.selectbox(
                        "Ana tür",
                        TURLER,
                        index=tur_index,
                        key=f"tur_{i}"
                    )

                    altlar = ALT_TURLER[yeni_tur]
                    mevcut_alt = kayit_alt_turu(kayit)
                    alt_index = altlar.index(mevcut_alt) if mevcut_alt in altlar else 0

                    yeni_alt = st.selectbox(
                        "Tam tür",
                        altlar,
                        index=alt_index,
                        key=f"alt_{i}"
                    )

                    mevcut_renk = kayit.get("renk", "Bilinmiyor")
                    renk_index = (
                        RENKLER.index(mevcut_renk)
                        if mevcut_renk in RENKLER
                        else RENKLER.index("Bilinmiyor")
                    )

                    yeni_renk = st.selectbox(
                        "Renk",
                        RENKLER,
                        index=renk_index,
                        key=f"renk_{i}"
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("💾 Değişiklikleri kaydet", key=f"kaydet_{i}"):
                            kayit["tur"] = yeni_tur
                            kayit["alt_tur"] = yeni_alt
                            kayit["renk"] = yeni_renk
                            dolabi_kaydet(dolap)
                            st.success("Kıyafet bilgileri güncellendi.")
                            st.rerun()

                    with col2:
                        if st.button("🗑️ Kıyafeti sil", key=f"sil_{i}"):
                            st.session_state[f"sil_onay_{i}"] = True

                    if st.session_state.get(f"sil_onay_{i}", False):
                        st.warning("Bu kıyafeti dolabından silmek istiyor musun?")
                        onay1, onay2 = st.columns(2)

                        with onay1:
                            if st.button("Evet, sil", key=f"evet_sil_{i}"):
                                dolap.pop(i)
                                dolabi_kaydet(dolap)
                                st.session_state["dolap"] = dolap
                                st.session_state.pop(f"sil_onay_{i}", None)
                                st.rerun()

                        with onay2:
                            if st.button("Vazgeç", key=f"vazgec_{i}"):
                                st.session_state.pop(f"sil_onay_{i}", None)
                                st.rerun()

# ---------------------------------------------------------
# KOMBİN ÖNER
# ---------------------------------------------------------

elif sayfa == "Kombin Öner":
    st.header("✨ Tarzına göre kombin oluştur")
    st.write("Bir tarz seç; dolabındaki uygun kıyafetlerle kombin hazırlayalım.")

    if not dolap:
        st.info("Önce Kıyafet Ekle bölümünden dolabına birkaç kıyafet ekle.")
    else:
        secilen_tarz = st.radio(
            "Hangi tarzı istiyorsun?",
            ["Günlük", "Spor", "Şık"],
            horizontal=True
        )

        if st.button("✨ Kombinimi oluştur", type="primary"):
            kombin, aciklamalar = kombin_olustur(dolap, secilen_tarz)
            st.session_state["son_kombin"] = kombin
            st.session_state["son_kombin_tarz"] = secilen_tarz
            st.session_state["son_kombin_aciklamalar"] = aciklamalar

        kombin = st.session_state.get("son_kombin", [])
        kombin_tarz = st.session_state.get("son_kombin_tarz", "")
        aciklamalar = st.session_state.get("son_kombin_aciklamalar", [])

        if kombin:
            st.subheader(f"{kombin_tarz} kombin önerin")

            kolaj = kolaj_olustur(kombin)
            if kolaj:
                st.image(kolaj, use_container_width=True)

            st.markdown("### Kombindeki parçalar")
            for kayit in kombin:
                st.write(
                    f"- **{kayit_alt_turu(kayit)}** — "
                    f"{kayit.get('renk', 'Bilinmiyor')}"
                )

            if aciklamalar:
                st.markdown("### Küçük notlar")
                for aciklama in aciklamalar:
                    st.write(f"- {aciklama}")

            if st.button("🔄 Aynı tarzda başka kombin dene"):
                st.session_state.pop("son_kombin", None)
                st.rerun()

        elif kombin_tarz:
            for aciklama in aciklamalar:
                st.warning(aciklama)

st.sidebar.divider()
st.sidebar.caption("Akıllı Dolabım • Yapay zekâ destekli kombin asistanı")
