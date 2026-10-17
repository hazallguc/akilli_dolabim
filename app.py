
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

RENKLER = [
    "Siyah", "Beyaz", "Gri", "Kırmızı", "Pembe", "Mavi",
    "Lacivert", "Yeşil", "Sarı", "Turuncu", "Mor",
    "Kahverengi", "Bej", "Pembe-Siyah", "Siyah-Beyaz",
    "Çok renkli", "Bilinmiyor"
]

# ---------------------------------------------------------
# VERİ KAYITLARI
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
            with open(DOSYA, "rb") as kaynak, open(YEDEK, "wb") as yedek:
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


def renk_al(kayit):
    return str(kayit.get("renk", "Bilinmiyor") or "Bilinmiyor")


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
# KIYAFET TÜRÜ TAHMİNİ
# ---------------------------------------------------------

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
        return None

    etiketler = {
        "Üst giyim": "a photo of a shirt, blouse or top",
        "Alt giyim": "a photo of trousers, jeans or a skirt",
        "Elbise": "a photo of a dress",
        "Dış giyim": "a photo of a coat or jacket",
        "Ayakkabı": "a photo of shoes or sneakers",
        "Çanta": "a photo of a handbag or backpack",
        "Aksesuar": "a photo of jewelry or a fashion accessory",
        "Diğer": "a photo of another fashion item"
    }

    try:
        tahminler = model(
            fotograf,
            candidate_labels=list(etiketler.values())
        )
        bulunan = tahminler[0]["label"]

        for tur, etiket in etiketler.items():
            if etiket == bulunan:
                return tur
    except Exception:
        pass

    return None


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

        resim = resim.crop((
            int(w * 0.10), int(h * 0.10),
            max(int(w * 0.90), 1),
            max(int(h * 0.90), 1)
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
            return f"{renk1}-{renk2}"

        if oran1 < 0.20:
            return "Çok renkli"

        return renk1

    except Exception:
        return "Bilinmiyor"


# ---------------------------------------------------------
# TARZ VE RENK UYUMU
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

    # Renklerden biri bilinmiyorsa parçayı otomatik eleme.
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


def dosya_adi(kayit):
    return str(kayit.get("dosya_adi", "")).lower()


def tarz_puani(kayit, tarz):
    """
    Dosya adındaki kelimeler ve kıyafet türü üzerinden
    tarza uygunluk için yaklaşık puan üretir.
    """

    tur = tur_al(kayit)
    ad = dosya_adi(kayit)
    puan = 0

    if tarz == "Günlük":
        kelimeler = [
            "basic", "jean", "kot", "tişört", "tisort",
            "sweat", "günlük", "gunluk", "spor ayakkabı"
        ]

        puan += sum(2 for kelime in kelimeler if kelime in ad)

        if tur in ["Üst giyim", "Alt giyim", "Ayakkabı"]:
            puan += 2

        if tur == "Elbise" and "günlük" not in ad and "gunluk" not in ad:
            puan -= 1

    elif tarz == "Spor":
        kelimeler = [
            "spor", "sneaker", "eşofman", "esofman", "sweat",
            "hoodie", "tayt", "koşu", "kosu", "antrenman"
        ]

        puan += sum(3 for kelime in kelimeler if kelime in ad)

        if tur == "Ayakkabı":
            puan += 5
        elif tur in ["Üst giyim", "Alt giyim"]:
            puan += 2

        if any(k in ad for k in ["topuklu", "klasik", "abiye"]):
            puan -= 4

    elif tarz == "Şık":
        kelimeler = [
            "blazer", "topuklu", "gömlek", "gomlek",
            "klasik", "loafer", "abiye", "şık", "sik",
            "ceket", "etek"
        ]

        puan += sum(3 for kelime in kelimeler if kelime in ad)

        if tur == "Dış giyim":
            puan += 3
        elif tur == "Çanta":
            puan += 2
        elif tur == "Ayakkabı":
            puan += 2
        elif tur == "Elbise":
            puan += 2

        if any(k in ad for k in ["eşofman", "esofman", "koşu", "kosu"]):
            puan -= 4

    return puan


def temel_kombin_puani(temel, tarz):
    puan = sum(tarz_puani(p, tarz) for p in temel)

    renkler = [renk_al(p) for p in temel]

    for i in range(len(renkler)):
        for j in range(i + 1, len(renkler)):
            puan += renk_uyum_puani(renkler[i], renkler[j])

    turler = [tur_al(p) for p in temel]

    # Üst + alt kombinini günlük ve spor tarzında öne çıkar.
    if "Üst giyim" in turler and "Alt giyim" in turler:
        if tarz in ["Günlük", "Spor"]:
            puan += 8
        else:
            puan += 5

    # Elbise, üst-alt kombini mevcutsa varsayılan olarak öne geçmesin.
    if "Elbise" in turler:
        puan -= 3

    return puan


def en_iyi_aday(adaylar, puan_fonksiyonu):
    """
    En yüksek puanlı adaylar arasından seçim yapar.
    Böylece puanları eşit olan kayıtlarda hep ilk fotoğraf seçilmez.
    """

    if not adaylar:
        return None

    puanli = [(puan_fonksiyonu(a), a) for a in adaylar]
    en_yuksek = max(p for p, _ in puanli)

    en_iyiler = [
        a for p, a in puanli
        if p >= en_yuksek - 2
    ]

    return random.choice(en_iyiler)


# ---------------------------------------------------------
# KOMBIN SEÇİMİ
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

    # Öncelik: üst giyim + alt giyim.
    temel_adaylari = []

    if ustler and altlar:
        for ust in ustler:
            for alt in altlar:
                temel_adaylari.append([ust, alt])

    # Elbiseyi ancak üst-alt kombinine alternatif olarak ekle.
    # Üst ve alt varsa günlük/spor için öncelik ikili kombindedir.
    if elbiseler and not (ustler and altlar):
        for elbise in elbiseler:
            temel_adaylari.append([elbise])

    # Hiç üst-alt ikilisi veya elbise yoksa açıklama ver.
    if not temel_adaylari:
        if not ustler and not elbiseler:
            return [], (
                "Tam kombin oluşturabilmem için dolabına üst giyim "
                "veya elbise eklemelisin."
            )

        if not altlar and not elbiseler:
            return [], (
                "Tam kombin oluşturabilmem için dolabına pantolon, "
                "etek veya elbise eklemelisin."
            )

        return [], (
            "Dolabında henüz tamamlanmış bir kombin için yeterli "
            "temel parça yok. Üst giyim + alt giyim veya elbise ekle."
        )

    temel = en_iyi_aday(
        temel_adaylari,
        lambda aday: temel_kombin_puani(aday, tarz)
    )

    sonuc = list(temel)

    # Tamamlayıcıları tek tek seç.
    # Renk uyumu yardımcı ölçüttür; tek başına kıyafeti elemez.
    gruplar = [
        dislar,
        ayakkabilar,
        cantalar,
        aksesuarlar
    ]

    for grup in gruplar:
        if not grup:
            continue

        def tamamlayici_puani(parca):
            puan = tarz_puani(parca, tarz)

            for mevcut in sonuc:
                puan += renk_uyum_puani(
                    renk_al(parca),
                    renk_al(mevcut)
                )

            return puan

        secilen = en_iyi_aday(
            [p for p in grup if p not in sonuc],
            tamamlayici_puani
        )

        if secilen is not None:
            sonuc.append(secilen)

    return sonuc, aciklama_olustur(sonuc, tarz)


def aciklama_olustur(kombin, tarz):
    turler = [tur_al(p) for p in kombin]

    renkler = list(dict.fromkeys(
        renk_al(p)
        for p in kombin
        if renk_al(p) != "Bilinmiyor"
    ))

    metin = f"**Seçilen tarz: {tarz}.** "

    if "Üst giyim" in turler and "Alt giyim" in turler:
        metin += (
            "Üst giyim ve alt giyimi temel alarak kombini oluşturdum."
        )
    elif "Elbise" in turler:
        metin += (
            "Dolabında uygun üst-alt ikilisi bulunmadığı için "
            "elbise üzerinden bir kombin oluşturdum."
        )
    else:
        metin += "Dolabındaki temel parçaları bir araya getirdim."

    if renkler:
        metin += " Renkler: " + ", ".join(renkler) + "."

    eksikler = []

    if "Dış giyim" not in turler:
        eksikler.append("ceket veya hırka")

    if "Ayakkabı" not in turler:
        eksikler.append("ayakkabı")

    if "Çanta" not in turler:
        eksikler.append("çanta")

    if "Aksesuar" not in turler:
        eksikler.append("takı veya kemer")

    if eksikler:
        metin += (
            "\n\n**Dolabında bulunmadığı için kolaja eklenemeyen "
            "tamamlayıcılar:** " + ", ".join(eksikler) + "."
        )

    return metin


# ---------------------------------------------------------
# TEK KARELİK KOLAJ
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

        etiket = f"{tur_al(parca)} • {renk_al(parca)}"

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
                with st.spinner("Analiz ediliyor..."):
                    tahmin_turu = tur_tahmin_et(fotograf)
                    tahmin_renk = renk_tahmin_et(fotograf)

                st.session_state["gecici_resim"] = fotograf_base64(fotograf)
                st.session_state["gecici_tur"] = tahmin_turu or "Üst giyim"
                st.session_state["gecici_renk"] = (
                    tahmin_renk
                    if tahmin_renk in RENKLER
                    else "Bilinmiyor"
                )
                st.session_state["analiz_ok"] = True

            if st.session_state.get("analiz_ok"):
                varsayilan_tur = st.session_state.get(
                    "gecici_tur", "Üst giyim"
                )
                varsayilan_renk = st.session_state.get(
                    "gecici_renk", "Bilinmiyor"
                )

                tur = st.selectbox(
                    "Kıyafet türünü kontrol et",
                    TURLER,
                    index=(
                        TURLER.index(varsayilan_tur)
                        if varsayilan_tur in TURLER else 0
                    )
                )

                renk = st.selectbox(
                    "Rengini kontrol et",
                    RENKLER,
                    index=(
                        RENKLER.index(varsayilan_renk)
                        if varsayilan_renk in RENKLER
                        else len(RENKLER) - 1
                    )
                )

                if st.button("➕ Dolabıma kaydet", type="primary"):
                    dolap.append({
                        "dosya_adi": yuklenen.name,
                        "tur": tur,
                        "renk": renk,
                        "resim": st.session_state["gecici_resim"],
                        "eklenme_tarihi": datetime.now().isoformat(
                            timespec="seconds"
                        )
                    })

                    if dolabi_kaydet(dolap):
                        st.session_state["analiz_ok"] = False
                        st.session_state.pop("gecici_resim", None)
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
        st.info(
            "Dolabın boş. Kıyafet Ekle bölümünden başlayabilirsin."
        )
    else:
        st.write(f"Toplam kayıt: **{len(dolap)}**")

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
                        st.image(
                            fotograf,
                            use_container_width=True
                        )

                with sag:
                    st.write(
                        f"**Dosya:** {parca.get('dosya_adi', 'Kıyafet')}"
                    )

                    eski_tur = tur_al(parca)
                    eski_renk = renk_al(parca)

                    yeni_tur = st.selectbox(
                        "Tür",
                        TURLER,
                        index=TURLER.index(eski_tur),
                        key=f"tur_{i}"
                    )

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
    st.write("Nasıl bir kombin istiyorsun? Bir tarza tıkla.")

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
        if st.button(
            "✨ Bu tarza göre kombin oluştur",
            type="primary"
        ):
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
                    st.write(
                        f"- **{tur_al(parca)}** — {renk_al(parca)}"
                    )

        elif aciklama:
            st.info(aciklama)


st.sidebar.divider()
st.sidebar.caption("Akıllı Dolabım • Kişisel kombin asistanı")
