
import streamlit as st
import json
import os
import base64
import io
import colorsys
from datetime import datetime
from collections import Counter
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np

# =========================================================
# AKILLI DOLABIM
# =========================================================

st.set_page_config(
    page_title="Akıllı Dolabım",
    page_icon="👗",
    layout="wide"
)

DOSYA = "dolap.json"
YEDEK_DOSYA = "dolap.json.bak"

KIYAFET_TURLERI = [
    "Üst giyim",
    "Alt giyim",
    "Elbise",
    "Dış giyim",
    "Ayakkabı",
    "Çanta",
    "Aksesuar",
    "Diğer"
]

RENKLER = [
    "Siyah", "Beyaz", "Gri", "Kırmızı", "Pembe",
    "Mavi", "Lacivert", "Yeşil", "Sarı", "Turuncu",
    "Mor", "Kahverengi", "Bej", "Pembe-Siyah",
    "Siyah-Beyaz", "Çok renkli", "Bilinmiyor"
]

# =========================================================
# DOSYA VE VERİ İŞLEMLERİ
# =========================================================

def dolabi_yukle():
    """Mevcut kıyafet kayıtlarını değiştirmeden yükler."""
    if not os.path.exists(DOSYA):
        return []

    try:
        with open(DOSYA, "r", encoding="utf-8") as dosya:
            veri = json.load(dosya)

        if isinstance(veri, list):
            return veri

        st.error("dolap.json dosyasının biçimi beklenenden farklı.")
        return []

    except json.JSONDecodeError:
        st.error(
            "dolap.json okunamadı. Dosyanın içeriğini kaybetmemek için "
            "otomatik olarak üzerine yazılmadı."
        )
        return []

    except Exception as hata:
        st.error(f"Dolap yüklenemedi: {hata}")
        return []


def dolabi_kaydet(veri):
    """Kaydetmeden önce mevcut JSON dosyasının yedeğini alır."""
    try:
        if os.path.exists(DOSYA):
            if not os.path.exists(YEDEK_DOSYA):
                with open(DOSYA, "rb") as kaynak:
                    with open(YEDEK_DOSYA, "wb") as yedek:
                        yedek.write(kaynak.read())

        gecici_dosya = DOSYA + ".tmp"

        with open(gecici_dosya, "w", encoding="utf-8") as dosya:
            json.dump(veri, dosya, ensure_ascii=False, indent=2)

        os.replace(gecici_dosya, DOSYA)
        return True

    except Exception as hata:
        st.error(f"Kaydetme hatası: {hata}")
        return False


def fotografi_base64_yap(fotograf):
    tampon = io.BytesIO()
    fotograf.save(tampon, format="JPEG", quality=90)
    return base64.b64encode(tampon.getvalue()).decode("utf-8")


def fotografi_ac(kayit):
    """Eski ve yeni kayıtların fotoğraflarını açmayı dener."""
    if not isinstance(kayit, dict):
        return None

    resim = kayit.get("resim")

    if not resim:
        return None

    try:
        veri = base64.b64decode(resim)
        return Image.open(io.BytesIO(veri)).convert("RGB")
    except Exception:
        return None


def kayit_turu(kayit):
    if not isinstance(kayit, dict):
        return "Diğer"

    tur = kayit.get("tur", "Diğer")

    if tur in KIYAFET_TURLERI:
        return tur

    return "Diğer"


def kayit_rengi(kayit):
    if not isinstance(kayit, dict):
        return "Bilinmiyor"

    renk = kayit.get("renk", "Bilinmiyor")

    return renk if renk in RENKLER else "Bilinmiyor"


# =========================================================
# YAPAY ZEKÂ İLE KIYAFET TÜRÜ TAHMİNİ
# =========================================================

@st.cache_resource(show_spinner=False)
def clip_modelini_yukle():
    """Model ilk gerektiğinde yüklenir."""
    try:
        from transformers import pipeline

        return pipeline(
            "zero-shot-image-classification",
            model="openai/clip-vit-base-patch32"
        )
    except Exception:
        return None


def kiyafet_turu_tahmin_et(fotograf):
    """
    CLIP modeliyle kıyafet türünü tahmin eder.
    Model yüklenemezse None döndürür.
    """
    model = clip_modelini_yukle()

    if model is None:
        return None

    etiketler = {
        "Üst giyim": "a photo of a shirt, blouse or top",
        "Alt giyim": "a photo of trousers, jeans or a skirt",
        "Elbise": "a photo of a dress",
        "Dış giyim": "a photo of a coat or jacket",
        "Ayakkabı": "a photo of shoes or sneakers",
        "Çanta": "a photo of a handbag or backpack",
        "Aksesuar": "a photo of jewelry, a belt or an accessory",
        "Diğer": "a photo of another fashion item"
    }

    try:
        tahminler = model(
            fotograf,
            candidate_labels=list(etiketler.values())
        )

        en_iyi = tahminler[0]["label"]

        for tur, etiket in etiketler.items():
            if etiket == en_iyi:
                return tur

    except Exception:
        return None

    return None


# =========================================================
# RENK ALGILAMA
# =========================================================

def piksel_rengini_siniflandir(rgb):
    """Bir RGB pikselini yaklaşık renk adına dönüştürür."""
    r, g, b = [int(x) for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)

    if v < 0.19:
        return "Siyah"

    if s < 0.15:
        if v > 0.88:
            return "Beyaz"
        if v < 0.48:
            return "Gri"
        return "Gri"

    derece = h * 360

    if derece < 12 or derece >= 345:
        return "Kırmızı"
    elif derece < 25:
        return "Turuncu"
    elif derece < 43:
        return "Sarı"
    elif derece < 80:
        return "Yeşil"
    elif derece < 165:
        return "Yeşil"
    elif derece < 205:
        return "Mavi"
    elif derece < 255:
        return "Lacivert" if v < 0.55 else "Mavi"
    elif derece < 290:
        return "Mor"
    elif derece < 330:
        return "Pembe"
    else:
        return "Kırmızı"


def renk_bul(fotograf):
    """
    Fotoğrafın merkez bölümündeki renkleri yaklaşık olarak analiz eder.
    İki renk belirgin ve birbirine yakınsa ikisini birlikte döndürür.
    """
    try:
        resim = ImageOps.exif_transpose(fotograf).convert("RGB")

        genislik, yukseklik = resim.size

        # Kenarlardaki arka plan etkisini azalt.
        sol = int(genislik * 0.10)
        ust = int(yukseklik * 0.10)
        sag = int(genislik * 0.90)
        alt = int(yukseklik * 0.90)

        resim = resim.crop((sol, ust, sag, alt))
        resim.thumbnail((120, 120))

        pikseller = np.asarray(resim).reshape(-1, 3)

        # Hesaplamayı hızlandırmak için örnekleme.
        if len(pikseller) > 4000:
            adim = max(1, len(pikseller) // 4000)
            pikseller = pikseller[::adim]

        renk_sayilari = Counter(
            piksel_rengini_siniflandir(piksel)
            for piksel in pikseller
        )

        toplam = max(1, sum(renk_sayilari.values()))
        sirali = renk_sayilari.most_common()

        en_baskin, adet = sirali[0]
        baskin_oran = adet / toplam

        # Pembe-siyah gibi iki renkli ayakkabıları yakala.
        ikinci_renk = None
        ikinci_oran = 0

        if len(sirali) > 1:
            ikinci_renk, ikinci_adet = sirali[1]
            ikinci_oran = ikinci_adet / toplam

        if (
            {en_baskin, ikinci_renk} == {"Pembe", "Siyah"}
            and ikinci_oran >= 0.15
            and baskin_oran <= ikinci_oran * 1.8
        ):
            return "Pembe-Siyah"

        if (
            {en_baskin, ikinci_renk} == {"Siyah", "Beyaz"}
            and ikinci_oran >= 0.20
        ):
            return "Siyah-Beyaz"

        # İki renk de belirginse ikisini birlikte göster.
        if (
            ikinci_renk
            and ikinci_renk != en_baskin
            and ikinci_oran >= 0.23
            and baskin_oran <= ikinci_oran * 1.7
        ):
            return f"{en_baskin}-{ikinci_renk}"

        if baskin_oran < 0.20:
            return "Çok renkli"

        return en_baskin

    except Exception:
        return "Bilinmiyor"


# =========================================================
# KOMBİN OLUŞTURMA
# =========================================================

def renkleri_ayir(renk):
    if not renk or renk == "Bilinmiyor":
        return []

    return [
        parca.strip()
        for parca in renk.replace("/", "-").split("-")
        if parca.strip()
    ]


def renk_uyumu(renk1, renk2):
    """Basit, açıklanabilir renk uyumu kontrolü."""
    renk1 = kayit_rengi({"renk": renk1})
    renk2 = kayit_rengi({"renk": renk2})

    nötrler = {
        "Siyah", "Beyaz", "Gri", "Bej",
        "Kahverengi", "Lacivert"
    }

    if renk1 == "Bilinmiyor" or renk2 == "Bilinmiyor":
        return True

    renk1_liste = renkleri_ayir(renk1)
    renk2_liste = renkleri_ayir(renk2)

    if set(renk1_liste) & set(renk2_liste):
        return True

    if any(r in nötrler for r in renk1_liste + renk2_liste):
        return True

    uyum_gruplari = [
        {"Mavi", "Lacivert", "Beyaz"},
        {"Pembe", "Gri", "Beyaz"},
        {"Yeşil", "Bej", "Kahverengi"},
        {"Kırmızı", "Siyah", "Beyaz"},
        {"Mor", "Gri", "Siyah"},
        {"Sarı", "Mavi", "Lacivert"},
        {"Turuncu", "Mavi", "Beyaz"}
    ]

    for grup in uyum_gruplari:
        if set(renk1_liste + renk2_liste).issubset(grup):
            return True

    return False


def uyumlu_mu(parca, secilenler):
    return all(
        renk_uyumu(parca.get("renk"), secilen.get("renk"))
        for secilen in secilenler
    )


def kombin_oner(dolap):
    """Dolaptaki mevcut parçalardan bir kombin oluşturmaya çalışır."""
    gecerli = [
        p for p in dolap
        if isinstance(p, dict) and fotografi_ac(p) is not None
    ]

    ustler = [p for p in gecerli if kayit_turu(p) == "Üst giyim"]
    altlar = [p for p in gecerli if kayit_turu(p) == "Alt giyim"]
    elbiseler = [p for p in gecerli if kayit_turu(p) == "Elbise"]

    dis_giyim = [p for p in gecerli if kayit_turu(p) == "Dış giyim"]
    ayakkabilar = [p for p in gecerli if kayit_turu(p) == "Ayakkabı"]
    cantalar = [p for p in gecerli if kayit_turu(p) == "Çanta"]
    aksesuarlar = [p for p in gecerli if kayit_turu(p) == "Aksesuar"]

    sonuc = []

    # Elbise varsa öncelikle elbise temelli kombin dene.
    if elbiseler:
        sonuc = [elbiseler[0]]

    elif ustler and altlar:
        en_iyi = None
        en_iyi_puan = -1

        for ust in ustler:
            for alt in altlar:
                puan = 1 if renk_uyumu(
                    ust.get("renk"), alt.get("renk")
                ) else 0

                if puan > en_iyi_puan:
                    en_iyi = [ust, alt]
                    en_iyi_puan = puan

        sonuc = en_iyi or [ustler[0], altlar[0]]

    elif ustler:
        sonuc = [ustler[0]]

    elif altlar:
        sonuc = [altlar[0]]

    if not sonuc:
        return [], "Kombin oluşturmak için dolabına önce kıyafet eklemelisin."

    # Önce ana kombinle renk uyumuna bakan tamamlayıcı parçaları ekle.
    for grup in [dis_giyim, ayakkabilar, cantalar, aksesuarlar]:
        adaylar = [
            parca for parca in grup
            if parca not in sonuc and uyumlu_mu(parca, sonuc)
        ]

        if adaylar:
            sonuc.append(adaylar[0])

    aciklama = kombin_aciklamasi(sonuc, gecerli)
    return sonuc, aciklama


def kombin_aciklamasi(kombin, dolap):
    if not kombin:
        return "Henüz kombin önerilemiyor."

    renkler = [
        kayit_rengi(p)
        for p in kombin
        if kayit_rengi(p) != "Bilinmiyor"
    ]

    if renkler:
        renk_metni = ", ".join(dict.fromkeys(renkler))
        metin = f"Bu kombin {renk_metni} renklerini bir araya getiriyor."
    else:
        metin = "Bu kombin dolabındaki mevcut parçalardan oluşturuldu."

    turler = [kayit_turu(p) for p in kombin]
    eksikler = []

    if "Elbise" not in turler and "Alt giyim" not in turler:
        eksikler.append("pantolon veya etek")

    if "Ayakkabı" not in turler:
        eksikler.append("kombinin renklerine uygun ayakkabı")

    if "Çanta" not in turler:
        eksikler.append("küçük bir çanta")

    if "Aksesuar" not in turler:
        eksikler.append("minimal takılar veya bir kemer")

    if "Dış giyim" not in turler:
        eksikler.append("hava serinse uyumlu bir ceket")

    if eksikler:
        metin += "\n\n**Tamamlayıcı öneriler:** " + ", ".join(eksikler) + " ekleyebilirsin."

    return metin


# =========================================================
# TEK KARELİK KOMBİN KOLAJI
# =========================================================

def kombin_kolaji_olustur(kombin, boyut=1000):
    """
    Kıyafetlerin gerçek fotoğraflarını tek karelik kolajda birleştirir.
    """
    tuval = Image.new("RGB", (boyut, boyut), "#f5f1ed")
    cizim = ImageDraw.Draw(tuval)

    try:
        baslik_font = ImageFont.truetype("arial.ttf", 35)
        etiket_font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        baslik_font = ImageFont.load_default()
        etiket_font = ImageFont.load_default()

    cizim.text(
        (boyut // 2, 30),
        "AKILLI DOLABIM",
        fill="#333333",
        font=baslik_font,
        anchor="mt"
    )

    if not kombin:
        return tuval

    kenar = 35
    baslik_yuksekligi = 95
    alt_bosluk = 35
    alan_yuksekligi = boyut - baslik_yuksekligi - alt_bosluk
    sutun = 2
    satir = (len(kombin) + sutun - 1) // sutun

    hucre_genislik = (boyut - 2 * kenar) // sutun
    hucre_yukseklik = alan_yuksekligi // max(1, satir)

    for i, parca in enumerate(kombin):
        resim = fotografi_ac(parca)

        if resim is None:
            continue

        sat = i // sutun
        sut = i % sutun

        x0 = kenar + sut * hucre_genislik
        y0 = baslik_yuksekligi + sat * hucre_yukseklik

        kutu_bosluk = 12
        kutu = (
            x0 + kutu_bosluk,
            y0 + kutu_bosluk,
            x0 + hucre_genislik - kutu_bosluk,
            y0 + hucre_yukseklik - kutu_bosluk
        )

        cizim.rounded_rectangle(
            kutu,
            radius=18,
            fill="#ffffff",
            outline="#e4ddd6",
            width=2
        )

        etiket_yuksekligi = 45
        resim_alani = (
            kutu[0] + 10,
            kutu[1] + 10,
            kutu[2] - 10,
            kutu[3] - etiket_yuksekligi
        )

        max_w = resim_alani[2] - resim_alani[0]
        max_h = resim_alani[3] - resim_alani[1]

        resim.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

        rx = resim_alani[0] + (max_w - resim.width) // 2
        ry = resim_alani[1] + (max_h - resim.height) // 2

        tuval.paste(resim, (rx, ry))

        tur = kayit_turu(parca)
        renk = kayit_rengi(parca)
        etiket = f"{tur} • {renk}"

        cizim.text(
            (
                (kutu[0] + kutu[2]) // 2,
                kutu[3] - etiket_yuksekligi // 2
            ),
            etiket,
            fill="#333333",
            font=etiket_font,
            anchor="mm"
        )

    return tuval


# =========================================================
# ARAYÜZ
# =========================================================

st.title("👗 Akıllı Dolabım")
st.caption("Kıyafetlerini düzenle, dolabını keşfet ve kombinlerini oluştur.")

dolap = dolabi_yukle()

sayfa = st.sidebar.radio(
    "Menü",
    ["Kıyafet Ekle", "Dolabım", "Kombin Öner"]
)

# =========================================================
# KIYAFET EKLE
# =========================================================

if sayfa == "Kıyafet Ekle":
    st.header("📸 Dolabına kıyafet ekle")
    st.write("Bir kıyafet fotoğrafı yükle; türünü ve rengini belirleyelim.")

    yuklenen = st.file_uploader(
        "Kıyafet fotoğrafını seç",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if yuklenen is not None:
        try:
            fotograf = Image.open(yuklenen)
            fotograf = ImageOps.exif_transpose(fotograf).convert("RGB")

            st.image(
                fotograf,
                caption="Yüklediğin fotoğraf",
                width=300
            )

            if st.button("🤖 Kıyafeti analiz et", type="primary"):
                with st.spinner("Kıyafet analiz ediliyor..."):
                    tahmin_turu = kiyafet_turu_tahmin_et(fotograf)
                    tahmin_rengi = renk_bul(fotograf)

                st.session_state["gecici_fotograf"] = fotografi_base64_yap(fotograf)
                st.session_state["gecici_tur"] = (
                    tahmin_turu if tahmin_turu else "Üst giyim"
                )
                st.session_state["gecici_renk"] = (
                    tahmin_rengi if tahmin_rengi in RENKLER else "Bilinmiyor"
                )
                st.session_state["analiz_tamam"] = True

                if tahmin_turu:
                    st.success(f"Tahmin edilen tür: {tahmin_turu}")
                else:
                    st.warning(
                        "Yapay zekâ modeli yüklenemedi. Türü aşağıdan "
                        "kendin seçebilirsin."
                    )

                st.info(f"Tahmin edilen renk: {tahmin_rengi}")

            if st.session_state.get("analiz_tamam", False):
                st.subheader("Kıyafet bilgilerini kontrol et")

                secili_tur = st.selectbox(
                    "Kıyafet türü",
                    KIYAFET_TURLERI,
                    index=(
                        KIYAFET_TURLERI.index(
                            st.session_state.get("gecici_tur", "Üst giyim")
                        )
                    )
                )

                tahmin_renk = st.session_state.get(
                    "gecici_renk", "Bilinmiyor"
                )

                secili_renk = st.selectbox(
                    "Kıyafet rengi",
                    RENKLER,
                    index=(
                        RENKLER.index(tahmin_renk)
                        if tahmin_renk in RENKLER else len(RENKLER) - 1
                    )
                )

                if st.button("➕ Dolabıma kaydet", type="primary"):
                    yeni_kayit = {
                        "dosya_adi": yuklenen.name,
                        "tur": secili_tur,
                        "renk": secili_renk,
                        "resim": st.session_state["gecici_fotograf"],
                        "eklenme_tarihi": datetime.now().isoformat(timespec="seconds")
                    }

                    dolap.append(yeni_kayit)

                    if dolabi_kaydet(dolap):
                        st.session_state["analiz_tamam"] = False
                        st.session_state.pop("gecici_fotograf", None)
                        st.success("Kıyafet dolabına kaydedildi!")
                        st.rerun()

        except Exception as hata:
            st.error(f"Fotoğraf işlenemedi: {hata}")


# =========================================================
# DOLABIM
# =========================================================

elif sayfa == "Dolabım":
    st.header("🧥 Dolabım")

    if not dolap:
        st.info("Dolabında henüz kıyafet yok. Kıyafet Ekle bölümünden başlayabilirsin.")
    else:
        st.write(f"Toplam **{len(dolap)}** kayıt")

        filtre = st.selectbox(
            "Türüne göre filtrele",
            ["Tümü"] + KIYAFET_TURLERI
        )

        for i, parca in enumerate(dolap):
            if not isinstance(parca, dict):
                continue

            tur = kayit_turu(parca)

            if filtre != "Tümü" and tur != filtre:
                continue

            with st.container(border=True):
                sol, sag = st.columns([1, 2])

                with sol:
                    fotograf = fotografi_ac(parca)
                    if fotograf is not None:
                        st.image(fotograf, use_container_width=True)
                    else:
                        st.warning("Fotoğraf açılamadı.")

                with sag:
                    st.write(f"**Dosya:** {parca.get('dosya_adi', 'Kıyafet')}")
                    st.write(f"**Mevcut tür:** {tur}")
                    st.write(f"**Renk:** {kayit_rengi(parca)}")

                    yeni_tur = st.selectbox(
                        "Türü düzelt",
                        KIYAFET_TURLERI,
                        index=(
                            KIYAFET_TURLERI.index(tur)
                            if tur in KIYAFET_TURLERI else len(KIYAFET_TURLERI) - 1
                        ),
                        key=f"tur_{i}"
                    )

                    yeni_renk = st.selectbox(
                        "Rengi düzelt",
                        RENKLER,
                        index=(
                            RENKLER.index(kayit_rengi(parca))
                            if kayit_rengi(parca) in RENKLER
                            else len(RENKLER) - 1
                        ),
                        key=f"renk_{i}"
                    )

                    kaydet_sol, sil_sag = st.columns(2)

                    with kaydet_sol:
                        if st.button("💾 Düzeltmeleri kaydet", key=f"kaydet_{i}"):
                            dolap[i]["tur"] = yeni_tur
                            dolap[i]["renk"] = yeni_renk

                            if dolabi_kaydet(dolap):
                                st.success("Kıyafet bilgileri güncellendi.")
                                st.rerun()

                    with sil_sag:
                        if st.button("🗑️ Kıyafeti sil", key=f"sil_{i}"):
                            st.session_state["silme_onayi"] = i

                    if st.session_state.get("silme_onayi") == i:
                        st.warning("Bu kıyafeti silmek istediğinden emin misin?")
                        evet, hayir = st.columns(2)

                        with evet:
                            if st.button("Evet, sil", key=f"evet_{i}"):
                                dolap.pop(i)

                                if dolabi_kaydet(dolap):
                                    st.session_state.pop("silme_onayi", None)
                                    st.success("Kıyafet silindi.")
                                    st.rerun()

                        with hayir:
                            if st.button("Vazgeç", key=f"hayir_{i}"):
                                st.session_state.pop("silme_onayi", None)
                                st.rerun()


# =========================================================
# KOMBİN ÖNER
# =========================================================

elif sayfa == "Kombin Öner":
    st.header("✨ Kombin Öner")
    st.write(
        "Dolabındaki kıyafetlerden bir kombin oluşturup parçaları "
        "tek karelik bir görselde birleştirelim."
    )

    if not dolap:
        st.info("Önce Kıyafet Ekle bölümünden dolabına birkaç parça ekle.")
    else:
        if st.button("✨ Kombinimi oluştur", type="primary"):
            kombin, aciklama = kombin_oner(dolap)

            st.session_state["son_kombin"] = kombin
            st.session_state["son_aciklama"] = aciklama

        kombin = st.session_state.get("son_kombin", [])
        aciklama = st.session_state.get("son_aciklama", "")

        if kombin:
            kolaj = kombin_kolaji_olustur(kombin)

            sol, sag = st.columns([1.2, 1])

            with sol:
                st.image(
                    kolaj,
                    caption="Dolabındaki parçalarla hazırlanan kombin kolajı",
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

                st.subheader("👚 Kombindeki parçalar")

                for parca in kombin:
                    st.write(
                        f"- **{kayit_turu(parca)}** — "
                        f"{kayit_rengi(parca)}"
                    )

        elif aciklama:
            st.info(aciklama)

st.sidebar.divider()
st.sidebar.caption("Akıllı Dolabım • Yapay zekâ destekli kombin asistanı")
