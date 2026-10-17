
import streamlit as st
import json
import os
import base64
import io
import colorsys

import numpy as np
from PIL import Image
from transformers import pipeline

# --------------------------------------------------
# SAYFA AYARLARI
# --------------------------------------------------

st.set_page_config(
    page_title="Akıllı Dolabım",
    page_icon="👗",
    layout="wide"
)

st.title("👗 Akıllı Dolabım")
st.write(
    "Kıyafetlerini dolabına ekle, renklerini keşfet "
    "ve yapay zekâ destekli kombinler oluştur."
)

DOSYA = "dolap.json"

KIYAFET_TURLERI = [
    "Üst giyim",
    "Alt giyim",
    "Elbise",
    "Ayakkabı",
    "Dış giyim",
    "Çanta",
    "Aksesuar"
]

# --------------------------------------------------
# YAPAY ZEKÂ MODELLERİ
# --------------------------------------------------

@st.cache_resource(show_spinner="Yapay zekâ modelleri yükleniyor...")
def modelleri_yukle():
    tur_modeli = pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32"
    )

    bolge_modeli = pipeline(
        "image-segmentation",
        model="mattmdjaga/segformer_b2_clothes"
    )

    return tur_modeli, bolge_modeli


# --------------------------------------------------
# DOLAP VERİLERİ
# --------------------------------------------------

def dolabi_yukle():
    if not os.path.exists(DOSYA):
        return []

    try:
        with open(DOSYA, "r", encoding="utf-8") as dosya:
            veriler = json.load(dosya)

        if not isinstance(veriler, list):
            st.error("dolap.json beklenen biçimde değil. Dosyayı silme.")
            return []

        duzeltilmis = []

        for sira, parca in enumerate(veriler, start=1):
            if not isinstance(parca, dict):
                st.warning(
                    f"{sira}. kayıt beklenen biçimde değil; "
                    "bu kayıt korunarak görüntüleme atlandı."
                )
                continue

            kayit = dict(parca)

            kayit.setdefault("dosya_adi", f"Kıyafet {sira}")
            kayit.setdefault("tur", "Üst giyim")
            kayit.setdefault("renk", "Bilinmiyor")
            kayit.setdefault("resim", "")

            if not isinstance(kayit["dosya_adi"], str) or not kayit["dosya_adi"].strip():
                kayit["dosya_adi"] = f"Kıyafet {sira}"

            if kayit["tur"] not in KIYAFET_TURLERI:
                kayit["tur"] = "Üst giyim"

            if not isinstance(kayit["renk"], str) or not kayit["renk"].strip():
                kayit["renk"] = "Bilinmiyor"

            if not isinstance(kayit["resim"], str):
                kayit["resim"] = ""

            duzeltilmis.append(kayit)

        return duzeltilmis

    except (json.JSONDecodeError, OSError) as hata:
        st.error("Dolap verileri okunamadı. dolap.json dosyanı kontrol et.")
        st.code(str(hata))
        return []


def dolabi_kaydet(veriler):
    gecici_dosya = DOSYA + ".tmp"

    with open(gecici_dosya, "w", encoding="utf-8") as dosya:
        json.dump(veriler, dosya, ensure_ascii=False, indent=2)

    os.replace(gecici_dosya, DOSYA)


def resim_base64(resim):
    tampon = io.BytesIO()
    resim.convert("RGB").save(tampon, format="JPEG")
    return base64.b64encode(tampon.getvalue()).decode("utf-8")


def base64_resim_ac(veri):
    if not veri:
        raise ValueError("Bu kayıtta fotoğraf verisi bulunmuyor.")

    ham_veri = base64.b64decode(veri)
    return Image.open(io.BytesIO(ham_veri)).convert("RGB")


def parca_adi(parca, sira=None):
    ad = parca.get("dosya_adi")

    if isinstance(ad, str) and ad.strip():
        return ad

    return f"Kıyafet {sira}" if sira is not None else "İsimsiz kıyafet"


# --------------------------------------------------
# KIYAFET TÜRÜ TESPİTİ
# --------------------------------------------------

def kiyafet_turu_bul(resim, tur_modeli):
    etiketler = [
        "a photograph of a T-shirt or shirt",
        "a photograph of trousers or jeans",
        "a photograph of a skirt",
        "a photograph of a dress",
        "a photograph of shoes or sneakers",
        "a photograph of a coat or jacket",
        "a photograph of a handbag or backpack",
        "a photograph of an accessory such as a hat or scarf"
    ]

    sonuc = tur_modeli(resim, candidate_labels=etiketler)
    etiket = sonuc[0]["label"].lower()

    if "shoes" in etiket or "sneakers" in etiket:
        return "Ayakkabı"
    if "trousers" in etiket or "jeans" in etiket:
        return "Alt giyim"
    if "skirt" in etiket:
        return "Alt giyim"
    if "dress" in etiket:
        return "Elbise"
    if "coat" in etiket or "jacket" in etiket:
        return "Dış giyim"
    if "handbag" in etiket or "backpack" in etiket:
        return "Çanta"
    if "accessory" in etiket:
        return "Aksesuar"

    return "Üst giyim"


# --------------------------------------------------
# RENK ANALİZİ
# --------------------------------------------------

def hsv_pikselleri(resim, maske=None, maksimum=12000):
    rgb = np.asarray(
        resim.convert("RGB").resize((120, 120))
    )
    pikseller = rgb.reshape(-1, 3).astype(np.float32) / 255.0

    if maske is not None:
        maske_resmi = Image.fromarray(
            (maske.astype(np.uint8) * 255)
        ).resize((120, 120), Image.Resampling.NEAREST)

        maske_dizisi = np.asarray(maske_resmi).reshape(-1) > 0

        if np.any(maske_dizisi):
            pikseller = pikseller[maske_dizisi]

    if len(pikseller) == 0:
        pikseller = rgb.reshape(-1, 3).astype(np.float32) / 255.0

    if len(pikseller) > maksimum:
        secilenler = np.linspace(
            0, len(pikseller) - 1, maksimum
        ).astype(int)
        pikseller = pikseller[secilenler]

    return np.array([
        colorsys.rgb_to_hsv(float(r), float(g), float(b))
        for r, g, b in pikseller
    ])


def pembe_siyah_kontrol(hsv):
    ton = hsv[:, 0]
    doygunluk = hsv[:, 1]
    parlaklik = hsv[:, 2]

    pembe = (
        (ton >= 0.86) & (ton < 0.95) &
        (doygunluk >= 0.28) & (parlaklik >= 0.35)
    )

    kirmizi = (
        ((ton < 0.04) | (ton >= 0.97)) &
        (doygunluk >= 0.28) & (parlaklik >= 0.25)
    )

    siyah = parlaklik < 0.20

    pembe_sayisi = int(np.sum(pembe))
    kirmizi_sayisi = int(np.sum(kirmizi))
    siyah_sayisi = int(np.sum(siyah))

    renkli_sayisi = int(
        np.sum((doygunluk >= 0.20) & (parlaklik >= 0.20))
    )

    if renkli_sayisi == 0:
        return False

    pembe_orani = pembe_sayisi / renkli_sayisi
    siyah_orani = siyah_sayisi / len(hsv)
    kirmizi_orani = kirmizi_sayisi / renkli_sayisi

    if kirmizi_orani >= 0.20:
        return False

    return (
        pembe_sayisi >= 20
        and siyah_sayisi >= 20
        and pembe_orani >= 0.15
        and siyah_orani >= 0.10
        and pembe_sayisi > kirmizi_sayisi * 1.5
    )


def piksel_renk_sinifi(r, g, b):
    h, s, v = colorsys.rgb_to_hsv(
        r / 255.0, g / 255.0, b / 255.0
    )

    if v < 0.20:
        return "Siyah"

    if s < 0.18:
        return "Beyaz" if v > 0.90 else "Gri"

    if h < 0.04 or h >= 0.97:
        return "Kırmızı"
    if h < 0.10:
        return "Turuncu"
    if h < 0.17:
        return "Sarı"
    if h < 0.43:
        return "Yeşil"
    if h < 0.56:
        return "Turkuaz"
    if h < 0.72:
        return "Mavi"
    if h < 0.82:
        return "Mor"
    if h < 0.95:
        return "Pembe"

    return "Kırmızı"


def renk_bul(resim, bolge_modeli, kiyafet_turu):
    analiz_resmi = resim.convert("RGB")
    maske = None

    try:
        bolgeler = bolge_modeli(analiz_resmi)

        if kiyafet_turu == "Ayakkabı":
            ayakkabi_maskeleri = []

            for bolge in bolgeler:
                etiket = bolge.get("label", "").lower()

                if "shoe" in etiket:
                    maske_resmi = bolge["mask"].convert("L")
                    maske_dizisi = np.asarray(maske_resmi) > 127

                    if np.sum(maske_dizisi) > 20:
                        ayakkabi_maskeleri.append(maske_resmi)

            if ayakkabi_maskeleri:
                birlesik = np.zeros(
                    (analiz_resmi.height, analiz_resmi.width),
                    dtype=bool
                )

                for maske_resmi in ayakkabi_maskeleri:
                    maske_resmi = maske_resmi.resize(
                        analiz_resmi.size,
                        Image.Resampling.NEAREST
                    )
                    birlesik |= np.asarray(maske_resmi) > 127

                maske = birlesik

    except Exception:
        maske = None

    hsv = hsv_pikselleri(analiz_resmi, maske)

    if kiyafet_turu == "Ayakkabı" and pembe_siyah_kontrol(hsv):
        return "Pembe-Siyah"

    renkli = hsv[
        (hsv[:, 1] >= 0.18) &
        (hsv[:, 2] >= 0.20)
    ]
    hsv_kullan = renkli if len(renkli) > 0 else hsv

    siniflar = []

    for h, s, v in hsv_kullan:
        r, g, b = colorsys.hsv_to_rgb(float(h), float(s), float(v))
        siniflar.append(
            piksel_renk_sinifi(
                int(r * 255), int(g * 255), int(b * 255)
            )
        )

    if not siniflar:
        return "Bilinmiyor"

    sayim = {}
    for sinif in siniflar:
        sayim[sinif] = sayim.get(sinif, 0) + 1

    return max(sayim, key=sayim.get)


# --------------------------------------------------
# KOMBİN ÖNERİLERİ
# --------------------------------------------------

def kombin_icin_renk_uyumu(renk1, renk2):
    temel_uyumlar = {
        "Siyah": ["Beyaz", "Gri", "Kırmızı", "Pembe", "Mavi", "Yeşil", "Sarı", "Mor", "Turuncu", "Siyah"],
        "Beyaz": ["Siyah", "Gri", "Mavi", "Pembe", "Kırmızı", "Yeşil", "Mor", "Turuncu", "Sarı", "Beyaz"],
        "Gri": ["Siyah", "Beyaz", "Pembe", "Kırmızı", "Mavi", "Mor", "Yeşil", "Gri"],
        "Pembe": ["Siyah", "Beyaz", "Gri", "Mavi", "Mor", "Pembe", "Kırmızı"],
        "Kırmızı": ["Siyah", "Beyaz", "Gri", "Mavi", "Kırmızı"],
        "Mavi": ["Beyaz", "Siyah", "Gri", "Pembe", "Mavi", "Turuncu", "Sarı"],
        "Yeşil": ["Beyaz", "Siyah", "Gri", "Sarı", "Yeşil"],
        "Sarı": ["Siyah", "Beyaz", "Mavi", "Gri", "Sarı"],
        "Mor": ["Beyaz", "Siyah", "Gri", "Pembe", "Mor"],
        "Turuncu": ["Siyah", "Beyaz", "Mavi", "Gri", "Turuncu"],
        "Turkuaz": ["Beyaz", "Siyah", "Gri", "Mavi", "Turkuaz"],
        "Pembe-Siyah": ["Beyaz", "Gri", "Siyah", "Pembe", "Mavi"],
        "Bilinmiyor": ["Siyah", "Beyaz", "Gri"]
    }

    if renk1 == "Pembe-Siyah":
        return renk2 in temel_uyumlar["Pembe-Siyah"]
    if renk2 == "Pembe-Siyah":
        return renk1 in temel_uyumlar["Pembe-Siyah"]

    return renk2 in temel_uyumlar.get(renk1, [])


def kombin_aciklamasi(parcalar, ortam):
    renkler = [p.get("renk", "Bilinmiyor") for p in parcalar]
    turler = [p.get("tur", "Üst giyim") for p in parcalar]

    if ortam == "Günlük":
        stil = "Rahat, kullanışlı ve günlük hayata uygun"
        ortam_metni = "okul, arkadaş buluşması veya günlük işler"
    elif ortam == "Şık":
        stil = "Dengeli, özenli ve daha şık"
        ortam_metni = "akşam buluşması veya özel bir etkinlik"
    else:
        stil = "Konforlu ve sportif"
        ortam_metni = "yürüyüş, gezi veya hareketli bir gün"

    if len(set(renkler)) == 1:
        renk_aciklama = (
            f"{renkler[0]} tonları bir araya gelerek "
            "bütünlüklü bir görünüm oluşturuyor."
        )
    else:
        renk_aciklama = (
            f"{', '.join(renkler)} renkleri bir arada kullanılıyor. "
            "Nötr renkler, diğer renklerin dengeli görünmesine yardımcı olur."
        )

    return {
        "stil": stil,
        "ortam": ortam_metni,
        "renk": renk_aciklama,
        "ayakkabi": (
            "Ayakkabı, kombinin alt kısmını tamamlıyor."
            if "Ayakkabı" in turler
            else "Ayakkabı ekleyerek kombini tamamlayabilirsin."
        ),
        "aksesuar": "Sade bir çanta ve minimal takılar görünümü tamamlayabilir.",
        "alternatif": (
            "Daha farklı bir görünüm için parçalardan birini "
            "siyah, beyaz veya gri gibi nötr bir renkle değiştirebilirsin."
        )
    }


def kombin_oner(dolap, ortam, adet=3):
    ustler = [x for x in dolap if x.get("tur") == "Üst giyim"]
    altlar = [x for x in dolap if x.get("tur") == "Alt giyim"]
    elbiseler = [x for x in dolap if x.get("tur") == "Elbise"]
    ayakkabilar = [x for x in dolap if x.get("tur") == "Ayakkabı"]

    kombinler = []

    for ust in ustler:
        for alt in altlar:
            if not kombin_icin_renk_uyumu(
                ust.get("renk", "Bilinmiyor"),
                alt.get("renk", "Bilinmiyor")
            ):
                continue

            if ayakkabilar:
                for ayakkabi in ayakkabilar:
                    if (
                        kombin_icin_renk_uyumu(
                            ust.get("renk", "Bilinmiyor"),
                            ayakkabi.get("renk", "Bilinmiyor")
                        )
                        or kombin_icin_renk_uyumu(
                            alt.get("renk", "Bilinmiyor"),
                            ayakkabi.get("renk", "Bilinmiyor")
                        )
                    ):
                        parcalar = [ust, alt, ayakkabi]
                        kombinler.append({
                            "parcalar": parcalar,
                            "aciklama": kombin_aciklamasi(parcalar, ortam)
                        })
            else:
                parcalar = [ust, alt]
                kombinler.append({
                    "parcalar": parcalar,
                    "aciklama": kombin_aciklamasi(parcalar, ortam)
                })

    for elbise in elbiseler:
        if ayakkabilar:
            for ayakkabi in ayakkabilar:
                if kombin_icin_renk_uyumu(
                    elbise.get("renk", "Bilinmiyor"),
                    ayakkabi.get("renk", "Bilinmiyor")
                ):
                    parcalar = [elbise, ayakkabi]
                    kombinler.append({
                        "parcalar": parcalar,
                        "aciklama": kombin_aciklamasi(parcalar, ortam)
                    })
        else:
            parcalar = [elbise]
            kombinler.append({
                "parcalar": parcalar,
                "aciklama": kombin_aciklamasi(parcalar, ortam)
            })

    benzersiz = []
    gorulen = set()

    for kombin in kombinler:
        anahtar = tuple(sorted(
            p.get("dosya_adi") or p.get("resim", "")[:40]
            for p in kombin["parcalar"]
        ))
        if anahtar not in gorulen:
            gorulen.add(anahtar)
            benzersiz.append(kombin)

    return benzersiz[:adet]


# --------------------------------------------------
# MODELİ BAŞLAT
# --------------------------------------------------

with st.spinner("Akıllı Dolabım hazırlanıyor..."):
    try:
        tur_modeli, bolge_modeli = modelleri_yukle()
    except Exception as hata:
        st.error(
            "Yapay zekâ modelleri yüklenemedi. "
            "İnternet bağlantısını ve requirements.txt dosyasını kontrol et."
        )
        st.code(str(hata))
        st.stop()

dolap = dolabi_yukle()

# --------------------------------------------------
# YAN MENÜ
# --------------------------------------------------

st.sidebar.header("📌 Menü")

sayfa = st.sidebar.radio(
    "Ne yapmak istiyorsun?",
    ["Kıyafet Ekle", "Dolabım", "Kombin Öner"]
)

# --------------------------------------------------
# KIYAFET EKLE
# --------------------------------------------------

if sayfa == "Kıyafet Ekle":
    st.header("📸 Dolabına kıyafet ekle")

    yuklenen = st.file_uploader(
        "Kıyafet fotoğrafını seç",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if yuklenen is not None:
        try:
            resim = Image.open(yuklenen).convert("RGB")
        except Exception:
            st.error("Bu fotoğraf açılamadı. JPG, PNG veya WEBP dosyası dene.")
            st.stop()

        st.image(resim, caption="Yüklediğin fotoğraf", width=300)

        tur_secimi = st.selectbox(
            "Kıyafet türü",
            ["Yapay zekâ belirlesin"] + KIYAFET_TURLERI
        )

        if st.button("✨ Kıyafeti analiz et ve dolaba ekle"):
            try:
                with st.spinner("Kıyafet analiz ediliyor..."):
                    if tur_secimi == "Yapay zekâ belirlesin":
                        tur = kiyafet_turu_bul(resim, tur_modeli)
                    else:
                        tur = tur_secimi

                    renk = renk_bul(resim, bolge_modeli, tur)

                    kayit = {
                        "dosya_adi": yuklenen.name,
                        "tur": tur,
                        "renk": renk,
                        "resim": resim_base64(resim)
                    }

                    # Güncel kayıtları diskten alıp ekle; eski kıyafetleri ezme.
                    guncel_dolap = dolabi_yukle()
                    guncel_dolap.append(kayit)
                    dolabi_kaydet(guncel_dolap)

                st.success("Kıyafet dolabına eklendi!")
                st.write(f"**Kıyafet türü:** {tur}")
                st.write(f"**Algılanan renk:** {renk}")

                if tur == "Ayakkabı" and renk == "Pembe-Siyah":
                    st.caption(
                        "İki renkli ayakkabı kontrolü uygulandı. "
                        "Sonuç fotoğrafın ışığına ve renklerine göre değişebilir."
                    )

                st.rerun()

            except Exception as hata:
                st.error("Kıyafet analiz edilirken bir hata oluştu.")
                st.code(str(hata))


# --------------------------------------------------
# DOLABIM
# --------------------------------------------------

elif sayfa == "Dolabım":
    st.header("🧥 Dolabımdaki kıyafetler")

    if not dolap:
        st.info(
            "Dolabın henüz boş. Kıyafet Ekle bölümünden "
            "ilk fotoğrafını yükleyebilirsin."
        )
    else:
        filtre = st.selectbox(
            "Kıyafet türüne göre filtrele",
            ["Tümü"] + KIYAFET_TURLERI
        )

        goruntulenecek = dolap

        if filtre != "Tümü":
            goruntulenecek = [
                x for x in dolap
                if x.get("tur", "Üst giyim") == filtre
            ]

        if not goruntulenecek:
            st.info("Bu türde kayıtlı kıyafet yok. Filtreyi 'Tümü' yapmayı dene.")
        else:
            sutun_sayisi = 3

            for baslangic in range(0, len(goruntulenecek), sutun_sayisi):
                sutunlar = st.columns(sutun_sayisi)

                for i, sutun in enumerate(sutunlar):
                    sira = baslangic + i

                    if sira >= len(goruntulenecek):
                        continue

                    parca = goruntulenecek[sira]
                    ad = parca_adi(parca, sira + 1)

                    with sutun:
                        try:
                            st.image(
                                base64_resim_ac(parca.get("resim", "")),
                                use_container_width=True
                            )
                        except Exception:
                            st.warning(
                                "Bu kaydın fotoğrafı eksik veya açılamıyor. "
                                "Kayıt otomatik olarak silinmedi."
                            )

                        st.markdown(f"**{ad}**")
                        st.write(f"Tür: {parca.get('tur', 'Üst giyim')}")
                        st.write(f"Renk: {parca.get('renk', 'Bilinmiyor')}")

                        if st.button("🗑️ Sil", key=f"sil_{sira}_{ad}"):
                            try:
                                hedef_indeks = dolap.index(parca)
                                dolap.pop(hedef_indeks)
                                dolabi_kaydet(dolap)
                                st.rerun()
                            except (ValueError, OSError) as hata:
                                st.error("Kıyafet silinemedi.")
                                st.code(str(hata))


# --------------------------------------------------
# KOMBİN ÖNER
# --------------------------------------------------

elif sayfa == "Kombin Öner":
    st.header("✨ Sana özel kombinler")

    if not dolap:
        st.info("Önce dolabına birkaç kıyafet eklemelisin.")
    else:
        ortam = st.selectbox(
            "Kombini hangi ortam için istiyorsun?",
            ["Günlük", "Şık", "Sportif"]
        )

        adet = st.slider(
            "Kaç kombin gösterilsin?",
            min_value=1,
            max_value=10,
            value=3
        )

        if st.button("👗 Kombinlerimi oluştur"):
            kombinler = kombin_oner(dolap, ortam, adet)

            if not kombinler:
                st.warning(
                    "Henüz uygun kombin oluşturulamadı. "
                    "Dolabına üst giyim, alt giyim veya elbise "
                    "ve mümkünse ayakkabı eklemeyi dene."
                )
            else:
                for sira, kombin in enumerate(kombinler, start=1):
                    st.divider()
                    st.subheader(f"💖 Kombin {sira}")

                    parcalar = kombin["parcalar"]
                    aciklama = kombin["aciklama"]
                    sutunlar = st.columns(len(parcalar))

                    for parca_sira, (sutun, parca) in enumerate(
                        zip(sutunlar, parcalar), start=1
                    ):
                        with sutun:
                            try:
                                st.image(
                                    base64_resim_ac(parca.get("resim", "")),
                                    use_container_width=True
                                )
                            except Exception:
                                st.warning("Bu parçanın fotoğrafı açılamadı.")

                            st.markdown(f"**{parca.get('tur', 'Üst giyim')}**")
                            st.write(f"Renk: {parca.get('renk', 'Bilinmiyor')}")
                            st.caption(parca_adi(parca, parca_sira))

                    st.markdown("#### 🪄 Kombin açıklaması")
                    st.write(aciklama["stil"] + " bir kombin.")

                    st.markdown("**🎯 Nerede giyebilirsin?**")
                    st.write(
                        "Bu kombin " + aciklama["ortam"] + " için uygun olabilir."
                    )

                    st.markdown("**🎨 Renk uyumu**")
                    st.write(aciklama["renk"])

                    st.markdown("**👟 Ayakkabı**")
                    st.write(aciklama["ayakkabi"])

                    st.markdown("**👜 Aksesuar önerisi**")
                    st.write(aciklama["aksesuar"])

                    st.markdown("**🔄 Alternatif öneri**")
                    st.write(aciklama["alternatif"])

st.sidebar.divider()
st.sidebar.caption("Akıllı Dolabım • Yapay zekâ destekli kombin asistanı")
