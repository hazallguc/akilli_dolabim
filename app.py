
import streamlit as st
from PIL import Image
from transformers import pipeline
import io
import json
import os
import base64
import colorsys
import numpy as np

# ==================================================
# SAYFA AYARLARI
# ==================================================

st.set_page_config(
    page_title="Akıllı Dolabım",
    page_icon="👗",
    layout="wide"
)

st.title("👗 Akıllı Dolabım")
st.write(
    "Yapay zekâ ile kıyafetlerini tanı, dolabını düzenle "
    "ve görselli, detaylı kombin önerileri al."
)

DOLAP_DOSYASI = "dolap.json"


# ==================================================
# DOLABI KAYDETME VE YÜKLEME
# ==================================================

def dolabi_kaydet(dolap):
    try:
        with open(DOLAP_DOSYASI, "w", encoding="utf-8") as dosya:
            json.dump(dolap, dosya, ensure_ascii=False)
    except Exception as hata:
        st.warning(f"Dolap kaydedilemedi: {hata}")


def dolabi_yukle():
    if os.path.exists(DOLAP_DOSYASI):
        try:
            with open(DOLAP_DOSYASI, "r", encoding="utf-8") as dosya:
                veri = json.load(dosya)
                return veri if isinstance(veri, list) else []
        except Exception:
            return []
    return []


def resmi_base64_yap(resim):
    tampon = io.BytesIO()
    resim.convert("RGB").save(tampon, format="JPEG", quality=92)
    return base64.b64encode(tampon.getvalue()).decode("utf-8")


def base64_resim_ac(veri):
    return Image.open(
        io.BytesIO(base64.b64decode(veri))
    ).convert("RGB")


# ==================================================
# YAPAY ZEKÂ MODELLERİ
# ==================================================

@st.cache_resource
def modelleri_yukle():
    kiyafet_modeli = pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32"
    )

    segmentasyon_modeli = pipeline(
        "image-segmentation",
        model="mattmdjaga/segformer_b2_clothes"
    )

    return kiyafet_modeli, segmentasyon_modeli


KIYAFETLER = [
    "a T-shirt",
    "a shirt",
    "a sweater",
    "a jacket",
    "a coat",
    "a dress",
    "a skirt",
    "a pair of trousers",
    "a pair of jeans",
    "a pair of shorts",
    "a pair of shoes",
    "a pair of sneakers",
    "a handbag",
    "a hat",
    "a scarf"
]

KIYAFET_CEVIRI = {
    "a T-shirt": "Tişört",
    "a shirt": "Gömlek",
    "a sweater": "Kazak",
    "a jacket": "Ceket",
    "a coat": "Mont / Kaban",
    "a dress": "Elbise",
    "a skirt": "Etek",
    "a pair of trousers": "Pantolon",
    "a pair of jeans": "Kot Pantolon",
    "a pair of shorts": "Şort",
    "a pair of shoes": "Ayakkabı",
    "a pair of sneakers": "Spor Ayakkabı",
    "a handbag": "Çanta",
    "a hat": "Şapka",
    "a scarf": "Atkı"
}


# ==================================================
# RENK ALGILAMA
# ==================================================

def renk_sayaci_olustur():
    return {
        "Siyah": 0,
        "Beyaz": 0,
        "Gri": 0,
        "Pembe": 0,
        "Kırmızı": 0,
        "Turuncu": 0,
        "Sarı": 0,
        "Yeşil": 0,
        "Mavi": 0,
        "Lacivert": 0,
        "Mor": 0,
        "Kahverengi": 0
    }


def piksel_renk_sinifi(r, g, b):
    r, g, b = r / 255, g / 255, b / 255
    ton, doygunluk, parlaklik = colorsys.rgb_to_hsv(r, g, b)

    if parlaklik < 0.20:
        return "Siyah"

    if doygunluk < 0.16:
        if parlaklik > 0.82:
            return "Beyaz"
        return "Gri"

    # Pembe tonları
    if 0.82 <= ton < 0.97:
        return "Pembe"

    if 0.75 <= ton < 0.82:
        return "Mor"

    if ton < 0.04 or ton >= 0.97:
        return "Kırmızı"

    if ton < 0.10:
        return "Kahverengi" if parlaklik < 0.60 else "Turuncu"

    if ton < 0.20:
        return "Kahverengi" if parlaklik < 0.60 else "Sarı"

    if ton < 0.43:
        return "Yeşil"

    if ton < 0.55:
        return "Mavi"

    if ton < 0.72:
        return "Lacivert" if parlaklik < 0.55 else "Mavi"

    return "Mor"


def piksellerin_renklerini_say(pikseller):
    sayac = renk_sayaci_olustur()

    for piksel in pikseller:
        renk = piksel_renk_sinifi(
            float(piksel[0]),
            float(piksel[1]),
            float(piksel[2])
        )
        sayac[renk] += 1

    return sayac


def pembe_siyah_kontrol(pikseller):
    """
    Pembe ve siyahı ayrı ayrı sayar.
    Gri ve beyaz pikseller iki rengin oranını
    hesaplarken paydaya dahil edilmez.
    """
    pembe = 0
    siyah = 0

    for piksel in pikseller:
        r, g, b = [float(x) / 255 for x in piksel[:3]]
        ton, doygunluk, parlaklik = colorsys.rgb_to_hsv(r, g, b)

        if (
            0.82 <= ton < 0.97
            and doygunluk >= 0.22
            and parlaklik >= 0.22
        ):
            pembe += 1

        elif parlaklik < 0.20:
            siyah += 1

    renkli_toplam = pembe + siyah

    if renkli_toplam == 0:
        return False

    return (
        pembe >= 10
        and siyah >= 10
        and pembe / renkli_toplam >= 0.10
        and siyah / renkli_toplam >= 0.10
    )


def segmentasyon_maskesi_al(sonuc, resim, tur):
    """
    Ayakkabı analizinde sol ve sağ ayakkabı maskelerini
    birlikte kullanır. Böylece arka planın renk sonucunu
    bozma ihtimalini azaltır.
    """
    maskeler = []

    ayakkabi_mi = "Ayakkabı" in (tur or "")

    kiyafet_etiketleri = {
        "Tişört": ["upper-clothes"],
        "Gömlek": ["upper-clothes"],
        "Kazak": ["upper-clothes"],
        "Ceket": ["upper-clothes", "coat"],
        "Mont / Kaban": ["coat", "upper-clothes"],
        "Elbise": ["dress"],
        "Etek": ["skirt"],
        "Pantolon": ["pants"],
        "Kot Pantolon": ["pants"],
        "Şort": ["pants"],
        "Ayakkabı": ["shoe"],
        "Spor Ayakkabı": ["shoe"],
        "Çanta": ["bag"],
        "Şapka": ["hat"],
        "Atkı": ["scarf"]
    }

    kabul_edilenler = (
        ["shoe"]
        if ayakkabi_mi
        else kiyafet_etiketleri.get(tur, [])
    )

    for parca in sonuc:
        etiket = str(parca.get("label", "")).lower().replace("_", "-")

        # Ayakkabı maskeleri left-shoe / right-shoe
        # şeklinde gelebilir.
        uygun = any(kelime in etiket for kelime in kabul_edilenler)

        if not uygun:
            continue

        maske = parca.get("mask")

        if maske is None:
            continue

        if not isinstance(maske, Image.Image):
            maske = Image.fromarray(np.array(maske))

        maske = maske.convert("L").resize(resim.size)
        maske_dizisi = np.array(maske) > 127

        if maske_dizisi.any():
            maskeler.append(maske_dizisi)

    if not maskeler:
        return None

    birlesik = np.logical_or.reduce(maskeler)

    # Aşırı küçük veya neredeyse tüm fotoğrafı kaplayan
    # maskeler güvenilir kabul edilmez.
    oran = float(birlesik.mean())

    if oran < 0.005 or oran > 0.95:
        return None

    return birlesik


def renk_bul(resim, segmentasyon_modeli=None, tur=None):
    resim = resim.convert("RGB")
    dizi = np.array(resim)
    h, w, _ = dizi.shape

    pikseller = None

    if segmentasyon_modeli is not None:
        try:
            sonuc = segmentasyon_modeli(resim)

            maske = segmentasyon_maskesi_al(
                sonuc,
                resim,
                tur
            )

            if maske is not None:
                pikseller = dizi[maske]

        except Exception:
            pikseller = None

    # Maske bulunamazsa resmin merkezine odaklan.
    # Fotoğrafın kenarları çoğunlukla arka plan içerir.
    if pikseller is None or len(pikseller) == 0:
        orta = dizi[
            int(h * 0.15):int(h * 0.85),
            int(w * 0.15):int(w * 0.85)
        ]

        pikseller = orta.reshape(-1, 3)

    if len(pikseller) > 20000:
        adim = max(1, len(pikseller) // 20000)
        pikseller = pikseller[::adim]

    # Özel iki renk kontrolü, genel renk seçiminden önce.
    if pembe_siyah_kontrol(pikseller):
        return "Pembe-Siyah"

    sayac = piksellerin_renklerini_say(pikseller)

    # Gri baskın görünse bile pembe ve siyah birlikteyse
    # iki renkli ürün etiketini koru.
    if pembe_siyah_kontrol(pikseller):
        return "Pembe-Siyah"

    return max(sayac, key=sayac.get)


# ==================================================
# KOMBİN ÖNERİSİ
# ==================================================

UYUMLU_RENKLER = {
    "Siyah": [
        "Beyaz", "Gri", "Pembe", "Kırmızı",
        "Mavi", "Lacivert", "Kot Pantolon", "Pembe-Siyah"
    ],
    "Beyaz": [
        "Siyah", "Mavi", "Lacivert", "Pembe",
        "Kahverengi", "Pembe-Siyah", "Kot Pantolon", "Gri"
    ],
    "Gri": [
        "Siyah", "Beyaz", "Pembe", "Mavi",
        "Kırmızı", "Pembe-Siyah", "Lacivert"
    ],
    "Pembe": [
        "Siyah", "Beyaz", "Gri", "Lacivert",
        "Kot Pantolon", "Pembe-Siyah"
    ],
    "Pembe-Siyah": [
        "Beyaz", "Gri", "Siyah", "Kot Pantolon", "Pembe"
    ],
    "Lacivert": [
        "Beyaz", "Gri", "Pembe", "Kahverengi", "Kot Pantolon"
    ],
    "Mavi": [
        "Beyaz", "Gri", "Siyah", "Kahverengi", "Pembe"
    ],
    "Kahverengi": [
        "Beyaz", "Bej", "Krem", "Lacivert", "Kot Pantolon"
    ],
    "Bej": ["Beyaz", "Kahverengi", "Siyah", "Lacivert"],
    "Kırmızı": ["Siyah", "Beyaz", "Gri", "Lacivert"],
    "Yeşil": ["Beyaz", "Siyah", "Bej", "Kahverengi"],
    "Mor": ["Siyah", "Beyaz", "Gri"],
    "Sarı": ["Siyah", "Beyaz", "Lacivert"],
    "Turuncu": ["Siyah", "Beyaz", "Kot Pantolon"]
}


def renk_uyumlu_mu(renk1, renk2):
    return (
        renk1 == renk2
        or renk2 in UYUMLU_RENKLER.get(renk1, [])
        or renk1 in UYUMLU_RENKLER.get(renk2, [])
    )


def kombin_oner(dolap):
    ust_turleri = ["Tişört", "Gömlek", "Kazak", "Ceket"]
    alt_turleri = ["Pantolon", "Kot Pantolon", "Etek", "Şort"]

    ustler = [x for x in dolap if x.get("tur") in ust_turleri]
    altlar = [x for x in dolap if x.get("tur") in alt_turleri]

    ayakkabilar = [
        x for x in dolap
        if "Ayakkabı" in x.get("tur", "")
    ]

    elbiseler = [x for x in dolap if x.get("tur") == "Elbise"]

    oneriler = []

    for ust in ustler:
        for alt in altlar:
            ust_rengi = ust.get("renk", "Bilinmiyor")
            alt_rengi = alt.get("renk", "Bilinmiyor")

            if not renk_uyumlu_mu(ust_rengi, alt_rengi):
                continue

            if ust_rengi == alt_rengi:
                renk_aciklamasi = (
                    "Benzer renkler sade ve bütünlüklü bir görünüm sağlar."
                )
            else:
                renk_aciklamasi = (
                    f"{ust_rengi} üst ile {alt_rengi} alt arasında "
                    "tamamlayıcı bir renk dengesi oluşturabilirsin."
                )

            ayakkabi = next(
                (
                    x for x in ayakkabilar
                    if renk_uyumlu_mu(
                        x.get("renk", "Bilinmiyor"),
                        ust_rengi
                    ) or renk_uyumlu_mu(
                        x.get("renk", "Bilinmiyor"),
                        alt_rengi
                    )
                ),
                ayakkabilar[0] if ayakkabilar else None
            )

            parcalar = [ust, alt]

            if ayakkabi:
                parcalar.append(ayakkabi)

            if ust["tur"] == "Gömlek":
                stil = "Smart casual"
                ortam = "Okul, kafe ve günlük buluşmalar"
                aksesuar = (
                    "İnce bir kemer, saat ve sade bir çanta "
                    "tercih edebilirsin."
                )
            elif ust["tur"] == "Kazak":
                stil = "Rahat ve klasik"
                ortam = "Serin havalar, okul ve günlük geziler"
                aksesuar = (
                    "Atkı, omuz çantası veya minimal takılar "
                    "kombini tamamlayabilir."
                )
            elif "Kot" in alt["tur"] or ust["tur"] == "Tişört":
                stil = "Günlük ve sportif"
                ortam = "Okul, alışveriş ve arkadaş buluşmaları"
                aksesuar = (
                    "Sırt çantası, saat veya küçük bir kolye "
                    "kullanabilirsin."
                )
            else:
                stil = "Günlük ve dengeli"
                ortam = "Günlük kullanım ve sosyal buluşmalar"
                aksesuar = (
                    "Küçük bir çanta ve sade takılarla görünümü "
                    "tamamlayabilirsin."
                )

            if ayakkabi and ayakkabi.get("renk") == "Pembe-Siyah":
                ayakkabi_notu = (
                    "Pembe-siyah ayakkabıyı kombinin vurgu parçası "
                    "olarak kullan. Siyah, beyaz veya gri bir üst "
                    "renkleri dengeli biçimde bir araya getirebilir."
                )
            elif ayakkabi:
                ayakkabi_notu = (
                    f"{ayakkabi.get('renk', 'Nötr')} ayakkabı, "
                    "kombinin genel renk dengesini tamamlayabilir."
                )
            else:
                ayakkabi_notu = (
                    "Nötr renkli bir ayakkabı seçerek kombini "
                    "tamamlayabilirsin."
                )

            oneriler.append({
                "baslik": f"{stil} kombin",
                "parcalar": parcalar,
                "stil": stil,
                "ortam": ortam,
                "renk_aciklamasi": renk_aciklamasi,
                "ayakkabi_notu": ayakkabi_notu,
                "aksesuar": aksesuar,
                "alternatif": (
                    f"{ust['tur']} yerine farklı bir üst deneyebilir "
                    f"veya {alt['tur']} ile daha açık ya da koyu "
                    "bir renk kullanabilirsin."
                )
            })

    for elbise in elbiseler:
        renk = elbise.get("renk", "Bilinmiyor")

        ayakkabi = next(
            (
                x for x in ayakkabilar
                if renk_uyumlu_mu(
                    renk,
                    x.get("renk", "Bilinmiyor")
                )
            ),
            ayakkabilar[0] if ayakkabilar else None
        )

        parcalar = [elbise]

        if ayakkabi:
            parcalar.append(ayakkabi)

        oneriler.append({
            "baslik": "Elbise kombini",
            "parcalar": parcalar,
            "stil": "Feminen ve dengeli",
            "ortam": "Günlük buluşmalar veya uygun aksesuarlarla davetler",
            "renk_aciklamasi": (
                f"{renk} elbiseyi ana parça olarak kullan. "
                "Diğer parçaları sade tutarak bütünlük oluştur."
            ),
            "ayakkabi_notu": (
                f"{ayakkabi.get('renk', '')} ayakkabı ile elbiseyi "
                "tamamlayabilirsin."
                if ayakkabi
                else "Elbiseyle uyumlu bir ayakkabı seçebilirsin."
            ),
            "aksesuar": "Küçük bir çanta ve zarif takılar tercih edebilirsin.",
            "alternatif": (
                "Farklı bir çanta veya dış giyim parçasıyla "
                "daha sportif ya da klasik bir görünüm deneyebilirsin."
            )
        })

    # Aynı parçaları kullanan aynı kombinleri tekrarlama
    benzersiz = []
    gorulenler = set()

    for oner in oneriler:
        anahtar = tuple(
            (
                p.get("dosya_adi", ""),
                p.get("tur", ""),
                p.get("renk", "")
            )
            for p in oner["parcalar"]
        )

        if anahtar not in gorulenler:
            gorulenler.add(anahtar)
            benzersiz.append(oner)

    if not benzersiz:
        benzersiz.append({
            "baslik": "Yeni renk eşleşmeleri dene",
            "parcalar": [],
            "stil": "Deneysel",
            "ortam": "Günlük kullanım",
            "renk_aciklamasi": (
                "Mevcut parçalar arasında doğrudan eşleşme bulunamadı. "
                "Bu, kıyafetlerinin kesinlikle uyumsuz olduğu anlamına "
                "gelmez. Siyah, beyaz, gri ve kot tonlarıyla yeni "
                "eşleştirmeler deneyebilirsin."
            ),
            "ayakkabi_notu": (
                "Dolabındaki ayakkabıları farklı üst ve altlarla deneyebilirsin."
            ),
            "aksesuar": "Sade bir çanta veya minimal takılar kullanabilirsin.",
            "alternatif": (
                "Dolabına farklı renklerde üst veya alt parçalar ekleyerek "
                "kombin seçeneklerini artırabilirsin."
            )
        })

    return benzersiz[:10]


# ==================================================
# OTURUM VE MEVCUT DOLAP
# ==================================================

if "dolap" not in st.session_state:
    st.session_state["dolap"] = dolabi_yukle()

dolap = st.session_state["dolap"]


# ==================================================
# FOTOĞRAF YÜKLEME
# ==================================================

st.subheader("📸 Dolabına kıyafet ekle")

yuklenenler = st.file_uploader(
    "Kıyafet veya ayakkabı fotoğraflarını seç",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="kiyafet_yukleyici"
)

if st.button("✨ Fotoğrafları analiz et ve dolaba ekle"):
    if not yuklenenler:
        st.warning("Önce en az bir fotoğraf yükle.")
    else:
        with st.spinner("Yapay zekâ modelleri yükleniyor ve fotoğraflar analiz ediliyor..."):
            try:
                kiyafet_modeli, segmentasyon_modeli = modelleri_yukle()

                eklenen = 0
                atlanan = 0

                for dosya in yuklenenler:
                    try:
                        resim = Image.open(dosya).convert("RGB")

                        if any(
                            p.get("dosya_adi") == dosya.name
                            for p in st.session_state["dolap"]
                        ):
                            atlanan += 1
                            continue

                        tahminler = kiyafet_modeli(
                            resim,
                            candidate_labels=KIYAFETLER
                        )

                        tahmin = tahminler[0]["label"]
                        tur = KIYAFET_CEVIRI.get(tahmin, "Diğer")

                        renk = renk_bul(
                            resim,
                            segmentasyon_modeli,
                            tur
                        )

                        st.session_state["dolap"].append({
                            "dosya_adi": dosya.name,
                            "tur": tur,
                            "renk": renk,
                            "resim": resmi_base64_yap(resim)
                        })

                        eklenen += 1

                    except Exception as hata:
                        st.warning(
                            f"{dosya.name} analiz edilemedi: {hata}"
                        )

                dolabi_kaydet(st.session_state["dolap"])

                if eklenen:
                    st.success(f"{eklenen} fotoğraf dolabına eklendi.")

                if atlanan:
                    st.info(
                        f"{atlanan} fotoğraf aynı dosya adıyla "
                        "zaten dolapta olduğu için eklenmedi."
                    )

            except Exception as hata:
                st.error(
                    "Yapay zekâ modelleri yüklenemedi. "
                    f"Hata ayrıntısı: {hata}"
                )


# ==================================================
# ESKİ GRİ AYAKKABILARI OTOMATİK KONTROL ET
# BUTON YOK. OTURUMDA YALNIZCA BİR KEZ ÇALIŞIR.
# ==================================================

if "eski_ayakkabi_kontrol_edildi" not in st.session_state:
    st.session_state["eski_ayakkabi_kontrol_edildi"] = True

    gri_ayakkabilar = [
        p for p in st.session_state["dolap"]
        if "Ayakkabı" in p.get("tur", "")
        and p.get("renk") == "Gri"
        and p.get("resim")
    ]

    if gri_ayakkabilar:
        try:
            with st.spinner("Mevcut ayakkabı renkleri kontrol ediliyor..."):
                _, segmentasyon_modeli = modelleri_yukle()

                for parca in gri_ayakkabilar:
                    try:
                        resim = base64_resim_ac(parca["resim"])
                        yeni_renk = renk_bul(
                            resim,
                            segmentasyon_modeli,
                            parca.get("tur")
                        )
                        parca["renk"] = yeni_renk
                    except Exception:
                        pass

                dolabi_kaydet(st.session_state["dolap"])

        except Exception:
            # Model yüklenemezse mevcut dolap verilerini koru.
            pass


# ==================================================
# DOLABI GÖSTER
# ==================================================

st.divider()
st.subheader("🧥 Benim Dolabım")

dolap = st.session_state["dolap"]

if not dolap:
    st.info("Dolabın henüz boş. Fotoğraf yükleyerek başlayabilirsin.")
else:
    st.write(f"Dolabında toplam **{len(dolap)} parça** var.")

    sutunlar = st.columns(3)
    silinecek = None

    for i, parca in enumerate(dolap):
        with sutunlar[i % 3]:
            try:
                st.image(
                    base64_resim_ac(parca["resim"]),
                    use_container_width=True
                )
            except Exception:
                st.warning("Bu fotoğraf açılamadı.")

            st.write(f"**Tür:** {parca.get('tur', 'Bilinmiyor')}")
            st.write(f"**Renk:** {parca.get('renk', 'Bilinmiyor')}")
            st.caption(parca.get("dosya_adi", ""))

            if st.button("🗑️ Dolabımdan kaldır", key=f"sil_{i}"):
                silinecek = i

    if silinecek is not None:
        st.session_state["dolap"].pop(silinecek)
        dolabi_kaydet(st.session_state["dolap"])
        st.rerun()


# ==================================================
# GÖRSELLİ VE DETAYLI KOMBİN ÖNERİLERİ
# ==================================================

st.divider()
st.subheader("💡 Sana Özel Kombin Önerileri")

dolap = st.session_state["dolap"]

if dolap:
    oneriler = kombin_oner(dolap)

    st.write(
        "Kombinlerin görsellerini inceleyebilir, parçaların neden "
        "uyumlu olduğunu ve nasıl tamamlayabileceğini öğrenebilirsin."
    )

    for i, oner in enumerate(oneriler, start=1):
        with st.container(border=True):
            st.markdown(f"### ✨ Kombin {i}: {oner['baslik']}")

            parcalar = oner["parcalar"]

            if parcalar:
                gorsel_sutunlari = st.columns(len(parcalar))

                for j, parca in enumerate(parcalar):
                    with gorsel_sutunlari[j]:
                        try:
                            st.image(
                                base64_resim_ac(parca["resim"]),
                                use_container_width=True
                            )
                        except Exception:
                            st.caption("Fotoğraf görüntülenemedi.")

                        st.markdown(
                            f"**{parca.get('tur', 'Parça')}**"
                        )
                        st.caption(
                            f"Renk: {parca.get('renk', 'Bilinmiyor')}"
                        )
            else:
                st.info(
                    "Henüz doğrudan eşleşen parça bulunamadı. "
                    "Dolabındaki renkleri birlikte değerlendirebilirsin."
                )

            st.divider()

            st.markdown(f"**👗 Stil:** {oner['stil']}")
            st.markdown(f"**📍 Kullanım ortamı:** {oner['ortam']}")

            st.markdown("**🎨 Renk uyumu**")
            st.write(oner["renk_aciklamasi"])

            st.markdown("**👟 Ayakkabı önerisi**")
            st.write(oner["ayakkabi_notu"])

            st.markdown("**👜 Aksesuar önerisi**")
            st.write(oner["aksesuar"])

            st.markdown("**🔁 Alternatif kombin**")
            st.write(oner["alternatif"])

    st.caption(
        "Kombinler dolabındaki gerçek fotoğraflar ve renk uyumu "
        "kuralları kullanılarak hazırlanır. Açıklamalar kural "
        "tabanlıdır; ayrı bir moda uzmanı modeli tarafından üretilmez."
    )
else:
    st.info("Kombin önerileri için önce dolabına kıyafet ekle.")


# ==================================================
# BİLGİ
# ==================================================

st.divider()
st.caption(
    "Dolap verileri dolap.json dosyasında saklanır. "
    "Streamlit Cloud üzerinde dosyanın kalıcılığı garanti edilmez."
)
