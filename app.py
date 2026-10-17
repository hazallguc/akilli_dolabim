
import streamlit as st
from PIL import Image
from transformers import pipeline
import io
import json
import os
import base64
import colorsys
import numpy as np

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
    "Yapay zekâ ile kıyafetlerini tanı, dolabını düzenle "
    "ve sana özel detaylı kombin önerileri al."
)

DOLAP_DOSYASI = "dolap.json"


# --------------------------------------------------
# DOLABI KAYDETME VE YÜKLEME
# --------------------------------------------------

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
                return json.load(dosya)
        except Exception:
            return []
    return []


def resmi_base64_yap(resim):
    tampon = io.BytesIO()
    resim.convert("RGB").save(tampon, format="JPEG")
    return base64.b64encode(tampon.getvalue()).decode("utf-8")


def base64_resim_ac(veri):
    return Image.open(
        io.BytesIO(base64.b64decode(veri))
    ).convert("RGB")


# --------------------------------------------------
# YAPAY ZEKÂ MODELLERİ
# --------------------------------------------------

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


# --------------------------------------------------
# RENK ALGILAMA YARDIMCILARI
# --------------------------------------------------

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


def piksellerin_renklerini_say(pikseller):
    sayac = renk_sayaci_olustur()

    for piksel in pikseller:
        r, g, b = [float(x) / 255 for x in piksel[:3]]
        ton, doygunluk, parlaklik = colorsys.rgb_to_hsv(r, g, b)

        # Siyah, beyaz ve gri
        if parlaklik < 0.20:
            sayac["Siyah"] += 1
            continue

        if doygunluk < 0.16:
            if parlaklik > 0.82:
                sayac["Beyaz"] += 1
            else:
                sayac["Gri"] += 1
            continue

        # Pembe
        if 0.82 <= ton < 0.97:
            sayac["Pembe"] += 1
            continue

        # Mor
        if 0.75 <= ton < 0.82:
            sayac["Mor"] += 1
            continue

        # Kırmızı
        if ton < 0.04 or ton >= 0.97:
            sayac["Kırmızı"] += 1
            continue

        # Turuncu ve kahverengi
        if ton < 0.10:
            if parlaklik < 0.60:
                sayac["Kahverengi"] += 1
            else:
                sayac["Turuncu"] += 1
            continue

        # Sarı
        if ton < 0.17:
            if parlaklik < 0.65:
                sayac["Kahverengi"] += 1
            else:
                sayac["Sarı"] += 1
            continue

        if ton < 0.20:
            if parlaklik < 0.55:
                sayac["Kahverengi"] += 1
            else:
                sayac["Sarı"] += 1
            continue

        # Yeşil, mavi ve lacivert
        if ton < 0.43:
            sayac["Yeşil"] += 1
        elif ton < 0.55:
            sayac["Mavi"] += 1
        elif ton < 0.72:
            if parlaklik < 0.55:
                sayac["Lacivert"] += 1
            else:
                sayac["Mavi"] += 1
        else:
            sayac["Mor"] += 1

    return sayac


def pembe_siyah_kontrol(pikseller):
    """
    İki renkli ayakkabılarda gri arka planın baskın
    gelmesini önlemek için renkli ve siyah pikselleri
    ayrı ayrı değerlendirir.
    """
    if len(pikseller) == 0:
        return False

    pembe = 0
    siyah = 0
    renkli_ve_siyah = 0

    for piksel in pikseller:
        r, g, b = [float(x) / 255 for x in piksel[:3]]
        ton, doygunluk, parlaklik = colorsys.rgb_to_hsv(r, g, b)

        # Pembe: belirgin doygunluk ve uygun renk tonu
        if (
            0.82 <= ton < 0.97
            and doygunluk >= 0.22
            and parlaklik >= 0.22
        ):
            pembe += 1
            renkli_ve_siyah += 1

        # Siyah: düşük parlaklık
        elif parlaklik < 0.20:
            siyah += 1
            renkli_ve_siyah += 1

    if renkli_ve_siyah == 0:
        return False

    # İki renk de belirgin miktarda bulunmalı.
    # Gri arka planın sayımı tek başına etkilememesi amaçlanır.
    pembe_orani = pembe / renkli_ve_siyah
    siyah_orani = siyah / renkli_ve_siyah

    return (
        pembe >= 20
        and siyah >= 20
        and pembe_orani >= 0.10
        and siyah_orani >= 0.10
    )


# --------------------------------------------------
# RENK ALGILAMA
# --------------------------------------------------

def renk_bul(resim, segmentasyon_modeli=None, tur=None):
    """
    Önce kıyafet maskesinden renkleri bulur.
    Ayakkabılarda öncelikle ayakkabı etiketli maskeleri
    kullanır. Maskeler uygun değilse görüntünün orta
    bölgesine geçer.
    """

    resim = resim.convert("RGB")
    resim_dizisi = np.array(resim)

    renk_pikselleri = None
    ayakkabi_mi = tur is not None and "Ayakkabı" in tur

    if segmentasyon_modeli is not None:
        try:
            sonuc = segmentasyon_modeli(resim)
            uygun_maskeler = []

            for parca in sonuc:
                etiket = str(parca.get("label", "")).lower()
                maske = parca.get("mask")

                if maske is None:
                    continue

                # Yüz, saç, ten ve arka planı kullanma
                if any(kelime in etiket for kelime in [
                    "background", "face", "hair", "skin",
                    "left-arm", "right-arm",
                    "left-leg", "right-leg"
                ]):
                    continue

                # Ayakkabı için yalnızca ayakkabı maskelerini kullan
                if ayakkabi_mi and "shoe" not in etiket:
                    continue

                if not ayakkabi_mi:
                    # Genel kıyafet kategorilerinde aksesuar
                    # veya ayakkabı maskelerini birleştirme.
                    kiyafet_etiketleri = [
                        "upper-clothes", "coat", "dress",
                        "skirt", "pants", "jumpsuit",
                        "scarf", "hat"
                    ]

                    if not any(k in etiket for k in kiyafet_etiketleri):
                        continue

                if not isinstance(maske, Image.Image):
                    maske = Image.fromarray(np.array(maske))

                maske = maske.convert("L").resize(resim.size)
                maske_dizisi = np.array(maske) > 127

                if maske_dizisi.any():
                    uygun_maskeler.append(maske_dizisi)

            if uygun_maskeler:
                birlesik_maske = np.logical_or.reduce(uygun_maskeler)
                aday_pikseller = resim_dizisi[birlesik_maske]

                # Maskenin resmin çok küçük bir kısmını kaplaması
                # veya neredeyse tüm resmi kapsaması şüphelidir.
                maske_orani = birlesik_maske.mean()

                if (
                    len(aday_pikseller) > 0
                    and 0.02 < maske_orani < 0.95
                ):
                    renk_pikselleri = aday_pikseller

        except Exception:
            renk_pikselleri = None

    # Maske bulunamadıysa orta bölgeyi kullan.
    # Ayakkabı fotoğraflarında ürünün merkezde olması beklenir.
    if renk_pikselleri is None or len(renk_pikselleri) == 0:
        h, w, _ = resim_dizisi.shape

        orta_bolge = resim_dizisi[
            int(h * 0.15):int(h * 0.85),
            int(w * 0.15):int(w * 0.85)
        ]

        renk_pikselleri = orta_bolge.reshape(-1, 3)

    # Performans için örnekleme
    if len(renk_pikselleri) > 20000:
        adim = max(1, len(renk_pikselleri) // 20000)
        renk_pikselleri = renk_pikselleri[::adim]

    # Pembe-siyah ayakkabı kontrolü, genel renk seçiminden önce
    if ayakkabi_mi and pembe_siyah_kontrol(renk_pikselleri):
        return "Pembe-Siyah"

    sayac = piksellerin_renklerini_say(renk_pikselleri)

    # Genel renk tespitinde de iki rengi kontrol et
    if pembe_siyah_kontrol(renk_pikselleri):
        return "Pembe-Siyah"

    toplam = sum(sayac.values())

    if toplam == 0:
        return "Bilinmiyor"

    sirali = sorted(
        sayac.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sirali[0][0]


# --------------------------------------------------
# DETAYLI KOMBİN ÖNERİSİ
# --------------------------------------------------

def kombin_oner(dolap):
    if len(dolap) < 2:
        return [{
            "baslik": "Dolabını oluşturmaya devam et",
            "parcalar": "Henüz yeterli parça yok.",
            "stil": "Başlangıç",
            "ortam": "Günlük",
            "aciklama": (
                "Üst ve alt giyimden en az birer parça "
                "eklediğinde daha anlamlı kombinler oluşturabiliriz."
            ),
            "aksesuar": "Minimal bir çanta veya saatle başlayabilirsin.",
            "alternatif": "Dolabına ikinci parçayı ekleyerek devam et."
        }]

    ustler = [
        x for x in dolap
        if x.get("tur") in [
            "Tişört", "Gömlek", "Kazak", "Ceket"
        ]
    ]

    altlar = [
        x for x in dolap
        if x.get("tur") in [
            "Pantolon", "Kot Pantolon", "Etek", "Şort"
        ]
    ]

    ayakkabilar = [
        x for x in dolap
        if "Ayakkabı" in x.get("tur", "")
    ]

    elbiseler = [
        x for x in dolap
        if x.get("tur") == "Elbise"
    ]

    uyumlu_renkler = {
        "Siyah": ["Beyaz", "Gri", "Pembe", "Kırmızı", "Mavi",
                  "Lacivert", "Kot Pantolon"],
        "Beyaz": ["Siyah", "Mavi", "Lacivert", "Pembe",
                  "Kahverengi", "Pembe-Siyah", "Kot Pantolon"],
        "Gri": ["Siyah", "Beyaz", "Pembe", "Mavi",
                "Kırmızı", "Pembe-Siyah", "Lacivert"],
        "Pembe": ["Siyah", "Beyaz", "Gri", "Lacivert",
                  "Kot Pantolon"],
        "Pembe-Siyah": ["Beyaz", "Gri", "Siyah",
                        "Kot Pantolon", "Pembe"],
        "Lacivert": ["Beyaz", "Gri", "Pembe",
                     "Kahverengi", "Kot Pantolon"],
        "Mavi": ["Beyaz", "Gri", "Siyah",
                 "Kahverengi", "Pembe"],
        "Kahverengi": ["Beyaz", "Bej", "Krem",
                       "Lacivert", "Kot Pantolon"],
        "Bej": ["Beyaz", "Kahverengi", "Siyah", "Lacivert"],
        "Kırmızı": ["Siyah", "Beyaz", "Gri", "Lacivert"],
        "Yeşil": ["Beyaz", "Siyah", "Bej", "Kahverengi"],
        "Mor": ["Siyah", "Beyaz", "Gri"],
        "Sarı": ["Siyah", "Beyaz", "Lacivert"],
        "Turuncu": ["Siyah", "Beyaz", "Kot Pantolon"],
        "Siyah-Beyaz": ["Siyah", "Beyaz", "Gri", "Kot Pantolon"]
    }

    def uyumlu_mu(renk1, renk2):
        return (
            renk1 == renk2
            or renk2 in uyumlu_renkler.get(renk1, [])
            or renk1 in uyumlu_renkler.get(renk2, [])
        )

    def ayakkabi_aciklamasi(renk):
        if renk == "Pembe-Siyah":
            return (
                "Pembe-siyah ayakkabıyı kombinin dikkat çekici "
                "parçası olarak kullan. Üst giyimde siyah, beyaz "
                "veya gri tercih ederek ayakkabıyla bütünlük sağla."
            )
        if renk in ["Siyah", "Beyaz", "Gri"]:
            return (
                f"{renk} ayakkabı, kombinin genel renk dengesini "
                "korumaya yardımcı olur."
            )
        return (
            f"{renk} ayakkabıyı kombinde vurgu rengi olarak "
            "kullanabilir, diğer parçaları daha sade tutabilirsin."
        )

    oneriler = []

    # Üst + alt kombinleri
    for ust in ustler:
        for alt in altlar:
            ust_rengi = ust.get("renk", "Bilinmiyor")
            alt_rengi = alt.get("renk", "Bilinmiyor")

            if not uyumlu_mu(ust_rengi, alt_rengi):
                continue

            if ust_rengi == alt_rengi:
                renk_aciklamasi = (
                    "Benzer renkleri kullanarak sade ve bütünlüklü "
                    "bir görünüm oluşturabilirsin."
                )
            else:
                renk_aciklamasi = (
                    f"{ust_rengi} ve {alt_rengi} birlikte kullanıldığında "
                    "renk kontrastı oluşturur. Parçalardan birini "
                    "ana renk, diğerini tamamlayıcı renk olarak düşün."
                )

            ayakkabi = ayakkabilar[0] if ayakkabilar else None

            parcalar = (
                f"{ust['tur']} ({ust_rengi}) + "
                f"{alt['tur']} ({alt_rengi})"
            )

            if ayakkabi:
                parcalar += (
                    f" + {ayakkabi['tur']} ({ayakkabi['renk']})"
                )

            if ust["tur"] == "Tişört" and "Kot" in alt["tur"]:
                stil = "Rahat ve sportif"
                ortam = "Günlük geziler, okul ve arkadaş buluşmaları"
                aksesuar = (
                    "Sade bir sırt çantası, saat veya küçük bir kolye "
                    "tercih edebilirsin."
                )
            elif ust["tur"] == "Gömlek":
                stil = "Smart casual"
                ortam = "Okul, kafe ve günlük buluşmalar"
                aksesuar = (
                    "İnce bir kemer, saat ve sade bir çanta "
                    "görünümü tamamlayabilir."
                )
            elif ust["tur"] == "Kazak":
                stil = "Rahat ve klasik"
                ortam = "Serin havalar, günlük geziler ve okul"
                aksesuar = (
                    "Atkı, sade bir omuz çantası veya minimal takılar "
                    "kullanabilirsin."
                )
            else:
                stil = "Günlük ve dengeli"
                ortam = "Günlük kullanım ve arkadaş buluşmaları"
                aksesuar = (
                    "Kombinin renklerini tekrar eden küçük bir aksesuar "
                    "seçerek bütünlük oluşturabilirsin."
                )

            if ayakkabi:
                ayakkabi_notu = ayakkabi_aciklamasi(ayakkabi["renk"])
            else:
                ayakkabi_notu = (
                    "Dolabında uygun bir ayakkabı varsa nötr renkli "
                    "bir modelle kombini tamamlayabilirsin."
                )

            alternatif = (
                f"{ust['tur']} yerine dolabındaki farklı bir üstü "
                f"deneyebilir veya {alt['tur']} ile daha açık ya da "
                "daha koyu bir renk kullanabilirsin."
            )

            oneriler.append({
                "baslik": f"{stil} kombin",
                "parcalar": parcalar,
                "stil": stil,
                "ortam": ortam,
                "aciklama": (
                    f"{renk_aciklamasi} "
                    f"{ayakkabi_notu}"
                ),
                "aksesuar": aksesuar,
                "alternatif": alternatif
            })

    # Elbise kombinleri
    for elbise in elbiseler:
        renk = elbise.get("renk", "Bilinmiyor")

        ayakkabi = None
        for aday in ayakkabilar:
            if uyumlu_mu(renk, aday.get("renk", "Bilinmiyor")):
                ayakkabi = aday
                break

        parcalar = f"Elbise ({renk})"

        if ayakkabi:
            parcalar += f" + {ayakkabi['tur']} ({ayakkabi['renk']})"

        oneriler.append({
            "baslik": "Elbise kombini",
            "parcalar": parcalar,
            "stil": "Feminen ve dengeli",
            "ortam": "Günlük buluşmalar veya uygun aksesuarlarla özel davetler",
            "aciklama": (
                f"{renk} elbiseyi ana parça olarak kullan. "
                "Görünümü sade tutarak elbisenin öne çıkmasını sağlayabilirsin. "
                + (
                    ayakkabi_aciklamasi(ayakkabi["renk"])
                    if ayakkabi else
                    "Elbiseyle uyumlu bir ayakkabı seçerek kombini tamamla."
                )
            ),
            "aksesuar": (
                "Küçük bir çanta, zarif küpeler veya ince bir kolye "
                "tercih edebilirsin."
            ),
            "alternatif": (
                "Aynı elbiseyi farklı bir çanta veya dış giyim parçasıyla "
                "daha sportif ya da daha klasik hale getirebilirsin."
            )
        })

    # Benzer kombinleri tekrar etme ve önerileri sınırla
    benzersiz = []
    gorulenler = set()

    for oner in oneriler:
        anahtar = oner["parcalar"]

        if anahtar not in gorulenler:
            gorulenler.add(anahtar)
            benzersiz.append(oner)

    if not benzersiz:
        benzersiz.append({
            "baslik": "Yeni bir renk uyumu dene",
            "parcalar": "Dolabındaki parçaları birlikte değerlendir.",
            "stil": "Deneysel",
            "ortam": "Günlük kullanım",
            "aciklama": (
                "Mevcut renk eşleştirmelerinde doğrudan uyum bulunamadı. "
                "Bu, parçaların kesinlikle uyumsuz olduğu anlamına gelmez. "
                "Siyah, beyaz, gri veya kot gibi nötr parçalarla yeni "
                "eşleştirmeler deneyebilirsin."
            ),
            "aksesuar": (
                "Sade bir çanta veya ayakkabı ile renk yoğunluğunu dengele."
            ),
            "alternatif": (
                "Dolabına farklı renklerde üst veya alt giyim ekleyerek "
                "daha fazla seçenek oluşturabilirsin."
            )
        })

    return benzersiz[:10]


# --------------------------------------------------
# OTURUM VE DOLAP
# --------------------------------------------------

if "dolap" not in st.session_state:
    st.session_state["dolap"] = dolabi_yukle()

dolap = st.session_state["dolap"]


# --------------------------------------------------
# FOTOĞRAF YÜKLEME
# --------------------------------------------------

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
        with st.spinner(
            "Yapay zekâ modelleri yükleniyor ve fotoğraflar analiz ediliyor..."
        ):
            try:
                kiyafet_modeli, segmentasyon_modeli = modelleri_yukle()

                eklenen = 0
                atlanan = 0

                for dosya in yuklenenler:
                    try:
                        resim = Image.open(dosya).convert("RGB")

                        if any(
                            parca.get("dosya_adi") == dosya.name
                            for parca in st.session_state["dolap"]
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

                        yeni_parca = {
                            "dosya_adi": dosya.name,
                            "tur": tur,
                            "renk": renk,
                            "resim": resmi_base64_yap(resim)
                        }

                        st.session_state["dolap"].append(yeni_parca)
                        eklenen += 1

                    except Exception as hata:
                        st.warning(
                            f"{dosya.name} analiz edilemedi: {hata}"
                        )

                dolabi_kaydet(st.session_state["dolap"])

                if eklenen:
                    st.success(f"{eklenen} fotoğraf dolabına eklendi!")

                if atlanan:
                    st.info(
                        f"{atlanan} fotoğraf aynı dosya adıyla "
                        "dolapta bulunduğu için eklenmedi."
                    )

            except Exception as hata:
                st.error(
                    "Yapay zekâ modelleri yüklenemedi. "
                    f"Hata ayrıntısı: {hata}"
                )


# --------------------------------------------------
# ESKİ FOTOĞRAFLARIN RENKLERİNİ YENİDEN ANALİZ ET
# --------------------------------------------------

st.divider()
st.subheader("🎨 Renk algılamasını düzelt")

st.write(
    "Daha önce dolabına eklediğin fotoğrafların renklerini "
    "yeniden analiz edebilirsin. Bu işlem fotoğrafları silmez."
)

if st.button("🔄 Dolabımdaki renkleri yeniden analiz et"):
    if not st.session_state["dolap"]:
        st.info("Dolabında henüz analiz edilecek parça yok.")
    else:
        with st.spinner("Kıyafet renkleri yeniden analiz ediliyor..."):
            try:
                _, segmentasyon_modeli = modelleri_yukle()
                duzeltilen = 0

                for parca in st.session_state["dolap"]:
                    try:
                        resim = base64_resim_ac(parca["resim"])

                        yeni_renk = renk_bul(
                            resim,
                            segmentasyon_modeli,
                            parca.get("tur")
                        )

                        parca["renk"] = yeni_renk
                        duzeltilen += 1

                    except Exception as hata:
                        st.warning(
                            f"{parca.get('dosya_adi', 'Fotoğraf')} "
                            f"yeniden analiz edilemedi: {hata}"
                        )

                dolabi_kaydet(st.session_state["dolap"])

                st.success(
                    f"{duzeltilen} parçanın renk analizi tamamlandı."
                )
                st.rerun()

            except Exception as hata:
                st.error(f"Renk analizi yapılamadı: {hata}")


# --------------------------------------------------
# DOLABI GÖSTER
# --------------------------------------------------

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
                resim = base64_resim_ac(parca["resim"])
                st.image(resim, use_container_width=True)
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


# --------------------------------------------------
# DETAYLI KOMBİN ÖNERİLERİ
# --------------------------------------------------

st.divider()
st.subheader("💡 Sana Özel Kombin Önerileri")

dolap = st.session_state["dolap"]

if dolap:
    oneriler = kombin_oner(dolap)

    st.write(
        "Kombinler dolabındaki kıyafetlerin renklerine ve türlerine "
        "göre hazırlanır. Her önerinin altında stil ve kullanım "
        "önerilerini de görebilirsin."
    )

    for i, oner in enumerate(oneriler, start=1):
        with st.expander(
            f"✨ Kombin {i}: {oner['baslik']}",
            expanded=(i == 1)
        ):
            st.markdown(f"**👗 Kombin parçaları:** {oner['parcalar']}")
            st.markdown(f"**🎨 Stil:** {oner['stil']}")
            st.markdown(f"**📍 Nerede giyilir?** {oner['ortam']}")

            st.markdown("**💬 Neden bu kombin?**")
            st.write(oner["aciklama"])

            st.markdown("**👜 Aksesuar önerisi**")
            st.write(oner["aksesuar"])

            st.markdown("**🔁 Alternatif kombin fikri**")
            st.write(oner["alternatif"])

    st.caption(
        "Not: Öneriler mevcut dolap verilerinden ve renk kurallarından "
        "üretilir. Bu sürüm, her kombini ayrıca değerlendiren bir moda "
        "uzmanı yapay zekâ modeli kullanmaz."
    )
else:
    st.info("Kombin önerileri için önce dolabına kıyafet ekle.")


# --------------------------------------------------
# BİLGİ
# --------------------------------------------------

st.divider()
st.caption(
    "Not: dolap.json dosyası dolap verilerini yerel dosya olarak saklar. "
    "Streamlit Cloud yeniden başlatıldığında veya yeniden dağıtımda "
    "bu dosyanın kalıcılığı garanti edilmez."
)
