
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

st.set_page_config(page_title="Akıllı Dolabım", page_icon="👗", layout="wide")

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
RENKLER = ["Siyah", "Beyaz", "Gri", "Kırmızı", "Bordo", "Pembe", "Pembe-Siyah", "Mavi", "Lacivert", "Yeşil", "Sarı", "Turuncu", "Mor", "Kahverengi", "Bej", "Krem", "Çok renkli", "Bilinmiyor"]

ALT_TUR_KATEGORI = {
    alt: ana for ana, altlar in ALT_TURLER.items() for alt in altlar
}

MODEL_ETIKETLERI = {
    "Tişört": "a t-shirt", "Bluz": "a blouse", "Gömlek": "a button-up shirt",
    "Crop top": "a crop top", "Sweatshirt": "a sweatshirt", "Hoodie": "a hoodie",
    "Kazak": "a sweater", "Spor üst": "a sports top", "Body": "a bodysuit",
    "Jean": "jeans", "Kumaş pantolon": "dress trousers", "Eşofman altı": "sweatpants",
    "Tayt": "leggings", "Spor tayt": "sports leggings", "Etek": "a skirt",
    "Şort": "shorts", "Spor şort": "sports shorts",
    "Günlük elbise": "a casual dress", "Şık elbise": "an elegant dress",
    "Abiye": "an evening gown", "Spor elbise": "a sporty dress",
    "Blazer ceket": "a blazer", "Günlük ceket": "a casual jacket",
    "Deri ceket": "a leather jacket", "Kot ceket": "a denim jacket",
    "Şişme mont": "a puffer jacket", "Mont": "a coat or jacket",
    "Kaban": "a long winter coat", "Hırka": "a cardigan",
    "Spor ceket": "a sports jacket", "Spor ayakkabı": "sports shoes",
    "Sneaker": "sneakers", "Topuklu ayakkabı": "high heels",
    "Babet": "ballet flats", "Loafer": "loafers", "Bot": "boots",
    "Sandalet": "sandals", "Klasik ayakkabı": "formal shoes",
    "Günlük çanta": "a casual handbag", "Şık çanta": "an elegant handbag",
    "Omuz çantası": "a shoulder bag", "El çantası": "a clutch bag",
    "Sırt çantası": "a backpack", "Spor çanta": "a sports bag",
    "Takı": "jewelry", "Kemer": "a belt", "Şapka": "a hat",
    "Atkı": "a scarf", "Gözlük": "glasses"
}

@st.cache_resource(show_spinner="Yapay zekâ modeli yükleniyor...")
def model_yukle():
    from transformers import pipeline
    return pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

def yapay_zeka_tahmin(image):
    try:
        model = model_yukle()
        sonuc = model(image, candidate_labels=list(MODEL_ETIKETLERI.values()))
        ters = {v: k for k, v in MODEL_ETIKETLERI.items()}
        alt = ters.get(sonuc[0]["label"], "Diğer")
        return ALT_TUR_KATEGORI.get(alt, "Diğer"), alt, float(sonuc[0].get("score", 0))
    except Exception:
        return "Diğer", "Diğer", 0.0

def renk_tahmin_et(image):
    try:
        img = image.convert("RGB")
        img.thumbnail((120, 120))
        sayac = Counter()
        for r, g, b in img.getdata():
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
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
            else:
                renk = "Pembe"
            sayac[renk] += 1

        en_cok = sayac.most_common(2)
        if not en_cok:
            return "Bilinmiyor"
        if len(en_cok) > 1:
            toplam = sum(sayac.values())
            if en_cok[0][1] / toplam < 0.60 and en_cok[1][1] / toplam > 0.20:
                ikili = {en_cok[0][0], en_cok[1][0]}
                if ikili == {"Pembe", "Siyah"}:
                    return "Pembe-Siyah"
                return f"{en_cok[0][0]}-{en_cok[1][0]}"
        return en_cok[0][0]
    except Exception:
        return "Bilinmiyor"

def dolabi_yukle():
    if not os.path.exists(DOSYA):
        return []
    try:
        with open(DOSYA, "r", encoding="utf-8") as f:
            veri = json.load(f)
        kayitlar = veri if isinstance(veri, list) else veri.get("dolap", [])
        if not isinstance(kayitlar, list):
            return []
        for k in kayitlar:
            k.setdefault("isim", k.get("dosya_adi", "Kıyafet"))
            k.setdefault("dosya_adi", k.get("isim", "Kıyafet"))
            k.setdefault("resim", "")
            k.setdefault("tur", "Diğer")
            k.setdefault("alt_tur", "")
            k.setdefault("renk", "Bilinmiyor")
            k.setdefault("guven", 0.0)
        return kayitlar
    except Exception:
        st.error("dolap.json okunamadı. Dosyanı silme; dosyanın biçimini kontrol etmek gerekiyor.")
        return []

def dolabi_kaydet(dolap):
    if os.path.exists(DOSYA):
        try:
            with open(DOSYA, "r", encoding="utf-8") as f:
                eski = f.read()
            with open(DOSYA + ".bak", "w", encoding="utf-8") as f:
                f.write(eski)
        except Exception:
            pass
    with open(DOSYA, "w", encoding="utf-8") as f:
        json.dump(dolap, f, ensure_ascii=False, indent=2)

def fotografi_ac(kayit):
    try:
        return Image.open(io.BytesIO(base64.b64decode(kayit.get("resim", "")))).convert("RGB")
    except Exception:
        return None

def fotografi_koda_cevir(image):
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def alt_tur_al(kayit):
    alt = kayit.get("alt_tur", "")
    if alt in ALT_TUR_KATEGORI:
        return alt
    ana = kayit.get("tur", "Diğer")
    secenekler = ALT_TURLER.get(ana, ["Diğer"])
    return secenekler[-1]

def kayitlar_tur(kayitlar, tur):
    return [k for k in kayitlar if k.get("tur") == tur]

# =========================================================
# TARZA UYGUN SEÇİM
# Önce tam kombin oluşturulur. Spor seçeneği, dolapta spor
# alt türü yoksa günlük uygun parçaları da değerlendirebilir.
# =========================================================

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

def oncelik_puani(kayit, liste):
    alt = alt_tur_al(kayit)
    return len(liste) - liste.index(alt) if alt in liste else 0

def secenekleri_sirala(kayitlar, tur, oncelikler):
    adaylar = kayitlar_tur(kayitlar, tur)
    # Tercih edilenler önce gelir; uygun alternatifler de tamamen elenmez.
    return sorted(
        adaylar,
        key=lambda k: (
            oncelik_puani(k, oncelikler),
            -len(k.get("isim", ""))
        ),
        reverse=True
    )

def kombin_imzasi(kombin):
    return tuple(sorted(
        str(k.get("isim", k.get("dosya_adi", ""))) + "|" +
        str(k.get("alt_tur", "")) + "|" +
        str(k.get("resim", "")[:24])
        for k in kombin
    ))

def kombin_olustur(kayitlar, tarz, onceki_imza=None, tur_sayisi=0):
    tercih = TERCIHLER[tarz]
    aciklamalar = []

    ustler = secenekleri_sirala(kayitlar, "Üst giyim", tercih["ust"])
    altlar = secenekleri_sirala(kayitlar, "Alt giyim", tercih["alt"])
    elbiseler = kayitlar_tur(kayitlar, "Elbise")

    # Üst + alt kombinlerini üret. Uygun bir parça varsa ikisi de seçilir.
    aday_kombinler = []

    if ustler and altlar:
        for ust, alt in product(ustler, altlar):
            puan = oncelik_puani(ust, tercih["ust"]) + oncelik_puani(alt, tercih["alt"])
            # Aynı parçanın iki kere seçilmesini önle.
            if ust is not alt:
                aday_kombinler.append((puan, [ust, alt]))

    # Elbise, tarzına uygun olduğunda alternatif tam kombin olabilir.
    for elbise in elbiseler:
        alt = alt_tur_al(elbise)
        uygun_mu = (
            (tarz == "Şık" and alt in ["Şık elbise", "Abiye", "Günlük elbise"])
            or (tarz == "Spor" and alt == "Spor elbise")
            or (tarz == "Günlük" and alt in ["Günlük elbise", "Spor elbise"])
        )
        if uygun_mu:
            elbise_puani = 100 if (
                (tarz == "Şık" and alt in ["Şık elbise", "Abiye"])
                or (tarz == "Spor" and alt == "Spor elbise")
            ) else 20
            aday_kombinler.append((elbise_puani, [elbise]))

    # Üst-alt bulunamasa bile dolaptaki uygun bir elbiseyi kullan.
    if not aday_kombinler:
        if ustler:
            aday_kombinler.extend((1, [u]) for u in ustler)
        if altlar:
            aday_kombinler.extend((1, [a]) for a in altlar)
        if elbiseler:
            aday_kombinler.extend((1, [e]) for e in elbiseler)

    if not aday_kombinler:
        return [], ["Dolabında kombin oluşturmak için uygun ana kıyafet bulunamadı."]

    # Yüksek puanlı kombinler önce; aynı kombini yeniden seçmemeye çalış.
    aday_kombinler.sort(key=lambda x: x[0], reverse=True)
    puanli = [k for _, k in aday_kombinler]

    # Aynı kıyafetlerle oluşturulmuş tekrarları kaldır.
    benzersiz = []
    gorulen = set()
    for k in puanli:
        imza = kombin_imzasi(k)
        if imza not in gorulen:
            gorulen.add(imza)
            benzersiz.append(k)

    if not benzersiz:
        return [], ["Kombin oluşturulamadı."]

    onceki_indeks = None
    if onceki_imza:
        for i, k in enumerate(benzersiz):
            if kombin_imzasi(k) == onceki_imza:
                onceki_indeks = i
                break

    if onceki_indeks is not None and len(benzersiz) > 1:
        secim_indeksi = (onceki_indeks + 1) % len(benzersiz)
    else:
        secim_indeksi = tur_sayisi % len(benzersiz)

    ana_kombin = list(benzersiz[secim_indeksi])

    # Ek parçaları, ana kıyafetlere göre uygun şekilde seç.
    disler = kayitlar_tur(kayitlar, "Dış giyim")
    ayakkabilar = kayitlar_tur(kayitlar, "Ayakkabı")
    cantalar = kayitlar_tur(kayitlar, "Çanta")
    aksesuarlar = kayitlar_tur(kayitlar, "Aksesuar")

    dis = secenekleri_sirala(kayitlar, "Dış giyim", tercih["dis"])
    ayakkabi = secenekleri_sirala(kayitlar, "Ayakkabı", tercih["ayakkabi"])
    canta = secenekleri_sirala(kayitlar, "Çanta", tercih["canta"])

    # Şık kombinde şişme montu tercih etme.
    if tarz == "Şık":
        dis = [k for k in dis if alt_tur_al(k) != "Şişme mont" and alt_tur_al(k) != "Mont" and alt_tur_al(k) != "Kaban"]

    # Spor kombinde topuklu ayakkabıyı kesinlikle kullanma.
    if tarz == "Spor":
        ayakkabi = [k for k in ayakkabi if alt_tur_al(k) in ["Spor ayakkabı", "Sneaker"]]

    # Ana parçayla aynı kayıt eklenmesin.
    ana_ids = {id(k) for k in ana_kombin}

    def ekle_bir_tane(adaylar):
        for k in adaylar:
            if id(k) not in ana_ids:
                ana_ids.add(id(k))
                ana_kombin.append(k)
                return k
        return None

    secilen_dis = ekle_bir_tane(dis)
    secilen_ayakkabi = ekle_bir_tane(ayakkabi)
    secilen_canta = ekle_bir_tane(canta)

    if not secilen_ayakkabi:
        aciklamalar.append("Bu tarz için uygun alt türde bir ayakkabı bulunamadı.")
    if not secilen_dis:
        aciklamalar.append("Bu tarz için uygun dış giyim bulunamadı; uygunsuz bir mont eklenmedi.")
    if not secilen_canta:
        aciklamalar.append("Bu tarz için uygun çanta bulunamadı.")

    # Aksesuarlar zorunlu değil. Kemer hiçbir zaman otomatik eklenmez.
    # Takı yalnızca şık kombinlerde, dolapta varsa eklenebilir.
    if tarz == "Şık":
        takilar = [k for k in aksesuarlar if alt_tur_al(k) == "Takı" and id(k) not in ana_ids]
        if takilar:
            ana_kombin.append(takilar[0])

    return ana_kombin, aciklamalar

# =========================================================
# KÜÇÜK KOLAJ
# =========================================================

def kolaj_olustur(kombin):
    kayitlar = []
    for k in kombin:
        img = fotografi_ac(k)
        if img is not None:
            kayitlar.append((k, img))

    if not kayitlar:
        return None

    # Önceki dev görünüm yerine küçük, sabit kartlar.
    kart_w, kart_h = 190, 205
    bosluk, baslik_h = 12, 42
    sutun = min(3, len(kayitlar))
    satir = (len(kayitlar) + sutun - 1) // sutun

    w = sutun * kart_w + (sutun + 1) * bosluk
    h = baslik_h + satir * kart_h + (satir + 1) * bosluk
    canvas = Image.new("RGB", (w, h), "#f7f4f1")
    draw = ImageDraw.Draw(canvas)
    draw.text((bosluk, 14), "AKILLI DOLABIM • KOMBİNİN", fill="#292421")

    for i, (kayit, img) in enumerate(kayitlar):
        x = bosluk + (i % sutun) * (kart_w + bosluk)
        y = baslik_h + bosluk + (i // sutun) * (kart_h + bosluk)

        draw.rounded_rectangle(
            [x, y, x + kart_w, y + kart_h],
            radius=10, fill="white", outline="#e1d9d3", width=1
        )

        img.thumbnail((kart_w - 16, 145))
        ix = x + (kart_w - img.width) // 2
        iy = y + 7
        canvas.paste(img, (ix, iy))

        alt = alt_tur_al(kayit)
        draw.text((x + 8, y + 158), alt[:26], fill="#292421")
        draw.text((x + 8, y + 178), kayit.get("renk", "Bilinmiyor")[:25], fill="#77706b")

    return canvas

def resim_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

# =========================================================
# STREAMLIT ARAYÜZÜ
# =========================================================

if "dolap" not in st.session_state:
    st.session_state["dolap"] = dolabi_yukle()

dolap = st.session_state["dolap"]

st.title("👗 Akıllı Dolabım")
st.caption("Kıyafetlerini ekle, dolabını düzenle ve tarzına uygun kombin oluştur.")

sayfa = st.sidebar.radio("Menü", ["Kıyafet Ekle", "Dolabım", "Kombin Öner"])

if sayfa == "Kıyafet Ekle":
    st.header("📸 Dolabına kıyafet ekle")
    dosya = st.file_uploader("Kıyafet fotoğrafını yükle", type=["jpg", "jpeg", "png", "webp"])

    if dosya:
        try:
            image = Image.open(dosya).convert("RGB")
            st.image(image, caption="Yüklediğin fotoğraf", width=280)

            if st.button("🤖 Fotoğrafı analiz et", type="primary"):
                with st.spinner("Fotoğraf analiz ediliyor..."):
                    tur, alt, guven = yapay_zeka_tahmin(image)
                    renk = renk_tahmin_et(image)
                st.session_state["son_analiz"] = {
                    "isim": dosya.name,
                    "resim": fotografi_koda_cevir(image),
                    "tur": tur,
                    "alt_tur": alt,
                    "renk": renk,
                    "guven": guven
                }

            analiz = st.session_state.get("son_analiz")
            if analiz and analiz.get("isim") == dosya.name:
                st.subheader("Bilgileri kontrol et")
                tur = st.selectbox(
                    "Ana tür", TURLER,
                    index=TURLER.index(analiz["tur"]) if analiz["tur"] in TURLER else 0,
                    key="ekle_tur"
                )
                altlar = ALT_TURLER[tur]
                alt = st.selectbox(
                    "Tam tür", altlar,
                    index=altlar.index(analiz["alt_tur"]) if analiz["alt_tur"] in altlar else 0,
                    key="ekle_alt"
                )
                renk = st.selectbox(
                    "Renk", RENKLER,
                    index=RENKLER.index(analiz["renk"]) if analiz["renk"] in RENKLER else len(RENKLER)-1,
                    key="ekle_renk"
                )

                if st.button("➕ Dolabıma kaydet", type="primary"):
                    dolap.append({
                        "isim": analiz["isim"],
                        "dosya_adi": analiz["isim"],
                        "resim": analiz["resim"],
                        "tur": tur,
                        "alt_tur": alt,
                        "renk": renk,
                        "guven": analiz.get("guven", 0.0)
                    })
                    st.session_state["dolap"] = dolap
                    dolabi_kaydet(dolap)
                    st.session_state.pop("son_analiz", None)
                    st.success("Kıyafet dolabına kaydedildi!")
                    st.rerun()
        except Exception as e:
            st.error(f"Fotoğraf açılamadı: {e}")

elif sayfa == "Dolabım":
    st.header("🧥 Dolabım")
    if not dolap:
        st.info("Dolabında henüz kıyafet yok.")
    else:
        st.write(f"Toplam **{len(dolap)}** kıyafet kayıtlı.")
        filtre = st.selectbox("Türüne göre filtrele", ["Tümü"] + TURLER)

        for i, kayit in enumerate(dolap):
            if filtre != "Tümü" and kayit.get("tur") != filtre:
                continue

            with st.container(border=True):
                sol, sag = st.columns([1, 2])
                with sol:
                    img = fotografi_ac(kayit)
                    if img:
                        st.image(img, width=180)

                with sag:
                    st.markdown(f"**{kayit.get('isim', kayit.get('dosya_adi', 'Kıyafet'))}**")
                    mevcut_tur = kayit.get("tur", "Diğer")
                    if mevcut_tur not in TURLER:
                        mevcut_tur = "Diğer"

                    yeni_tur = st.selectbox(
                        "Ana tür", TURLER, index=TURLER.index(mevcut_tur), key=f"tur_{i}"
                    )
                    altlar = ALT_TURLER[yeni_tur]
                    mevcut_alt = alt_tur_al(kayit)
                    yeni_alt = st.selectbox(
                        "Tam tür", altlar,
                        index=altlar.index(mevcut_alt) if mevcut_alt in altlar else 0,
                        key=f"alt_{i}"
                    )
                    mevcut_renk = kayit.get("renk", "Bilinmiyor")
                    yeni_renk = st.selectbox(
                        "Renk", RENKLER,
                        index=RENKLER.index(mevcut_renk) if mevcut_renk in RENKLER else len(RENKLER)-1,
                        key=f"renk_{i}"
                    )

                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("💾 Kaydet", key=f"kaydet_{i}"):
                            kayit["tur"] = yeni_tur
                            kayit["alt_tur"] = yeni_alt
                            kayit["renk"] = yeni_renk
                            dolabi_kaydet(dolap)
                            st.success("Bilgiler kaydedildi.")
                            st.rerun()
                    with c2:
                        if st.button("🗑️ Sil", key=f"sil_{i}"):
                            st.session_state[f"sil_onay_{i}"] = True

                    if st.session_state.get(f"sil_onay_{i}", False):
                        st.warning("Bu kıyafeti silmek istiyor musun?")
                        a, b = st.columns(2)
                        with a:
                            if st.button("Evet, sil", key=f"evet_{i}"):
                                dolap.pop(i)
                                dolabi_kaydet(dolap)
                                st.session_state["dolap"] = dolap
                                st.rerun()
                        with b:
                            if st.button("Vazgeç", key=f"vazgec_{i}"):
                                st.session_state.pop(f"sil_onay_{i}", None)
                                st.rerun()

elif sayfa == "Kombin Öner":
    st.header("✨ Tarzına göre kombin oluştur")

    if not dolap:
        st.info("Önce Kıyafet Ekle bölümünden dolabına kıyafet ekle.")
    else:
        tarz = st.radio("Hangi tarzı istiyorsun?", ["Günlük", "Spor", "Şık"], horizontal=True)

        if "kombin_sayaci" not in st.session_state:
            st.session_state["kombin_sayaci"] = 0

        if st.button("✨ Kombinimi oluştur", type="primary"):
            st.session_state["kombin_sayaci"] = 0
            kombin, notlar = kombin_olustur(dolap, tarz)
            st.session_state["son_kombin"] = kombin
            st.session_state["son_kombin_imza"] = kombin_imzasi(kombin) if kombin else None
            st.session_state["son_kombin_tarz"] = tarz
            st.session_state["son_kombin_notlar"] = notlar

        kombin = st.session_state.get("son_kombin", [])
        onceki_tarz = st.session_state.get("son_kombin_tarz", "")

        if kombin and onceki_tarz == tarz:
            st.subheader(f"{tarz} kombin önerin")

            kolaj = kolaj_olustur(kombin)
            if kolaj:
                # Büyük ekrana yayılmasını engelle.
                st.image(kolaj, width=min(kolaj.width, 640))

                # İndirme butonu
                st.download_button(
                    label="⬇️ Kombin görselini indir",
                    data=resim_bytes(kolaj),
                    file_name=f"{tarz.lower()}_kombini.png",
                    mime="image/png",
                    key="kombin_indir"
                )

            st.markdown("### Kombindeki parçalar")
            for k in kombin:
                st.write(f"- **{alt_tur_al(k)}** — {k.get('renk', 'Bilinmiyor')}")

            for not_metni in st.session_state.get("son_kombin_notlar", []):
                st.info(not_metni)

            if st.button("🔄 Aynı tarzda başka kombin dene"):
                eski_imza = st.session_state.get("son_kombin_imza")
                st.session_state["kombin_sayaci"] += 1
                yeni_kombin, yeni_notlar = kombin_olustur(
                    dolap,
                    tarz,
                    onceki_imza=eski_imza,
                    tur_sayisi=st.session_state["kombin_sayaci"]
                )
                st.session_state["son_kombin"] = yeni_kombin
                st.session_state["son_kombin_imza"] = kombin_imzasi(yeni_kombin) if yeni_kombin else None
                st.session_state["son_kombin_tarz"] = tarz
                st.session_state["son_kombin_notlar"] = yeni_notlar
                st.rerun()

        elif not kombin:
            st.caption("Tarzını seçip **Kombinimi oluştur** düğmesine bas.")
