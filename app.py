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
