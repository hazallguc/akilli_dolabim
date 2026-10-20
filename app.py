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

# CLIP'e gösterilen adaylar; ana tür ve alt tür birlikte tahmin edilir.
MODEL_ADAYLARI = [
    ("Üst giyim", "Tişört", "a short sleeve casual t shirt"),
    ("Üst giyim", "Bluz", "a women's blouse"),
    ("Üst giyim", "Gömlek", "a button up shirt"),
    ("Üst giyim", "Crop top", "a cropped top"),
    ("Üst giyim", "Sweatshirt", "a crewneck sweatshirt"),
    ("Üst giyim", "Hoodie", "a hooded sweatshirt"),
    ("Üst giyim", "Kazak", "a knitted sweater"),
    ("Üst giyim", "Spor üst", "an athletic sports top"),
    ("Üst giyim", "Body", "a fitted bodysuit"),
    ("Üst giyim", "Diğer üst giyim", "another type of upper body clothing"),
    ("Alt giyim", "Jean", "a pair of denim jeans"),
    ("Alt giyim", "Kumaş pantolon", "a pair of tailored fabric trousers"),
    ("Alt giyim", "Eşofman altı", "a pair of sweatpants"),
    ("Alt giyim", "Tayt", "a pair of casual leggings"),
    ("Alt giyim", "Spor tayt", "a pair of athletic leggings"),
    ("Alt giyim", "Etek", "a skirt"),
    ("Alt giyim", "Şort", "a pair of casual shorts"),
    ("Alt giyim", "Spor şort", "a pair of athletic shorts"),
    ("Alt giyim", "Diğer alt giyim", "another type of lower body clothing"),
    ("Elbise", "Günlük elbise", "a casual everyday dress"),
    ("Elbise", "Şık elbise", "an elegant formal dress"),
    ("Elbise", "Abiye", "a long evening gown"),
    ("Elbise", "Spor elbise", "a sporty athletic dress"),
    ("Elbise", "Diğer elbise", "another style of dress"),
    ("Dış giyim", "Blazer ceket", "a tailored blazer jacket"),
    ("Dış giyim", "Günlük ceket", "a casual everyday jacket"),
    ("Dış giyim", "Deri ceket", "a leather jacket"),
    ("Dış giyim", "Kot ceket", "a denim jean jacket"),
    ("Dış giyim", "Şişme mont", "a puffer jacket"),
    ("Dış giyim", "Mont", "a warm winter jacket"),
    ("Dış giyim", "Kaban", "a long wool coat"),
    ("Dış giyim", "Hırka", "a knitted cardigan"),
    ("Dış giyim", "Spor ceket", "an athletic track jacket"),
    ("Dış giyim", "Diğer dış giyim", "another type of outerwear"),
    ("Ayakkabı", "Spor ayakkabı", "athletic running shoes"),
    ("Ayakkabı", "Sneaker", "casual fashion sneakers"),
    ("Ayakkabı", "Topuklu ayakkabı", "high heel shoes"),
    ("Ayakkabı", "Babet", "flat ballet shoes"),
    ("Ayakkabı", "Loafer", "loafer shoes"),
    ("Ayakkabı", "Bot", "ankle boots"),
    ("Ayakkabı", "Sandalet", "sandals"),
    ("Ayakkabı", "Klasik ayakkabı", "formal dress shoes"),
    ("Ayakkabı", "Diğer ayakkabı", "another type of footwear"),
    ("Çanta", "Günlük çanta", "an everyday handbag"),
    ("Çanta", "Şık çanta", "an elegant evening handbag"),
    ("Çanta", "Omuz çantası", "a shoulder bag"),
    ("Çanta", "El çantası", "a small hand held handbag"),
    ("Çanta", "Sırt çantası", "a backpack"),
    ("Çanta", "Spor çanta", "a gym sports bag"),
    ("Çanta", "Diğer çanta", "another type of bag"),
    ("Aksesuar", "Takı", "jewelry"),
    ("Aksesuar", "Kemer", "a belt"),
    ("Aksesuar", "Şapka", "a hat"),
    ("Aksesuar", "Atkı", "a scarf"),
    ("Aksesuar", "Gözlük", "eyeglasses or sunglasses"),
    ("Aksesuar", "Diğer aksesuar", "another fashion accessory"),
