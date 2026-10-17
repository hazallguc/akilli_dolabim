import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Akıllı Dolabım",
    page_icon="👗",
    layout="wide"
)

st.title("👗 Akıllı Dolabım")
st.subheader("Yapay zekâ destekli kombin asistanın")

st.write(
    "Kıyafetlerini yükle, dijital dolabını oluştur "
    "ve sana uygun kombinleri keşfet!"
)

st.divider()

st.header("📸 Kıyafetlerini dolabına ekle")

fotograflar = st.file_uploader(
    "Kıyafet fotoğraflarını seç",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if fotograflar:
    st.success(f"{len(fotograflar)} fotoğraf yükledin!")

    sutunlar = st.columns(3)

    for i, fotograf in enumerate(fotograflar):
        with sutunlar[i % 3]:
            resim = Image.open(fotograf).convert("RGB")
            st.image(resim, use_container_width=True)
            st.caption(fotograf.name)

st.divider()
st.caption("Akıllı Dolabım | İlk prototip")

%%writefile app.py
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Akıllı Dolabım",
    page_icon="👗",
    layout="wide"
)

st.title("👗 Akıllı Dolabım")
st.subheader("Yapay zekâ destekli kombin asistanın")

st.write(
    "Kıyafetlerini yükle, dijital dolabını oluştur "
    "ve sana uygun kombinleri keşfet!"
)

st.divider()

st.header("📸 Kıyafetlerini dolabına ekle")

fotograflar = st.file_uploader(
    "Kıyafet fotoğraflarını seç",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if fotograflar:
    st.success(f"{len(fotograflar)} fotoğraf yükledin!")

    sutunlar = st.columns(3)

    for i, fotograf in enumerate(fotograflar):
        with sutunlar[i % 3]:
            resim = Image.open(fotograf).convert("RGB")
            st.image(resim, use_container_width=True)
            st.caption(fotograf.name)

st.divider()
st.caption("Akıllı Dolabım | İlk prototip")
