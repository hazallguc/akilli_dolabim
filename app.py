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
from supabase import create_client

st.set_page_config(page_title="Akıllı Dolabım", page_icon="👗", layout="wide")

# ------------------------------
# SUPABASE VE HESAPLAR
# ------------------------------
def supabase_client():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception as e:
        st.error("Supabase ayarları bulunamadı. Streamlit Cloud > Settings > Secrets bölümünü kontrol et.")
        st.caption(f"Ayrıntı: {e}")
        st.stop()

if "supabase" not in st.session_state:
    st.session_state.supabase = supabase_client()

sb = st.session_state.supabase

def oturumu_temizle():
    for key in ("supabase_user", "dolap", "son_analiz", "son_kombin",
                "kombin_imza", "kombin_aciklamalari", "hava", "aktif_kullanici"):
        st.session_state.pop(key, None)

def giris_ekrani():
    st.title("👗 Akıllı Dolabım")
    st.caption("Yapay zekâ destekli kişisel kıyafet dolabın.")
    giris, kayit = st.tabs(["Giriş yap", "Yeni hesap oluştur"])

    with giris:
        with st.form("giris_formu"):
            email = st.text_input("E-posta", key="giris_email")
            parola = st.text_input("Şifre", type="password", key="giris_parola")
            gonder = st.form_submit_button("Giriş yap", type="primary", use_container_width=True)
        if gonder:
            try:
                sonuc = sb.auth.sign_in_with_password({"email": email.strip(), "password": parola})
                if sonuc.user:
                    st.session_state.supabase_user = sonuc.user
                    st.session_state.aktif_kullanici = sonuc.user.id
                    st.session_state.dolap = veritabanindan_yukle(sonuc.user.id)
                    st.rerun()
                else:
                    st.error("Giriş yapılamadı. E-posta ve şifreni kontrol et.")
            except Exception as e:
                st.error(f"Giriş yapılamadı: {e}")

    with kayit:
        with st.form("kayit_formu"):
            email2 = st.text_input("E-posta adresin", key="kayit_email")
            parola2 = st.text_input("Şifre (en az 6 karakter önerilir)", type="password", key="kayit_parola")
            parola3 = st.text_input("Şifreyi tekrar yaz", type="password", key="kayit_parola_tekrar")
            kayit_gonder = st.form_submit_button("Hesap oluştur", type="primary", use_container_width=True)
        if kayit_gonder:
            if not email2.strip() or not parola2:
                st.warning("E-posta ve şifre alanlarını doldur.")
            elif parola2 != parola3:
                st.warning("Şifreler aynı değil.")
            elif len(parola2) < 6:
                st.warning("Şifren en az 6 karakter olsun.")
            else:
                try:
                    sonuc = sb.auth.sign_up({"email": email2.strip(), "password": parola2})
                    if sonuc.session and sonuc.user:
                        st.session_state.supabase_user = sonuc.user
                        st.session_state.aktif_kullanici = sonuc.user.id
                        st.session_state.dolap = veritabanindan_yukle(sonuc.user.id)
                        st.success("Hesabın oluşturuldu!")
                        st.rerun()
                    else:
                        st.success("Hesap oluşturuldu. E-postana gelen doğrulama bağlantısına tıkla, ardından giriş yap.")
                except Exception as e:
                    st.error(f"Hesap oluşturulamadı: {e}")

def fotografi_koda_cevir(image):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def fotografi_ac(kayit):
    try:
