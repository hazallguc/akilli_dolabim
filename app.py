    if not kayitlar:
        st.info("Dolabın henüz boş. Kıyafet Ekle bölümünden fotoğraf yükleyebilirsin.")
    else:
        st.write(f"Toplam **{len(kayitlar)}** kıyafetin var.")
        arama = st.text_input("🔎 Kıyafet ara", placeholder="Örneğin: siyah tişört")
        filtre = st.selectbox("Türüne göre filtrele", ["Tümü"] + TURLER)
        gosterilecekler = []
        for i, kayit in enumerate(kayitlar):
            if filtre != "Tümü" and kayit.get("tur") != filtre:
                continue
            aranacak = (
                str(kayit.get("isim", "")) + " " +
                str(kayit.get("alt_tur", "")) + " " +
                str(kayit.get("renk", ""))
            ).lower()
            if arama.strip().lower() in aranacak:
                gosterilecekler.append((i, kayit))
        if not gosterilecekler:
            st.warning("Bu aramaya uygun kıyafet bulunamadı.")
        else:
            sutunlar = st.columns(4)
            for sira, (i, kayit) in enumerate(gosterilecekler):
                with sutunlar[sira % 4]:
                    with st.container(border=True):
                        resim = fotografi_ac(kayit)
                        if resim:
                            st.image(resim, use_container_width=True)
                        else:
                            st.info("Fotoğraf bulunamadı.")
                        st.markdown(f"**{gorunen_kiyafet_ismi(kayit)}**")
                        st.caption(f"{kayit.get('renk', 'Bilinmiyor')} · {kayit.get('tur', 'Diğer')}")
                        with st.expander("Düzenle / Sil"):
                            yeni_isim = st.text_input("Kıyafet adı", value=alt_tur_al(kayit), key=f"galeri_isim_{i}")
                            mevcut_tur = kayit.get("tur", "Diğer")
                            tur_index = TURLER.index(mevcut_tur) if mevcut_tur in TURLER else 0
                            yeni_tur = st.selectbox("Tür", TURLER, index=tur_index, key=f"galeri_tur_{i}")
                            altlar = ALT_TURLER[yeni_tur]
                            mevcut_alt = kayit.get("alt_tur", altlar[0])
                            alt_index = altlar.index(mevcut_alt) if mevcut_alt in altlar else 0
                            yeni_alt = st.selectbox("Alt tür", altlar, index=alt_index, key=f"galeri_alt_{i}")
                            mevcut_renk = kayit.get("renk", "Bilinmiyor")
                            renk_index = RENKLER.index(mevcut_renk) if mevcut_renk in RENKLER else 0
                            yeni_renk = st.selectbox("Renk", RENKLER, index=renk_index, key=f"galeri_renk_{i}")
                            if st.button("💾 Kaydet", key=f"galeri_kaydet_{i}", use_container_width=True):
                                eski_kayitlar = [dict(k) for k in st.session_state.dolap]
                                eski_kayitlar[i].update({
                                    "isim": yeni_isim, "tur": yeni_tur,
                                    "alt_tur": yeni_alt, "renk": yeni_renk
                                })
                                if dolabi_kaydet(eski_kayitlar):
                                    st.session_state.dolap = eski_kayitlar
                                    st.session_state.son_kombin = None
                                    st.success("Kıyafet güncellendi.")
                                    st.rerun()
                            if st.button("🗑️ Sil", key=f"galeri_sil_{i}", use_container_width=True):
                                st.session_state["silinecek"] = i
                            if st.session_state.get("silinecek") == i:
                                st.warning("Bu kıyafeti silmek istediğine emin misin?")
                                evet, vazgec = st.columns(2)
                                with evet:
                                    if st.button("Evet, sil", key=f"galeri_evet_{i}"):
                                        yeni_kayitlar = [dict(k) for k in st.session_state.dolap]
                                        yeni_kayitlar.pop(i)
                                        if dolabi_kaydet(yeni_kayitlar):
                                            st.session_state.dolap = yeni_kayitlar
                                            st.session_state.pop("silinecek", None)
                                            st.session_state.son_kombin = None
                                            st.rerun()
                                with vazgec:
                                    if st.button("Vazgeç", key=f"galeri_vazgec_{i}"):
                                        st.session_state.pop("silinecek", None)
                                        st.rerun()

elif sayfa == "Kombin Öner":
    st.header("✨ Kombin önerileri")
    if not st.session_state.dolap:
        st.info("Önce Kıyafet Ekle bölümünden dolabına kıyafet ekle.")
    else:
        tarz = st.selectbox("Nasıl bir kombin istiyorsun?", ["Günlük", "Spor", "Şık"])
        hava = st.session_state.get("hava")
        if hava:
            st.info(f"🌡️ {hava['sehir']}: {hava['sicaklik']:.1f}°C — Yağış ihtimali %{hava['yagis_ihtimali']}")
        else:
            st.warning("Hava durumu alınmadı. Sol menüden şehir seçip hava durumunu getir.")
        if st.button("Kombinimi oluştur", type="primary"):
            kombin, aciklamalar = kombin_olustur(st.session_state.dolap, tarz, hava=hava)
            st.session_state.son_kombin = kombin
            st.session_state.kombin_aciklamalari = aciklamalar
            st.session_state.kombin_imza = tuple(sorted(id(x) for x in kombin)) if kombin else None
        kombin = st.session_state.son_kombin
        if kombin:
            st.subheader("Senin kombin önerin")
            kolaj = kolaj_olustur(kombin)
            st.image(kolaj, use_container_width=True)
            st.download_button("Kombin görselini indir", data=resim_bytes(kolaj), file_name="akilli_dolabim_kombin.png", mime="image/png")
            st.subheader("Kombindeki parçalar")
            for kayit in kombin:
                st.write(f"- **{gorunen_kiyafet_ismi(kayit)}**")
            for aciklama in st.session_state.get("kombin_aciklamalari", []):
                st.caption(aciklama)
            if st.button("Aynı tarzda başka kombin dene"):
                yeni_kombin, aciklamalar = kombin_olustur(
                    st.session_state.dolap, tarz, hava=hava,
                    onceki_imza=st.session_state.kombin_imza
                )
                st.session_state.son_kombin = yeni_kombin
                st.session_state.kombin_aciklamalari = aciklamalar
                st.session_state.kombin_imza = tuple(sorted(id(x) for x in yeni_kombin)) if yeni_kombin else None
                st.rerun()
        elif "kombin_aciklamalari" in st.session_state:
            for aciklama in st.session_state.kombin_aciklamalari:
                st.warning(aciklama)
