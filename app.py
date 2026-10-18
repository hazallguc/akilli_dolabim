# ------------------------------
elif sayfa == "Dolabım":
    st.header("🧥 Dolabım")
    clothes = st.session_state.dolap
    if not clothes:
        st.info("Dolabın henüz boş. Kıyafet Ekle bölümünden fotoğraf yükleyebilirsin.")
    else:
        st.write(f"Toplam **{len(clothes)}** kıyafetin var.")
        search = st.text_input("🔎 Kıyafet ara", placeholder="Örneğin: siyah tişört")
        filter_type = st.selectbox("Türüne göre filtrele", ["Tümü"]+TURLER)
        visible = []
        for i, item in enumerate(clothes):
            if filter_type != "Tümü" and item.get("tur") != filter_type: continue
            searchable = (item.get("isim","")+" "+item.get("alt_tur","")+" "+item.get("renk","")).lower()
            if search.strip().lower() not in searchable: continue
            visible.append((i,item))
        if not visible: st.warning("Bu aramaya uygun kıyafet bulunamadı.")
        else:
            cols = st.columns(4)
            for order, (i,item) in enumerate(visible):
                with cols[order%4]:
                    with st.container(border=True):
                        image = fotografi_ac(item)
                        if image: st.image(image, width="stretch")
                        else: st.info("Fotoğraf bulunamadı.")
                        st.markdown(f"**{gorunen_kiyafet_ismi(item)}**")
                        st.caption(f"{item.get('renk','Bilinmiyor')} · {item.get('tur','Diğer')}")
                        with st.expander("Düzenle / Sil"):
                            new_name = st.text_input("Kıyafet adı", value=alt_tur_al(item), key=f"edit_name_{i}")
                            old_type = item.get("tur","Diğer")
                            new_type = st.selectbox("Tür", TURLER, index=TURLER.index(old_type) if old_type in TURLER else 0, key=f"edit_type_{i}")
                            subs = ALT_TURLER[new_type]
                            old_sub = item.get("alt_tur",subs[0])
                            new_sub = st.selectbox("Alt tür", subs, index=subs.index(old_sub) if old_sub in subs else 0, key=f"edit_sub_{i}")
                            old_color = item.get("renk","Bilinmiyor")
                            new_color = st.selectbox("Renk", RENKLER, index=RENKLER.index(old_color) if old_color in RENKLER else 0, key=f"edit_color_{i}")
                            if st.button("💾 Kaydet", key=f"save_{i}", width="stretch"):
                                item.update({"isim":new_name,"tur":new_type,"alt_tur":new_sub,"renk":new_color})
                                if dolabi_kaydet(clothes):
                                    st.success("Kıyafet güncellendi.")
                                    st.rerun()
                            if st.button("🗑️ Sil", key=f"delete_{i}", width="stretch"):
                                st.session_state["silinecek"] = item.get("_db_id") or f"new_{i}"
                            if st.session_state.get("silinecek") == (item.get("_db_id") or f"new_{i}"):
                                st.warning("Bu kıyafeti silmek istediğine emin misin?")
                                yes, no = st.columns(2)
                                with yes:
                                    if st.button("Evet, sil", key=f"confirm_delete_{i}"):
                                        clothes.pop(i)
                                        st.session_state.pop("silinecek",None)
                                        if dolabi_kaydet(clothes): st.rerun()
                                with no:
                                    if st.button("Vazgeç", key=f"cancel_delete_{i}"):
                                        st.session_state.pop("silinecek",None); st.rerun()

# ------------------------------
# KOMBİN ÖNER
# ------------------------------
elif sayfa == "Kombin Öner":
    st.header("✨ Kombin önerileri")
    if not st.session_state.dolap:
        st.info("Önce Kıyafet Ekle bölümünden dolabına kıyafet ekle.")
    else:
        style = st.selectbox("Nasıl bir kombin istiyorsun?", ["Günlük","Spor","Şık"])
        weather = st.session_state.get("hava")
        if weather:
            st.info(f"🌡️ {weather['sehir']}: {weather['sicaklik']:.1f}°C — Yağış ihtimali %{weather['yagis_ihtimali']}")
        else:
            st.warning("Hava durumu alınmadı. Sol menüden şehir seçip hava durumunu getir.")
        if st.button("Kombinimi oluştur", type="primary"):
            combo, notes = kombin_olustur(st.session_state.dolap, style, hava=weather)
            st.session_state.son_kombin = combo
            st.session_state.kombin_aciklamalari = notes
            st.session_state.kombin_imza = tuple(sorted(id(x) for x in combo)) if combo else None
        combo = st.session_state.get("son_kombin")
        if combo:
            st.subheader("Senin kombin önerin")
            collage = kolaj_olustur(combo)
            st.image(collage, width="stretch")
            st.download_button("Kombin görselini indir", data=resim_bytes(collage), file_name="akilli_dolabim_kombin.png", mime="image/png")
            st.subheader("Kombindeki parçalar")
            for item in combo: st.write(f"- **{gorunen_kiyafet_ismi(item)}**")
            for note in st.session_state.get("kombin_aciklamalari",[]): st.caption(note)
            if st.button("Aynı tarzda başka kombin dene"):
                new_combo, notes = kombin_olustur(st.session_state.dolap, style, hava=weather, onceki_imza=st.session_state.kombin_imza)
                st.session_state.son_kombin = new_combo
                st.session_state.kombin_aciklamalari = notes
                st.session_state.kombin_imza = tuple(sorted(id(x) for x in new_combo)) if new_combo else None
                st.rerun()
        elif "kombin_aciklamalari" in st.session_state:
            for note in st.session_state.kombin_aciklamalari: st.warning(note)

# ------------------------------
# YEDEK / AKTARIM
# ------------------------------
elif sayfa == "Yedek / Aktarım":
    st.header("📦 Yedek ve aktarım")
    st.write("Bu hesapta kayıtlı kıyafetlerini JSON olarak yedekleyebilirsin.")
    export_data = [{k:v for k,v in item.items() if k != "_db_id"} for item in st.session_state.dolap]
    st.download_button("Kıyafet verilerimi yedekle (JSON)", data=json.dumps(export_data, ensure_ascii=False, indent=2).encode("utf-8"), file_name="akilli_dolabim_yedek.json", mime="application/json", width="stretch")
    st.divider()
    st.write("Önceki uygulamandan indirdiğin JSON yedeği varsa buradan bu hesaba aktarabilirsin.")
    backup_file = st.file_uploader("Eski dolap JSON yedeğini seç", type=["json"])
    if backup_file and st.button("Yedeği hesabıma aktar", type="primary"):
        try:
            imported = json.load(backup_file)
            if isinstance(imported, dict): imported = imported.get("dolap", [])
            if not isinstance(imported, list): raise ValueError("JSON içinde kıyafet listesi bulunamadı.")
            valid = []
            for item in imported:
                if not isinstance(item, dict): continue
                valid.append({
                    "isim": str(item.get("isim") or item.get("alt_tur") or "Kıyafet"),
                    "tur": str(item.get("tur") or "Diğer"),
                    "alt_tur": str(item.get("alt_tur") or "Diğer"),
                    "renk": str(item.get("renk") or "Bilinmiyor"),
                    "resim": str(item.get("resim") or "")
                })
            if valid:
                st.session_state.dolap.extend(valid)
                if dolabi_kaydet(st.session_state.dolap):
                    st.success(f"{len(valid)} kıyafet bu hesabına aktarıldı.")
                    st.rerun()
            else:
                st.warning("Yedekte aktarılacak kıyafet bulunamadı.")
        except Exception as e:
            st.error(f"Yedek okunamadı: {e}")
