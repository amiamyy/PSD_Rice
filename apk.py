import streamlit as st

st.title("Klasifikasi Jenis Beras")
st.sidebar.title("Navigasi")
st.sidebar.markdown("Pilih halaman berikut:")

page = st.sidebar.radio("Pilih Halaman:", ["Penjelasan Dataset", "Prediksi"])

if page == "Penjelasan Dataset":
    st.rerun()
    import pages.analisisdata_page

if page == "Prediksi":
    st.rerun()
    import pages.prediksi_page
