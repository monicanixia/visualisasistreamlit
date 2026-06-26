"""
PART 001
streamlit_app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Analisis Sentimen Mobil Listrik Indonesia",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main{
    background-color:#f5f7fb;
}
.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0 2px 10px rgba(0,0,0,.08);
}
h1,h2,h3{
    color:#1565C0;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🚗 Analisis Sentimen Mobil Listrik")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Pilih Halaman",
    [
        "🏠 Dashboard",
        "📁 Dataset",
        "📊 Distribusi Label",
        "☁️ Word Cloud",
        "🔤 Top Words",
        "📈 Evaluasi Model"
    ]
)

if menu=="🏠 Dashboard":
    st.title("Dashboard Analisis Sentimen Mobil Listrik Indonesia")
    st.info("Part 001 hanya berisi kerangka aplikasi. Seluruh fitur akan dilengkapi pada part berikutnya.")

    c1,c2,c3,c4=st.columns(4)

    c1.metric("Total Data","-")
    c2.metric("Youtube","-")
    c3.metric("TikTok","-")
    c4.metric("Akurasi Terbaik","-")

    st.markdown("---")

    kiri,kanan=st.columns([2,1])

    with kiri:
        st.subheader("Ringkasan")
        st.write("""
Dashboard ini nantinya akan menampilkan:

- Total komentar
- Total Youtube
- Total TikTok
- Distribusi Sentimen
- Grafik Interaktif
- Statistik Dataset
- Insight otomatis
""")

    with kanan:
        st.subheader("Menu")
        st.write("""
✅ Dashboard

✅ Dataset

✅ Distribusi Label

✅ WordCloud

✅ Top Words

✅ Evaluasi Model
""")

elif menu=="📁 Dataset":
    st.title("Dataset")
    st.warning("Fitur akan dilengkapi pada Part 002.")

elif menu=="📊 Distribusi Label":
    st.title("Distribusi Label")
    st.warning("Fitur akan dilengkapi pada Part 003.")

elif menu=="☁️ Word Cloud":
    st.title("Word Cloud")
    st.warning("Fitur akan dilengkapi pada Part 004.")

elif menu=="🔤 Top Words":
    st.title("Top Words")
    st.warning("Fitur akan dilengkapi pada Part 005.")

elif menu=="📈 Evaluasi Model":
    st.title("Evaluasi Model")
    st.warning("Fitur akan dilengkapi pada Part 006.")
