"""
PART 001
streamlit_app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
from streamlit.components.v1 import html

@st.cache_data
def load_data():

    yt = pd.read_excel("dataset_sentimen_final_Youtube Baru.xlsx")
    tt = pd.read_excel("dataset_sentimen_final Tiktok Baru.xlsx")

    aspek_yt = pd.read_excel("dataset_final_aspek_Youtube.xlsx")
    aspek_tt = pd.read_excel("dataset_final_aspek Tiktok.xlsx")

    lda_yt = pd.read_excel("top_words_lda Youtube.xlsx")
    lda_tt = pd.read_excel("top_words_lda Tiktok.xlsx")

    xgb_yt = pd.read_excel("ringkasan_xgboost_Youtube.xlsx")
    xgb_tt = pd.read_excel("ringkasan_xgboost_Tiktok.xlsx")

    lstm_yt = pd.read_excel("ringkasan_lstm_Youtube.xlsx")
    lstm_tt = pd.read_excel("ringkasan_lstm_Tiktok.xlsx")
    word_yt = pd.read_excel("dataset_wordcloud_Youtube.xlsx")
    word_tt = pd.read_excel("dataset_wordcloud_Tiktok.xlsx")

    return (
        yt,
        tt,
        aspek_yt,
        aspek_tt,
        lda_yt,
        lda_tt,
        xgb_yt,
        xgb_tt,
        lstm_yt,
        lstm_tt,
        word_yt,
        word_tt
    )


# =========================
# TIDAK ADA INDENTASI LAGI
# =========================

(
    yt,
    tt,
    aspek_yt,
    aspek_tt,
    lda_yt,
    lda_tt,
    xgb_yt,
    xgb_tt,
    lstm_yt,
    lstm_tt,
    word_yt,
    word_tt
) = load_data()


dataset = pd.concat([yt, tt], ignore_index=True)


def hitung_statistik(data):

    total = len(data)

    positif = len(data[data["sentiment_label_final"] == "Positif"])

    netral = len(data[data["sentiment_label_final"] == "Netral"])

    negatif = len(data[data["sentiment_label_final"] == "Negatif"])

    return total, positif, netral, negatif
 


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

    total, positif, netral, negatif = hitung_statistik(dataset)
    
    c1,c2,c3 = st.columns(3)

    c1.metric("📊 Total", total)

    c2.metric("▶️ Youtube", len(yt))

    c3.metric("🎵 TikTok", len(tt))

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

    st.title("📁 Dataset Hasil Preprocessing")

    pilihan = st.radio(
        "Pilih Dataset",
        ["📺 YouTube", "🎵 TikTok"],
        horizontal=True
    )

    if pilihan == "📺 YouTube":
        data = yt
        nama = "YouTube"

    else:
        data = tt
        nama = "TikTok"

    st.markdown(f"### Dataset {nama}")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Jumlah Data", len(data))

    with col2:
        st.metric("Jumlah Kolom", len(data.columns))

    st.markdown("---")

    st.subheader("Preview Dataset")

    st.dataframe(
        data,
        use_container_width=True,
        height=500
    )

    st.download_button(
        label="⬇ Download Dataset",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=f"dataset_preprocessing_{nama}.csv",
        mime="text/csv"
    )

elif menu=="📊 Distribusi Label":

    st.title("📊 Distribusi Label Sentimen")

    pilihan = st.radio(
        "Pilih Dataset",
        ["📺 YouTube", "🎵 TikTok"],
        horizontal=True
    )

    if pilihan == "📺 YouTube":
        data = yt
        nama = "YouTube"
    else:
        data = tt
        nama = "TikTok"

    st.subheader(f"Distribusi Label Dataset {nama}")

    # Menghitung jumlah label
    distribusi = (
        data["sentiment_label_final"]
        .value_counts()
        .reindex(["Positif", "Negatif", "Netral"], fill_value=0)
        .reset_index()
    )

    distribusi.columns = ["Sentimen", "Jumlah"]

    # Warna batang
    warna = {
        "Positif": "#2ecc71",
        "Negatif": "#e74c3c",
        "Netral": "#95a5a6"
    }

    fig = px.bar(
        distribusi,
        x="Sentimen",
        y="Jumlah",
        color="Sentimen",
        text="Jumlah",
        color_discrete_map=warna
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        title=f"Distribusi Sentimen Dataset {nama}",
        xaxis_title="Label Sentimen",
        yaxis_title="Jumlah Data",
        showlegend=False,
        height=550
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "😊 Positif",
        int(distribusi.loc[
            distribusi["Sentimen"]=="Positif",
            "Jumlah"
        ].values[0])
    )

    c2.metric(
        "😠 Negatif",
        int(distribusi.loc[
            distribusi["Sentimen"]=="Negatif",
            "Jumlah"
        ].values[0])
    )

    c3.metric(
        "😐 Netral",
        int(distribusi.loc[
            distribusi["Sentimen"]=="Netral",
            "Jumlah"
        ].values[0])
    )

    st.markdown("### Tabel Distribusi Label")

    st.dataframe(
        distribusi,
        use_container_width=True,
        hide_index=True
    )

elif menu=="☁️ Word Cloud":

    st.title("☁️ Word Cloud Sentimen")

    st.markdown("""
Visualisasi Word Cloud digunakan untuk melihat kata-kata yang paling sering muncul
berdasarkan **dataset** dan **label sentimen**.
""")

    st.markdown("---")

    # ==========================================
    # MEMILIH DATASET DAN LABEL
    # ==========================================

    col1, col2 = st.columns(2)

    with col1:

        dataset_pilih = st.selectbox(
            "📂 Pilih Dataset",
            [
                "YouTube",
                "TikTok"
            ]
        )

    with col2:

        label_pilih = st.selectbox(
            "😊 Pilih Sentimen",
            [
                "Positif",
                "Netral",
                "Negatif"
            ]
        )

    # ==========================================
    # MEMILIH DATASET
    # ==========================================

    if dataset_pilih == "YouTube":

        data = word_yt.copy()

    else:

        data = word_tt.copy()

    # ==========================================
    # FILTER BERDASARKAN LABEL
    # ==========================================

    data = data[
        data["sentiment_label_final"] == label_pilih
    ].copy()

    st.markdown(f"## {dataset_pilih} - {label_pilih}")

    # ==========================================
    # JIKA DATA KOSONG
    # ==========================================

    if len(data) == 0:

        st.warning("Data untuk sentimen ini tidak ditemukan.")

    else:

        # ======================================
        # MEMILIH KOLOM WORDCLOUD
        # ======================================

        # Prioritas:
        # wordcloud
        # stemming
        # tokenizing
        # clean_normalized
        # text

        if "wordcloud" in data.columns:

            text = " ".join(

                data["wordcloud"]

                .fillna("")

                .astype(str)

            )

        elif "stemming" in data.columns:

            text = " ".join(

                data["stemming"]

                .fillna("")

                .astype(str)

            )

        elif "tokenizing" in data.columns:

            text = " ".join(

                data["tokenizing"]

                .fillna("")

                .astype(str)

                .str.replace("[", "", regex=False)

                .str.replace("]", "", regex=False)

                .str.replace("'", "", regex=False)

                .str.replace(",", " ", regex=False)

            )

        elif "clean_normalized" in data.columns:

            text = " ".join(

                data["clean_normalized"]

                .fillna("")

                .astype(str)

            )

        else:

            text = " ".join(

                data["text"]

                .fillna("")

                .astype(str)

            )

        # ======================================
        # MEMBERSIHKAN SPASI
        # ======================================

        text = " ".join(text.split())

        # ======================================
        # CUSTOM STOPWORDS
        # ======================================

        stop_words = {

            "mobil",
            "listrik",
            "kendaraan",
            "indonesia",
            "ev",

            "yg",
            "nya",
            "aja",
            "nih",
            "sih",
            "bang",
            "bro",
            "deh",
            "dong",
            "kan",

            "saya",
            "aku",
            "kami",
            "kita",
            "orang",

            "buat",
            "jadi",
            "karena",
            "kalau",
            "kalo",

            "lebih",
            "masih",
            "cukup",
            "banyak",

            "itu",
            "ini",

            "the",
            "and"
        }
        # ======================================
        # MEMBUAT WORD CLOUD
        # ======================================

        wc = WordCloud(
            width=1800,
            height=900,
            background_color="white",
            stopwords=stop_words,
            max_words=200,
            collocations=False,
            contour_width=1,
            contour_color="steelblue",
            colormap="viridis"
        ).generate(text)

        fig, ax = plt.subplots(figsize=(18, 9))

        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")

        st.pyplot(fig, use_container_width=True)

        st.markdown("---")

        # ======================================
        # METRIC
        # ======================================

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Jumlah Komentar",
            len(data)
        )

        c2.metric(
            "Jumlah Kata Unik",
            len(wc.words_)
        )

        c3.metric(
            "Total Kata",
            len(text.split())
        )

        st.markdown("---")

        # ======================================
        # TOP 20 WORD
        # ======================================

        st.subheader("20 Kata yang Paling Sering Muncul")

        top_word = (
            pd.DataFrame(
                wc.words_.items(),
                columns=["Kata", "Frekuensi"]
            )
            .sort_values(
                "Frekuensi",
                ascending=False
            )
            .head(20)
        )

        fig2 = px.bar(
            top_word,
            x="Frekuensi",
            y="Kata",
            orientation="h",
            text="Frekuensi",
            color="Frekuensi",
            color_continuous_scale="viridis"
        )

        fig2.update_layout(
            height=700,
            yaxis=dict(
                categoryorder="total ascending"
            ),
            coloraxis_showscale=False,
            title="Top 20 Kata"
        )

        fig2.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        st.markdown("---")

        # ======================================
        # TABEL
        # ======================================

        with st.expander("Lihat Tabel Frekuensi Kata"):

            st.dataframe(
                top_word,
                use_container_width=True,
                hide_index=True
            )

        # ======================================
        # DOWNLOAD
        # ======================================

        csv = top_word.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Top Words",
            data=csv,
            file_name=f"top_word_{dataset_pilih}_{label_pilih}.csv",
            mime="text/csv"
        )

elif menu=="🔤 Top Words":
    st.title("Top Words")
    st.warning("Fitur akan dilengkapi pada Part 005.")

elif menu=="📈 Evaluasi Model":
    st.title("Evaluasi Model")
    st.warning("Fitur akan dilengkapi pada Part 006.")
