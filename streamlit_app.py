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
Visualisasi **Word Cloud** digunakan untuk melihat kata-kata yang
paling sering muncul berdasarkan **dataset** dan **label sentimen**.
""")

    st.markdown("---")

    # =====================================================
    # PILIH DATASET
    # =====================================================

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

    st.markdown("---")

    # =====================================================
    # MEMILIH DATASET
    # =====================================================

    if dataset_pilih == "YouTube":

        data = word_yt.copy()

    else:

        data = word_tt.copy()

    # =====================================================
    # NORMALISASI NAMA KOLOM
    # =====================================================

    data.columns = (

        data.columns

        .str.strip()

        .str.lower()

    )

    # =====================================================
    # VALIDASI KOLOM
    # =====================================================

    if "sentimen" not in data.columns:

        st.error(

            "Kolom 'sentimen' tidak ditemukan."

        )

        st.write(data.columns.tolist())

        st.stop()

    if "text_wordcloud" not in data.columns:

        st.error(

            "Kolom 'text_wordcloud' tidak ditemukan."

        )

        st.write(data.columns.tolist())

        st.stop()

    # =====================================================
    # MEMBERSIHKAN LABEL SENTIMEN
    # =====================================================

    data["sentimen"] = (

        data["sentimen"]

        .fillna("")

        .astype(str)

        .str.strip()

        .str.title()

    )

    # =====================================================
    # FILTER SENTIMEN
    # =====================================================

    data = data[

        data["sentimen"] == label_pilih

    ].copy()

    # =====================================================
    # JIKA DATA KOSONG
    # =====================================================

    if data.empty:

        st.warning(

            f"Tidak ada data sentimen {label_pilih}."

        )

        st.stop()

    # =====================================================
    # MEMBERSIHKAN TEXT
    # =====================================================

    text = " ".join(

        data["text_wordcloud"]

        .fillna("")

        .astype(str)

    )

    text = (

        text

        .replace("[", " ")

        .replace("]", " ")

        .replace("'", " ")

        .replace('"', " ")

        .replace(",", " ")

    )

    text = " ".join(text.split())

    # =====================================================
    # STOPWORDS
    # =====================================================

    stop_words = {

        "mobil",

        "listrik",

        "kendaraan",

        "ev",

        "indonesia",

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

        "aku",

        "saya",

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

    # =====================================================
    # LANJUT PART 2
    # =====================================================
    # =====================================================
    # MEMBUAT WORD CLOUD
    # =====================================================

    wc = WordCloud(

        width=1800,

        height=900,

        background_color="white",

        stopwords=stop_words,

        max_words=200,

        collocations=False,

        contour_width=1,

        contour_color="steelblue",

        colormap="viridis",

        random_state=42

    ).generate(text)

    # =====================================================
    # TAMPILKAN WORD CLOUD
    # =====================================================

    st.subheader(

        f"☁️ Word Cloud {dataset_pilih} - {label_pilih}"

    )

    fig, ax = plt.subplots(

        figsize=(18,9)

    )

    ax.imshow(

        wc,

        interpolation="bilinear"

    )

    ax.axis("off")

    st.pyplot(

        fig,

        use_container_width=True

    )

    plt.close(fig)

    st.markdown("---")

    # =====================================================
    # HITUNG FREKUENSI KATA
    # =====================================================

    from collections import Counter

    semua_kata = []

    for kalimat in data["text_wordcloud"]:

        kata = (

            str(kalimat)

            .lower()

            .split()

        )

        kata = [

            x

            for x in kata

            if x not in stop_words

            and len(x) > 1

        ]

        semua_kata.extend(kata)

    counter = Counter(

        semua_kata

    )

    top_word = (

        pd.DataFrame(

            counter.items(),

            columns=[

                "Kata",

                "Frekuensi"

            ]

        )

        .sort_values(

            by="Frekuensi",

            ascending=False

        )

        .reset_index(

            drop=True

        )

    )

    if top_word.empty:

        st.warning(

            "Tidak ada kata yang dapat ditampilkan."

        )

        st.stop()

    st.markdown("---")

    # =====================================================
    # METRIC
    # =====================================================

    c1, c2, c3 = st.columns(3)

    c1.metric(

        "Jumlah Komentar",

        len(data)

    )

    c2.metric(

        "Jumlah Kata",

        len(semua_kata)

    )

    c3.metric(

        "Kata Unik",

        len(counter)

    )

    st.markdown("---")

    # =====================================================
    # TOP 20 WORD
    # =====================================================

    st.subheader(

        "📈 Top 20 Kata"

    )

    top20 = top_word.head(20)

    fig2 = px.bar(

        top20,

        x="Frekuensi",

        y="Kata",

        orientation="h",

        color="Frekuensi",

        text="Frekuensi",

        color_continuous_scale="viridis"

    )

    fig2.update_layout(

        height=700,

        yaxis=dict(

            categoryorder="total ascending"

        ),

        title=f"Top 20 Kata ({dataset_pilih} - {label_pilih})",

        coloraxis_showscale=False

    )

    fig2.update_traces(

        textposition="outside"

    )

    st.plotly_chart(

        fig2,

        use_container_width=True

    )

    st.markdown("---")

    # =====================================================
    # LANJUT PART 3
    # =====================================================
    # =====================================================
    # TABEL TOP 20 KATA
    # =====================================================

    st.subheader("📋 Tabel Top 20 Kata")

    tampil_top = top20.copy()

    tampil_top.index = tampil_top.index + 1

    tampil_top.index.name = "No"

    st.dataframe(

        tampil_top,

        use_container_width=True,

        height=500

    )

    st.markdown("---")

    # =====================================================
    # PENCARIAN KATA
    # =====================================================

    st.subheader("🔍 Pencarian Kata")

    keyword = st.text_input(

        "Masukkan kata",

        placeholder="Contoh : baterai"

    )

    if keyword:

        hasil = top_word[

            top_word["Kata"]

            .str.contains(

                keyword,

                case=False,

                na=False

            )

        ]

        if hasil.empty:

            st.warning(

                "Kata tidak ditemukan."

            )

        else:

            st.success(

                f"Ditemukan {len(hasil)} kata."

            )

            st.dataframe(

                hasil,

                use_container_width=True,

                hide_index=True

            )

    st.markdown("---")

    # =====================================================
    # DOWNLOAD TOP WORD
    # =====================================================

    csv = top20.to_csv(

        index=False

    ).encode(

        "utf-8"

    )

    st.download_button(

        label="⬇️ Download Top 20 Kata",

        data=csv,

        file_name=f"top20_{dataset_pilih}_{label_pilih}.csv",

        mime="text/csv"

    )

    st.markdown("---")

    # =====================================================
    # DISTRIBUSI SENTIMEN
    # =====================================================

    st.subheader("📊 Distribusi Sentimen Dataset")

    distribusi = (

        word_yt

        if dataset_pilih == "YouTube"

        else word_tt

    )

    distribusi = (

        distribusi["sentimen"]

        .value_counts()

        .rename_axis("Sentimen")

        .reset_index(name="Jumlah")

    )

    warna = {

        "Positif":"#2ecc71",

        "Netral":"#f1c40f",

        "Negatif":"#e74c3c"

    }

    fig3 = px.pie(

        distribusi,

        names="Sentimen",

        values="Jumlah",

        hole=0.45,

        color="Sentimen",

        color_discrete_map=warna

    )

    fig3.update_layout(

        height=500,

        legend_title="Sentimen"

    )

    st.plotly_chart(

        fig3,

        use_container_width=True

    )

    st.markdown("---")

    # =====================================================
    # DATASET HASIL FILTER
    # =====================================================

    with st.expander(

        "📄 Dataset Hasil Filter"

    ):

        st.dataframe(

            data,

            use_container_width=True,

            hide_index=True,

            height=500

        )

    st.markdown("---")

    # =====================================================
    # DOWNLOAD DATASET FILTER
    # =====================================================

    csv_filter = data.to_csv(

        index=False

    ).encode(

        "utf-8"

    )

    st.download_button(

        label="⬇️ Download Dataset Filter",

        data=csv_filter,

        file_name=f"dataset_{dataset_pilih}_{label_pilih}.csv",

        mime="text/csv"

    )

    st.markdown("---")

    # =====================================================
    # RINGKASAN
    # =====================================================

    st.info(

        f"""

**Dataset :** {dataset_pilih}

**Sentimen :** {label_pilih}

**Jumlah Komentar :** {len(data):,}

**Jumlah Kata :** {len(semua_kata):,}

**Jumlah Kata Unik :** {len(counter):,}

**Kata Terbanyak :** {top20.iloc[0]['Kata']}

**Frekuensi :** {top20.iloc[0]['Frekuensi']}

"""

    )

    st.success(

        "✅ Visualisasi Word Cloud berhasil dibuat."

    )


elif menu=="🔤 Top Words":
    st.title("Top Words")
    st.warning("Fitur akan dilengkapi pada Part 005.")

elif menu=="📈 Evaluasi Model":
    st.title("Evaluasi Model")
    st.warning("Fitur akan dilengkapi pada Part 006.")
