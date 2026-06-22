!pip install -q streamlit pandas openpyxl plotly matplotlib wordcloud pyngrok
# ==========================================================
# IMPORT LIBRARY
# ==========================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from streamlit.components.v1 import html

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Analisis Sentimen Mobil Listrik",
    page_icon="🚗",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

[data-testid="stSidebar"]{
    background-color:#111827;
}

h1,h2,h3,h4{
    color:white;
}

.metric-card{
    background-color:#1F2937;
    padding:20px;
    border-radius:15px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# PATH
# ==========================================================

import os
from google.colab import drive

# =====================================================
# MOUNT GOOGLE DRIVE
# =====================================================

drive.mount('/content/drive')
BASE_PATH = "/content/drive/MyDrive/STREAMLIT_SKRIPSI"


# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    yt_sentimen = pd.read_excel(
        f"{BASE_PATH}/dataset_sentimen_final_Youtube.xlsx"
    )

    tt_sentimen = pd.read_excel(
        f"{BASE_PATH}/dataset_sentimen_final Tiktok Baru.xlsx"
    )

    yt_aspek = pd.read_excel(
        f"{BASE_PATH}/dataset_final_aspek_Youtube.xlsx"
    )

    tt_aspek = pd.read_excel(
        f"{BASE_PATH}/dataset_final_aspek Tiktok.xlsx"
    )

    lda_yt = pd.read_excel(
        f"{BASE_PATH}/top_words_lda Youtube.xlsx"
    )

    lda_tt = pd.read_excel(
        f"{BASE_PATH}/top_words_lda Tiktok.xlsx"
    )

    xgb_yt = pd.read_excel(
        f"{BASE_PATH}/ringkasan_xgboost_Youtube.xlsx"
    )

    xgb_tt = pd.read_excel(
        f"{BASE_PATH}/ringkasan_xgboost_Tiktok.xlsx"
    )

    lstm_yt = pd.read_excel(
        f"{BASE_PATH}/ringkasan_lstm_Youtube.xlsx"
    )

    lstm_tt = pd.read_excel(
        f"{BASE_PATH}/ringkasan_lstm_Tiktok.xlsx"
    )

    return (
        yt_sentimen,
        tt_sentimen,
        yt_aspek,
        tt_aspek,
        lda_yt,
        lda_tt,
        xgb_yt,
        xgb_tt,
        lstm_yt,
        lstm_tt
    )

(
yt_sentimen,
tt_sentimen,
yt_aspek,
tt_aspek,
lda_yt,
lda_tt,
xgb_yt,
xgb_tt,
lstm_yt,
lstm_tt
)=load_data()

# ==========================================================
# DATA GABUNGAN
# ==========================================================

dataset = pd.concat(
    [yt_sentimen, tt_sentimen],
    ignore_index=True
)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🚗 Analisis Sentimen")

menu = st.sidebar.selectbox(
    "Pilih Menu",
    [
        "Dashboard",
        "Distribusi Sentimen",
        "Perbandingan Dataset",
        "Analisis Aspek",
        "WordCloud",
        "Topik LDA",
        "PyLDAvis",
        "Evaluasi XGBoost",
        "Evaluasi LSTM",
        "Perbandingan Model",
        "Data Komentar"
    ]
)

# ==========================================================
# DASHBOARD
# ==========================================================

if menu == "Dashboard":

    st.title("🚗 Dashboard Analisis Sentimen Mobil Listrik")

    total_data = len(dataset)
    total_yt = len(yt_sentimen)
    total_tt = len(tt_sentimen)

    positif = (
        dataset["sentiment_label_final"]
        .astype(str)
        .str.contains("Positif", case=False)
        .sum()
    )

    netral = (
        dataset["sentiment_label_final"]
        .astype(str)
        .str.contains("Netral", case=False)
        .sum()
    )

    negatif = (
        dataset["sentiment_label_final"]
        .astype(str)
        .str.contains("Negatif", case=False)
        .sum()
    )

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "Total Data",
        f"{total_data:,}"
    )

    c2.metric(
        "Youtube",
        f"{total_yt:,}"
    )

    c3.metric(
        "TikTok",
        f"{total_tt:,}"
    )

    c4,c5,c6 = st.columns(3)

    c4.metric(
        "Positif",
        f"{positif:,}"
    )

    c5.metric(
        "Netral",
        f"{netral:,}"
    )

    c6.metric(
        "Negatif",
        f"{negatif:,}"
    )

    st.divider()

    st.subheader("Distribusi Sentimen Keseluruhan")

    sentimen = (
        dataset["sentiment_label_final"]
        .value_counts()
        .reset_index()
    )

    sentimen.columns = [
        "Sentimen",
        "Jumlah"
    ]

    fig = px.bar(
        sentimen,
        x="Sentimen",
        y="Jumlah",
        text="Jumlah",
        color="Sentimen",
        title="Distribusi Sentimen Seluruh Dataset"
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
# ==========================================================
# DISTRIBUSI SENTIMEN
# ==========================================================

elif menu == "Distribusi Sentimen":

    st.title("📊 Distribusi Sentimen")

    sumber = st.radio(
        "Pilih Dataset",
        ["Gabungan", "Youtube", "TikTok"],
        horizontal=True
    )

    if sumber == "Youtube":
        data = yt_sentimen

    elif sumber == "TikTok":
        data = tt_sentimen

    else:
        data = dataset

    sentimen = (
        data["sentiment_label_final"]
        .value_counts()
        .reset_index()
    )

    sentimen.columns = [
        "Sentimen",
        "Jumlah"
    ]

    fig = px.bar(
        sentimen,
        x="Sentimen",
        y="Jumlah",
        text="Jumlah",
        color="Sentimen",
        title=f"Distribusi Sentimen Dataset {sumber}"
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        sentimen,
        use_container_width=True
    )

# ==========================================================
# PERBANDINGAN DATASET
# ==========================================================

elif menu == "Perbandingan Dataset":

    st.title("⚖️ Perbandingan Youtube vs TikTok")

    yt = (
        yt_sentimen["sentiment_label_final"]
        .value_counts()
        .reset_index()
    )

    yt.columns = [
        "Sentimen",
        "Jumlah"
    ]

    yt["Dataset"] = "Youtube"

    tt = (
        tt_sentimen["sentiment_label_final"]
        .value_counts()
        .reset_index()
    )

    tt.columns = [
        "Sentimen",
        "Jumlah"
    ]

    tt["Dataset"] = "TikTok"

    gabung = pd.concat(
        [yt, tt],
        ignore_index=True
    )

    fig = px.bar(
        gabung,
        x="Sentimen",
        y="Jumlah",
        color="Dataset",
        barmode="group",
        text="Jumlah",
        title="Perbandingan Sentimen Youtube dan TikTok"
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        gabung,
        use_container_width=True
    )

# ==========================================================
# ANALISIS ASPEK
# ==========================================================

elif menu == "Analisis Aspek":

    st.title("🎯 Analisis Aspek")

    sumber = st.radio(
        "Pilih Dataset",
        ["Gabungan", "Youtube", "TikTok"],
        horizontal=True
    )

    if sumber == "Youtube":
        data = yt_sentimen

    elif sumber == "TikTok":
        data = tt_sentimen

    else:
        data = dataset

    aspek = (
        data["label_aspect"]
        .value_counts()
        .reset_index()
    )

    aspek.columns = [
        "Aspek",
        "Jumlah"
    ]

    fig = px.bar(
        aspek,
        x="Aspek",
        y="Jumlah",
        text="Jumlah",
        color="Jumlah",
        title=f"Distribusi Aspek Dataset {sumber}"
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        aspek,
        use_container_width=True
    )

# ==========================================================
# WORDCLOUD
# ==========================================================

elif menu == "WordCloud":

    st.title("☁️ WordCloud")

    sumber = st.radio(
        "Dataset",
        ["Gabungan", "Youtube", "TikTok"],
        horizontal=True
    )

    if sumber == "Youtube":
        data = yt_sentimen

    elif sumber == "TikTok":
        data = tt_sentimen

    else:
        data = dataset

    pilihan = st.selectbox(
        "Pilih Sentimen",
        ["Positif", "Netral", "Negatif"]
    )

    temp = data[
        data["sentiment_label_final"]
        .astype(str)
        .str.lower()
        == pilihan.lower()
    ]

    teks = " ".join(
        temp["text"].astype(str)
    )

    if len(teks.strip()) == 0:

        st.warning(
            "Tidak ada data untuk ditampilkan."
        )

    else:

        wc = WordCloud(
            width=1500,
            height=800,
            background_color="black",
            colormap="viridis",
            max_words=200
        ).generate(teks)

        fig, ax = plt.subplots(
            figsize=(16,8)
        )

        ax.imshow(wc)

        ax.axis("off")

        st.pyplot(fig)

        st.caption(
            f"WordCloud Sentimen {pilihan} - Dataset {sumber}"
        )
    # ==========================================================
# TOPIK LDA
# ==========================================================

elif menu == "Topik LDA":

    st.title("🧠 Topic Modeling LDA")

    sumber = st.radio(
        "Pilih Dataset",
        ["Youtube","TikTok"],
        horizontal=True
    )

    if sumber == "Youtube":

        st.subheader("Topik LDA Youtube")

        st.dataframe(
            lda_yt,
            use_container_width=True
        )

    else:

        st.subheader("Topik LDA TikTok")

        st.dataframe(
            lda_tt,
            use_container_width=True
        )

# ==========================================================
# PYLDAVIS
# ==========================================================

elif menu == "PyLDAvis":

    st.title("📌 Visualisasi PyLDAvis")

    sumber = st.radio(
        "Pilih Dataset",
        ["Youtube","TikTok"],
        horizontal=True
    )

    if sumber == "Youtube":

        file_html = (
            f"{BASE_PATH}/pyldavis_k10_Youtube (2).html"
        )

    else:

        file_html = (
            f"{BASE_PATH}/pyldavis_k10 Tiktok.html"
        )

    with open(
        file_html,
        "r",
        encoding="utf-8"
    ) as f:

        source_code = f.read()

    html(
        source_code,
        height=900,
        scrolling=True
    )

# ==========================================================
# EVALUASI XGBOOST
# ==========================================================

elif menu == "Evaluasi XGBoost":

    st.title("🚀 Evaluasi XGBoost")

    tab1, tab2 = st.tabs(
        ["Youtube","TikTok"]
    )

    with tab1:

        st.subheader(
            "Hasil Evaluasi Youtube"
        )

        st.dataframe(
            xgb_yt,
            use_container_width=True
        )

    with tab2:

        st.subheader(
            "Hasil Evaluasi TikTok"
        )

        st.dataframe(
            xgb_tt,
            use_container_width=True
        )

# ==========================================================
# EVALUASI LSTM
# ==========================================================

elif menu == "Evaluasi LSTM":

    st.title("🤖 Evaluasi LSTM")

    tab1, tab2 = st.tabs(
        ["Youtube","TikTok"]
    )

    with tab1:

        st.subheader(
            "Hasil Evaluasi Youtube"
        )

        st.dataframe(
            lstm_yt,
            use_container_width=True
        )

    with tab2:

        st.subheader(
            "Hasil Evaluasi TikTok"
        )

        st.dataframe(
            lstm_tt,
            use_container_width=True
        )

# ==========================================================
# PERBANDINGAN MODEL
# ==========================================================

elif menu == "Perbandingan Model":

    st.title("⚖️ Perbandingan Model")

    st.subheader(
        "Ringkasan XGBoost"
    )

    st.dataframe(
        xgb_yt,
        use_container_width=True
    )

    st.subheader(
        "Ringkasan LSTM"
    )

    st.dataframe(
        lstm_yt,
        use_container_width=True
    )

# ==========================================================
# DATA KOMENTAR
# ==========================================================

elif menu == "Data Komentar":

    st.title("📝 Data Komentar")

    sumber = st.radio(
        "Pilih Dataset",
        ["Gabungan","Youtube","TikTok"],
        horizontal=True
    )

    if sumber == "Youtube":

        data = yt_sentimen

    elif sumber == "TikTok":

        data = tt_sentimen

    else:

        data = dataset

    st.dataframe(
        data,
        use_container_width=True
    )

    csv = data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Dataset",
        data=csv,
        file_name=f"{sumber}.csv",
        mime="text/csv"
    )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    """
    <center>
    <h5>
    Dashboard Analisis Sentimen Mobil Listrik Indonesia
    <br>
    Menggunakan LDA, XGBoost dan LSTM
    </h5>
    </center>
    """,
    unsafe_allow_html=True
)
