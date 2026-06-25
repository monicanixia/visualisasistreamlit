"""
PART 1A - streamlit_app.py
====================================================
Dashboard Analisis Sentimen Mobil Listrik Indonesia
====================================================
"""

import os
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from streamlit.components.v1 import html

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Analisis Sentimen Mobil Listrik",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CSS
# --------------------------------------------------

st.markdown("""
<style>

#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}

.block-container{
padding-top:1rem;
padding-left:2rem;
padding-right:2rem;
padding-bottom:1rem;
background:#F5F7FA;
}

div[data-testid="stMetric"]{
background:white;
padding:15px;
border-radius:12px;
border:1px solid #E3E8EF;
box-shadow:0 2px 8px rgba(0,0,0,.08);
}

h1,h2,h3{
color:#1565C0;
}

section[data-testid="stSidebar"]{
background:#1565C0;
}

section[data-testid="stSidebar"] *{
color:white;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():

    yt_sentimen = pd.read_excel("dataset_sentimen_final_Youtube.xlsx")
    tt_sentimen = pd.read_excel("dataset_sentimen_final Tiktok Baru.xlsx")

    yt_aspek = pd.read_excel("dataset_final_aspek_Youtube.xlsx")
    tt_aspek = pd.read_excel("dataset_final_aspek Tiktok.xlsx")

    lda_yt = pd.read_excel("top_words_lda Youtube.xlsx")
    lda_tt = pd.read_excel("top_words_lda Tiktok.xlsx")

    xgb_yt = pd.read_excel("ringkasan_xgboost_Youtube.xlsx")
    xgb_tt = pd.read_excel("ringkasan_xgboost_Tiktok.xlsx")

    lstm_yt = pd.read_excel("ringkasan_lstm_Youtube.xlsx")
    lstm_tt = pd.read_excel("ringkasan_lstm_Tiktok.xlsx")

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

dataset = pd.concat([yt_sentimen,tt_sentimen],ignore_index=True)

TEXT_COL="text"
LABEL_COL="sentiment_label_final"

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🚗 Analisis Sentimen")
st.sidebar.caption("Dashboard Skripsi")

menu = st.sidebar.radio(
    "Menu",
    [
        "🏠 Dashboard",
        "📂 Dataset",
        "📊 Distribusi Sentimen",
        "📌 Distribusi Aspek",
        "☁️ WordCloud",
        "🔤 Top Words LDA",
        "🌐 PyLDAvis",
        "🤖 XGBoost",
        "🧠 LSTM",
        "📈 Perbandingan Model",
        "📋 Kesimpulan"
    ]
)

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if menu=="🏠 Dashboard":

    st.title("🚗 Dashboard Analisis Sentimen Mobil Listrik Indonesia")

    st.caption(
        "Analisis komentar YouTube dan TikTok menggunakan "
        "LDA, XGBoost, dan LSTM."
    )

    c1,c2,c3,c4=st.columns(4)

    c1.metric("Total Data",f"{len(dataset):,}")
    c2.metric("YouTube",f"{len(yt_sentimen):,}")
    c3.metric("TikTok",f"{len(tt_sentimen):,}")
    c4.metric("Topik LDA",lda_yt["topic_id"].nunique())

    st.divider()

    dist = (
        dataset[LABEL_COL]
        .value_counts()
        .reset_index()
    )

    dist.columns=["Sentimen","Jumlah"]

    fig = px.bar(
        dist,
        x="Sentimen",
        y="Jumlah",
        color="Sentimen",
        text="Jumlah"
    )

    fig.update_layout(
        template="plotly_white",
        height=500,
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Perbandingan Dataset")

    yt = yt_sentimen[LABEL_COL].value_counts().reset_index()
    yt.columns=["Sentimen","Jumlah"]
    yt["Dataset"]="YouTube"

    tt = tt_sentimen[LABEL_COL].value_counts().reset_index()
    tt.columns=["Sentimen","Jumlah"]
    tt["Dataset"]="TikTok"

    gabung = pd.concat([yt,tt])

    fig2 = px.bar(
        gabung,
        x="Sentimen",
        y="Jumlah",
        color="Dataset",
        barmode="group",
        text="Jumlah"
    )

    fig2.update_layout(
        template="plotly_white",
        height=500
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.info(
        "Lanjutkan dengan PART 1B untuk halaman berikutnya."
    )
"""
PART 1B - streamlit_app.py
Lanjutkan setelah PART 1A
"""

# ============================================================
# DATASET
# ============================================================

elif menu == "📂 Dataset":

    st.title("📂 Dataset Hasil Preprocessing")

    dataset_opsi = st.selectbox(
        "Pilih Dataset",
        ["YouTube","TikTok"]
    )

    data = yt_sentimen if dataset_opsi=="YouTube" else tt_sentimen

    a,b,c,d = st.columns(4)

    a.metric("Jumlah Data", len(data))
    b.metric("Jumlah Kolom", len(data.columns))
    c.metric("Missing Value", int(data.isnull().sum().sum()))
    d.metric("Duplicate", int(data.duplicated().sum()))

    st.divider()

    st.subheader("Informasi Dataset")
    st.write(data.dtypes)

    st.subheader("Pilih Kolom")

    kolom = st.multiselect(
        "Kolom yang ditampilkan",
        list(data.columns),
        default=list(data.columns)
    )

    tampil = data[kolom]

    keyword = st.text_input("🔍 Cari komentar")

    if keyword:
        tampil = tampil[
            tampil.astype(str)
            .apply(lambda x: x.str.contains(keyword, case=False, na=False))
            .any(axis=1)
        ]

    jumlah = st.slider(
        "Jumlah baris",
        10,
        min(500, len(tampil)),
        min(100, len(tampil))
    )

    st.dataframe(
        tampil.head(jumlah),
        use_container_width=True,
        height=600
    )

    csv = tampil.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Dataset",
        csv,
        file_name=f"{dataset_opsi}.csv",
        mime="text/csv"
    )

# ============================================================
# DISTRIBUSI SENTIMEN
# ============================================================

elif menu == "📊 Distribusi Sentimen":

    st.title("📊 Distribusi Sentimen")

    dataset_opsi = st.selectbox(
        "Pilih Dataset",
        ["YouTube","TikTok"]
    )

    data = yt_sentimen if dataset_opsi=="YouTube" else tt_sentimen

    distribusi = (
        data[LABEL_COL]
        .value_counts()
        .reset_index()
    )

    distribusi.columns = [
        "Sentimen",
        "Jumlah"
    ]

    distribusi["Persentase"] = (
        distribusi["Jumlah"] /
        distribusi["Jumlah"].sum() * 100
    ).round(2)

    positif = distribusi.loc[
        distribusi["Sentimen"].str.lower()=="positif",
        "Jumlah"
    ].sum()

    netral = distribusi.loc[
        distribusi["Sentimen"].str.lower()=="netral",
        "Jumlah"
    ].sum()

    negatif = distribusi.loc[
        distribusi["Sentimen"].str.lower()=="negatif",
        "Jumlah"
    ].sum()

    c1,c2,c3 = st.columns(3)

    c1.metric("😊 Positif", int(positif))
    c2.metric("😐 Netral", int(netral))
    c3.metric("😠 Negatif", int(negatif))

    st.divider()

    fig = px.bar(
        distribusi,
        x="Sentimen",
        y="Jumlah",
        color="Sentimen",
        text="Jumlah"
    )

    fig.update_layout(
        template="plotly_white",
        height=550,
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Persentase Sentimen")

    fig2 = px.bar(
        distribusi,
        y="Sentimen",
        x="Persentase",
        orientation="h",
        color="Sentimen",
        text="Persentase"
    )

    fig2.update_layout(
        template="plotly_white",
        height=400,
        showlegend=False
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.subheader("Tabel Distribusi")

    st.dataframe(
        distribusi,
        use_container_width=True
    )

    st.download_button(
        "⬇ Download Distribusi",
        distribusi.to_csv(index=False).encode(),
        "distribusi_sentimen.csv",
        "text/csv"
    )

# ============================================================
# DISTRIBUSI ASPEK
# ============================================================

elif menu == "📌 Distribusi Aspek":

    st.title("📌 Distribusi Aspek")

    dataset_opsi = st.selectbox(
        "Pilih Dataset",
        ["YouTube","TikTok"]
    )

    aspek = yt_aspek if dataset_opsi=="YouTube" else tt_aspek

    aspek_count = (
        aspek["label_aspect"]
        .value_counts()
        .reset_index()
    )

    aspek_count.columns = [
        "Aspek",
        "Jumlah"
    ]

    fig = px.bar(
        aspek_count,
        y="Aspek",
        x="Jumlah",
        orientation="h",
        color="Jumlah",
        text="Jumlah"
    )

    fig.update_layout(
        template="plotly_white",
        height=600
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        aspek_count,
        use_container_width=True
    )
"""
PART 2A - streamlit_app.py
WordCloud + Top Words LDA
Tempelkan setelah PART 1B
"""

from collections import Counter

# ============================================================
# WORD CLOUD
# ============================================================

elif menu == "☁️ WordCloud":

    st.title("☁️ WordCloud Sentimen")

    c1,c2=st.columns(2)

    with c1:
        dataset_opsi=st.selectbox(
            "Pilih Dataset",
            ["YouTube","TikTok"]
        )

    with c2:
        label_opsi=st.selectbox(
            "Pilih Label",
            ["Positif","Netral","Negatif"]
        )

    data=yt_sentimen if dataset_opsi=="YouTube" else tt_sentimen

    data=data[
        data[LABEL_COL].astype(str).str.lower()
        ==
        label_opsi.lower()
    ]

    st.metric("Jumlah Komentar",len(data))

    if len(data)==0:

        st.warning("Data tidak tersedia.")

    else:

        text=" ".join(
            data[TEXT_COL].fillna("").astype(str)
        )

        wc=WordCloud(
            width=1800,
            height=900,
            background_color="white",
            colormap="Blues",
            max_words=300
        ).generate(text)

        fig,ax=plt.subplots(figsize=(16,8))
        ax.imshow(wc)
        ax.axis("off")

        st.pyplot(fig,use_container_width=True)

        kata=text.split()

        top=Counter(kata).most_common(20)

        top=pd.DataFrame(
            top,
            columns=["Kata","Frekuensi"]
        )

        kiri,kanan=st.columns(2)

        with kiri:

            st.subheader("Top 20 Kata")

            st.dataframe(
                top,
                use_container_width=True,
                height=500
            )

        with kanan:

            fig2=px.bar(
                top,
                y="Kata",
                x="Frekuensi",
                orientation="h",
                text="Frekuensi"
            )

            fig2.update_layout(
                template="plotly_white",
                height=500,
                showlegend=False
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

# ============================================================
# TOP WORDS LDA
# ============================================================

elif menu=="🔤 Top Words LDA":

    st.title("🔤 Top Words LDA")

    dataset_opsi=st.selectbox(
        "Dataset",
        ["YouTube","TikTok"]
    )

    lda=lda_yt if dataset_opsi=="YouTube" else lda_tt

    topic=st.selectbox(
        "Pilih Topik",
        sorted(lda["topic_id"].unique())
    )

    hasil=lda[
        lda["topic_id"]==topic
    ]

    st.dataframe(
        hasil,
        use_container_width=True,
        height=500
    )

    if "top_words" in hasil.columns:

        kata=hasil.iloc[0]["top_words"]

        daftar=[i.strip() for i in str(kata).split(",")]

        df=pd.DataFrame({
            "Top Words":daftar,
            "Urutan":range(1,len(daftar)+1)
        })

        st.subheader("Daftar Top Words")

        st.dataframe(
            df,
            use_container_width=True
        )

        fig=px.bar(
            df,
            x="Urutan",
            y="Top Words",
            orientation="h"
        )

        fig.update_layout(
            template="plotly_white",
            height=600
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.download_button(
        "⬇ Download Top Words",
        hasil.to_csv(index=False).encode(),
        "top_words.csv",
        "text/csv"
    )
"""
PART 2B - streamlit_app.py
PyLDAvis + Evaluasi XGBoost
Tempelkan setelah PART 2A
"""

# ============================================================
# PYLDAVIS
# ============================================================

elif menu == "🌐 PyLDAvis":

    st.title("🌐 Visualisasi Topik (PyLDAvis)")

    dataset_opsi = st.selectbox(
        "Pilih Dataset",
        ["YouTube","TikTok"]
    )

    html_file = (
        "pyldavis_k10_Youtube.html"
        if dataset_opsi=="YouTube"
        else "pyldavis_k10_Tiktok.html"
    )

    if os.path.exists(html_file):
        with open(html_file,"r",encoding="utf-8") as f:
            html(f.read(),height=900,scrolling=True)
    else:
        st.error(f"File {html_file} tidak ditemukan.")

# ============================================================
# XGBOOST
# ============================================================

elif menu == "🤖 XGBoost":

    st.title("🤖 Evaluasi Model XGBoost")

    dataset_opsi = st.selectbox(
        "Pilih Dataset",
        ["YouTube","TikTok"]
    )

    ringkasan = (
        xgb_yt if dataset_opsi=="YouTube"
        else xgb_tt
    )

    rasio = st.selectbox(
        "Split Data",
        ringkasan["Rasio"].tolist()
    )

    hasil = ringkasan[
        ringkasan["Rasio"]==rasio
    ]

    st.subheader("Ringkasan Evaluasi")

    st.dataframe(
        hasil,
        use_container_width=True
    )

    c1,c2,c3,c4=st.columns(4)

    c1.metric(
        "Accuracy",
        f'{hasil["Test Accuracy"].iloc[0]:.4f}'
    )

    c2.metric(
        "Precision",
        f'{hasil["Macro Precision"].iloc[0]:.4f}'
    )

    c3.metric(
        "Recall",
        f'{hasil["Macro Recall"].iloc[0]:.4f}'
    )

    c4.metric(
        "F1 Score",
        f'{hasil["Macro F1"].iloc[0]:.4f}'
    )

    st.divider()

    st.subheader("Confusion Matrix")

    file_map = {
        ("YouTube","80:20"):
            "cm_xgboost_80_20_youtube.png",
        ("YouTube","70:30"):
            "cm_xgboost_70_30_youtube.png",
        ("YouTube","60:40"):
            "cm_xgboost_60_40_youtube.png",
        ("TikTok","80:20"):
            "cm_xgboost_80_20_tiktok.png",
        ("TikTok","70:30"):
            "cm_xgboost_70_30_tiktok.png",
        ("TikTok","60:40"):
            "cm_xgboost_60_40_tiktok.png",
    }

    img=file_map.get((dataset_opsi,rasio))

    if img and os.path.exists(img):
        st.image(img,use_container_width=True)
    else:
        st.warning("Confusion Matrix belum ditemukan.")

    st.download_button(
        "⬇ Download Ringkasan",
        ringkasan.to_csv(index=False).encode("utf-8"),
        file_name="ringkasan_xgboost.csv",
        mime="text/csv"
    )
"""
PART 3A - streamlit_app.py
LSTM + Perbandingan Model
Tempelkan setelah PART 2B
"""

# ============================================================
# LSTM
# ============================================================

elif menu == "🧠 LSTM":

    st.title("🧠 Evaluasi Model LSTM")

    dataset_opsi = st.selectbox("Pilih Dataset",["YouTube","TikTok"])

    ringkasan = lstm_yt if dataset_opsi=="YouTube" else lstm_tt

    rasio = st.selectbox("Split Data", ringkasan["Rasio"].tolist())

    hasil = ringkasan[ringkasan["Rasio"]==rasio]

    st.dataframe(hasil,use_container_width=True)

    c1,c2,c3,c4=st.columns(4)

    c1.metric("Accuracy",f'{hasil["Accuracy"].iloc[0]:.4f}')
    c2.metric("Precision",f'{hasil["Macro Precision"].iloc[0]:.4f}')
    c3.metric("Recall",f'{hasil["Macro Recall"].iloc[0]:.4f}')
    c4.metric("F1 Score",f'{hasil["Macro F1"].iloc[0]:.4f}')

    st.divider()

    file_map={
        ("YouTube","80:20"):"cm_lstm_80_20_youtube.png",
        ("YouTube","70:30"):"cm_lstm_70_30_youtube.png",
        ("YouTube","60:40"):"cm_lstm_60_40_youtube.png",
        ("TikTok","80:20"):"cm_lstm_80_20_tiktok.png",
        ("TikTok","70:30"):"cm_lstm_70_30_tiktok.png",
        ("TikTok","60:40"):"cm_lstm_60_40_tiktok.png",
    }

    img=file_map.get((dataset_opsi,rasio))

    st.subheader("Confusion Matrix")

    if img and os.path.exists(img):
        st.image(img,use_container_width=True)
    else:
        st.warning("Confusion Matrix belum ditemukan.")

# ============================================================
# PERBANDINGAN MODEL
# ============================================================

elif menu == "📈 Perbandingan Model":

    st.title("📈 Perbandingan Model")

    dataset_opsi=st.selectbox(
        "Dataset",
        ["YouTube","TikTok"]
    )

    xgb=xgb_yt if dataset_opsi=="YouTube" else xgb_tt
    lstm=lstm_yt if dataset_opsi=="YouTube" else lstm_tt

    banding=xgb[["Rasio","Test Accuracy","Macro Precision","Macro Recall","Macro F1"]].copy()

    banding.columns=[
        "Rasio",
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]

    banding["Model"]="XGBoost"

    lst=lstm[["Rasio","Accuracy","Macro Precision","Macro Recall","Macro F1"]].copy()

    lst.columns=[
        "Rasio",
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]

    lst["Model"]="LSTM"

    gabung=pd.concat([banding,lst],ignore_index=True)

    metrik=st.selectbox(
        "Pilih Metrik",
        ["Accuracy","Precision","Recall","F1"]
    )

    fig=px.bar(
        gabung,
        x="Rasio",
        y=metrik,
        color="Model",
        barmode="group",
        text=metrik
    )

    fig.update_layout(
        template="plotly_white",
        height=550
    )

    st.plotly_chart(fig,use_container_width=True)

    st.dataframe(gabung,use_container_width=True)

    st.download_button(
        "⬇ Download Perbandingan",
        gabung.to_csv(index=False).encode("utf-8"),
        file_name="perbandingan_model.csv",
        mime="text/csv"
    )
"""
PART 3B - streamlit_app.py
Kesimpulan + Footer + Utility
Tempelkan setelah PART 3A
"""

# ============================================================
# KESIMPULAN
# ============================================================

elif menu == "📋 Kesimpulan":

    st.title("📋 Kesimpulan Analisis")

    dataset_opsi = st.selectbox(
        "Pilih Dataset",
        ["YouTube","TikTok"]
    )

    xgb = xgb_yt if dataset_opsi=="YouTube" else xgb_tt
    lstm = lstm_yt if dataset_opsi=="YouTube" else lstm_tt

    best_xgb = xgb.sort_values("Test Accuracy", ascending=False).iloc[0]
    best_lstm = lstm.sort_values("Accuracy", ascending=False).iloc[0]

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🏆 XGBoost Terbaik")
        st.metric("Rasio", best_xgb["Rasio"])
        st.metric("Accuracy", f'{best_xgb["Test Accuracy"]:.4f}')
        st.metric("Macro F1", f'{best_xgb["Macro F1"]:.4f}')

    with c2:
        st.subheader("🏆 LSTM Terbaik")
        st.metric("Rasio", best_lstm["Rasio"])
        st.metric("Accuracy", f'{best_lstm["Accuracy"]:.4f}')
        st.metric("Macro F1", f'{best_lstm["Macro F1"]:.4f}')

    st.divider()

    st.subheader("Ringkasan Dataset")

    info = pd.DataFrame({
        "Informasi":[
            "Jumlah Data YouTube",
            "Jumlah Data TikTok",
            "Jumlah Topik LDA",
            "Jumlah Aspek"
        ],
        "Nilai":[
            len(yt_sentimen),
            len(tt_sentimen),
            lda_yt["topic_id"].nunique(),
            yt_aspek["label_aspect"].nunique()
        ]
    })

    st.dataframe(info, use_container_width=True)

    st.subheader("Catatan")

    st.success("""
    Dashboard ini menyajikan hasil analisis sentimen komentar
    mengenai mobil listrik pada platform YouTube dan TikTok.

    Tahapan analisis meliputi:

    • Text Preprocessing
    • Topic Modeling LDA
    • Klasifikasi XGBoost
    • Klasifikasi LSTM
    • Evaluasi Accuracy, Precision, Recall dan Macro F1
    """)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
"""
<div style="text-align:center;
padding:15px;
color:#666;
font-size:14px">

<b>Dashboard Analisis Sentimen Mobil Listrik Indonesia</b><br>

Dibuat menggunakan Streamlit • Plotly • Pandas • WordCloud

</div>
""",
unsafe_allow_html=True
)
