import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from streamlit.components.v1 import html

st.set_page_config(page_title="Analisis Sentimen Mobil Listrik", page_icon="🚗", layout="wide")

@st.cache_data
def load_data():
    yt_sentimen = pd.read_excel("dataset_sentimen_final_Youtube.xlsx")
    tt_sentimen = pd.read_excel("dataset_sentimen_final_Tiktok.xlsx")
    yt_aspek = pd.read_excel("dataset_final_aspek_Youtube.xlsx")
    tt_aspek = pd.read_excel("dataset_final_aspek_Tiktok.xlsx")
    lda_yt = pd.read_excel("top_words_lda_Youtube.xlsx")
    lda_tt = pd.read_excel("top_words_lda_Tiktok.xlsx")
    xgb_yt = pd.read_excel("ringkasan_xgboost_Youtube.xlsx")
    xgb_tt = pd.read_excel("ringkasan_xgboost_Tiktok.xlsx")
    lstm_yt = pd.read_excel("ringkasan_lstm_Youtube.xlsx")
    lstm_tt = pd.read_excel("ringkasan_lstm_Tiktok.xlsx")
    return yt_sentimen, tt_sentimen, yt_aspek, tt_aspek, lda_yt, lda_tt, xgb_yt, xgb_tt, lstm_yt, lstm_tt

yt_sentimen, tt_sentimen, yt_aspek, tt_aspek, lda_yt, lda_tt, xgb_yt, xgb_tt, lstm_yt, lstm_tt = load_data()
dataset = pd.concat([yt_sentimen, tt_sentimen], ignore_index=True)

st.sidebar.title("🚗 Analisis Sentimen Mobil Listrik")
menu = st.sidebar.selectbox("Menu",[
    "Dashboard","Distribusi Sentimen","Perbandingan Dataset","Analisis Aspek",
    "WordCloud","Topik LDA","PyLDAvis","Evaluasi XGBoost",
    "Evaluasi LSTM","Perbandingan Model","Data Komentar"
])

if menu == "Dashboard":
    st.title("Dashboard Analisis Sentimen Mobil Listrik")
    st.metric("Total Data", len(dataset))

    if "sentiment_label_final" in dataset.columns:
        sent = dataset["sentiment_label_final"].value_counts().reset_index()
        sent.columns = ["Sentimen","Jumlah"]
        fig = px.bar(sent,x="Sentimen",y="Jumlah",color="Sentimen",text="Jumlah")
        st.plotly_chart(fig,use_container_width=True)

elif menu == "Distribusi Sentimen":
    sumber = st.radio("Dataset",["Gabungan","Youtube","TikTok"])
    data = dataset if sumber=="Gabungan" else yt_sentimen if sumber=="Youtube" else tt_sentimen
    sent = data["sentiment_label_final"].value_counts().reset_index()
    sent.columns=["Sentimen","Jumlah"]
    st.dataframe(sent,use_container_width=True)
    st.plotly_chart(px.pie(sent,names="Sentimen",values="Jumlah"),use_container_width=True)

elif menu == "Perbandingan Dataset":
    yt = yt_sentimen["sentiment_label_final"].value_counts().reset_index()
    yt.columns=["Sentimen","Jumlah"]
    yt["Dataset"]="Youtube"
    tt = tt_sentimen["sentiment_label_final"].value_counts().reset_index()
    tt.columns=["Sentimen","Jumlah"]
    tt["Dataset"]="TikTok"
    gabung = pd.concat([yt,tt])
    st.plotly_chart(px.bar(gabung,x="Sentimen",y="Jumlah",color="Dataset",barmode="group"),use_container_width=True)

elif menu == "Analisis Aspek":
    sumber = st.radio("Dataset",["Youtube","TikTok"])
    data = yt_aspek if sumber=="Youtube" else tt_aspek
    st.dataframe(data,use_container_width=True)

elif menu == "WordCloud":
    sumber = st.radio("Dataset",["Gabungan","Youtube","TikTok"])
    data = dataset if sumber=="Gabungan" else yt_sentimen if sumber=="Youtube" else tt_sentimen
    if "text" in data.columns:
        text = " ".join(data["text"].astype(str))
        wc = WordCloud(width=1200,height=600,background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(12,6))
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

elif menu == "Topik LDA":
    sumber = st.radio("Dataset",["Youtube","TikTok"])
    st.dataframe(lda_yt if sumber=="Youtube" else lda_tt,use_container_width=True)

elif menu == "PyLDAvis":
    sumber = st.radio("Dataset",["Youtube","TikTok"])
    file_html = "pyldavis_k10_Youtube.html" if sumber=="Youtube" else "pyldavis_k10_Tiktok.html"
    with open(file_html,"r",encoding="utf-8") as f:
        html(f.read(),height=900,scrolling=True)

elif menu == "Evaluasi XGBoost":
    st.dataframe(xgb_yt,use_container_width=True)
    st.dataframe(xgb_tt,use_container_width=True)

elif menu == "Evaluasi LSTM":
    st.dataframe(lstm_yt,use_container_width=True)
    st.dataframe(lstm_tt,use_container_width=True)

elif menu == "Perbandingan Model":
    st.subheader("XGBoost")
    st.dataframe(xgb_yt,use_container_width=True)
    st.subheader("LSTM")
    st.dataframe(lstm_yt,use_container_width=True)

elif menu == "Data Komentar":
    st.dataframe(dataset,use_container_width=True)

st.markdown("---")
st.markdown("### Dashboard Analisis Sentimen Mobil Listrik Indonesia")
