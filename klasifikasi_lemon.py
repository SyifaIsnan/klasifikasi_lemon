import streamlit as st
import pandas as pd
import joblib

model_tersimpan = joblib.load('klasifikasi_lemon.joblib')

st.set_page_config(
	page_title='Klasifikasi Lemon'
)

st.title('Klasifikasi Lemon')

diameter = st.slider('Diameter', 46.0, 57.0, 70.0)
berat = st.slider('Berat', 70.1, 103.1, 146.0)
tebal_kulit = st.slider('Tebal Kulit', 3.5, 4.3, 7.0)
kadar_gula = st.slider('Kadar Gula', 6.8, 8.0, 9.0)
asal_daerah = st.pills('Asal Daerah', ['California', 'Malang', 'Medan'])
musim_panen = st.pills('Musim Panen', ['Awal', 'Puncak', 'Akhir'])
warna = st.pills('Warna', ['Hijau pekat', 'Kuning kehijauan', 'Kuning cerah'])

if st.button('prediksi'):

	data_baru = pd.DataFrame([[diameter, berat, tebal_kulit, kadar_gula, asal_daerah, musim_panen,warna]], columns=['diameter', 'berat', 'tebal_kulit', 'kadar_gula', 'asal_daerah', 'musim_panen', 'warna'])
	prediksi = model_tersimpan.predict(data_baru)[0]
	akurasi = max( model_tersimpan.predict_proba(data_baru)[0])
	st.success(f"Model memprediksi {prediksi} dengan tingkat keyakinan {akurasi*100:.2f}")
	st.balloons()

st.divider()
