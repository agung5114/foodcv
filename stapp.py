import streamlit as st
import pandas as pd
import numpy as np
# import plotly_express as px
from PIL import Image
import streamlit.components.v1 as components
# import matplotlib.pyplot as plt
from tensorflow import keras
import joblib
import operator
import sys

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet import decode_predictions

from PIL import Image
sys.modules['Image'] = Image 
# [theme]
base="light"
primaryColor="purple"

model = keras.models.load_model('foodclass.h5')

def getPrediction(file):
    img = Image.open(file)
    newsize = (224, 224)
    image = img.resize(newsize)
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = yhat[0]
    a,b,c,d = np.round(label[0]*100,2),np.round(label[1]*100,2),np.round(label[2]*100,2),np.round(label[3]*100,2)
    top = dict(zip(['bakso','pempek','sate','soto'], [a,b,c,d]))
    top3 = dict(sorted(top.items(), key=operator.itemgetter(1), reverse=True)[:3])
    return top3

st.set_page_config(layout='wide')

def main():
    menu = ["Verdict Prediction","Dispute Prediction","Metric Scores"]
    
    choice = st.sidebar.selectbox("Select Menu", menu)

        
    if choice == "Verdict Prediction":
        st.subheader("Verdict Prediction")
        data = st.file_uploader('Upload Foto')
        if data == None:
            st.write('Silakan Upload Foto')

        if st.button('Prediksi'):
            hasil = getPrediction(data)
            st.image(data)
            st.write(hasil)

        # st.subheader("Features")
        # #Intializing
        # c1,c2 = st.beta_columns((1,1))
        # with c1:
        #     sl = st.number_input(label="FP Lengkap",value=1,min_value=0, max_value=1, step=1)
        #     sw = st.number_input(label="FP Tepat Waktu",value=1,min_value=0, max_value=1, step=1)
        #     pl = st.number_input(label="Keterangan FP Sesuai",value=0,min_value=0, max_value=1, step=1)
        #     dm1 = st.number_input(label="FP Diganti Dibatalkan",value=1,min_value=0, max_value=1, step=1)
        #     dm2 = st.number_input(label="FP Tidak Double Kredit",value=1,min_value=0, max_value=1, step=1)
        # with c2:
        #     dm0 = st.number_input(label="Lawan PKP",value=1,min_value=0, max_value=1, step=1)
        #     dm3 = st.number_input(label="Lawan Disanksi",value=1,min_value=0, max_value=1, step=1)
        #     dm4 = st.number_input(label="Lawan Lapor",value=1,min_value=0, max_value=1, step=1)
        #     dm5 = st.number_input(label="Minta Tanggung Jawab Lawan",value=1,min_value=0, max_value=1, step=1)
        #     pw = st.number_input(label="PPN telah dibayar",value=0,min_value=0, max_value=1, step=1)

        # if st.button("Click Here to Classify"):
        #     dfvalues = pd.DataFrame(list(zip([sl],[sw],[pl],[pw])),columns =['lengkap', 'tepatwaktu', 'ketsesuai', 'adapembayaran'])
        #     input_variables = np.array(dfvalues[['lengkap', 'tepatwaktu', 'ketsesuai', 'adapembayaran']])
        #     prediction = knn.predict(input_variables)
        #     if prediction == 'ditolak':
        #         st.subheader('Prediksi Hasil Verdict')
        #         st.title('Permohonan Banding Ditolak')
        #     elif prediction =='sebagian':
        #         st.subheader('Prediksi Hasil Verdict')
        #         st.title('Permohonan Banding Diterima Sebagian')
        #     else:
        #         st.subheader('Prediksi Hasil Verdict')
        #         st.title('Permohonan Banding Diterima Seluruhnya')
    
    elif choice == "Dispute Prediction":
        st.title("Dispute Predicction")

    elif choice == "Metric Scores":
        st.title("Metric Scores")
        

if __name__=='__main__':
    main()
