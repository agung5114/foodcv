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

model = keras.models.load_model('fcvmodel.h5')
dbfood = pd.read_csv('dbfood.csv',sep=";")
food = dbfood['nama'].tolist()
# def getPrediction(path):
#     img = Image.open(path)
#     newsize = (224, 224)
#     image = img.resize(newsize)
#     image = img_to_array(image)
#     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#     image = preprocess_input(image)
#     yhat = model.predict(image)
#     label = yhat[0]
#     # a,b,c,d = np.round(label[0]*100,2),np.round(label[1]*100,2),np.round(label[2]*100,2),np.round(label[3]*100,2)
#     # top = dict(zip(['bakso','pempek','sate','soto'], [a,b,c,d]))
#     # a,b,c,d = np.round(label[0]*100,2),np.round(label[1]*100,2),np.round(label[2]*100,2),np.round(label[3]*100,2)
#     prob = []
#     for i in range(len(label)):
#         prob.append(np.round(label[i]*100,2))
#     top = dict(zip(['Ayam','Bakso','Bubur_Ayam','Gado_Gado','Gudeg','Hamburger','Kentang_Goreng','Ketoprak','Mie_Goreng','Nasi_Goreng','Pempek','Rawon','Sate','Sayur_Asam','Siomay','Soto','Tongseng'], prob))
#     top3 = dict(sorted(top.items(), key=operator.itemgetter(1), reverse=True)[:3])
#     return top3

def getPrediction(data,model):
    img = Image.open(data)
    newsize = (224, 224)
    image = img.resize(newsize)
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = yhat[0]
    prob = []
    for i in range(len(label)):
        prob.append(np.round(label[i]*100,2))
    top = dict(zip(food, prob))
    top3 = dict(sorted(top.items(), key=operator.itemgetter(1), reverse=True)[:3])
    return top3

st.set_page_config(layout='wide')

def main():
    # menu = ["Food Calorie Estimator","Profile","Recommendations"]
    
    # choice = st.sidebar.selectbox("Select Menu", menu)

    # if choice == "Food Calorie Estimator":
    st.subheader("Food Calorie Estimator")
    data = st.file_uploader('Upload Foto')
    if data == None:
        st.write('Silakan Upload Foto')
    else:
        st.image(data)

    if st.button('Prediksi'):
        hasil = getPrediction(data,model)
        # dfhasil = pd.DataFrame.from_dict(hasil)
        keys = list(hasil.keys())
        dbkal = dbfood[dbfood['nama'].isin(keys)]
        # db = dbkal.join(dfhasil,how='left',on)
        st.write(keys[0])
        # st.write(dfhasil)
        st.write(dbkal)

    # elif choice == "Profile":
    #     st.title("Profile")

    # elif choice == "Recommendations":
    #     st.title("Recommendations")
        

if __name__=='__main__':
    main()
