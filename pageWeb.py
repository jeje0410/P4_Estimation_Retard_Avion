import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime

USHolydays = [datetime.date(2016,1,1),datetime.date(2016,1,18),datetime.date(2016,2,15),datetime.date(2016,5,30),datetime.date(2016,7,4),datetime.date(2016,9,5),datetime.date(2016,10,10),datetime.date(2016,11,11),datetime.date(2016,11,24),datetime.date(2016,12,26),datetime.date(2016,12,31)]
def daystoholydays(day,month):
    return min([abs((x - datetime.date(2016, month, day)).days) for x in USHolydays])

def transformSin(nombre, frequence):
    return np.sin(2*np.pi*nombre/frequence)

def transformCos(nombre, frequence):
    return np.cos(2*np.pi*nombre/frequence)

st.title('Prédiction retard Avion')



carrier_name = st.selectbox('Selectionnez une compagnie',{'DL', 'B6', 'AA', 'AS', 'EV', 'F9', 'VX', 'UA', 'OO', 'WN', 'HA','NK'})

state_dep = st.selectbox('Selectionnez un état de départ',{'AK','AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 
'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV','NY', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TT', 'TX', 'UT',
 'VA', 'VI', 'VT', 'WA', 'WI', 'WV', 'WY'})
state_arr = st.selectbox('Selectionnez un état d\'arrivé',{'AK','AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 
'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV','NY', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TT', 'TX', 'UT',
 'VA', 'VI', 'VT', 'WA', 'WI', 'WV', 'WY'})

monthDep = st.slider("Mois de départ", 1, 12)
dayDep = st.slider("Jour de départ", 1, 31)

hourDep = st.slider("Heure de départ", 0, 23)
hourArr = st.slider("Heure d'arrivée", 0, 23)


@st.cache(persist=True)
def load_models():
    model = {}
    with open('Modeles/arbreDecision.pickle', 'rb') as file:
        model['ArbreDecision'] = pickle.load(file)
    with open('Modeles/RegressionLineaire.pickle', 'rb') as file:
        model['RegressionLineaire'] = pickle.load(file)
    return model

def load_scaler():
    scaler = {}
    with open('Modeles/scaler.pickle', 'rb') as file:
        scaler['Scaler'] = pickle.load(file)
    return scaler
    
#Lecture du dataframe etat d'origine
with open('Modeles/df_std_state_origin.pickle', 'rb') as file:
    df_std_state_origin = pickle.load(file)
    
#Lecture du dataframe etat de destination
with open('Modeles/df_std_state_dest.pickle', 'rb') as file:
    df_std_state_dest = pickle.load(file)
    
#Lecture du dataframe compagnie
with open('Modeles/df_std_carrier.pickle', 'rb') as file:
    df_std_carrier = pickle.load(file)
    
#Lecture du dataframe Distancce
with open('Modeles/df_mean_distance.pickle', 'rb') as file:
    df_mean_distance = pickle.load(file)
    
#Lecture du dataframe temps prévu
with open('Modeles/df_mean_time.pickle', 'rb') as file:
    df_mean_time = pickle.load(file)
    
#Lecture du dataframe retard au départ
with open('Modeles/df_mean_by_delay.pickle', 'rb') as file:
    df_mean_by_delay = pickle.load(file) 

#Lecture du modele
model = load_models()
scaler = load_scaler()

try:
    retard_depart = df_mean_by_delay.loc[carrier_name].loc[state_dep].loc[state_arr].values[0]
except KeyError:
    retard_depart = 0

try:
    timeto_fly = df_mean_time.loc[state_dep].loc[state_arr].values[0]
except KeyError:
    timeto_fly = 0
    
try:
    distanceto_fly = df_mean_distance.loc[state_dep].loc[state_arr].values[0]
except KeyError:
    distanceto_fly = 0

# Ordre d'appel du molele DEP_DELAY, CRS_ELAPSED_TIME, DISTANCE, daysToHolydays, sin_month, cos_month, sin_day, cos_day, sin_hour_dep, cos_hour_dep, sin_hour_arr, cos_hour_arr, meanByStateOrigin, meanByStateDest, meanByCarrier
variable_std = scaler['Scaler'].transform([[retard_depart, timeto_fly, distanceto_fly,  daystoholydays(dayDep,monthDep), transformSin(monthDep,12),
       transformCos(monthDep,12), transformSin(dayDep,30) , transformCos(dayDep,30),  transformSin(hourDep,24), transformCos(hourDep,24),
       transformSin(hourArr,24), transformCos(hourArr,24), df_std_state_origin.loc[state_dep].values[0],df_std_state_dest.loc[state_arr].values[0],df_std_carrier.loc[carrier_name].values[0]]])

#retard = model['ArbreDecision'].predict(variable_std)
st.write('Le retard prédit avec la regression linéaire est de ', model['RegressionLineaire'].predict(variable_std)) 
st.write('Le retard prédit avec l\'arbre de décision est de ', model['ArbreDecision'].predict(variable_std)) 

#Impact sur les performances d'utiliser la moyenne par rapport à la valeur brute