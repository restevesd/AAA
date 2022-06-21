import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from pycaret.classification import *
import numpy as np


def app():
    st.title("AAA Casados")
    st.info('Analizador de RED AAA CASADOS')
    genero =  st.selectbox("Genero",("Masculino","Femenino"))
    edad   = st.text_input("Edad")
    estado_civil = st.selectbox("Estado Civil",("Casado","Unión Libre"))
    cargas = st.text_input("Número de cargas")
    vivienda = st.selectbox("Tipo de Vivienda",("Propia","Arrendada","Hipotecada","Prestada","Vive con familiares"))
    domicilio = st.text_input("Tiempo Domicilio")
    educacion = st.selectbox("Nivel de Educación",("Secundaria","Técnica","Superior","Postgrado","Especialización","Doctorado"))
    antiguedad = st.text_input("Antigüedad Laboral")
    edad_cyg = st.text_input("Edad Cónyugue")
    educacion_cyg = st.selectbox("Nivel de Educación Conyugue",("Secundaria","Técnica","Superior","Postgrado","Especialización","Doctorado"))
    antiguedad_lab_cyg = st.text_input("Antigüedad laboral cónyugue")    
    total_ingresos = st.text_input("Total Ingresos")
    total_egresos = st.text_input("Total Egresos")
    total_activos = st.text_input("Total Activos")
    total_pasivos = st.text_input("Total Pasivos")
    total_patrimonio =st.text_input("Total Patrimonio")
    #st.write("Total Patrimonio",float(total_activos)-float(total_pasivos))
    tipo_credito = st.selectbox("Tipo de Crédito",("Consumo","Vivienda"))
    tipo_garantia = st.selectbox("Tipo de Garantía",("HIP","PRE","QUI","Sin Garantía"))
    relacion_laboral = st.selectbox("Relación Laboral",("Dependiente","Independiente"))
    score = st.text_input("Score")
    relacion_laboral_cyg = st.selectbox("Relación Laboral Cónyugue",("Dependiente","Independiente"))
    score_cyg = st.text_input("Score Conyugue")
    cuota_estimada = st.text_input("Couta estimada")
    cuota_sistema_financiero = st.text_input("Couta Sistema Financiero")
 
    if st.button("Analizar"):
        st.info('analizando')


        columns={"genero":genero,"edad":int(edad),"estado_civil":estado_civil,
                 "cargas":int(cargas),"vivienda":vivienda,"domicilio":domicilio,"educacion":educacion,
                 "antiguedad":antiguedad,"edad_cyg":int(edad_cyg),"educacion_cyg":educacion_cyg,
                 "antiguedad_lab":int(antiguedad_lab_cyg),"total_ingresos":float(total_ingresos),
                 "total_egresos":float(total_egresos),"total_activos":float(total_activos),
                 "total_pasivos":float(total_pasivos),"total_patrimonio":float(total_patrimonio),
                 "tipo_credito":tipo_credito,"tipo_garantia":tipo_garantia,
                 "relacion_laboral":relacion_laboral,"score":float(score),
                 "relacion_laboral_cyg":relacion_laboral_cyg,"score_cyg":float(score_cyg),
                 "cuota_estimada":float(cuota_estimada),"cuota_sistema_financiero":float(cuota_sistema_financiero)}
        data = pd.DataFrame([columns])
        print(data.info())
        model = load_model("LGBM")
        label = ["AAA","REVISION"]
        resultado =np.argmax(model.predict_proba(data))   
        st.success("El resultado es: "+label[resultado])
        print("Resultado: ",label[resultado])
if __name__ == "__main__":
	app()
