import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import io
import base64


st.set_page_config(page_title = "Exploracion de datos",
                   page_icon = ":hammer_and_wrench:",
                   layout = "wide")

st.title("Geo-Localizacion:")
st.markdown("")
st.info( """

        ###### **Importante:**

        * ###### Es necesario que los datos se encuentren normalizados.

        * ###### La cantidad de datos por columna deben coincidir en todas las columnas.

        * ###### Los valores numericos deben ser del tipo numero, no string.

        * ###### Los valores de longitud y latitud deben estar presentes para los graficos.

        """)

# Coordenadas de Luján, Buenos Aires, Argentina
lat = -34.5703
lon = -59.1059

def cargar():
    files = st.file_uploader("Datos para el análisis.", accept_multiple_files=True)
    return files

def evaluar_clustering(df, columna, max_clusters):
    # Seleccionar la columna para realizar el clustering
    data = df[[columna]]

    # Inicializar listas para almacenar los valores de codo y silueta
    valores_codo = []
    valores_silueta = []

    # Iterar sobre diferentes valores de k
    for k in range(2, max_clusters+1):
        # Crear el modelo de K-means
        kmeans = KMeans(n_clusters=k, random_state=42)

        # Ajustar el modelo a los datos
        kmeans.fit(data)

        # Calcular el valor de codo (inercia)
        valores_codo.append(kmeans.inertia_)

        # Calcular el valor de silueta
        labels = kmeans.predict(data)
        score = silhouette_score(data, labels)
        valores_silueta.append(score)

    # Crear la figura con subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Graficar el codo
    axs[0].plot(range(2, max_clusters+1), valores_codo, marker='o')
    axs[0].set_xlabel('Número de Clusters (k)')
    axs[0].set_ylabel('Valor de Codo')
    axs[0].set_title('Gráfico de Codo')

    # Graficar la silueta
    axs[1].plot(range(2, max_clusters+1), valores_silueta, marker='o')
    axs[1].set_xlabel('Número de Clusters (k)')
    axs[1].set_ylabel('Valor de Silueta')
    axs[1].set_title('Gráfico de Silueta')

    # Ajustar el espaciado entre subplots
    plt.tight_layout()

    # Mostrar la figura en Streamlit
    #st.pyplot(fig)

    # Convertir la figura en una imagen
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Convertir la imagen en base64
    image_base64 = base64.b64encode(buffer.read()).decode()

    # Almacenar la imagen en 'session_state'
    st.session_state['grafico_evaluacion'] = image_base64

def realizar_clustering(df, columna, num_clusters):
    # Seleccionar la columna para realizar el clustering
    data = df[[columna]]

    # Crear el modelo de K-means con el número de clusters deseado
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    # Ajustar el modelo a los datos
    kmeans.fit(data)

    # Obtener las etiquetas de los clusters asignadas a cada punto de datos
    labels = kmeans.labels_

    # Agregar las etiquetas de clusters al DataFrame original
    df['Cluster'] = labels

    # Devolver el DataFrame con las etiquetas de clusters
    return df

def generar_mapa(df, lat, lon):

    # Crear el mapa utilizando Plotly Express
    fig = px.scatter_mapbox(df, lat='Latitude', lon='Longitude', hover_name='Name', hover_data=['Volumen de Ventas'], color='Cluster', zoom=10)

    # Ajustar el tamaño de los puntos
    fig.update_traces(marker=dict(size=10))

    # Configurar el estilo del mapa y el tamaño de la figura
    fig.update_layout(
        mapbox_style='open-street-map',
        mapbox_zoom=13,
        mapbox_center={'lat': lat, 'lon': lon},
        height=750
    )

    # Mostrar el mapa
    st.plotly_chart(fig, use_container_width=True)

def grafico_mapa_3d(df, columna_volumen):
    # Crear el mapa 3D utilizando Plotly Express
    fig = px.scatter_3d(df, x='Longitude', y='Latitude', z=columna_volumen, color='Cluster',
                        size=columna_volumen, size_max=20, opacity=0.7,
                        title='Mapa 3D: Volumen de Ventas en Luján')

    # Configurar el diseño del mapa 3D
    fig.update_layout(scene=dict(
        xaxis=dict(title='Longitud'),
        yaxis=dict(title='Latitud'),
        zaxis=dict(title=columna_volumen)
    ))

    # Ajustar la altura del gráfico
    fig.update_layout(height=750)

    # Mostrar el mapa 3D
    st.plotly_chart(fig, use_container_width=True)

def agregar_mapa_calor(df, lat, lon, columna_volumen):
    # Crear el mapa utilizando Plotly Express con puntos de mapa de calor
    fig = px.density_mapbox(df, lat='Latitude', lon='Longitude', z=columna_volumen, radius=30,
                            center=dict(lat=lat, lon=lon), zoom=13,
                            mapbox_style='open-street-map', hover_name='Name',
                            title='Mapa de Calor')

    # Configurar el tamaño de la figura
    fig.update_layout(height=750)

    # Mostrar el mapa en Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Párrafo de introducción
st.write('Realizar analisis de clustering en base a una columna seleccionada.')

files = cargar()


# Inicializar el estado del botón en 'session_state'
if 'realizar_clustering_button_state' not in st.session_state:
    st.session_state['realizar_clustering_button_state'] = False

if files:
        selected_file = st.multiselect("Seleccionar archivo", options=[i.name for i in files], key=[uploaded_file for uploaded_file in files])
       
        # Definir la clave para almacenar el estado del botón en 'session_state'
        button_state_key = 'realizar_clustering_button_state'

        for uploaded_file in files:
            if uploaded_file.name in selected_file:
                df = pd.read_excel(uploaded_file)

                # Imprimir el DataFrame
                st.dataframe(df, use_container_width=True)

                # Obtener la lista de columnas del DataFrame
                columnas = df.columns.tolist()

                # Selector de columnas
                columna_seleccionada = st.selectbox('Seleccione una columna', columnas)  
                
                # Botón para ejecutar el clustering
                if st.button('Realizar analisis de Clustering'):

                     # Almacenar el estado del botón en 'session_state'
                    st.session_state[button_state_key] = True

                    # Aquí puedes realizar el clustering con la columna seleccionada
                    st.write('Clustering realizado con la columna:', columna_seleccionada)

                    # Imprimir el analisis
                    evaluar_clustering(df, columna_seleccionada, 10)

                    st.info(
                        """
                        * ###### El gráfico de codo se enfoca en la varianza explicada por los clusters adicionales. 
                            El objetivo es identificar el punto donde el incremento en la varianza explicada se estabiliza o disminuye significativamente, 
                            lo que indica que agregar más clusters no aporta una mejora sustancial al modelo.

                        * ###### El gráfico de silueta evalúa la calidad de los clusters existentes. 
                            Calcula una medida de cuán bien cada punto se ajusta a su propio cluster en comparación con otros clusters. 
                            Un valor de silueta cercano a 1 indica que el punto está bien asignado a su cluster, 
                            mientras que un valor cercano a -1 indica que el punto podría estar mejor asignado a otro cluster. 
                            El objetivo es encontrar el valor de k donde la media de las siluetas sea máxima.

                        En general, se recomienda considerar ambos gráficos y combinar la información que proporcionan. 
                        Si ambos gráficos sugieren un valor de k similar, eso puede ser una señal fuerte. 
                        Sin embargo, si hay discrepancias entre los gráficos, se puede realizar un análisis más detallado, 
                        como la interpretación de los clusters individuales y la evaluación de la coherencia y la interpretabilidad de los resultados, 
                        para tomar una decisión informada sobre el número óptimo de clusters.""")
                    
                # Obtener el gráfico almacenado en 'session_state'
                if 'grafico_evaluacion' in st.session_state and st.session_state[button_state_key] == True:
                    # Decodificar la imagen base64
                    image_base64 = st.session_state['grafico_evaluacion']
                    image = base64.b64decode(image_base64)

                    # Mostrar la imagen en Streamlit
                    st.image(image, use_column_width=True)
                
                    # Selector de número de clusters
                    num_clusters = st.selectbox('Seleccione el número de clusters', range(2, 11))

                # Botón para ejecutar el clustering
                if st.button('Realizar Clustering'):

                    # Coordenadas
                    lat = df['Latitude'].mean()
                    lon = df['Longitude'].mean()

                    # Aquí puedes realizar el clustering con la columna seleccionada
                    st.write('Clustering realizado con la columna:', columna_seleccionada)

                    df_cluster = realizar_clustering(df, columna_seleccionada, num_clusters)

                    # Imprimir el DataFrame
                    st.dataframe(df_cluster, use_container_width=True)

                    # Párrafo de introducción
                    st.write('Visualizar datos en mapa.')

                    # Imprimir el mapa
                    generar_mapa(df_cluster, lat, lon)

                    # Imprimir el mapa
                    grafico_mapa_3d(df_cluster, columna_seleccionada)

                    # Párrafo de introducción
                    st.write('Mapa de temperatura segun los valores de la columna seleccionada.')

                    # Imprimir el mapa
                    agregar_mapa_calor(df_cluster, lat, lon, columna_seleccionada)