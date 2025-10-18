import pandas as pd
import folium

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns

from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from yellowbrick.cluster import SilhouetteVisualizer

import streamlit as st
from streamlit.components.v1 import html

st.subheader("Cluster analysis of Ozurgeti city economic activities")

data = pd.read_excel("Ozurgeti_dataset_coordinates_filtered.xlsx")
NACE_REV2 = pd.read_excel("NACE Rev2.xlsx")

data[['Code1','Code2','Code3']]=data['საქმიანობის კოდი NACE Rev.2'].str.split(pat='.', expand=True)

data.drop(columns=['Code2', 'Code3'], inplace=True)
NACE_REV2['Rev 2.1 code']=NACE_REV2['Rev 2.1 code'].astype(str)

data=data.merge(NACE_REV2, left_on='Code1', right_on='Rev 2.1 code',how='left')

Columns=['პირადი ნომერი',
                   'დაბის/თემის/სოფლის საკრებულო (ფაქტობრივი)',
                   'ფაქტობრივი მისამართი',
                   'აქტიური ეკონომიკური სუბიექტები',
                   'Rev 2.1 code',
                    'Rev 2 code',
                    'საქმიანობის დასახელება NACE Rev.2',
                    'საქმიანობის კოდი NACE Rev.2',
                    'Code1',
                    'ნომერი',
                    'ქუჩა',
                    'ქუჩა_OPS',
                    'similarity_score',
                    'matched_street',
                    'კოორდინატები',
                    ]
Droppable = [col for col in Columns if col in data.columns]
data.drop(columns=Droppable, inplace=True)

df=pd.DataFrame(data[~data['Name'].isna()].value_counts('Name')).reset_index()

Top_10=df.sort_values(by='count', ascending=False).head(10)

df_top_10 = data[data['Name'].isin(Top_10['Name'])]

# 1. Define your colors
colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'skyblue', 'pink', 'gray', 'brown']

# 2. Map each category to a color
unique_names = df_top_10['Name'].unique()
name_to_color = {name: colors[i] for i, name in enumerate(unique_names)}

# 3. Create the map
ozurgeti_map = folium.Map(
    location=[41.925, 42.003],
    tiles='CartoDB positron',
    zoom_start=14.5,
)

# 4. Add markers
for idx, row in df_top_10.iterrows():
    folium.CircleMarker(
        location=[row['Long'], row['Lat']],  # Note: folium uses Lat, Long order
        radius=5,
        color=name_to_color.get(row['Name'], 'white'),  # fallback color
        fill=True,
        fill_color=name_to_color.get(row['Name'], 'white'),
        fill_opacity=0.6,
        popup=row['Name'],
        tooltip=row['Name']
    ).add_to(ozurgeti_map)

html(ozurgeti_map._repr_html_(), height=600)

data=data[~data['Name'].isna()]

ord_enc = OrdinalEncoder()
data["Name_code"] = ord_enc.fit_transform(data[["Name"]])

scaler=StandardScaler()

data[['Long_T',	'Lat_T', 'Name_code_T']]= scaler.fit_transform(data[['Long',	'Lat', 'Name_code']])

def optimize_Kmeans(data, max_k):
    means=[]
    inertias=[]
    for k in range(1, max_k):
        kmeans=KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        means.append(k)
        inertias.append(kmeans.inertia_)

    fig=plt.subplots(figsize=(10,5))
    plt.plot(means, inertias)
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertias')
    plt.grid(True)
    # st.pyplot(plt)

optimize_Kmeans(data[['Long_T',	'Lat_T', 'Name_code_T']],10)

kmeans = KMeans(n_clusters=4
                , random_state=42)
kmeans.fit(data[['Long_T',	'Lat_T', 'Name_code_T']])
data['Kmeans_labels']=kmeans.labels_

sns.pairplot(data[['Long','Lat','Kmeans_labels']], hue='Kmeans_labels')
st.pyplot(plt)


# 1. Compute overall silhouette score
overall_score = silhouette_score(data[['Long_T',	'Lat_T', 'Name_code_T']], data['Kmeans_labels'])

# 2. Compute silhouette score for each sample
sample_scores = silhouette_samples(data[['Long_T',	'Lat_T', 'Name_code_T']], data['Kmeans_labels'])

# 3. Add sample scores to your DataFrame
data['silhouette_score'] = sample_scores


# 4. Compute average silhouette score per cluster
per_cluster_scores = data.groupby('Kmeans_labels')['silhouette_score'].mean()


kmeans = KMeans(n_clusters=4, random_state=0)

visualizer = SilhouetteVisualizer(kmeans)
visualizer.fit(data[['Long_T',	'Lat_T', 'Name_code_T']])

visualizer.poof()

data.sort_values(by='silhouette_score', ascending=False)


# 1. Define your colors
colors = ['red', 'blue', 'yellow', 'brown' ]

# 2. Map each category to a color
unique_names = data['Kmeans_labels'].unique()
name_to_color = {name: colors[i] for i, name in enumerate(unique_names)}


# 3. Create the map
ozurgeti_map = folium.Map(
    location=[41.925, 42.003],
    tiles='CartoDB positron',
    zoom_start=14.5,
)

# 4. Add markers
for idx, row in data.iterrows():
    folium.CircleMarker(
        location=[row['Long'], row['Lat']],  # Note: folium uses Lat, Long order
        radius=5,
        color=name_to_color.get(row['Kmeans_labels'], 'white'),  # fallback color
        fill=True,
        fill_color=name_to_color.get(row['Kmeans_labels'], 'white'),
        fill_opacity=0.6,
        popup=row['Name'],
        tooltip=row['Name']
    ).add_to(ozurgeti_map)

html(ozurgeti_map._repr_html_(), height=600)





