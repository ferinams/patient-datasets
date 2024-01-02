import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

df = pd.read_csv('patient_dataset.csv')

df.rename(index=str, columns={
    'chest_pain_type' : 'chestpain',
    'blood_pressure' : 'blood',
    'max_heart_rate' : 'maxheart',
    'plasma_glucose' : 'glucose',
    'skin_thickness' : 'thickness',
    'heart_disease' : 'heartdisease'


}, inplace=True)

x = df.drop(['gender', 'insulin', 'bmi', 'diabetes_pedigree', 'exercise_angina', 'residence_type', 'smoking_status'], axis=1)

imputer = SimpleImputer(strategy='mean')
x_imputed = pd.DataFrame(imputer.fit_transform(x), columns=x.columns)

print("Columns:", x_imputed.columns)

print("Number of Infinity values in x:", np.isinf(x_imputed).sum().sum())

st.header("isi dataset")
st.write(x)

## menampilkan Panah elbow
imputer = SimpleImputer(strategy='mean')
x_imputed = imputer.fit_transform(x)

clusters= []
for i in range(1, 11):
    km =KMeans(n_clusters=i).fit(x_imputed)
    clusters.append(km.inertia_)


fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=list(range(1,11)), y=clusters, ax=ax)
ax.set_title('mencari elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')

#Panah elbow
ax.annotate('possible elbow point', xy=(3, 4.5), xytext=(3, 2.5), xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

ax.annotate('possible elbow point', xy=(4, 4.5), xytext=(4, 2.5), xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

st.set_option('deprecation.showPyplotGlobalUse', False)
elbo_plot = st.pyplot()


st.sidebar.subheader("Nilai Jumlah K")
selected_columns = ['blood', 'age']
print("Selected Columns:", selected_columns)
clust = st.sidebar.slider("Pilih Jumlah Cluster:", 2,10,3,1)
print("Number of Clusters:", clust)


def k_means(n_clust):
    x_selected = x_imputed[['blood', 'age']].copy()
    kmean = KMeans(n_clusters=n_clust).fit(x_selected)
    x_selected['labels'] = kmean.labels_

    # Visualisasi Scatter Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=x_selected['blood'], y=x_selected['age'], hue=x_selected['labels'], palette=sns.color_palette('hls', n_colors=n_clust))

    for label in range(n_clust):
        plt.annotate(label,
            (x_selected[x_selected['labels'] == label]['blood'].mean(),
            x_selected[x_selected['labels'] == label]['age'].mean()),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center')

    st.header('Plot Klaster')

# Memanggil fungsi k_means tanpa menambahkan st.pyplot() di dalamnya
k_means(clust)

# Menampilkan plot di luar fungsi
st.pyplot()