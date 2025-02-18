import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
#from sklearn.decomposition import PCA
#importação da base de dados/ data import
data_df = pd.read_csv('data/bank_data.csv')
#esse print é importante para identificarmos o tipos das colunas e se existe valores nulos/ this print is important to identify the types of columns and if there are null values
print(data_df.info())
#É possivel observar que as colunas MINIMUM_PAYMENTS e CREDIT_LIMIT tem itens nulos./ It is possible to observe that the columns MINIMUM_PAYMENTS and CREDIT_LIMIT have null items.
#Baseado nisso podemos deletar essas linhas ou podemos subistituir os valores nulos pela media dos valores da coluna./ Based on this we can delete these lines or we can replace the null values by the average of the column values.
data_df['MINIMUM_PAYMENTS'].fillna(data_df['MINIMUM_PAYMENTS'].mean(), inplace=True)
data_df['CREDIT_LIMIT'].fillna(data_df['CREDIT_LIMIT'].mean(), inplace=True)
#Agora podemos deletar a coluna CUST_ID pois ela não é relevante para a análise./ Now we can delete the CUST_ID column because it is not relevant for the analysis.
data_df.drop('CUST_ID', axis=1, inplace=True)
#vamos usar o print novamente para checar se os valores nulos foram substituidos./ let's use the print again to check if the null values were replaced.
print(data_df.info())
#agora vamos checar os valores duplicados./ now let's check the duplicate values.
print(data_df.duplicated().sum())

#Caso queira fazer uma analise de corelação das colunas, podemos usar o código abaixo./ If you want to do a correlation analysis of the columns, we can use the code below.
#correlations = creditcard_df.corr()
#f, ax = plt.subplots(figsize=(20,20))
#sns.heatmap(correlations, annot=True);
#Agora vamos normalizar os dados para que tenham média 0 e desvio padrão 1./ Now let's normalize the data so that they have an average of 0 and a standard deviation of 1.
scaler = StandardScaler()
data_df_scaled = scaler.fit_transform(data_df)
#apos essa transformação os dados vão estar em um array numpy/ after this transformation the data will be in a numpy array.
#o código a baixo faz a contagem de clusters para identificar o melhor número de clusters para o KMeans./ the code below counts clusters to identify the best number of clusters for KMeans.
clusters_ = []
range_values = range(1, 20)
for i in range_values:
  kmeans = KMeans(n_clusters=i)
  kmeans.fit(data_df_scaled)
  clusters_.append(kmeans.inertia_)
#o gráfico ajudará a identificar o melhor número de clusters atraves do Elbow Method/ the graph will help identify the best number of clusters through the Elbow Method
plt.plot(clusters_, 'bx-')
plt.xlabel('Clusters')
plt.ylabel('WCSS')
plt.show()
#com base no gráfico acima, o melhor número de clusters é 10./ based on the graph above, the best number of clusters is 10.
kmeans = KMeans(n_clusters=10)
kmeans.fit(data_df_scaled)
labels = kmeans.labels_
#vamos checar a densidade de cada cluster/ let's check the density of each cluster
np.unique(labels, return_counts=True)
#vamos criar um data frame como os centroides de cada cluster/ let's create a data frame like the centroids of each cluster
cluster_centers = pd.DataFrame(data = kmeans.cluster_centers_, columns = [data_df.columns])
#print(cluster_centers)
#para conseguirmos fazer uma análise dos clusters precisamos fazer a inversão da normalização/ to be able to analyze the clusters we need to reverse the normalization
cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data = cluster_centers, columns = [data_df.columns])
#print(cluster_centers)
#criar a classificação para cada cluster (temporario)

#vamos adicionar a coluna de cluster ao data frame original/ let's add the cluster column to the original data frame
data_df_cluster = pd.concat([data_df, pd.DataFrame({'cluster': labels})], axis = 1)
#Com ajuda de gráficos podemos fazer uma análise visual dos clusters/ With the help of graphs we can make a visual analysis of the clusters
#for i in data_df.columns:
#  plt.figure(figsize=(35,5))
#  for j in range(10):
#    plt.subplot(1, 10, j + 1)
#    cluster = data_df_cluster[data_df_cluster['cluster'] == j]
#    cluster[i].hist(bins = 20)
#    plt.title('{} \nCluster {}'.format(i, j))
#  plt.show()

# Agora organizamos os clusters em ordem / Now we organize the clusters in order
data_ordered = data_df_cluster.sort_values(by = 'cluster')

# Caminho para salvar o arquivo (substitua pelo seu)/
caminho = "cluster.csv"  # Windows

# Exporta para CSV (sem o índice numérico do Pandas)/
data_ordered.to_csv(caminho, index=False, encoding='utf-8')

