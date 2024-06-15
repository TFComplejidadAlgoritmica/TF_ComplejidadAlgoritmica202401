import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys
from math import radians, sin, cos, sqrt, atan2

# Función para calcular la distancia Haversine entre dos puntos geográficos
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# Clase para el grafo y el algoritmo de Kruskal
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def apply_union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def kruskal_algo(self):
        result = [] 
        i, e, tot_weight = 0, 0, 0.0  
        self.graph = sorted(self.graph, key=lambda item: item[2]) 
        parent = []
        rank = []
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
        while e < self.V - 1: 
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.apply_union(parent, rank, x, y)
                tot_weight += w

        return result, tot_weight

def main(num_datos):
    ubicacion_base = [-24.1858, -65.2992]

    archivo = 'dataset-jujuy.csv'
    datos = pd.read_csv(archivo, nrows=num_datos)

    datos[['longitud', 'latitud']] = datos['geojson'].str.split(',', expand=True)

    datos['latitud'] = datos['latitud'].astype(float)
    datos['longitud'] = datos['longitud'].astype(float)

    nodos_medio = []
    nodos_alta = []
    ubicaciones = [ubicacion_base]  

    for indice, fila in datos.iterrows():
        ubicacion = [fila['latitud'], fila['longitud']]
        ubicaciones.append(ubicacion)
        
        if fila['tension'] <= 20:
            nodos_medio.append(ubicacion)
        elif fila['tension'] == 33:
            nodos_alta.append(ubicacion)

    g = Graph(len(ubicaciones))
    for i in range(len(ubicaciones)):
        for j in range(i + 1, len(ubicaciones)):
            dist = haversine(ubicaciones[i][0], ubicaciones[i][1], ubicaciones[j][0], ubicaciones[j][1])
            g.add_edge(i, j, dist)

    mst_edges, mst_total_weight = g.kruskal_algo()

    G = nx.Graph()

    for i, ubicacion in enumerate(ubicaciones):
        G.add_node(i, pos=(ubicacion[1], ubicacion[0])) 

    for edge in mst_edges:
        u, v, w = edge
        G.add_edge(u, v, weight=w)

    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    node_pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos=node_pos, with_labels=False, node_size=20, edge_color='blue')
    plt.title('Grafo Original')

    plt.subplot(1, 2, 2)
    mst_pos = nx.spring_layout(G, seed=42)  # Layout para mostrar el MST de manera más clara
    nx.draw(G, pos=mst_pos, with_labels=False, node_size=20, edge_color='red')
    plt.title('MST de Kruskal\nPeso total: {:.2f} km'.format(mst_total_weight))

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python finalKruskalNetworkx.py <num_datos>")
    else:
        num_datos = int(sys.argv[1])
        main(num_datos)
