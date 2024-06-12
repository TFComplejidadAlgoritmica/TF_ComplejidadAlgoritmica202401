import heapq
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lon2, lat2):
    R = 6371.0  
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)] for row in range(vertices)]
        
    def add_edge(self, u, v, w):
        self.graph[u][v] = w
        self.graph[v][u] = w

    def min_key(self, key, mst_set):
        min = float('inf')
        for v in range(self.V):
            if key[v] < min and mst_set[v] == False:
                min = key[v]
                min_index = v
        return min_index

    def print_mst(self, parent):
        print("Edge \tWeight")
        for i in range(1, self.V):
            print(parent[i], "-", i, "\t", self.graph[i][parent[i]])

    def prim_algo(self):
        key = [float('inf')] * self.V
        parent = [None] * self.V
        key[0] = 0
        mst_set = [False] * self.V
        parent[0] = -1
        mst_edges = []

        for cout in range(self.V):
            u = self.min_key(key, mst_set)
            mst_set[u] = True
            for v in range(self.V):
                if self.graph[u][v] > 0 and mst_set[v] == False and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u

        for i in range(1, self.V):
            mst_edges.append((parent[i], i, self.graph[i][parent[i]]))

        self.print_mst(parent)
        return mst_edges

ubicacion_base = [-24.1858, -65.2992]

archivo = 'dataset-jujuy.csv'
num_filas = int(sys.argv[1])

datos = pd.read_csv(archivo).head(num_filas)
datos[['longitud', 'latitud']] = datos['geojson'].str.strip(' "').str.split(',', expand=True)
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

mst_edges = g.prim_algo()

G = nx.Graph()

for i, ubicacion in enumerate(ubicaciones):
    G.add_node(i, pos=(ubicacion[1], ubicacion[0]))

for i in range(len(ubicaciones)):
    for j in range(i + 1, len(ubicaciones)):
        dist = haversine(ubicaciones[i][0], ubicaciones[i][1], ubicaciones[j][0], ubicaciones[j][1])
        G.add_edge(i, j, weight=dist)

MST = nx.Graph()
for edge in mst_edges:
    u, v, w = edge
    MST.add_edge(u, v, weight=w)

plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
node_pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos=node_pos, with_labels=False, node_size=20, edge_color='blue')
plt.title('Grafo Original')

plt.subplot(1, 2, 2)
mst_pos = nx.spring_layout(MST, seed=42)
nx.draw(MST, pos=mst_pos, with_labels=False, node_size=20, edge_color='red')
plt.title('MST de Prim')

plt.show()
