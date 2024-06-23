import pandas as pd
import folium
import webbrowser
import sys
from math import radians, sin, cos, sqrt, atan2
import tkinter as tk
from tkinter import messagebox

def haversine(lat1, lon1, lat2, lon2):
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

    def kruskal_algo(self, nodos_alta, ubicaciones):
        result = []
        i, e, tot_weight = 0, 0, 0
        conexiones = []  # Lista para almacenar las conexiones de alta tensión
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
                # Verificar si ambos nodos son de alta tensión y agregar a conexiones
                if ubicaciones[u] in nodos_alta and ubicaciones[v] in nodos_alta:
                    conexiones.append((u, v, w))

    # Llamar a mostrar_advertencia aquí si se desea mostrar las advertencias inmediatamente después de encontrarlas
    # mostrar_advertencia(conexiones)

        for u, v, weight in result:
            tot_weight += weight
            print("%d - %d: %f" % (u, v, weight))

        print("Costo total del MST para el grafo: ", tot_weight)

        return result, conexiones  # Devolver también las conexiones


def mostrar_advertencia(conexiones):
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal de Tkinter
    for conexion in conexiones:
        torre1, torre2, distancia = conexion
        mensaje = f"Conexión de alta tensión entre torres: {torre1} - {torre2}: {distancia}"
        messagebox.showwarning("Advertencia", mensaje)
    root.destroy()    
   
def cargar_y_procesar_datos(archivo, num_filas):
    ubicacion_base = [-24.1858, -65.2992]

    datos = pd.read_csv(archivo, nrows=num_filas)
    datos[['longitud', 'latitud']] = datos['latitud_y_longitud'].str.strip(' "').str.split(',', expand=True)
    datos['latitud'] = datos['latitud'].astype(float)
    datos['longitud'] = datos['longitud'].astype(float)

    ubicaciones = [ubicacion_base]
    nodos_medio = []
    nodos_alta = []

    for indice, fila in datos.iterrows():
        ubicacion = [fila['latitud'], fila['longitud']]
        ubicaciones.append(ubicacion)
        if fila['tension'] <= 20:
            nodos_medio.append(ubicacion)
        elif fila['tension'] == 33:
            nodos_alta.append(ubicacion)

    g = Graph(len(ubicaciones))

    if nodos_alta:
        for nodo_alta in nodos_alta:
            dist = haversine(ubicacion_base[0], ubicacion_base[1], nodo_alta[0], nodo_alta[1])
            g.add_edge(0, ubicaciones.index(nodo_alta), dist)
            
    else:
        for nodo_medio in nodos_medio:
            dist = haversine(ubicacion_base[0], ubicacion_base[1], nodo_medio[0], nodo_medio[1])
            g.add_edge(0, ubicaciones.index(nodo_medio), dist)

    for i in range(1, len(ubicaciones)):
        for j in range(i + 1, len(ubicaciones)):
            dist = haversine(ubicaciones[i][0], ubicaciones[i][1], ubicaciones[j][0], ubicaciones[j][1])
            g.add_edge(i, j, dist)

    return g, ubicaciones, nodos_alta

def main():
    num_datos = int(sys.argv[1])

    mapa = folium.Map(location=[-24.1858, -65.2992], zoom_start=13)
    imagen_personalizada = 'plantaPrincipal.jpg'
    icono_personalizado = folium.features.CustomIcon(icon_image=imagen_personalizada, icon_size=(70, 70))
    folium.Marker(location=[-24.1858, -65.2992], icon=icono_personalizado).add_to(mapa)

    archivo = 'dataset-jujuy.csv'
    g, ubicaciones, nodos_alta = cargar_y_procesar_datos(archivo, num_datos)

    print (nodos_alta)
    
    for ubicacion in ubicaciones[1:]:
        folium.Marker(location=ubicacion, icon=folium.Icon(color='orange')).add_to(mapa)

    ubicacion_base = [-24.1858, -65.2992]

    radio_metros = 10  # 10 metros

    if nodos_alta:
        for ubicacion in nodos_alta:
            # Calcula la distancia en kilómetros y conviértela a metros
            dist_km = haversine(ubicacion_base[0], ubicacion_base[1], ubicacion[0], ubicacion[1])
            dist_metros = dist_km * 1000  # Convierte a metros

            # Calcula el factor de escala para ajustar el radio a 10 metros
            factor_escala = radio_metros / dist_metros

            # Ajusta el radio del círculo en metros
            radio_ajustado = int(dist_metros * factor_escala)

            folium.Circle(
                location=ubicacion,
                radius=radio_ajustado,
                color='red',
                fill=True,
                fill_color='red'
            ).add_to(mapa)



    mst_edges, conexiones_alta_tension = g.kruskal_algo(nodos_alta, ubicaciones)

    for edge in mst_edges:
        u, v, w = edge
        folium.PolyLine([ubicaciones[u], ubicaciones[v]], color="blue", weight=2.5, opacity=1).add_to(mapa)

    mostrar_advertencia(conexiones_alta_tension)
        
    mapa.save('mapaGrafo.html')
    webbrowser.open('mapaGrafo.html')

if __name__ == "__main__":
    main()
