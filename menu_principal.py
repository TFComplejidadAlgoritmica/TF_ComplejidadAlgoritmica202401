import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import pandas as pd
import subprocess

class ScriptExecutorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Trabajo final Complejidad Algorítmica Kruskal vs Prim")

        self.label = tk.Label(root, text="Seleccione una opción para ejecutar el script correspondiente:")
        self.label.pack(pady=10)

        self.kruskal_folium_btn = tk.Button(root, text="Ejecutar Kruskal con Google Maps", command=lambda: self.solicitar_num_datos('finalKruskalFolium.py'))
        self.kruskal_folium_btn.pack(pady=5)

        self.kruskal_networkx_btn = tk.Button(root, text="Ejecutar Visualizar Kruskal en NetworkX", command=lambda: self.solicitar_num_datos('finalKruskalNetworkx.py'))
        self.kruskal_networkx_btn.pack(pady=5)

        self.prim_folium_btn = tk.Button(root, text="Ejecutar Prim con Google Maps", command=lambda: self.solicitar_num_datos('PrimFolium.py'))
        self.prim_folium_btn.pack(pady=5)

        self.prim_networkx_btn = tk.Button(root, text="Ejecutar Visualizar Prim en NetworkX", command=lambda: self.solicitar_num_datos('PrimNetworkx.py'))
        self.prim_networkx_btn.pack(pady=5)

        self.salir_btn = tk.Button(root, text="Salir", command=root.quit)
        self.salir_btn.pack(pady=20)

        self.opciones_frame = tk.LabelFrame(root, text="Opciones de Filtrado:")
        self.opciones_frame.pack(pady=10)

        self.opcion_var = tk.IntVar()

        opciones = [
            ("Todos tensión alta", 1),
            ("Todos tensión media", 2),
            ("Todos nodos", 3)
        ]

        self.opciones_radio = []
        for opcion, valor in opciones:
            radio_btn = tk.Radiobutton(self.opciones_frame, text=opcion, variable=self.opcion_var, value=valor)
            radio_btn.pack(anchor='w')
            self.opciones_radio.append(radio_btn)

        self.table_frame = tk.Frame(root)
        self.table_frame.pack(pady=10)

    def ejecutar_script(self, nombre_script, datos_filtrados):
        opcion = self.opcion_var.get()

        if opcion == 1:
            datos_filtrados = datos_filtrados[datos_filtrados['tension'] == 33].copy()
        elif opcion == 2:
            datos_filtrados = datos_filtrados[datos_filtrados['tension'] <= 20].copy()
        elif opcion == 3:
            datos_filtrados = datos_filtrados.copy()

        try:
            datos_filtrados.to_csv('datos_filtrados.csv', index=False)
            subprocess.run(['python', nombre_script, str(len(datos_filtrados))], check=True)
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Error al ejecutar {nombre_script}: {e}")

    def solicitar_num_datos(self, script):
        self.script_seleccionado = script
        num_datos = simpledialog.askinteger("Cantidad de Datos", "Ingrese la cantidad de datos a procesar:")
        if num_datos is not None:
            self.mostrar_datos(num_datos, script)

    def mostrar_datos(self, num_datos, script):
        archivo = 'dataset-jujuy.csv'
        self.datos = pd.read_csv(archivo, nrows=num_datos)

        self.datos['poder de voltaje'] = self.datos['tension'].apply(lambda x: 'ALTO' if x == 33 else 'MEDIO')

        for widget in self.table_frame.winfo_children():
            widget.destroy()

        self.tree = ttk.Treeview(self.table_frame, selectmode='extended')
        self.tree.pack()

        self.tree["columns"] = list(self.datos.columns)
        self.tree["show"] = "headings"

        for col in self.datos.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor='center')

        for index, row in self.datos.iterrows():
            self.tree.insert("", "end", iid=index, values=list(row))

        eliminar_btn = tk.Button(self.table_frame, text="Eliminar Seleccionados", command=self.eliminar_filas)
        eliminar_btn.pack(pady=5)

        confirmar_btn = tk.Button(self.table_frame, text="Confirmar y Ejecutar Script", command=lambda: self.ejecutar_script(script, self.datos))
        confirmar_btn.pack(pady=10)


  
    def eliminar_filas(self):
        selected_items = self.tree.selection()
        for item in selected_items:
            self.datos.drop(int(item), inplace=True)
            self.tree.delete(item)

        self.datos.reset_index(drop=True, inplace=True)

def main():
    root = tk.Tk()
    app = ScriptExecutorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
