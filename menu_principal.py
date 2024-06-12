import tkinter as tk
from tkinter import messagebox, simpledialog
import subprocess

def ejecutar_script(nombre_script, num_datos):
    try:
        subprocess.run(['python', nombre_script, str(num_datos)], check=True)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error al ejecutar {nombre_script}: {e}")

def solicitar_num_datos(script):
    num_datos = simpledialog.askinteger("Cantidad de Datos", "Ingrese la cantidad de datos a procesar:")
    if num_datos is not None:
        ejecutar_script(script, num_datos)

def main():
    root = tk.Tk()
    root.title("Menu de Scripts")

    tk.Label(root, text="Seleccione una opción para ejecutar el script correspondiente:").pack(pady=10)

    tk.Button(root, text="Ejecutar finalKruskalFolium.py", command=lambda: solicitar_num_datos('finalKruskalFolium.py')).pack(pady=5)
    tk.Button(root, text="Ejecutar finalKruskalNetworkx.py", command=lambda: solicitar_num_datos('finalKruskalNetworkx.py')).pack(pady=5)
    tk.Button(root, text="Ejecutar PrimFolium.py", command=lambda: solicitar_num_datos('PrimFolium.py')).pack(pady=5)
    tk.Button(root, text="Ejecutar PrimNetworkx.py", command=lambda: solicitar_num_datos('PrimNetworkx.py')).pack(pady=5)
    tk.Button(root, text="Salir", command=root.quit).pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
