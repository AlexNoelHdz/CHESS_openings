import pandas as pd
import tkinter as tk
from tkinter import ttk
import datetime
from tkinter import font
import subprocess
import sys

class ChessLogger:
    def __init__(self, path, filename_format="chess_log_%Y-%m-%d_%H-%M.txt"):
        self.path = path
        self.filename_format = filename_format
        self.filename = self.generate_filename()
    
    def generate_filename(self):
        """Genera el nombre del archivo basado en la fecha actual y el formato especificado."""
        date_str = datetime.datetime.now().strftime(self.filename_format)
        return f"{self.path}\{date_str}"
    
    def write(self, text):
        """Escribe el texto proporcionado en el archivo de log."""
        with open(self.filename, 'a') as log_file:
            log_file.write(text + "\n")

def open_file(ruta_archivo):
    if sys.platform == "win32":
        subprocess.run(["start", ruta_archivo], shell=True)
    elif sys.platform == "darwin": #IOS
        subprocess.run(["open", ruta_archivo])
    elif sys.platform == "linux":
        subprocess.run(["xdg-open", ruta_archivo])

# Función para mostrar el DataFrame en un cuadro de diálogo emergente
def show_df_in_window(df, title):
    # Crear una ventana emergente
    window = tk.Tk()
    window.title(title)

    frame = ttk.Frame(window)
    frame.pack(fill='both', expand=True)

    tree = ttk.Treeview(frame, columns=df.columns, show='headings')
    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    tree.pack(side='left', fill='both', expand=True)
    vsb.pack(side='right', fill='y')
    hsb.pack(side='bottom', fill='x')

    # Usar tkinter.font para crear una instancia de fuente
    fonte = font.Font(family="Helvetica", size=10)  # Crea una fuente para usar en la medición

    # Configurar las cabeceras
    for col in df.columns:
        tree.heading(col, text=col)
        # Ajustar el ancho de las columnas
        # Calcular el ancho máximo de los datos en cada columna
        max_width = max([fonte.measure(str(val)) for val in df[col]] + [fonte.measure(col)])
        tree.column(col, width=max_width)

    # Agregar los datos del DataFrame al Treeview
    for index, row in df.iterrows():
        tree.insert("", "end", values=list(row))

    window.mainloop()
