import pandas as pd
import tkinter as tk
from tkinter import ttk
import datetime

class ChessLogger:
    def __init__(self, path, filename_format="chess_log_%Y-%m-%d.txt"):
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

# Función para mostrar el DataFrame en un cuadro de diálogo emergente
def show_df_in_window(df, title):
    # Crear una ventana emergente
    ventana = tk.Tk()
    ventana.title(title)

    # Crear un Frame para el TreeView
    frame = ttk.Frame(ventana)
    frame.pack(fill='both', expand=True)

    # Configurar el TreeView con las columnas del DataFrame
    tv = ttk.Treeview(frame, columns=list(df.columns), show='headings')
    tv.pack(fill='both', expand=True)

    # Definir los encabezados del TreeView basándose en las columnas del DataFrame
    for columna in df.columns:
        tv.heading(columna, text=columna)
        tv.column(columna, width=100)  # Configurar el ancho de la columna si es necesario

    # Agregar los datos del DataFrame al TreeView
    for _, fila in df.iterrows():
        tv.insert('', 'end', values=list(fila))

    # Ejecutar la ventana emergente
    ventana.mainloop()