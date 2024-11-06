import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

class ReconocedorVocales:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocedor de Vocales")
        self.root.geometry("800x600")
        self.root.configure(bg="#333333")
        
        # Plantillas de vocales 16x16
        self.templates = {
            'A': np.array([
                [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0],
                [0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0],
                [0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0],
                [0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0],
                [0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1]
            ], dtype=np.float32),
            
            'E': np.array([
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
            ], dtype=np.float32),
            
            'I': np.array([
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
            ], dtype=np.float32),
            
            'O': np.array([
                [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
                [0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0],
                [1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1],
                [0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
                [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0]
            ], dtype=np.float32),
            
            'U': np.array([
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1],
                [0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
                [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0]
            ], dtype=np.float32)
        }
        
        # También agregamos las plantillas en minúsculas
        lowercase_templates = {}
        for key, value in self.templates.items():
            lowercase_templates[key.lower()] = value * 0.8  # Versión más pequeña para minúsculas
        self.templates.update(lowercase_templates)
        
        # Crear interfaz
        self.create_widgets()
        
        self.current_image = None
        
    def create_widgets(self):
        # Estilos
        button_style = {"bg": "#4a4a4a", "fg": "white", "font": ("Helvetica", 12),
                       "relief": "flat", "padx": 20, "pady": 10}
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg="#333333")
        main_frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Botones
        self.load_button = tk.Button(main_frame, text="Cargar Imagen",
                                   command=self.load_image, **button_style)
        self.load_button.pack(pady=10)
        
        self.process_button = tk.Button(main_frame, text="Procesar",
                                      command=self.process_image, **button_style)
        self.process_button.pack(pady=10)
        
        # Canvas para la imagen
        self.canvas = tk.Canvas(main_frame, width=500, height=500,
                              bg="white", highlightthickness=0)
        self.canvas.pack(pady=20)
        
        # Label para resultados
        self.result_label = tk.Label(main_frame, text="", bg="#333333",
                                   fg="white", font=("Helvetica", 14))
        self.result_label.pack(pady=10)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if file_path:
            # Cargar imagen con OpenCV
            img = cv2.imread(file_path)
            if img is None:
                messagebox.showerror("Error", "No se pudo cargar la imagen")
                return
                
            # Verificar tamaño
            height, width = img.shape[:2]
            if width > 512 or height > 512:
                img = cv2.resize(img, (512, 512))
            
            self.current_image = img
            
            # Mostrar en canvas
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            
            self.canvas.config(width=img_pil.width, height=img_pil.height)
            self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
            self.canvas.image = img_tk  # Mantener referencia
            
    def process_image(self):
        if self.current_image is None:
            messagebox.showerror("Error", "Por favor cargue una imagen primero")
            return
            
        # Convertir a escala de grises
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
# Redimensionar a 16x16
        resized = cv2.resize(gray, (16, 16))
        
        # Binarizar la imagen
        _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
        
        # Normalizar a valores 0-1
        normalized = binary / 255.0
        
        # Comparar con cada plantilla
        results = {}
        for letra, template in self.templates.items():
            # Calcular correlación
            correlation = np.sum(normalized * template) / (16 * 16)
            results[letra] = correlation
        
        # Encontrar la mejor coincidencia
        best_match = max(results.items(), key=lambda x: x[1])
        confidence = best_match[1]
        
        if confidence > 0.75:  # Umbral de confianza
            result_text = f"Vocal detectada: {best_match[0]}\nConfianza: {confidence:.2%}"
        else:
            result_text = "No se detectó ninguna vocal clara"
        
        self.result_label.config(text=result_text)
        
        # Mostrar imagen procesada
        processed_display = cv2.resize(binary, (200, 200))
        img_pil = Image.fromarray(processed_display)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # Actualizar canvas con la imagen procesada
        self.canvas.config(width=200, height=200)
        self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
        self.canvas.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = ReconocedorVocales(root)
    root.mainloop()