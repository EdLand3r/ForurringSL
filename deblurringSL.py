import customtkinter as ctk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np

# Configuración de apariencia
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ForensicDeblurApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Forensic Image Restorer - Prototipo")
        self.geometry("900x600")
        
        self.cv_img_original = None
        self.cv_img_processed = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # --- Panel Lateral de Controles ---
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)
        
        self.btn_load = ctk.CTkButton(self.sidebar, text="Cargar Imagen", command=self.load_image)
        self.btn_load.pack(pady=20, padx=10)
        
        self.label_slider = ctk.CTkLabel(self.sidebar, text="Intensidad de Enfoque: 0")
        self.label_slider.pack(pady=(20, 0))
        
        self.slider_intensity = ctk.CTkSlider(self.sidebar, from_=0, to=10, number_of_steps=10, command=self.process_image)
        self.slider_intensity.set(0)
        self.slider_intensity.pack(pady=10, padx=10)
        
        # --- Panel Central de Visualización ---
        self.preview_frame = ctk.CTkFrame(self)
        self.preview_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)
        
        self.image_label = ctk.CTkLabel(self.preview_frame, text="Por favor, carga una imagen para comenzar")
        self.image_label.pack(expand=True)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            # Leer imagen con OpenCV
            self.cv_img_original = cv2.imread(file_path)
            # Redimensionar si es muy grande para que quepa bien en la pantalla de desarrollo
            max_dim = 500
            h, w = self.cv_img_original.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                self.cv_img_original = cv2.resize(self.cv_img_original, (int(w * scale), int(h * scale)))
                
            self.slider_intensity.set(0)
            self.label_slider.configure(text="Intensidad de Enfoque: 0")
            self.display_image(self.cv_img_original)

    def process_image(self, value):
        if self.cv_img_original is None:
            return
            
        intensity = int(float(value))
        self.label_slider.configure(text=f"Intensidad de Enfoque: {intensity}")
        
        if intensity == 0:
            self.display_image(self.cv_img_original)
            return
            
        # Algoritmo de Enfoque (Sharpening Kernel) adaptativo según el slider
        # Enfoque forense tradicional usando una matriz de convolución
        center_value = 1 + 4 * (intensity / 10)
        edge_value = -(intensity / 10)
        
        kernel = np.array([
            [0, edge_value, 0],
            [edge_value, center_value, edge_value],
            [0, edge_value, 0]
        ])
        
        # Aplicar el filtro a la imagen original
        self.cv_img_processed = cv2.filter2D(self.cv_img_original, -1, kernel)
        self.display_image(self.cv_img_processed)

    def display_image(self, cv_img):
        # Convertir de BGR (OpenCV) a RGB (Pillow/Tkinter)
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(img_pil.width, img_pil.height))
        
        self.image_label.configure(image=img_tk, text="")
        self.image_label.image = img_tk  # Mantener referencia en memoria

if __name__ == "__main__":
    app = ForensicDeblurApp()
    app.mainloop()