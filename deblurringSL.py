import customtkinter as ctk
from tkinter import filedialog
import cv2
from PIL import Image
import numpy as np
from scipy.signal import wiener

# Configuración de apariencia
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ForensicDeblurApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Forensic Image Restorer - Deconvolución de Wiener")
        self.geometry("1000x650")
        
        self.cv_img_original = None
        self.cv_img_processed = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # --- Panel Lateral de Controles ---
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)
        
        # Botón de carga
        self.btn_load = ctk.CTkButton(self.sidebar, text="Cargar Imagen", command=self.load_image)
        self.btn_load.pack(pady=20, padx=15)
        
        # Separador / Título de sección
        self.label_section = ctk.CTkLabel(self.sidebar, text="Filtro de Wiener", font=ctk.CTkFont(weight="bold"))
        self.label_section.pack(pady=(10, 5))
        
        # Slider 1: Tamaño del Núcleo (Debe ser impar)
        self.label_size = ctk.CTkLabel(self.sidebar, text="Tamaño del Filtro (Ventana): 3")
        self.label_size.pack(pady=(15, 0))
        
        # Mapeamos los pasos del slider a números impares: 3, 5, 7, 9, 11
        self.slider_size = ctk.CTkSlider(self.sidebar, from_=1, to=5, number_of_steps=4, command=self.on_slider_change)
        self.slider_size.set(1) # Corresponde a 3
        self.slider_size.pack(pady=5, padx=15)
        
        # Slider 2: Estimación de Ruido
        self.label_noise = ctk.CTkLabel(self.sidebar, text="Estimación de Ruido: 0.01")
        self.label_noise.pack(pady=(15, 0))
        
        self.slider_noise = ctk.CTkSlider(self.sidebar, from_=0.001, to=0.5, command=self.on_slider_change)
        self.slider_noise.set(0.01)
        self.slider_noise.pack(pady=5, padx=15)
        
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
            
            # Redimensionar si es muy grande para agilizar el procesamiento matemático en tiempo real
            max_dim = 600
            h, w = self.cv_img_original.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                self.cv_img_original = cv2.resize(self.cv_img_original, (int(w * scale), int(h * scale)))
                
            # Resetear controles
            self.slider_size.set(1)
            self.slider_noise.set(0.01)
            self.label_size.configure(text="Tamaño del Filtro (Ventana): 3")
            self.label_noise.configure(text="Estimación de Ruido: 0.01")
            
            self.display_image(self.cv_img_original)

    def on_slider_change(self, value):
        """Captura el movimiento de cualquier slider y ejecuta el procesamiento"""
        if self.cv_img_original is None:
            return
            
        # Calcular el tamaño impar basado en el paso del slider (1->3, 2->5, 3->7, 4->9, 5->11)
        steps_value = int(float(self.slider_size.get()))
        filter_size = 2 * steps_value + 1
        self.label_size.configure(text=f"Tamaño del Filtro (Ventana): {filter_size}")
        
        # Capturar el valor del ruido
        noise_value = float(self.slider_noise.get())
        self.label_noise.configure(text=f"Estimación de Ruido: {noise_value:.3f}")
        
        # Ejecutar el algoritmo forense
        self.process_image_wiener(filter_size, noise_value)

    def process_image_wiener(self, mysize, noise):
        # 1. Convertir la imagen a flotante (0.0 a 1.0) para precisión matemática
        img_float = self.cv_img_original.astype(np.float64) / 255.0
        
        # 2. Dividir en canales de color (B, G, R)
        channels = cv2.split(img_float)
        processed_channels = []
        
        # 3. Aplicar el filtro de Wiener de Scipy a cada canal de forma independiente
        for ch in channels:
            ch_deblurred = wiener(ch, mysize=mysize, noise=noise)
            processed_channels.append(ch_deblurred)
            
        # 4. Fusionar los canales y regresar al formato estándar de imagen (0-255)
        result = cv2.merge(processed_channels)
        self.cv_img_processed = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        # Mostrar en pantalla
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