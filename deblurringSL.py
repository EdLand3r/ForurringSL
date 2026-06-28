import customtkinter as ctk
from tkinter import filedialog
import cv2
from PIL import Image
import numpy as np
from scipy.signal import wiener
from skimage import restoration
import math

# Configuración de apariencia
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ForensicDeblurApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Forensic Image Restorer - Avanzado")
        self.geometry("1000x700")
        
        self.cv_img_original = None
        self.cv_img_processed = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # --- Panel Lateral de Controles ---
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)
        
        # Botón de carga
        self.btn_load = ctk.CTkButton(self.sidebar, text="Cargar Imagen", command=self.load_image)
        self.btn_load.pack(pady=20, padx=15)
        
        self.label_section = ctk.CTkLabel(self.sidebar, text="Algoritmo", font=ctk.CTkFont(weight="bold"))
        self.label_section.pack(pady=(10, 5))
        
        self.algorithm_var = ctk.StringVar(value="Wiener Espacial (Ligero)")
        self.algorithm_dropdown = ctk.CTkOptionMenu(
            self.sidebar, 
            values=["Wiener Espacial (Ligero)", "Wiener No Supervisado", "Richardson-Lucy"], 
            variable=self.algorithm_var, 
            command=self.on_algorithm_change
        )
        self.algorithm_dropdown.pack(pady=5, padx=15)
        
        # Contenedor para controles dinámicos
        self.dynamic_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.dynamic_frame.pack(fill="x", pady=10)
        
        self.btn_apply = ctk.CTkButton(self.sidebar, text="Aplicar Algoritmo", command=self.apply_algorithm)
        # Oculto por defecto, se muestra para algoritmos pesados
        
        # --- Panel Central de Visualización ---
        self.preview_frame = ctk.CTkFrame(self)
        self.preview_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)
        
        self.image_label = ctk.CTkLabel(self.preview_frame, text="Por favor, carga una imagen para comenzar")
        self.image_label.pack(expand=True)
        
        # Inicializar controles
        self.on_algorithm_change("Wiener Espacial (Ligero)")

    def clear_dynamic_frame(self):
        for widget in self.dynamic_frame.winfo_children():
            widget.destroy()

    def on_algorithm_change(self, algo_name):
        self.clear_dynamic_frame()
        self.btn_apply.pack_forget() # Ocultar botón aplicar por defecto
        
        if algo_name == "Wiener Espacial (Ligero)":
            # Controles antiguos (en tiempo real)
            self.label_size = ctk.CTkLabel(self.dynamic_frame, text="Tamaño de Ventana: 3")
            self.label_size.pack(pady=(10, 0))
            self.slider_size = ctk.CTkSlider(self.dynamic_frame, from_=1, to=5, number_of_steps=4, command=self.on_slider_change_light)
            self.slider_size.set(1)
            self.slider_size.pack(pady=5, padx=15)
            
            self.label_noise = ctk.CTkLabel(self.dynamic_frame, text="Ruido: 0.010")
            self.label_noise.pack(pady=(10, 0))
            self.slider_noise = ctk.CTkSlider(self.dynamic_frame, from_=0.001, to=0.5, command=self.on_slider_change_light)
            self.slider_noise.set(0.01)
            self.slider_noise.pack(pady=5, padx=15)
            
        else:
            # Algoritmos Pesados (Richardson-Lucy, Wiener No Supervisado)
            self.label_psf_type = ctk.CTkLabel(self.dynamic_frame, text="Tipo de Desenfoque (PSF)")
            self.label_psf_type.pack(pady=(10, 0))
            
            self.psf_type_var = ctk.StringVar(value="Gaussian")
            self.psf_dropdown = ctk.CTkOptionMenu(
                self.dynamic_frame, 
                values=["Gaussian", "Motion Blur"], 
                variable=self.psf_type_var,
                command=self.on_psf_change
            )
            self.psf_dropdown.pack(pady=5, padx=15)
            
            self.label_psf_size = ctk.CTkLabel(self.dynamic_frame, text="Tamaño del PSF: 5")
            self.label_psf_size.pack(pady=(10, 0))
            self.slider_psf_size = ctk.CTkSlider(self.dynamic_frame, from_=1, to=15, number_of_steps=14, command=self.update_labels)
            self.slider_psf_size.set(2) # Equivale a 5
            self.slider_psf_size.pack(pady=5, padx=15)
            
            # Ángulo para Motion Blur (oculto por defecto)
            self.label_angle = ctk.CTkLabel(self.dynamic_frame, text="Ángulo (grados): 0")
            self.slider_angle = ctk.CTkSlider(self.dynamic_frame, from_=0, to=180, number_of_steps=180, command=self.update_labels)
            self.slider_angle.set(0)
            
            # Mostrar controles de Motion Blur si ya está seleccionado
            if self.psf_type_var.get() == "Motion Blur":
                self.label_angle.pack(pady=(10, 0))
                self.slider_angle.pack(pady=5, padx=15)
            
            if algo_name == "Richardson-Lucy":
                self.label_iter = ctk.CTkLabel(self.dynamic_frame, text="Iteraciones: 15")
                self.label_iter.pack(pady=(10, 0))
                self.slider_iter = ctk.CTkSlider(self.dynamic_frame, from_=1, to=50, number_of_steps=49, command=self.update_labels)
                self.slider_iter.set(15)
                self.slider_iter.pack(pady=5, padx=15)
                
            self.btn_apply.pack(pady=20, padx=15)
            
        # Si ya hay imagen, limpiar el procesado al cambiar de algoritmo
        if self.cv_img_original is not None:
            self.display_image(self.cv_img_original)

    def on_psf_change(self, psf_type):
        if psf_type == "Motion Blur":
            self.label_angle.pack(pady=(10, 0))
            self.slider_angle.pack(pady=5, padx=15)
        else:
            self.label_angle.pack_forget()
            self.slider_angle.pack_forget()
            
    def update_labels(self, value=None):
        if hasattr(self, 'slider_psf_size') and self.slider_psf_size.winfo_exists():
            psf_size = 2 * int(self.slider_psf_size.get()) + 1
            self.label_psf_size.configure(text=f"Tamaño del PSF: {psf_size}")
            
        if hasattr(self, 'slider_angle') and self.slider_angle.winfo_exists():
            angle = int(self.slider_angle.get())
            self.label_angle.configure(text=f"Ángulo (grados): {angle}")
            
        if hasattr(self, 'slider_iter') and self.slider_iter.winfo_exists():
            iters = int(self.slider_iter.get())
            self.label_iter.configure(text=f"Iteraciones: {iters}")

    def on_slider_change_light(self, value):
        if self.cv_img_original is None:
            return
            
        steps_value = int(float(self.slider_size.get()))
        filter_size = 2 * steps_value + 1
        self.label_size.configure(text=f"Tamaño de Ventana: {filter_size}")
        
        noise_value = float(self.slider_noise.get())
        self.label_noise.configure(text=f"Ruido: {noise_value:.3f}")
        
        self.process_image_wiener_light(filter_size, noise_value)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.cv_img_original = cv2.imread(file_path)
            max_dim = 600
            h, w = self.cv_img_original.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                self.cv_img_original = cv2.resize(self.cv_img_original, (int(w * scale), int(h * scale)))
                
            self.display_image(self.cv_img_original)

    def generate_psf(self, psf_type, size, angle=0):
        psf = np.zeros((size, size))
        center = size // 2
        if psf_type == "Gaussian":
            x, y = np.mgrid[-center:center+1, -center:center+1]
            sigma = size / 6.0 if size > 1 else 1.0
            g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            psf = g / g.sum()
        elif psf_type == "Motion Blur":
            # Crear una línea horizontal
            psf[center, :] = 1.0
            # Rotar
            matrix = cv2.getRotationMatrix2D((center, center), angle, 1)
            psf = cv2.warpAffine(psf, matrix, (size, size))
            psf = psf / psf.sum()
        return psf

    def process_image_wiener_light(self, mysize, noise):
        img_float = self.cv_img_original.astype(np.float64) / 255.0
        channels = cv2.split(img_float)
        processed_channels = []
        for ch in channels:
            ch_deblurred = wiener(ch, mysize=mysize, noise=noise)
            processed_channels.append(ch_deblurred)
        result = cv2.merge(processed_channels)
        self.cv_img_processed = np.clip(result * 255, 0, 255).astype(np.uint8)
        self.display_image(self.cv_img_processed)

    def apply_algorithm(self):
        if self.cv_img_original is None:
            return
            
        algo = self.algorithm_var.get()
        psf_type = self.psf_type_var.get()
        psf_size = 2 * int(self.slider_psf_size.get()) + 1
        angle = int(self.slider_angle.get()) if hasattr(self, 'slider_angle') and self.slider_angle.winfo_exists() else 0
        
        psf = self.generate_psf(psf_type, psf_size, angle)
        
        # Convertir a float
        img_float = self.cv_img_original.astype(np.float64) / 255.0
        
        channels = cv2.split(img_float)
        processed_channels = []
        
        self.btn_apply.configure(text="Procesando...", state="disabled")
        self.update() # Forzar actualización de UI
        
        try:
            for ch in channels:
                if algo == "Wiener No Supervisado":
                    ch_deblurred, _ = restoration.unsupervised_wiener(ch, psf)
                elif algo == "Richardson-Lucy":
                    iters = int(self.slider_iter.get())
                    ch_deblurred = restoration.richardson_lucy(ch, psf, num_iter=iters, clip=False)
                processed_channels.append(ch_deblurred)
                
            result = cv2.merge(processed_channels)
            self.cv_img_processed = np.clip(result * 255, 0, 255).astype(np.uint8)
            self.display_image(self.cv_img_processed)
        except Exception as e:
            print(f"Error en el procesamiento: {e}")
        finally:
            self.btn_apply.configure(text="Aplicar Algoritmo", state="normal")

    def display_image(self, cv_img):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(img_pil.width, img_pil.height))
        
        self.image_label.configure(image=img_tk, text="")
        self.image_label.image = img_tk

if __name__ == "__main__":
    app = ForensicDeblurApp()
    app.mainloop()