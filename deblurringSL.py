import customtkinter as ctk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
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
        
        self.title("Forensic Image Restorer - Avanzado (Zoom & ROI)")
        self.geometry("1100x750")
        
        self.cv_img_original = None
        self.cv_img_processed = None
        self.tk_img = None
        
        # Variables de estado para Zoom, Paneo y ROI
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.roi = None # (x1, y1, x2, y2) en coordenadas de la imagen original
        self.roi_start = None # (x, y) en coordenadas del canvas
        
        self.setup_ui()
        
    def setup_ui(self):
        # --- Panel Lateral de Controles ---
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)
        
        # Botones superiores
        self.btn_load = ctk.CTkButton(self.sidebar, text="Cargar Imagen", command=self.load_image)
        self.btn_load.pack(pady=(20, 5), padx=15)
        
        self.btn_clear_roi = ctk.CTkButton(self.sidebar, text="Limpiar Selección (ROI)", command=self.clear_roi, fg_color="#7a2b2b", hover_color="#9c3b3b")
        self.btn_clear_roi.pack(pady=5, padx=15)
        
        # Controles de Algoritmo
        self.label_section = ctk.CTkLabel(self.sidebar, text="Algoritmo", font=ctk.CTkFont(weight="bold"))
        self.label_section.pack(pady=(15, 5))
        
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
        
        self.btn_apply = ctk.CTkButton(self.sidebar, text="Aplicar Algoritmo", command=self.apply_heavy_algorithm)
        
        # Instrucciones de uso
        self.label_instructions = ctk.CTkLabel(
            self.sidebar, 
            text="Controles:\n- Rueda Ratón: Zoom\n- Clic Derecho + Arrastrar: Mover\n- Clic Izquierdo + Arrastrar: Seleccionar Área",
            text_color="gray",
            justify="left"
        )
        self.label_instructions.pack(side="bottom", pady=20, padx=15)
        
        # --- Panel Central de Visualización (Canvas) ---
        self.preview_frame = ctk.CTkFrame(self)
        self.preview_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)
        
        self.canvas = ctk.CTkCanvas(self.preview_frame, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(expand=True, fill="both")
        
        # Eventos del ratón en el Canvas
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        
        # Paneo (Mover la imagen)
        self.canvas.bind("<ButtonPress-3>", self.start_pan)
        self.canvas.bind("<B3-Motion>", self.do_pan)
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.do_pan)
        
        # ROI (Seleccionar área)
        self.canvas.bind("<ButtonPress-1>", self.start_roi)
        self.canvas.bind("<B1-Motion>", self.do_roi)
        self.canvas.bind("<ButtonRelease-1>", self.end_roi)
        
        # Asegurar redibujado si se cambia el tamaño de la ventana
        self.canvas.bind("<Configure>", lambda e: self.redraw_canvas())
        
        # Inicializar controles
        self.on_algorithm_change("Wiener Espacial (Ligero)")

    def clear_roi(self):
        self.roi = None
        self.cv_img_processed = None
        self.trigger_light_processing()
        self.redraw_canvas()

    def clear_dynamic_frame(self):
        for widget in self.dynamic_frame.winfo_children():
            widget.destroy()

    def on_algorithm_change(self, algo_name):
        self.clear_dynamic_frame()
        self.btn_apply.pack_forget()
        
        if algo_name == "Wiener Espacial (Ligero)":
            self.label_size = ctk.CTkLabel(self.dynamic_frame, text="Tamaño de Ventana: 3")
            self.label_size.pack(pady=(10, 0))
            self.slider_size = ctk.CTkSlider(self.dynamic_frame, from_=1, to=5, number_of_steps=4, command=self.trigger_light_processing)
            self.slider_size.set(1)
            self.slider_size.pack(pady=5, padx=15)
            
            self.label_noise = ctk.CTkLabel(self.dynamic_frame, text="Ruido: 0.010")
            self.label_noise.pack(pady=(10, 0))
            self.slider_noise = ctk.CTkSlider(self.dynamic_frame, from_=0.001, to=0.5, command=self.trigger_light_processing)
            self.slider_noise.set(0.01)
            self.slider_noise.pack(pady=5, padx=15)
            
            self.trigger_light_processing()
            
        else:
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
            self.slider_psf_size.set(2)
            self.slider_psf_size.pack(pady=5, padx=15)
            
            self.label_angle = ctk.CTkLabel(self.dynamic_frame, text="Ángulo (grados): 0")
            self.slider_angle = ctk.CTkSlider(self.dynamic_frame, from_=0, to=180, number_of_steps=180, command=self.update_labels)
            self.slider_angle.set(0)
            
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
            self.cv_img_processed = None
            self.redraw_canvas()

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

    # --- Mouse Events para Canvas ---
    
    def on_mouse_wheel(self, event):
        if self.cv_img_original is None: return
        
        # Coordenadas relativas al canvas
        x = event.x
        y = event.y
        
        # Coordenadas en la imagen antes del zoom
        ix = (x - self.pan_x) / self.zoom_factor
        iy = (y - self.pan_y) / self.zoom_factor
        
        if event.delta > 0:
            self.zoom_factor *= 1.2
        else:
            self.zoom_factor /= 1.2
            
        # Limitar el alejamiento excesivo
        self.zoom_factor = max(0.1, min(self.zoom_factor, 20.0))
            
        # Ajustar el paneo para que el zoom sea hacia el puntero
        self.pan_x = x - ix * self.zoom_factor
        self.pan_y = y - iy * self.zoom_factor
        
        self.redraw_canvas()

    def start_pan(self, event):
        self.canvas.config(cursor="fleur")
        self.pan_start_x = event.x - self.pan_x
        self.pan_start_y = event.y - self.pan_y

    def do_pan(self, event):
        self.pan_x = event.x - self.pan_start_x
        self.pan_y = event.y - self.pan_start_y
        self.redraw_canvas()
        
    def end_pan(self, event):
        self.canvas.config(cursor="")

    def start_roi(self, event):
        if self.cv_img_original is None: return
        self.roi_start = (event.x, event.y)
        self.roi = None
        self.cv_img_processed = None
        self.canvas.delete("roi_rect")
        self.redraw_canvas() # Limpia procesamientos previos si se inicia un nuevo ROI

    def do_roi(self, event):
        if self.cv_img_original is None or not self.roi_start: return
        self.canvas.delete("temp_roi")
        self.canvas.create_rectangle(
            self.roi_start[0], self.roi_start[1], event.x, event.y, 
            outline="#00ffcc", width=2, dash=(4, 4), tags="temp_roi"
        )

    def end_roi(self, event):
        if self.cv_img_original is None or not self.roi_start: return
        self.canvas.delete("temp_roi")
        
        x1 = min(self.roi_start[0], event.x)
        x2 = max(self.roi_start[0], event.x)
        y1 = min(self.roi_start[1], event.y)
        y2 = max(self.roi_start[1], event.y)
        
        if x2 - x1 < 10 or y2 - y1 < 10:
            self.roi = None
        else:
            # Convertir de coordenadas del canvas a coordenadas de la imagen
            ix1 = int((x1 - self.pan_x) / self.zoom_factor)
            ix2 = int((x2 - self.pan_x) / self.zoom_factor)
            iy1 = int((y1 - self.pan_y) / self.zoom_factor)
            iy2 = int((y2 - self.pan_y) / self.zoom_factor)
            
            h, w = self.cv_img_original.shape[:2]
            ix1, ix2 = max(0, min(w, ix1)), max(0, min(w, ix2))
            iy1, iy2 = max(0, min(h, iy1)), max(0, min(h, iy2))
            
            if ix2 > ix1 and iy2 > iy1:
                self.roi = (ix1, iy1, ix2, iy2)
            else:
                self.roi = None
            
        self.roi_start = None
        self.trigger_light_processing()
        self.redraw_canvas()

    def redraw_canvas(self):
        if self.cv_img_original is None: return
        
        self.canvas.delete("all")
        
        img_to_draw = self.cv_img_processed if self.cv_img_processed is not None else self.cv_img_original
        
        h, w = img_to_draw.shape[:2]
        new_w = int(w * self.zoom_factor)
        new_h = int(h * self.zoom_factor)
        
        if new_w <= 0 or new_h <= 0: return
        
        img_rgb = cv2.cvtColor(img_to_draw, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Redimensionado eficiente: Nearest para gran zoom para ver píxeles, Bilinear para alejamiento
        resample_filter = Image.Resampling.NEAREST if self.zoom_factor > 2.0 else Image.Resampling.BILINEAR
        pil_img = pil_img.resize((new_w, new_h), resample_filter)
        
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(self.pan_x, self.pan_y, anchor="nw", image=self.tk_img)
        
        # Dibujar ROI si existe
        if self.roi:
            x1, y1, x2, y2 = self.roi
            cx1 = x1 * self.zoom_factor + self.pan_x
            cy1 = y1 * self.zoom_factor + self.pan_y
            cx2 = x2 * self.zoom_factor + self.pan_x
            cy2 = y2 * self.zoom_factor + self.pan_y
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="#00ffcc", width=2, dash=(4, 4), tags="roi_rect")

    # --- Procesamiento de Imágenes ---

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.cv_img_original = cv2.imread(file_path)
            
            # Ahora que tenemos ROI y Zoom, podemos permitir imágenes más grandes. Limitamos a 2000px por seguridad de RAM.
            max_dim = 2000
            h, w = self.cv_img_original.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                self.cv_img_original = cv2.resize(self.cv_img_original, (int(w * scale), int(h * scale)))
                
            self.cv_img_processed = None
            self.roi = None
            
            # Resetear vista al centro
            self.zoom_factor = 1.0
            self.canvas.update()
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            
            h, w = self.cv_img_original.shape[:2]
            # Ajustar zoom para que la imagen quepa inicialmente
            scale_w = cw / w
            scale_h = ch / h
            self.zoom_factor = min(scale_w, scale_h) * 0.9 if min(scale_w, scale_h) < 1 else 1.0
            
            self.pan_x = (cw - (w * self.zoom_factor)) / 2
            self.pan_y = (ch - (h * self.zoom_factor)) / 2
            
            self.trigger_light_processing()
            self.redraw_canvas()

    def generate_psf(self, psf_type, size, angle=0):
        psf = np.zeros((size, size))
        center = size // 2
        if psf_type == "Gaussian":
            x, y = np.mgrid[-center:center+1, -center:center+1]
            sigma = size / 6.0 if size > 1 else 1.0
            g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            psf = g / g.sum()
        elif psf_type == "Motion Blur":
            psf[center, :] = 1.0
            matrix = cv2.getRotationMatrix2D((center, center), angle, 1)
            psf = cv2.warpAffine(psf, matrix, (size, size))
            psf = psf / psf.sum()
        return psf

    def get_image_to_process(self):
        if self.roi:
            x1, y1, x2, y2 = self.roi
            return self.cv_img_original[y1:y2, x1:x2]
        return self.cv_img_original

    def set_processed_result(self, result):
        if self.roi:
            x1, y1, x2, y2 = self.roi
            if self.cv_img_processed is None:
                self.cv_img_processed = self.cv_img_original.copy()
            self.cv_img_processed[y1:y2, x1:x2] = result
        else:
            self.cv_img_processed = result
        self.redraw_canvas()

    def trigger_light_processing(self, value=None):
        if self.cv_img_original is None: return
        if self.algorithm_var.get() != "Wiener Espacial (Ligero)": return
        
        steps_value = int(float(self.slider_size.get()))
        filter_size = 2 * steps_value + 1
        self.label_size.configure(text=f"Tamaño de Ventana: {filter_size}")
        
        noise_value = float(self.slider_noise.get())
        self.label_noise.configure(text=f"Ruido: {noise_value:.3f}")
        
        img_target = self.get_image_to_process()
        if img_target.size == 0: return
        
        img_float = img_target.astype(np.float64) / 255.0
        channels = cv2.split(img_float)
        processed_channels = []
        for ch in channels:
            ch_deblurred = wiener(ch, mysize=filter_size, noise=noise_value)
            processed_channels.append(ch_deblurred)
            
        result = cv2.merge(processed_channels)
        result_uint8 = np.clip(result * 255, 0, 255).astype(np.uint8)
        self.set_processed_result(result_uint8)

    def apply_heavy_algorithm(self):
        if self.cv_img_original is None: return
        
        algo = self.algorithm_var.get()
        psf_type = self.psf_type_var.get()
        psf_size = 2 * int(self.slider_psf_size.get()) + 1
        angle = int(self.slider_angle.get()) if hasattr(self, 'slider_angle') and self.slider_angle.winfo_exists() else 0
        
        psf = self.generate_psf(psf_type, psf_size, angle)
        img_target = self.get_image_to_process()
        if img_target.size == 0: return
        
        img_float = img_target.astype(np.float64) / 255.0
        channels = cv2.split(img_float)
        processed_channels = []
        
        self.btn_apply.configure(text="Procesando...", state="disabled")
        self.update() 
        
        try:
            for ch in channels:
                if algo == "Wiener No Supervisado":
                    ch_deblurred, _ = restoration.unsupervised_wiener(ch, psf)
                elif algo == "Richardson-Lucy":
                    iters = int(self.slider_iter.get())
                    ch_deblurred = restoration.richardson_lucy(ch, psf, num_iter=iters, clip=False)
                processed_channels.append(ch_deblurred)
                
            result = cv2.merge(processed_channels)
            result_uint8 = np.clip(result * 255, 0, 255).astype(np.uint8)
            self.set_processed_result(result_uint8)
        except Exception as e:
            print(f"Error en el procesamiento: {e}")
        finally:
            self.btn_apply.configure(text="Aplicar Algoritmo", state="normal")

if __name__ == "__main__":
    app = ForensicDeblurApp()
    app.mainloop()