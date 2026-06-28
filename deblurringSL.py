import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
from scipy.signal import wiener
from skimage import restoration
import math
import os
import urllib.request
import hashlib
from datetime import datetime

# Configuración de apariencia
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ForensicDeblurApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Forensic Image Restorer - Pro Suite")
        self.geometry("1250x800")
        
        self.cv_img_backup = None # Imagen original intacta para poder resetear
        self.cv_img_original = None # Imagen actual (después de geometría/pre-proceso)
        self.cv_img_processed = None # Imagen después de filtros
        self.tk_img = None
        
        # Auditoría (Cadena de Custodia)
        self.audit_log = []
        self.original_hash = ""
        
        # Variables de estado para Zoom, Paneo, ROI y Perspectiva
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.roi = None 
        self.roi_start = None 
        
        self.perspective_mode = False
        self.perspective_points = []
        
        self.setup_ui()
        
    def setup_ui(self):
        # --- Panel Lateral de Controles ---
        self.sidebar = ctk.CTkFrame(self, width=350, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)
        
        # Botones Carga
        self.btn_load = ctk.CTkButton(self.sidebar, text="Cargar Imagen Única", command=self.load_image)
        self.btn_load.pack(pady=(15, 5), padx=15)
        
        self.btn_load_multi = ctk.CTkButton(self.sidebar, text="Promediar Múltiples (Stacking)", command=self.load_multiple_images, fg_color="#458265", hover_color="#5a9b7c")
        self.btn_load_multi.pack(pady=5, padx=15)
        
        self.btn_reset = ctk.CTkButton(self.sidebar, text="Restaurar Imagen Original", command=self.reset_image, fg_color="#a86d32", hover_color="#c7823c")
        self.btn_reset.pack(pady=5, padx=15)
        
        self.btn_clear_roi = ctk.CTkButton(self.sidebar, text="Limpiar Selección (ROI)", command=self.clear_roi, fg_color="#7a2b2b", hover_color="#9c3b3b")
        self.btn_clear_roi.pack(pady=(5, 15), padx=15)
        
        # TABS (4 Pestañas)
        self.tabview = ctk.CTkTabview(self.sidebar)
        self.tabview.pack(padx=10, pady=5, fill="both", expand=True)
        
        self.tab_geo = self.tabview.add("1. Geometría")
        self.tab_pre = self.tabview.add("2. Pre-Proceso")
        self.tab_deblur = self.tabview.add("3. Filtros Clásicos")
        self.tab_ai = self.tabview.add("4. IA y Reporte")
        
        # --- TAB 1: GEOMETRÍA ---
        self.label_geo = ctk.CTkLabel(self.tab_geo, text="Corrección de Perspectiva", font=ctk.CTkFont(weight="bold"))
        self.label_geo.pack(pady=(15, 5))
        self.label_geo_desc = ctk.CTkLabel(self.tab_geo, text="Marca 4 esquinas de la matrícula\npara aplanarla frontalmente.", text_color="gray")
        self.label_geo_desc.pack(pady=5)
        self.btn_start_persp = ctk.CTkButton(self.tab_geo, text="Iniciar Selección (4 Puntos)", command=self.start_perspective)
        self.btn_start_persp.pack(pady=10, padx=15)
        self.btn_cancel_persp = ctk.CTkButton(self.tab_geo, text="Cancelar", command=self.cancel_perspective, state="disabled", fg_color="gray")
        self.btn_cancel_persp.pack(pady=5, padx=15)
        
        # --- TAB 2: PRE-PROCESO ---
        self.label_pre = ctk.CTkLabel(self.tab_pre, text="Ajustes Globales", font=ctk.CTkFont(weight="bold"))
        self.label_pre.pack(pady=(15, 10))
        
        self.btn_clahe = ctk.CTkButton(self.tab_pre, text="Mejorar Contraste (CLAHE)", command=self.apply_clahe)
        self.btn_clahe.pack(pady=10, padx=15)
        
        self.btn_denoise = ctk.CTkButton(self.tab_pre, text="Reducción de Ruido (NL Means)", command=self.apply_denoise)
        self.btn_denoise.pack(pady=10, padx=15)
        
        # --- TAB 3: DEBLURRING ---
        self.algorithm_var = ctk.StringVar(value="Wiener Espacial (Ligero)")
        self.algorithm_dropdown = ctk.CTkOptionMenu(
            self.tab_deblur, 
            values=["Wiener Espacial (Ligero)", "Wiener No Supervisado", "Richardson-Lucy"], 
            variable=self.algorithm_var, 
            command=self.on_algorithm_change
        )
        self.algorithm_dropdown.pack(pady=(15, 5), padx=15)
        
        self.dynamic_frame = ctk.CTkFrame(self.tab_deblur, fg_color="transparent")
        self.dynamic_frame.pack(fill="x", pady=5)
        
        self.btn_apply = ctk.CTkButton(self.tab_deblur, text="Aplicar Algoritmo", command=self.apply_heavy_algorithm)
        
        # --- TAB 4: IA Y REPORTE ---
        self.label_ai = ctk.CTkLabel(self.tab_ai, text="Super Resolución (LapSRN)", font=ctk.CTkFont(weight="bold"))
        self.label_ai.pack(pady=(15, 5))
        
        self.btn_ai_x4 = ctk.CTkButton(self.tab_ai, text="Mejorar (x4)", command=lambda: self.apply_ai(4))
        self.btn_ai_x4.pack(pady=5, padx=15)
        
        self.btn_ai_x8 = ctk.CTkButton(self.tab_ai, text="Mejorar (x8)", command=lambda: self.apply_ai(8))
        self.btn_ai_x8.pack(pady=5, padx=15)
        
        self.label_ai_status = ctk.CTkLabel(self.tab_ai, text="", text_color="green")
        self.label_ai_status.pack(pady=5)
        
        self.label_report = ctk.CTkLabel(self.tab_ai, text="Cadena de Custodia", font=ctk.CTkFont(weight="bold"))
        self.label_report.pack(pady=(20, 5))
        self.btn_export_log = ctk.CTkButton(self.tab_ai, text="Exportar Reporte Forense", command=self.export_report, fg_color="#3b3b3b")
        self.btn_export_log.pack(pady=5, padx=15)
        
        # Instrucciones
        self.label_instructions = ctk.CTkLabel(
            self.sidebar, 
            text="Controles:\n- Rueda: Zoom\n- Clic Der: Mover\n- Clic Izq: ROI / Perspectiva",
            text_color="gray",
            justify="left"
        )
        self.label_instructions.pack(side="bottom", pady=10, padx=15)
        
        # --- Panel Central de Visualización (Canvas) ---
        self.preview_frame = ctk.CTkFrame(self)
        self.preview_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)
        
        self.canvas = ctk.CTkCanvas(self.preview_frame, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(expand=True, fill="both")
        
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<ButtonPress-3>", self.start_pan)
        self.canvas.bind("<B3-Motion>", self.do_pan)
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.do_pan)
        self.canvas.bind("<ButtonPress-1>", self.on_left_click)
        self.canvas.bind("<B1-Motion>", self.do_roi)
        self.canvas.bind("<ButtonRelease-1>", self.end_roi)
        self.canvas.bind("<Configure>", lambda e: self.redraw_canvas())
        
        self.on_algorithm_change("Wiener Espacial (Ligero)")

    # --- AUDITORÍA (Cadena de Custodia) ---
    def calculate_hash(self, filepath):
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def log_action(self, action):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {action}"
        self.audit_log.append(entry)
        print(entry)

    def export_report(self):
        if not self.audit_log:
            messagebox.showinfo("Reporte", "No hay acciones para reportar.")
            return
        
        filepath = filedialog.asksaveasfilename(defaultextension=".txt", initialfile="Reporte_Forense.txt", filetypes=[("Archivos de Texto", "*.txt")])
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=== REPORTE DE ANÁLISIS FORENSE DE IMAGEN ===\n")
                f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Herramienta: Forensic Image Restorer - Pro Suite\n\n")
                f.write("--- CADENA DE EVENTOS ---\n")
                for log in self.audit_log:
                    f.write(f"{log}\n")
            messagebox.showinfo("Exportado", f"Reporte guardado exitosamente en:\n{filepath}")

    # --- CARGA Y RESETEO ---
    def init_image_state(self, img, filepath="Multiple Images (Averaged)", single_hash=None):
        self.cv_img_backup = img.copy()
        self.cv_img_original = img.copy()
        self.cv_img_processed = None
        self.roi = None
        self.perspective_mode = False
        self.perspective_points = []
        self.audit_log = []
        
        if single_hash:
            self.original_hash = single_hash
            self.log_action(f"IMAGEN CARGADA: {filepath}")
            self.log_action(f"HASH MD5: {self.original_hash}")
        else:
            self.log_action(f"STACKING CARGADO: {filepath}")
            
        self.label_ai_status.configure(text="")
        
        self.zoom_factor = 1.0
        self.canvas.update()
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        h, w = self.cv_img_original.shape[:2]
        if w > 0 and h > 0:
            scale_w = cw / w
            scale_h = ch / h
            self.zoom_factor = min(scale_w, scale_h) * 0.9 if min(scale_w, scale_h) < 1 else 1.0
            self.pan_x = (cw - (w * self.zoom_factor)) / 2
            self.pan_y = (ch - (h * self.zoom_factor)) / 2
            
        self.trigger_light_processing()
        self.redraw_canvas()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            img = cv2.imread(file_path)
            md5_hash = self.calculate_hash(file_path)
            
            max_dim = 2500
            h, w = img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
                
            self.init_image_state(img, file_path, md5_hash)

    def load_multiple_images(self):
        file_paths = filedialog.askopenfilenames(title="Selecciona múltiples fotogramas para promediar", filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")])
        if not file_paths or len(file_paths) < 2:
            messagebox.showinfo("Aviso", "Debes seleccionar al menos 2 imágenes para realizar el promediado.")
            return
            
        try:
            images = []
            for path in file_paths:
                img = cv2.imread(path)
                if img is not None:
                    images.append(img.astype(np.float32))
            
            if not images: return
            
            # Verificar que todas tengan el mismo tamaño
            shape = images[0].shape
            for img in images:
                if img.shape != shape:
                    messagebox.showerror("Error", "Todas las imágenes deben tener exactamente la misma resolución para promediarlas.")
                    return
                    
            # Promediar
            avg_img = np.zeros(shape, dtype=np.float32)
            for img in images:
                avg_img += img
            avg_img /= len(images)
            
            final_img = np.clip(avg_img, 0, 255).astype(np.uint8)
            self.init_image_state(final_img, f"Secuencia de {len(images)} imágenes", None)
            self.log_action("OPERACIÓN: Promediado de Fotogramas (Frame Averaging) aplicado exitosamente.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Fallo al promediar imágenes:\n{e}")

    def reset_image(self):
        if self.cv_img_backup is not None:
            self.cv_img_original = self.cv_img_backup.copy()
            self.cv_img_processed = None
            self.roi = None
            self.perspective_mode = False
            self.perspective_points = []
            self.log_action("ESTADO: Imagen restaurada a su estado original.")
            self.trigger_light_processing()
            self.redraw_canvas()

    # --- FUNCIONES DE TABS (Clásicas) ---
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
            self.label_size.pack(pady=(5, 0))
            self.slider_size = ctk.CTkSlider(self.dynamic_frame, from_=1, to=5, number_of_steps=4, command=self.trigger_light_processing)
            self.slider_size.set(1)
            self.slider_size.pack(pady=5, padx=15)
            
            self.label_noise = ctk.CTkLabel(self.dynamic_frame, text="Ruido: 0.010")
            self.label_noise.pack(pady=(5, 0))
            self.slider_noise = ctk.CTkSlider(self.dynamic_frame, from_=0.001, to=0.5, command=self.trigger_light_processing)
            self.slider_noise.set(0.01)
            self.slider_noise.pack(pady=5, padx=15)
            
            self.trigger_light_processing()
            
        else:
            self.label_psf_type = ctk.CTkLabel(self.dynamic_frame, text="Tipo de Desenfoque (PSF)")
            self.label_psf_type.pack(pady=(5, 0))
            
            self.psf_type_var = ctk.StringVar(value="Gaussian")
            self.psf_dropdown = ctk.CTkOptionMenu(
                self.dynamic_frame, 
                values=["Gaussian", "Motion Blur"], 
                variable=self.psf_type_var,
                command=self.on_psf_change
            )
            self.psf_dropdown.pack(pady=5, padx=15)
            
            self.label_psf_size = ctk.CTkLabel(self.dynamic_frame, text="Tamaño del PSF: 5")
            self.label_psf_size.pack(pady=(5, 0))
            self.slider_psf_size = ctk.CTkSlider(self.dynamic_frame, from_=1, to=15, number_of_steps=14, command=self.update_labels)
            self.slider_psf_size.set(2)
            self.slider_psf_size.pack(pady=5, padx=15)
            
            self.label_angle = ctk.CTkLabel(self.dynamic_frame, text="Ángulo (grados): 0")
            self.slider_angle = ctk.CTkSlider(self.dynamic_frame, from_=-90, to=90, number_of_steps=180, command=self.update_labels)
            self.slider_angle.set(0)
            
            if self.psf_type_var.get() == "Motion Blur":
                self.label_angle.pack(pady=(5, 0))
                self.slider_angle.pack(pady=5, padx=15)
            
            if algo_name == "Richardson-Lucy":
                self.label_iter = ctk.CTkLabel(self.dynamic_frame, text="Iteraciones: 15")
                self.label_iter.pack(pady=(5, 0))
                self.slider_iter = ctk.CTkSlider(self.dynamic_frame, from_=1, to=50, number_of_steps=49, command=self.update_labels)
                self.slider_iter.set(15)
                self.slider_iter.pack(pady=5, padx=15)
                
            self.btn_apply.pack(pady=15, padx=15)
            self.cv_img_processed = None
            self.redraw_canvas()

    def on_psf_change(self, psf_type):
        if psf_type == "Motion Blur":
            self.label_angle.pack(pady=(5, 0))
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

    # --- TAB 1: PERSPECTIVA ---
    def start_perspective(self):
        if self.cv_img_original is None: return
        self.perspective_mode = True
        self.perspective_points = []
        self.btn_start_persp.configure(state="disabled", text="Seleccionando (0/4)...")
        self.btn_cancel_persp.configure(state="normal")
        self.roi = None
        self.cv_img_processed = None
        self.canvas.config(cursor="crosshair")
        self.redraw_canvas()

    def cancel_perspective(self):
        self.perspective_mode = False
        self.perspective_points = []
        self.btn_start_persp.configure(state="normal", text="Iniciar Selección (4 Puntos)")
        self.btn_cancel_persp.configure(state="disabled")
        self.canvas.config(cursor="")
        self.redraw_canvas()

    def add_perspective_point(self, event):
        if self.cv_img_original is None: return
        
        # Convertir coord de canvas a coord de imagen
        ix = int((event.x - self.pan_x) / self.zoom_factor)
        iy = int((event.y - self.pan_y) / self.zoom_factor)
        
        h, w = self.cv_img_original.shape[:2]
        if 0 <= ix < w and 0 <= iy < h:
            self.perspective_points.append([ix, iy])
            self.btn_start_persp.configure(text=f"Seleccionando ({len(self.perspective_points)}/4)...")
            self.redraw_canvas()
            
            if len(self.perspective_points) == 4:
                self.apply_perspective()

    def apply_perspective(self):
        pts1 = np.float32(self.perspective_points)
        
        # Calcular dimensiones del nuevo rectángulo basado en distancias
        width_A = np.sqrt(((pts1[2][0] - pts1[3][0]) ** 2) + ((pts1[2][1] - pts1[3][1]) ** 2))
        width_B = np.sqrt(((pts1[1][0] - pts1[0][0]) ** 2) + ((pts1[1][0] - pts1[0][0]) ** 2))
        maxWidth = max(int(width_A), int(width_B))

        height_A = np.sqrt(((pts1[1][0] - pts1[2][0]) ** 2) + ((pts1[1][1] - pts1[2][1]) ** 2))
        height_B = np.sqrt(((pts1[0][0] - pts1[3][0]) ** 2) + ((pts1[0][1] - pts1[3][1]) ** 2))
        maxHeight = max(int(height_A), int(height_B))
        
        # Puntos de destino (Rectángulo plano)
        pts2 = np.float32([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]])
        
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(self.cv_img_original, matrix, (maxWidth, maxHeight))
        
        self.cv_img_original = warped
        self.log_action(f"OPERACIÓN: Corrección de Perspectiva aplicada. Transformados 4 puntos a un rectángulo de {maxWidth}x{maxHeight}.")
        
        self.cancel_perspective() # Limpiar estados
        self.trigger_light_processing()

    # --- TAB 2: PRE-PROCESO ---
    def apply_clahe(self):
        if self.cv_img_original is None: return
        
        img = self.cv_img_original
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        self.cv_img_original = final
        self.log_action("OPERACIÓN: Mejora de Contraste Local (CLAHE) aplicada.")
        self.trigger_light_processing()
        self.redraw_canvas()

    def apply_denoise(self):
        if self.cv_img_original is None: return
        self.btn_denoise.configure(text="Limpiando...", state="disabled")
        self.update()
        
        try:
            # Filtro profesional que preserva bordes
            denoised = cv2.fastNlMeansDenoisingColored(self.cv_img_original, None, 10, 10, 7, 21)
            self.cv_img_original = denoised
            self.log_action("OPERACIÓN: Reducción de Ruido (Non-Local Means) aplicada.")
        except Exception as e:
            print(f"Error Denoise: {e}")
        finally:
            self.btn_denoise.configure(text="Reducción de Ruido (NL Means)", state="normal")
            self.trigger_light_processing()
            self.redraw_canvas()

    # --- Mouse Events para Canvas ---
    def on_left_click(self, event):
        if self.perspective_mode:
            self.add_perspective_point(event)
        else:
            self.start_roi(event)
            
    def on_mouse_wheel(self, event):
        if self.cv_img_original is None: return
        x, y = event.x, event.y
        ix = (x - self.pan_x) / self.zoom_factor
        iy = (y - self.pan_y) / self.zoom_factor
        
        if event.delta > 0: self.zoom_factor *= 1.2
        else: self.zoom_factor /= 1.2
            
        self.zoom_factor = max(0.1, min(self.zoom_factor, 20.0))
        self.pan_x = x - ix * self.zoom_factor
        self.pan_y = y - iy * self.zoom_factor
        self.redraw_canvas()

    def start_pan(self, event):
        if not self.perspective_mode:
            self.canvas.config(cursor="fleur")
        self.pan_start_x = event.x - self.pan_x
        self.pan_start_y = event.y - self.pan_y

    def do_pan(self, event):
        self.pan_x = event.x - self.pan_start_x
        self.pan_y = event.y - self.pan_start_y
        self.redraw_canvas()
        
    def end_pan(self, event):
        if not self.perspective_mode:
            self.canvas.config(cursor="")

    def start_roi(self, event):
        if self.cv_img_original is None: return
        self.roi_start = (event.x, event.y)
        self.roi = None
        self.cv_img_processed = None
        self.canvas.delete("roi_rect")
        self.redraw_canvas() 

    def do_roi(self, event):
        if self.cv_img_original is None or not self.roi_start or self.perspective_mode: return
        self.canvas.delete("temp_roi")
        self.canvas.create_rectangle(
            self.roi_start[0], self.roi_start[1], event.x, event.y, 
            outline="#00ffcc", width=2, dash=(4, 4), tags="temp_roi"
        )

    def end_roi(self, event):
        if self.cv_img_original is None or not self.roi_start or self.perspective_mode: return
        self.canvas.delete("temp_roi")
        
        x1 = min(self.roi_start[0], event.x)
        x2 = max(self.roi_start[0], event.x)
        y1 = min(self.roi_start[1], event.y)
        y2 = max(self.roi_start[1], event.y)
        
        if x2 - x1 < 10 or y2 - y1 < 10:
            self.roi = None
        else:
            ix1 = int((x1 - self.pan_x) / self.zoom_factor)
            ix2 = int((x2 - self.pan_x) / self.zoom_factor)
            iy1 = int((y1 - self.pan_y) / self.zoom_factor)
            iy2 = int((y2 - self.pan_y) / self.zoom_factor)
            
            h, w = self.cv_img_original.shape[:2]
            ix1, ix2 = max(0, min(w, ix1)), max(0, min(w, ix2))
            iy1, iy2 = max(0, min(h, iy1)), max(0, min(h, iy2))
            
            if ix2 > ix1 and iy2 > iy1:
                self.roi = (ix1, iy1, ix2, iy2)
                self.log_action(f"OPERACIÓN: Selección de ROI configurada en coordenadas ({ix1}, {iy1}) a ({ix2}, {iy2}).")
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
        
        resample_filter = Image.Resampling.NEAREST if self.zoom_factor > 2.0 else Image.Resampling.BILINEAR
        pil_img = pil_img.resize((new_w, new_h), resample_filter)
        
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(self.pan_x, self.pan_y, anchor="nw", image=self.tk_img)
        
        # Dibujar ROI
        if self.roi:
            x1, y1, x2, y2 = self.roi
            cx1 = x1 * self.zoom_factor + self.pan_x
            cy1 = y1 * self.zoom_factor + self.pan_y
            cx2 = x2 * self.zoom_factor + self.pan_x
            cy2 = y2 * self.zoom_factor + self.pan_y
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="#00ffcc", width=2, dash=(4, 4), tags="roi_rect")
            
        # Dibujar Puntos de Perspectiva
        if self.perspective_mode and self.perspective_points:
            for pt in self.perspective_points:
                cx = pt[0] * self.zoom_factor + self.pan_x
                cy = pt[1] * self.zoom_factor + self.pan_y
                self.canvas.create_oval(cx-4, cy-4, cx+4, cy+4, fill="red", outline="white")

    # --- Lógica de IA ---
    def download_model(self, scale):
        model_name = f"LapSRN_x{scale}.pb"
        url = f"https://raw.githubusercontent.com/fannymonori/TF-LapSRN/master/export/{model_name}"
        if not os.path.exists(model_name):
            self.label_ai_status.configure(text=f"Descargando {model_name}...", text_color="orange")
            self.update()
            try:
                urllib.request.urlretrieve(url, model_name)
                self.label_ai_status.configure(text="Descarga exitosa.", text_color="green")
            except Exception as e:
                self.label_ai_status.configure(text=f"Error descarga.", text_color="red")
                print(f"Download Error: {e}")
                return False
        return True

    def apply_ai(self, scale):
        if self.cv_img_original is None: return
        
        success = self.download_model(scale)
        if not success: return
        
        source = self.cv_img_processed if self.cv_img_processed is not None else self.cv_img_original
        
        if self.roi:
            x1, y1, x2, y2 = self.roi
            img_crop = source[y1:y2, x1:x2]
        else:
            img_crop = source
            
        if img_crop.size == 0: return

        self.label_ai_status.configure(text="Procesando IA...", text_color="orange")
        self.btn_ai_x4.configure(state="disabled")
        self.btn_ai_x8.configure(state="disabled")
        self.update()
        
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(f"LapSRN_x{scale}.pb")
            sr.setModel("lapsrn", scale)
            result = sr.upsample(img_crop)
            
            self.cv_img_original = result
            self.cv_img_processed = None
            self.roi = None
            
            self.log_action(f"OPERACIÓN: IA Super Resolución LapSRN x{scale} aplicada.")
            
            # Reset vista
            self.zoom_factor = 1.0
            self.canvas.update()
            cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
            h, w = result.shape[:2]
            scale_w = cw / w
            scale_h = ch / h
            self.zoom_factor = min(scale_w, scale_h) * 0.9 if min(scale_w, scale_h) < 1 else 1.0
            self.pan_x = (cw - (w * self.zoom_factor)) / 2
            self.pan_y = (ch - (h * self.zoom_factor)) / 2
            
            self.label_ai_status.configure(text="¡Imagen Mejorada!", text_color="green")
            self.redraw_canvas()
        except Exception as e:
            self.label_ai_status.configure(text="Error de IA", text_color="red")
            print(f"AI Error: {e}")
        finally:
            self.btn_ai_x4.configure(state="normal")
            self.btn_ai_x8.configure(state="normal")


    # --- Procesamiento Clásico ---
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
        
        # Avoid spamming log for real-time sliders. Log only when heavy algorithm applied.

    def apply_heavy_algorithm(self):
        if self.cv_img_original is None: return
        
        algo = self.algorithm_var.get()
        psf_type = self.psf_type_var.get()
        psf_size = 2 * int(self.slider_psf_size.get()) + 1
        angle = int(self.slider_angle.get()) if hasattr(self, 'slider_angle') and self.slider_angle.winfo_exists() else 0
        iters = int(self.slider_iter.get()) if hasattr(self, 'slider_iter') and self.slider_iter.winfo_exists() else 0
        
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
                    ch_deblurred = restoration.richardson_lucy(ch, psf, num_iter=iters, clip=False)
                processed_channels.append(ch_deblurred)
                
            result = cv2.merge(processed_channels)
            result_uint8 = np.clip(result * 255, 0, 255).astype(np.uint8)
            self.set_processed_result(result_uint8)
            
            log_str = f"OPERACIÓN: Filtro {algo} aplicado. PSF={psf_type}, Tamaño={psf_size}, Ángulo={angle}"
            if algo == "Richardson-Lucy": log_str += f", Iters={iters}"
            self.log_action(log_str)
            
        except Exception as e:
            print(f"Error en el procesamiento: {e}")
        finally:
            self.btn_apply.configure(text="Aplicar Algoritmo", state="normal")

if __name__ == "__main__":
    app = ForensicDeblurApp()
    app.mainloop()