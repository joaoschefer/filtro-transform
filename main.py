import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import os

# muda o tamanho da imagem
def cv_to_tk(cv_img, max_dim=600):
    h, w = cv_img.shape[:2]
    scale = min(max_dim / w, max_dim / h)
    resized = cv2.resize(cv_img, (int(w * scale), int(h * scale)))
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(resized)
    return ImageTk.PhotoImage(img)

# filtros
def aplicar_filtro(imagem, tipo):
    if tipo == "mediana":
        return cv2.medianBlur(imagem, 5)
    elif tipo == "laplaciano":
        lap = cv2.Laplacian(imagem, cv2.CV_64F)
        lap = cv2.convertScaleAbs(lap)
        return lap
    return imagem

# transformações
def ajustar_contraste(imagem, valor):
    alpha = 1 + (valor / 100.0)
    return cv2.convertScaleAbs(imagem, alpha=alpha, beta=0)

def tons_de_cinza(imagem):
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Editor de Imagem: Mediana, Laplaciano, Contraste, Tons de Cinza")
        self.original_img = None
        self.processed_img = None

        self.image_paths = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"]
        self.create_widgets()

    def create_widgets(self):
        frame_top = tk.Frame(self.root)
        frame_top.pack(pady=5)

        for path in self.image_paths:
            btn = tk.Button(frame_top, text=os.path.basename(path),
                            command=lambda p=path: self.load_image(p))
            btn.pack(side=tk.LEFT, padx=5)

        image_frame = tk.Frame(self.root)
        image_frame.pack(pady=10)

        self.canvas_original = tk.Label(image_frame)
        self.canvas_original.pack(side=tk.LEFT, padx=10, pady=10)

        self.canvas_modified = tk.Label(image_frame)
        self.canvas_modified.pack(side=tk.LEFT, padx=10, pady=10)

        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        btn_mediana = tk.Button(control_frame, text="Filtro Mediana", command=lambda: self.aplicar_filtro("mediana"))
        btn_mediana.pack(side=tk.LEFT, padx=5)

        btn_laplaciano = tk.Button(control_frame, text="Filtro Laplaciano", command=lambda: self.aplicar_filtro("laplaciano"))
        btn_laplaciano.pack(side=tk.LEFT, padx=5)

        btn_cinza = tk.Button(control_frame, text="Tons de Cinza", command=self.cinza)
        btn_cinza.pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="Contraste").pack(side=tk.LEFT, padx=5)
        self.slider_contraste = ttk.Scale(control_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                          command=lambda val: self.ajustar_contraste(int(float(val))))
        self.slider_contraste.pack(side=tk.LEFT, padx=5)
        btn_salvar = tk.Button(self.root, text="Salvar Imagem", command=self.salvar_imagem)
        btn_salvar.pack(pady=10)

    def load_image(self, path):
        self.original_img = cv2.imread(path)
        self.processed_img = self.original_img.copy()
        self.update_display()

    def update_display(self):
        if self.original_img is not None:
            orig_img_tk = cv_to_tk(self.original_img)
            mod_img_tk = cv_to_tk(self.processed_img)
            self.canvas_original.config(image=orig_img_tk)
            self.canvas_original.image = orig_img_tk
            self.canvas_modified.config(image=mod_img_tk)
            self.canvas_modified.image = mod_img_tk

    def aplicar_filtro(self, tipo):
        if self.original_img is not None:
            self.processed_img = aplicar_filtro(self.original_img.copy(), tipo)
            self.update_display()

    def ajustar_contraste(self, valor):
        if self.original_img is not None:
            self.processed_img = ajustar_contraste(self.original_img.copy(), valor)
            self.update_display()

    def cinza(self):
        if self.original_img is not None:
            self.processed_img = tons_de_cinza(self.original_img.copy())
            self.update_display()

    def salvar_imagem(self):
        if self.processed_img is not None:
            filename = filedialog.asksaveasfilename(defaultextension=".png",
                                                    filetypes=[("PNG files", "*.png"),
                                                               ("JPG files", "*.jpg"),
                                                               ("BMP files", "*.bmp")])
            if filename:
                cv2.imwrite(filename, self.processed_img)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
