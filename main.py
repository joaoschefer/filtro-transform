import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import os

def cv_to_tk(cv_img, max_dim=600):
    h, w = cv_img.shape[:2]
    scale = min(max_dim / w, max_dim / h)
    resized = cv2.resize(cv_img, (int(w * scale), int(h * scale)))
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(resized)
    return ImageTk.PhotoImage(img)

def aplicar_filtro(imagem, tipo):
    if tipo == "mediana":
        return cv2.medianBlur(imagem, 5)
    elif tipo == "laplaciano":
        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap = cv2.convertScaleAbs(lap)
        return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)
    elif tipo == "gaussiano":
        return cv2.GaussianBlur(imagem, (5, 5), 0)
    elif tipo == "sobel":
        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel = cv2.magnitude(sobelx, sobely)
        sobel = np.uint8(np.clip(sobel, 0, 255))
        return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
    return imagem

def ajustar_contraste(imagem, valor):
    alpha = 1 + (valor / 100.0)
    return cv2.convertScaleAbs(imagem, alpha=alpha, beta=0)

def ajustar_brilho(imagem, valor):
    return cv2.convertScaleAbs(imagem, alpha=1, beta=valor - 50)

def binarizar(imagem, limiar):
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    _, binarizada = cv2.threshold(gray, limiar, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binarizada, cv2.COLOR_GRAY2BGR)

def tons_de_cinza(imagem):
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def inverter_cores(imagem):
    return cv2.bitwise_not(imagem)

def isolar_cor_vermelha(imagem):
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (160, 100, 100), (179, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    color = cv2.bitwise_and(imagem, imagem, mask=mask)
    gray = tons_de_cinza(imagem)
    return np.where(mask[:, :, np.newaxis] == 0, gray, color)

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Editor de Imagem com Filtros e Transformações")
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

        filtro_frame = tk.LabelFrame(self.root, text="Filtros")
        filtro_frame.pack(pady=5)
        for nome in ["Mediana", "Laplaciano", "Gaussiano", "Sobel"]:
            btn = tk.Button(filtro_frame, text=nome, command=lambda n=nome.lower(): self.aplicar_filtro(n))
            btn.pack(side=tk.LEFT, padx=5)

        transform_frame = tk.LabelFrame(self.root, text="Transformações")
        transform_frame.pack(pady=5)

        tk.Label(transform_frame, text="Contraste").pack(side=tk.LEFT)
        self.slider_contraste = ttk.Scale(transform_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                          command=lambda val: self.aplicar_contraste(int(float(val))))
        self.slider_contraste.set(0)
        self.slider_contraste.pack(side=tk.LEFT, padx=5)

        tk.Label(transform_frame, text="Brilho").pack(side=tk.LEFT)
        self.slider_brilho = ttk.Scale(transform_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                       command=lambda val: self.aplicar_brilho(int(float(val))))
        self.slider_brilho.set(50)
        self.slider_brilho.pack(side=tk.LEFT, padx=5)

        tk.Label(transform_frame, text="Binarização").pack(side=tk.LEFT)
        self.slider_bin = ttk.Scale(transform_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                    command=lambda val: self.aplicar_binarizacao(int(float(val))))
        self.slider_bin.set(128)
        self.slider_bin.pack(side=tk.LEFT, padx=5)

        outros_frame = tk.Frame(self.root)
        outros_frame.pack(pady=5)
        tk.Button(outros_frame, text="Tons de Cinza", command=self.cinza).pack(side=tk.LEFT, padx=5)
        tk.Button(outros_frame, text="Inverter Cores", command=self.inverter).pack(side=tk.LEFT, padx=5)
        tk.Button(outros_frame, text="Isolar Vermelho", command=self.isolar_vermelho).pack(side=tk.LEFT, padx=5)

        tk.Button(self.root, text="Salvar Imagem", command=self.salvar_imagem).pack(pady=10)

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

    def aplicar_contraste(self, valor):
        if self.original_img is not None:
            self.processed_img = ajustar_contraste(self.original_img.copy(), valor)
            self.update_display()

    def aplicar_brilho(self, valor):
        if self.original_img is not None:
            self.processed_img = ajustar_brilho(self.original_img.copy(), valor)
            self.update_display()

    def aplicar_binarizacao(self, limiar):
        if self.original_img is not None:
            self.processed_img = binarizar(self.original_img.copy(), limiar)
            self.update_display()

    def cinza(self):
        if self.original_img is not None:
            self.processed_img = tons_de_cinza(self.original_img.copy())
            self.update_display()

    def inverter(self):
        if self.original_img is not None:
            self.processed_img = inverter_cores(self.original_img.copy())
            self.update_display()

    def isolar_vermelho(self):
        if self.original_img is not None:
            self.processed_img = isolar_cor_vermelha(self.original_img.copy())
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
