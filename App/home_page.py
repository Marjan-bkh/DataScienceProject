import tkinter as tk
from PIL import Image, ImageTk
import os

class HomePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.images = {}  # اینجا عکس‌ها رو نگه می‌داریم

        tk.Label(self, text="ML Model Explorer", font=("Arial", 20, "bold")).pack(pady=40)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=20)

        self._make_button(btn_frame, "classification", "Classification", 0)
        self._make_button(btn_frame, "regression", "Regression", 1)
        self._make_button(btn_frame, "clustering", "Clustering", 2)

    def _make_button(self, parent, category_key, label_text, column):
        image_path = os.path.join("assets", f"{category_key}.png")
        img = Image.open(image_path)
        img = img.resize((120, 120))
        photo = ImageTk.PhotoImage(img)

        self.images[category_key] = photo  # جلوگیری از پاک شدن عکس از حافظه

        btn = tk.Button(
            parent,
            image=photo,
            text=label_text,
            compound="top",
            width=150,
            height=160,
            font=("Arial", 11),
            command=lambda: self.controller.show_frame("category", category=category_key)
        )
        btn.grid(row=0, column=column, padx=20)