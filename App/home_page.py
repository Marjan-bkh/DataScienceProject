import tkinter as tk
from PIL import Image, ImageTk
import os

class HomePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.images = {}  # اینجا عکس‌ها رو نگه می‌داریم


        logo_path = os.path.join("assets", "sematec_logo.png")
        logo_img = Image.open(logo_path)
        logo_img = logo_img.resize((200, 100))
        logo_photo = ImageTk.PhotoImage(logo_img)
        self.images["logo"] = logo_photo  # جلوگیری از پاک شدن عکس از حافظه

        tk.Label(self, image=logo_photo).pack(pady=(20, 5))

        tk.Label(self, text="ML Model Explorer", font=("Arial", 20, "bold")).pack(pady=(5, 40))

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=20)

        self._make_button(btn_frame, "classification", "Classification", 0)
        self._make_button(btn_frame, "regression", "Regression", 1)
        self._make_button(btn_frame, "clustering", "Clustering", 2)

        # ---------- بخش اطلاعات دانشجو ----------
        info_frame = tk.Frame(self)
        info_frame.pack(side="bottom", pady=30)

        tk.Label(info_frame, text="Student Name: ", font=("Arial", 11, "bold")).grid(row=0, column=0, sticky="w", pady=2)
        tk.Label(info_frame, text="Marjan Bakhtiari", font=("Arial", 11)).grid(row=0, column=1, sticky="w", pady=2)

        tk.Label(info_frame, text="Instructor: ", font=("Arial", 11, "bold")).grid(row=1, column=0, sticky="w", pady=2)
        tk.Label(info_frame, text="Vahid Ghorbani", font=("Arial", 11)).grid(row=1, column=1, sticky="w", pady=2)

        tk.Label(info_frame, text="Course: ", font=("Arial", 11, "bold")).grid(row=2, column=0, sticky="w", pady=2)
        tk.Label(info_frame, text="Data Science", font=("Arial", 11)).grid(row=2, column=1, sticky="w", pady=2)

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