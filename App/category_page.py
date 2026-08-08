import tkinter as tk

class CategoryPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.title_label = tk.Label(self, font=("Arial", 18, "bold"))
        self.title_label.pack(pady=20)

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("home")).pack(anchor="w", padx=15)

        self.list_frame = tk.Frame(self)
        self.list_frame.pack(pady=20)

    def on_show(self, category):
        self.category = category
        self.title_label.config(text=category.capitalize())

        for widget in self.list_frame.winfo_children():
            widget.destroy()

        if category == "regression":
            tk.Button(self.list_frame, text="Bike Rental Demand", width=30,
                      command=lambda: self.controller.show_frame("prediction", dataset="bike_rental")).pack(pady=5)
            tk.Button(self.list_frame, text="Auto MPG", width=30,
                      command=lambda: self.controller.show_frame("prediction_auto_mpg")).pack(pady=5)
            tk.Button(self.list_frame, text="Concrete Compressive Strength", width=30,
                      command=lambda: self.controller.show_frame("prediction_concrete")).pack(pady=5)
            tk.Button(self.list_frame, text="Beijing PM2.5 Air Pollution", width=30,
                      command=lambda: self.controller.show_frame("prediction_forecast_pollution")).pack(pady=5)
            tk.Button(self.list_frame, text="Daily Demand Forecasting", width=30,
                      command=lambda: self.controller.show_frame("prediction_daily_orders")).pack(pady=5)
            tk.Button(self.list_frame, text="Dow Jones Index", width=30,
                      command=lambda: self.controller.show_frame("prediction_dow_jones")).pack(pady=5)
        else:
            tk.Label(self.list_frame, text="No datasets added yet.").pack()