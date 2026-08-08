import tkinter as tk
from home_page import HomePage
from category_page import CategoryPage
from App.prediction_pages.bike_rental import PredictionPage
from App.prediction_pages.auto_mpg import PredictionPageAutoMPG
from prediction_pages.concrete import PredictionPageConcrete
from prediction_pages.forecast_pollution import PredictionPageForecastPollution
from prediction_pages.daily_orders import PredictionPageDailyOrders
from prediction_pages.dow_jones import PredictionPageDowJones


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ML Model Explorer")
        self.geometry("700x650")
        window_width = 700
        window_height = 650

        x = int(self.winfo_screenwidth() / 2 - window_width / 2)
        y = int(self.winfo_screenheight() / 2 - window_height / 2)

        self.geometry(f"{window_width}x{window_height}+{x}+{y}")


        container = tk.Frame(self)
        container.pack(fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # حالا کلید دیکشنری یه رشته‌ی ساده‌ست، نه خود کلاس
        self.frames = {}
        self.frames["home"] = HomePage(container, self)
        self.frames["category"] = CategoryPage(container, self)
        self.frames["prediction"] = PredictionPage(container, self)
        self.frames["prediction_auto_mpg"] = PredictionPageAutoMPG(container, self)
        self.frames["prediction_concrete"] = PredictionPageConcrete(container, self)
        self.frames["prediction_forecast_pollution"] = PredictionPageForecastPollution(container, self)
        self.frames["prediction_daily_orders"] = PredictionPageDailyOrders(container, self)
        self.frames["prediction_dow_jones"] = PredictionPageDowJones(container, self)


        for frame in self.frames.values():
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("home")

    def show_frame(self, page_name, **kwargs):
        frame = self.frames[page_name]
        if hasattr(frame, "on_show"):
            frame.on_show(**kwargs)
        frame.tkraise()


if __name__ == "__main__":
    app = App()
    app.mainloop()