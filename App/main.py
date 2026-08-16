import tkinter as tk
from home_page import HomePage
from category_page import CategoryPage
from App.prediction_pages.bike_rental import PredictionPage
from App.prediction_pages.auto_mpg import PredictionPageAutoMPG
from prediction_pages.concrete import PredictionPageConcrete
from prediction_pages.forecast_pollution import PredictionPageForecastPollution
from prediction_pages.daily_orders import PredictionPageDailyOrders
from prediction_pages.dow_jones import PredictionPageDowJones
from prediction_pages.abalone import PredictionPageAbalone
from prediction_pages.real_estate import PredictionPageRealEstate
from prediction_pages.online_news import PredictionPageOnlineNews
from prediction_pages.banknote import PredictionPageBanknote
from prediction_pages.blood_transfusion import PredictionPageBloodTransfusion
from prediction_pages.car_evaluation import PredictionPageCarEvaluation
from prediction_pages.bankruptcy import PredictionPageQualitativeBankruptcy
from prediction_pages.heart_disease import PredictionPageHeartDisease
from prediction_pages.heart_attack import PredictionPageEchocardiogram
from prediction_pages.hepatitis import PredictionPageHepatitis
from prediction_pages.autism import PredictionPageAutism
from prediction_pages.persons_income import PredictionPagePersonsIncome
from prediction_pages.default_payment import PredictionPageCreditDefault
from prediction_pages.glass import PredictionPageGlass
from prediction_pages.room_occupancy import PredictionPageRoomOccupancy
from prediction_pages.covid_risk import PredictionPageCovidRisk
from prediction_pages.online_news_classification import PredictionPageOnlineNewsClassification
from prediction_pages.liver_disorder import PredictionPageLiverDisorder
from prediction_pages.student_knowledge import PredictionPageStudentKnowledge
from prediction_pages.wholesale_customers import PredictionPageWholesaleCustomers
from prediction_pages.power_consumption import PredictionPagePowerConsumption
from prediction_pages.travel_review import PredictionPageTravelReview


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
        self.frames["prediction_abalone"] = PredictionPageAbalone(container, self)
        self.frames["prediction_real_estate"] = PredictionPageRealEstate(container, self)
        self.frames["prediction_online_news"] = PredictionPageOnlineNews(container, self)
        self.frames["prediction_banknote"] = PredictionPageBanknote(container, self)
        self.frames["prediction_blood_transfusion"] = PredictionPageBloodTransfusion(container, self)
        self.frames["prediction_car_evaluation"] = PredictionPageCarEvaluation(container, self)
        self.frames["prediction_qualitative_bankruptcy"] = PredictionPageQualitativeBankruptcy(container, self)
        self.frames["prediction_heart_disease"] = PredictionPageHeartDisease(container, self)
        self.frames["prediction_echocardiogram"] = PredictionPageEchocardiogram(container, self)
        self.frames["prediction_hepatitis"] = PredictionPageHepatitis(container, self)
        self.frames["prediction_autism"] = PredictionPageAutism(container, self)
        self.frames["prediction_persons_income"] = PredictionPagePersonsIncome(container, self)
        self.frames["prediction_credit_default"] = PredictionPageCreditDefault(container, self)
        self.frames["prediction_glass"] = PredictionPageGlass(container, self)
        self.frames["prediction_room_occupancy"] = PredictionPageRoomOccupancy(container, self)
        self.frames["prediction_covid_risk"] = PredictionPageCovidRisk(container, self)
        self.frames["prediction_online_news_classification"] = PredictionPageOnlineNewsClassification(container, self)
        self.frames["prediction_liver_disorder"] = PredictionPageLiverDisorder(container, self)
        self.frames["prediction_student_knowledge"] = PredictionPageStudentKnowledge(container, self)
        self.frames["prediction_wholesale_customers"] = PredictionPageWholesaleCustomers(container, self)
        self.frames["prediction_power_consumption"] = PredictionPagePowerConsumption(container, self)
        self.frames["prediction_travel_review"] = PredictionPageTravelReview(container, self)







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