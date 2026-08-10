import tkinter as tk

class ScrollableFrame(tk.Frame):
    """
    یک فریم اسکرول‌شونده. هر چیزی که داخل self.scrollable_frame بذاری،
    اگه از ارتفاع صفحه بیشتر بشه، خودکار اسکرول‌بار میاد.
    """
    def __init__(self, parent):
        super().__init__(parent)

        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas_window = canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # وقتی عرض Canvas عوض شد، عرض فریم داخلی رو هم همونقدر کن (برای وسط‌چین شدن درست محتوا)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))