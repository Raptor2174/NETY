import tkinter as tk
from ui.taskbar import TaskbarView
from ui.dashboard import DashboardView

def launch_app():
    root = tk.Tk()
    root.title("NETY Dashboard")
    root.geometry("900x600")
    root.resizable(True, True)

    # Layout en 3 lignes (haut, milieu, bas)
    root.rowconfigure(1, weight=1)  # la zone du milieu prend l'espace disponible
    root.columnconfigure(0, weight=1)

    top_frame = tk.Frame(root)
    middle_frame = tk.Frame(root)
    bottom_frame = tk.Frame(root)

    top_frame.grid(row=0, column=0, sticky="ew")
    middle_frame.grid(row=1, column=0, sticky="nsew")
    bottom_frame.grid(row=2, column=0, sticky="ew")

    top_view(top_frame)
    middle_view(middle_frame)
    bottom_view(bottom_frame)

    root.mainloop()

def top_view(root):
    dashboard = TaskbarView(root)
    return dashboard

def middle_view(root):
    dashboard = DashboardView(root)
    return dashboard

def bottom_view(root):
    text = tk.Label(
        root,
        text="NETY Dashboard - Debug View",
        font=("Arial", 10)
    )
    text.pack(pady=4)
    return text