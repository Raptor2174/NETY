import tkinter as tk
from .taskbar import TaskbarView
from .dashboard import DashboardView
from .admin_room import AdminRoomView
from .error_404 import NotFoundView
from app.services.nety_adapter import NetyAdapter

def launch_app():
    root = tk.Tk()
    root.title("NETY Dashboard")
    root.geometry("900x600")
    root.resizable(True, True)

    # Créer une instance unique de NetyAdapter
    nety_adapter = NetyAdapter()

    # Layout en 3 lignes (haut, milieu, bas)
    root.rowconfigure(1, weight=1)  # la zone du milieu prend l'espace disponible
    root.columnconfigure(0, weight=1)

    top_frame = tk.Frame(root)
    middle_frame = tk.Frame(root)
    bottom_frame = tk.Frame(root)

    top_frame.grid(row=0, column=0, sticky="ew")
    middle_frame.grid(row=1, column=0, sticky="nsew")
    bottom_frame.grid(row=2, column=0, sticky="ew")

    # Passer middle_frame à middle_view pour pouvoir le modifier
    middle_view_container = middle_view(middle_frame, nety_adapter)
    
    # Callback pour changer de page
    def on_page_change(page_id):
        # Effacer le contenu actuel
        for widget in middle_frame.winfo_children():
            widget.destroy()
        
        # Charger la nouvelle page avec l'adaptateur
        if page_id == "dashboard":
            DashboardView(middle_frame, nety_adapter)
        elif page_id == "admin_room":
            AdminRoomView(middle_frame, nety_adapter)
        elif page_id == "404":
            NotFoundView(middle_frame)

    taskbar = TaskbarView(bottom_frame, on_page_change=on_page_change)

    root.mainloop()

def middle_view(root, nety_adapter):
    dashboard = DashboardView(root, nety_adapter)
    return dashboard

def bottom_view(root):
    text = tk.Label(
        root,
        text="NETY Dashboard - Debug View",
        font=("Arial", 10)
    )
    text.pack(pady=4)
    return text