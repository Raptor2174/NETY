import tkinter as tk
from services.nety_adapter import get_modules_status

class DashboardView:
    def __init__(self, root):
        self.root = root

        title = tk.Label(
            root,
            text="NETY – Module Dashboard - Debug View",
            font=("Arial", 18, "bold")
        )
        title.pack(pady=10)

        self.container = tk.Frame(root)
        self.container.pack(fill="both", expand=True)

        self.refresh()

    def refresh(self):
        for widget in self.container.winfo_children():
            widget.destroy()

        modules = get_modules_status()

        for mod in modules:
            label = tk.Label(
                self.container,
                text=f"[{mod['type']}] {mod['name']} → {mod['status']}",
                anchor="w"
            )
            label.pack(fill="x", padx=20, pady=2)

