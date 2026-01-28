import tkinter as tk
from app.services.nety_adapter import get_modules_status, NetyAdapter

class DashboardView:
    def __init__(self, root, nety_adapter=None):
        self.root = root
        self.nety_adapter = nety_adapter or NetyAdapter()
        self.ai_status_label = None
        self.toggle_button = None

        title = tk.Label(
            root,
            text="NETY – Dashboard",
            font=("Arial", 18, "bold")
        )
        title.pack(pady=10)

        # Section de contrôle de l'IA
        control_frame = tk.Frame(root, bg="#f0f0f0", relief="raised", borderwidth=2)
        control_frame.pack(fill="x", padx=20, pady=10)

        control_label = tk.Label(
            control_frame,
            text="🤖 Contrôle de l'IA NETY",
            font=("Arial", 14, "bold"),
            bg="#f0f0f0"
        )
        control_label.pack(pady=5)

        # Frame pour le statut et le bouton
        status_frame = tk.Frame(control_frame, bg="#f0f0f0")
        status_frame.pack(pady=5)

        # Indicateur de statut
        self.ai_status_label = tk.Label(
            status_frame,
            text="⚫ IA ARRÊTÉE",
            font=("Arial", 12, "bold"),
            fg="red",
            bg="#f0f0f0"
        )
        self.ai_status_label.pack(side="left", padx=10)

        # Bouton toggle
        self.toggle_button = tk.Button(
            status_frame,
            text="▶ Démarrer l'IA",
            command=self.toggle_nety_ai,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 11, "bold"),
            padx=20,
            pady=5
        )
        self.toggle_button.pack(side="left", padx=10)

        # Section des modules
        self.container = tk.Frame(root)
        self.container.pack(fill="both", expand=True, padx=20, pady=10)

        modules_title = tk.Label(
            self.container,
            text="📊 État des Modules",
            font=("Arial", 14, "bold")
        )
        modules_title.pack(anchor="w", pady=(0, 10))

        self.modules_frame = tk.Frame(self.container)
        self.modules_frame.pack(fill="both", expand=True)

        self.refresh()
        self.update_ai_status()

    def toggle_nety_ai(self):
        """Bascule l'état de l'IA NETY (marche/arrêt)"""
        if self.nety_adapter.is_running():
            self.nety_adapter.stop_nety()
        else:
            self.nety_adapter.start_nety()
        
        self.update_ai_status()

    def update_ai_status(self):
        """Met à jour l'affichage du statut de l'IA"""
        if self.nety_adapter.is_running():
            self.ai_status_label.config(text="🟢 IA EN MARCHE", fg="green")
            self.toggle_button.config(
                text="⏸ Arrêter l'IA",
                bg="#f44336"  # Rouge
            )
        else:
            self.ai_status_label.config(text="⚫ IA ARRÊTÉE", fg="red")
            self.toggle_button.config(
                text="▶ Démarrer l'IA",
                bg="#4CAF50"  # Vert
            )

    def refresh(self):
        """Rafraîchit la liste des modules"""
        for widget in self.modules_frame.winfo_children():
            widget.destroy()

        modules = get_modules_status()

        for mod in modules:
            # Créer un frame pour chaque module
            mod_frame = tk.Frame(self.modules_frame, relief="groove", borderwidth=1)
            mod_frame.pack(fill="x", pady=2)

            # Icône de statut
            status_icon = "🟢" if mod['status'] == "running" else "🟡" if mod['status'] == "idle" else "⚫"
            
            label = tk.Label(
                mod_frame,
                text=f"{status_icon} [{mod['type']}] {mod['name']} → {mod['status']}",
                anchor="w",
                font=("Arial", 10)
            )
            label.pack(fill="x", padx=10, pady=5)

