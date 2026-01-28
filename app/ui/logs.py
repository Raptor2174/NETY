import tkinter as tk
from tkinter import scrolledtext


class LogsView:
    """Vue pour afficher les logs/terminal de l'IA NETY"""
    
    def __init__(self, root, nety_adapter=None):
        self.root = root
        self.nety_adapter = nety_adapter
        self.logs_display = None
        self.auto_scroll = True
        
        # Crée l'interface directement
        self.create_ui(root)
        
        # Charger les logs initiaux
        self.refresh_logs()

    def create_ui(self, parent_frame):
        """Crée l'interface utilisateur de la page Logs"""
        
        # Frame principal
        main_frame = tk.Frame(parent_frame, bg="#1e1e1e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # En-tête
        header_frame = tk.Frame(main_frame, bg="#1e1e1e")
        header_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = tk.Label(
            header_frame,
            text="📊 Logs & Terminal - Actions de l'IA NETY",
            font=("Courier New", 16, "bold"),
            fg="#00FF00",
            bg="#1e1e1e"
        )
        title_label.pack(side="left")

        # Boutons de contrôle
        control_frame = tk.Frame(header_frame, bg="#1e1e1e")
        control_frame.pack(side="right")

        # Bouton rafraîchir
        refresh_btn = tk.Button(
            control_frame,
            text="🔄 Rafraîchir",
            command=self.refresh_logs,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5
        )
        refresh_btn.pack(side="left", padx=5)

        # Bouton effacer
        clear_btn = tk.Button(
            control_frame,
            text="🗑️ Effacer",
            command=self.clear_logs,
            bg="#f44336",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5
        )
        clear_btn.pack(side="left", padx=5)

        # Variable pour auto-scroll checkbox
        self.auto_scroll_var = tk.BooleanVar(value=True)
        auto_scroll_cb = tk.Checkbutton(
            control_frame,
            text="Auto-scroll",
            variable=self.auto_scroll_var,
            command=self._toggle_auto_scroll,
            bg="#1e1e1e",
            fg="white",
            selectcolor="#333333",
            font=("Arial", 10)
        )
        auto_scroll_cb.pack(side="left", padx=5)

        # Indicateur de statut
        self.status_label = tk.Label(
            main_frame,
            text=self._get_status_text(),
            font=("Courier New", 11),
            fg="#00FF00",
            bg="#1e1e1e"
        )
        self.status_label.pack(pady=(0, 5))

        # Zone d'affichage des logs (style terminal)
        self.logs_display = scrolledtext.ScrolledText(
            main_frame,
            height=25,
            width=100,
            state=tk.DISABLED,
            bg="#000000",
            fg="#00FF00",
            font=("Courier New", 10),
            insertbackground="white",
            wrap=tk.WORD
        )
        self.logs_display.pack(fill=tk.BOTH, expand=True)

        # Footer
        footer_label = tk.Label(
            main_frame,
            text="Logs en temps réel - Les actions de l'IA apparaissent ici",
            font=("Arial", 9, "italic"),
            fg="#888888",
            bg="#1e1e1e"
        )
        footer_label.pack(pady=(5, 0))

    def _get_status_text(self):
        """Retourne le texte d'état de l'IA"""
        if self.nety_adapter and self.nety_adapter.is_running():
            return "● Statut: IA EN MARCHE - Surveillance active"
        else:
            return "○ Statut: IA ARRÊTÉE - Aucune activité"

    def _toggle_auto_scroll(self):
        """Bascule l'auto-scroll"""
        self.auto_scroll = self.auto_scroll_var.get()

    def refresh_logs(self):
        """Rafraîchit l'affichage des logs"""
        if not self.logs_display or not self.nety_adapter:
            return

        # Mettre à jour le statut
        if hasattr(self, 'status_label'):
            self.status_label.config(text=self._get_status_text())

        # Récupérer les logs
        logs = self.nety_adapter.get_logs()

        # Effacer et remplir
        self.logs_display.config(state=tk.NORMAL)
        self.logs_display.delete("1.0", tk.END)

        if logs:
            for log in logs:
                self.logs_display.insert(tk.END, log + "\n")
        else:
            self.logs_display.insert(tk.END, "[Aucun log disponible]\n")

        # Auto-scroll vers le bas si activé
        if self.auto_scroll:
            self.logs_display.see(tk.END)

        self.logs_display.config(state=tk.DISABLED)

    def clear_logs(self):
        """Efface tous les logs"""
        if self.nety_adapter:
            self.nety_adapter.clear_logs()
            self.refresh_logs()
