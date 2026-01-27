import tkinter as tk

class TaskbarView:
    def __init__(self, root, on_page_change=None):
        self.root = root
        self.on_page_change = on_page_change

        taskbar = tk.Frame(root, bg="#333333", height=40)
        taskbar.pack(side="bottom", fill="x")

        label = tk.Label(
            taskbar,
            text="NETY Taskbar - Debug View",
            fg="#FFFFFF",
            bg="#333333",
            font=("Arial", 12)
        )
        label.pack(side="left", padx=10, pady=10)

        # Boutons de navigation
        button_frame = tk.Frame(taskbar, bg="#333333")
        button_frame.pack(side="left", padx=10)

        pages = [
            ("Dashboard", "dashboard"),
            ("Admin Room", "admin_room"),
            ("404", "404")
        ]

        for page_name, page_id in pages:
            btn = tk.Button(
                button_frame,
                text=page_name,
                bg="#555555",
                fg="#FFFFFF",
                relief="raised",
                command=lambda pid=page_id: self._on_button_click(pid)
            )
            btn.pack(side="left", padx=5)

    def _on_button_click(self, page_id):
        if self.on_page_change:
            self.on_page_change(page_id)

