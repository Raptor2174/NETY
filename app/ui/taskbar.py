import tkinter as tk

class TaskbarView:
    def __init__(self, root):
        self.root = root

        taskbar = tk.Frame(root, bg="#333333", height=40)
        taskbar.pack(side="bottom", fill="x")

        label = tk.Label(
            taskbar,
            text="NETY Taskbar - Debug View",
            fg="#FFFFFF",
            bg="#333333",
            font=("Arial", 12)
        )
        label.pack(pady=10)

