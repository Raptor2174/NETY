


import tkinter as tk


class NotFoundView:
    def __init__(self, root):
        self.frame = root
        self.frame.configure(bg="white")

        self.label = tk.Label(
            self.frame,
            text="404 - Page Not Found",
            font=("Arial", 24),
            fg="red",
            bg="white"
        )
        self.label.pack(expand=True)