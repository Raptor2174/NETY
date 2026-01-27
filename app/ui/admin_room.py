import tkinter as tk
from tkinter import scrolledtext, messagebox
from datetime import datetime

class AdminRoomView:
    def __init__(self, root, nety_adapter=None):
        self.root = root
        self.nety_adapter = nety_adapter
        self.chat_display = None
        self.message_input = None
        self.prompt_input = None
        
        # Crée l'interface directement
        self.create_ui(root)

    def create_ui(self, parent_frame):
        """Crée l'interface utilisateur du panneau AdminRoom"""
        
        # Frame principal
        main_frame = tk.Frame(parent_frame, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ===== CHATBOX =====
        chat_label = tk.Label(main_frame, text="💬 Chatbox", font=("Arial", 12, "bold"), bg="#f0f0f0")
        chat_label.pack(anchor="w", pady=(0, 5))

        # Affichage des messages
        self.chat_display = scrolledtext.ScrolledText(
            main_frame, height=10, width=60, state=tk.DISABLED, bg="white", font=("Arial", 10)
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Frame pour l'entrée de message
        input_frame = tk.Frame(main_frame, bg="#f0f0f0")
        input_frame.pack(fill=tk.X, pady=(0, 15))

        self.message_input = tk.Entry(input_frame, font=("Arial", 10), bg="white")
        self.message_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.message_input.bind("<Return>", lambda e: self.send_message())

        send_btn = tk.Button(
            input_frame, text="Envoyer", command=self.send_message, bg="#4CAF50", fg="white", font=("Arial", 10, "bold")
        )
        send_btn.pack(side=tk.LEFT)

        # ===== PROMPT_EXPEDITOR =====
        prompt_label = tk.Label(main_frame, text="🤖 Prompt Expeditor (vers IA)", font=("Arial", 12, "bold"), bg="#f0f0f0")
        prompt_label.pack(anchor="w", pady=(0, 5))

        # Zone de texte pour les prompts
        self.prompt_input = scrolledtext.ScrolledText(
            main_frame, height=6, width=60, bg="white", font=("Arial", 10)
        )
        self.prompt_input.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Bouton d'envoi du prompt
        prompt_btn = tk.Button(
            main_frame, text="Envoyer vers l'IA", command=self.send_prompt, bg="#2196F3", fg="white", font=("Arial", 10, "bold")
        )
        prompt_btn.pack(fill=tk.X)

    def send_message(self):
        """Envoie un message via le chatbox"""
        if not self.message_input:
            messagebox.showwarning("Attention", "L'interface n'est pas initialisée")
            return
        
        message = self.message_input.get().strip()
        if not message:
            messagebox.showwarning("Attention", "Veuillez entrer un message")
            return

        # Affiche le message de l'utilisateur
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.display_message(f"[{timestamp}] Vous: {message}\n")
        self.message_input.delete(0, tk.END)

    def send_prompt(self):
        """Envoie un prompt vers l'IA via nety_adapter"""
        if not self.prompt_input:
            messagebox.showwarning("Attention", "L'interface n'est pas initialisée")
            return
        
        prompt = self.prompt_input.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Attention", "Veuillez entrer un prompt")
            return

        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.display_message(f"[{timestamp}] 🚀 Prompt envoyé:\n{prompt}\n")
            
            # Envoie le prompt à l'IA via nety_adapter
            if self.nety_adapter:
                response = self.nety_adapter.process_prompt(prompt)
                self.display_message(f"[{timestamp}] 🤖 Réponse IA:\n{response}\n")
            
            self.prompt_input.delete("1.0", tk.END)
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'envoi: {str(e)}")

    def display_message(self, text):
        """Affiche un message dans le chatbox"""
        if not self.chat_display:
            return
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, text)
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)