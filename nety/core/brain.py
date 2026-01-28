"""
NETY Brain - Cerveau central de l'IA
Coordonne tous les modules et prend les décisions
"""

from typing import Dict, Any, Optional
from ..modules.machinelearning.ml_engine import MLEngine
from .config import ModuleTags


class Brain:
    """
    Cerveau central de NETY
    - Analyse les entrées
    - Route vers les bons modules
    - Combine les résultats
    """
    
    def __init__(self):
        """Initialise le cerveau et ses composants"""
        print("🧠 Initialisation du Brain NETY...")
        
        # Moteur de ML
        self.ml_engine = MLEngine()
        
        # Mémoire à court terme (conversation en cours)
        self.short_term_memory = []
        
        # État des modules
        self.modules_status = self._init_modules()
        
        print("✓ Brain initialisé avec succès")
    
    # ==========================================
    # 🎯 FONCTION PRINCIPALE
    # ==========================================
    def think(self, input_data: str) -> str:
        """
        Fonction principale de réflexion
        
        Args:
            input_data: Texte d'entrée de l'utilisateur
            
        Returns:
            Résultat du traitement
        """
        print(f"🧠 Brain réfléchit à : {input_data}")
        
        # 1. Prétraitement
        cleaned_input = self._preprocess(input_data)
        
        # 2. Analyser l'intention
        intent = self._analyze_intent(cleaned_input)
        
        # 3. Router vers le bon module
        result = self._route_to_module(intent, cleaned_input)
        
        # 4. Post-traitement
        final_output = self._postprocess(result)
        
        # 5. Mémoriser
        self._memorize(input_data, final_output)
        
        return final_output
    
    # ==========================================
    # 🔍 ANALYSE D'INTENTION
    # ==========================================
    def _analyze_intent(self, text: str) -> str:
        """
        Détermine ce que l'utilisateur veut faire
        
        Intentions possibles :
        - transform_text : Réécrire/transformer du texte
        - question : Poser une question
        - command : Donner une commande
        - conversation : Discussion naturelle
        """
        text_lower = text.lower()
        
        # Détection simple (tu pourras améliorer avec le ML)
        if any(word in text_lower for word in ["réécris", "transforme", "corrige"]):
            return "transform_text"
        
        if "?" in text or any(word in text_lower for word in ["quoi", "comment", "pourquoi"]):
            return "question"
        
        if any(word in text_lower for word in ["fait", "exécute", "lance"]):
            return "command"
        
        return "conversation"
    
    # ==========================================
    # 🚦 ROUTAGE VERS LES MODULES
    # ==========================================
    def _route_to_module(self, intent: str, data: str) -> str:
        """
        Route la requête vers le bon module selon l'intention
        """
        print(f"📍 Routage vers module : {intent}")
        
        if intent == "transform_text":
            # Utiliser le ML Engine pour transformer
            return self.ml_engine.transform_text(data)
        
        elif intent == "question":
            # Utiliser le module de compréhension
            return self._answer_question(data)
        
        elif intent == "command":
            # Exécuter une commande système
            return self._execute_command(data)
        
        else:  # conversation
            # Génération de réponse naturelle
            return self._generate_conversation(data)
    
    # ==========================================
    # 🛠️ FONCTIONS MÉTIER
    # ==========================================
    def _answer_question(self, question: str) -> str:
        """Répond à une question"""
        # Pour l'instant simple, tu pourras utiliser le ML plus tard
        return f"Réponse à '{question}': Fonction en développement."
    
    def _execute_command(self, command: str) -> str:
        """Exécute une commande"""
        # Exemple : "active le module TPM"
        return f"Commande '{command}' exécutée."
    
    def _generate_conversation(self, text: str) -> str:
        """Génère une réponse conversationnelle"""
        # Utiliser le ML Engine
        return self.ml_engine.generate_response(text)
    
    # ==========================================
    # 🧹 PRÉ/POST TRAITEMENT
    # ==========================================
    def _preprocess(self, text: str) -> str:
        """Nettoie le texte d'entrée"""
        # Supprime espaces multiples, normalise
        return " ".join(text.strip().split())
    
    def _postprocess(self, text: str) -> str:
        """Finalise la sortie"""
        # Ajoute la ponctuation, met en forme
        if not text.endswith((".", "!", "?")):
            text += "."
        return text
    
    # ==========================================
    # 💾 MÉMOIRE
    # ==========================================
    def _memorize(self, input_data: str, output: str):
        """Stocke l'interaction dans la mémoire court terme"""
        self.short_term_memory.append({
            "input": input_data,
            "output": output
        })
        
        # Limite la mémoire à 10 interactions
        if len(self.short_term_memory) > 10:
            self.short_term_memory.pop(0)
    
    def get_context(self) -> list:
        """Récupère le contexte de conversation"""
        return self.short_term_memory
    
    # ==========================================
    # 🔧 GESTION DES MODULES
    # ==========================================
    def _init_modules(self) -> Dict[str, str]:
        """Initialise l'état des modules"""
        modules = {}
        for group in ModuleTags.all_group_tags():
            submodules = ModuleTags.SUBMODULES.get(group, {})
            for name, code in submodules.items():
                modules[code] = "inactive"  # Par défaut inactif
        return modules
    
    def activate_module(self, module_code: str):
        """Active un module spécifique"""
        if module_code in self.modules_status:
            self.modules_status[module_code] = "active"
            print(f"✓ Module {module_code} activé")
    
    def get_modules_status(self) -> Dict[str, str]:
        """Retourne l'état de tous les modules"""
        return self.modules_status