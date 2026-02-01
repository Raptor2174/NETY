# nety/core/smart_prompt_builder.py

"""
Système de prompts adaptatifs multi-niveaux
Optimisé pour réduire la consommation de tokens Groq
"""

from typing import Dict, List, Tuple

class SmartPromptBuilder:
    """
    Construit des prompts de 3 niveaux selon le besoin :
    - MINIMAL : Questions simples (100-200 tokens)
    - STANDARD : Conversation normale (300-400 tokens)
    - ENRICHI : Questions sur mémoire/émotions (800-1200 tokens)
    """
    
    # Mots-clés déclenchant un prompt enrichi
    ENRICHED_KEYWORDS = [
        # Questions sur la mémoire
        "souvenir", "mémoire", "rappelle", "rappeler", "te souvient",
        # Questions sur l'identité
        "qui es-tu", "dis-moi qui", "parle de toi", "traits", "personnalité",
        # Questions sur les émotions
        "émotion", "ressens", "sentiment", "comment tu vas",
        # Questions sur la configuration
        "cortex", "limbique", "configuration", "esprit", "culture", "normand",
        # Demandes spécifiques
        "key_info", "corrélation", "profil"
    ]
    
    # Mots-clés pour prompts minimaux (questions simples)
    MINIMAL_KEYWORDS = [
        "bonjour", "salut", "hello", "merci", "ok", "d'accord",
        "oui", "non", "peut-être", "bye", "au revoir"
    ]
    
    def detect_prompt_level(self, message: str) -> str:
        """
        Détecte le niveau de prompt nécessaire
        
        Returns:
            "minimal", "standard", ou "enriched"
        """
        message_lower = message.lower()
        
        # Niveau ENRICHI (questions complexes)
        if any(keyword in message_lower for keyword in self.ENRICHED_KEYWORDS):
            return "enriched"
        
        # Niveau MINIMAL (salutations simples)
        if any(keyword in message_lower for keyword in self.MINIMAL_KEYWORDS):
            if len(message.split()) <= 5:  # Message court
                return "minimal"
        
        # Niveau STANDARD par défaut
        return "standard"
    
    def build_prompt(
        self, 
        message: str, 
        context: Dict, 
        limbic_filter: Dict,
        level: str = "auto"
    ) -> Tuple[str, int]:
        """
        Construit un prompt adaptatif
        
        Returns:
            Tuple[prompt, estimated_tokens]
        """
        
        # Auto-détection du niveau
        if level == "auto":
            level = self.detect_prompt_level(message)
        
        if level == "minimal":
            return self._build_minimal(message, context), 150
        elif level == "enriched":
            return self._build_enriched(message, context, limbic_filter), 1200
        else:  # standard
            return self._build_standard(message, context, limbic_filter), 350
    
    # ═══════════════════════════════════════════════════
    # NIVEAU 1: MINIMAL (100-200 tokens)
    # ═══════════════════════════════════════════════════
    
    def _build_minimal(self, message: str, context: Dict) -> str:
        """
        Prompt ultra-léger pour questions simples
        Économise jusqu'à 80% de tokens
        """
        parts = []
        
        # Juste le dernier échange si disponible
        history = context.get('history', [])
        if history:
            last = history[-1]
            parts.append(f"User: {last.get('input', '')[:50]}...")
            parts.append(f"NETY: {last.get('output', '')[:50]}...")
            parts.append("")
        
        parts.append(f"User: {message}")
        
        return "\n".join(parts)
    
    # ═══════════════════════════════════════════════════
    # NIVEAU 2: STANDARD (300-400 tokens)
    # ═══════════════════════════════════════════════════
    
    def _build_standard(self, message: str, context: Dict, limbic_filter: Dict) -> str:
        """
        Prompt équilibré pour conversations normales
        Version améliorée de l'actuel
        """
        parts = []
        
        # Historique récent (2 derniers échanges)
        history = context.get('history', [])
        if history:
            parts.append("Contexte:")
            for interaction in history[-2:]:
                user_msg = interaction.get('input', '')
                bot_msg = interaction.get('output', '')
                if user_msg and bot_msg:
                    # Tronquer pour économiser
                    parts.append(f"User: {user_msg[:80]}...")
                    parts.append(f"NETY: {bot_msg[:80]}...")
            parts.append("")
        
        # Profil utilisateur (condensé)
        profile = context.get('user_profile', {})
        if profile:
            profile_str = ", ".join([f"{k}: {v}" for k, v in list(profile.items())[:3]])
            parts.append(f"Profil: {profile_str}")
            parts.append("")
        
        # Top 3 souvenirs pertinents (actuel)
        memories = context.get('personal_memory', [])
        if memories:
            parts.append("Souvenirs:")
            for mem in memories[:3]:
                text = mem.get('text', '')
                if text:
                    parts.append(f"- {text[:100]}")
            parts.append("")
        
        parts.append(f"User: {message}")
        
        return "\n".join(parts)
    
    # ═══════════════════════════════════════════════════
    # NIVEAU 3: ENRICHI (800-1200 tokens)
    # ═══════════════════════════════════════════════════
    
    def _build_enriched(self, message: str, context: Dict, limbic_filter: Dict) -> str:
        """
        Prompt complet SEULEMENT pour questions sur mémoire/identité/émotions
        """
        sections = []
        
        # ─── IDENTITÉ (si demandée) ───
        if any(kw in message.lower() for kw in ["culture", "normand", "traits", "identité"]):
            sections.append("=== IDENTITÉ DE NETY ===")
            
            cultural = limbic_filter.get('cultural_traits', {})
            if cultural:
                sections.append("Traits culturels:")
                for trait, value in cultural.items():
                    if value > 0.5:
                        sections.append(f"  - {trait}: {value:.2f}")
            
            cognitive = limbic_filter.get('cognitive_traits', {})
            if cognitive:
                sections.append("Traits cognitifs:")
                for trait, value in cognitive.items():
                    if value > 0.5:
                        sections.append(f"  - {trait}: {value:.2f}")
            
            sections.append("")
        
        # ─── ÉMOTIONS (si demandées) ───
        if any(kw in message.lower() for kw in ["émotion", "ressens", "sentiment", "cortex", "limbique"]):
            emotional_state = limbic_filter.get('emotional_state', {})
            if emotional_state:
                sections.append("=== ÉTAT ÉMOTIONNEL ===")
                state = emotional_state.get('state', '')
                dominant = emotional_state.get('dominant_emotion', '')
                intensity = emotional_state.get('intensity', 0.0)
                
                sections.append(f"État: {state}")
                sections.append(f"Dominant: {dominant} ({intensity:.2f})")
                
                # Top 3 émotions seulement
                all_emotions = emotional_state.get('all_emotions', {})
                top_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                for emotion, value in top_emotions:
                    sections.append(f"  - {emotion}: {value:.2f}")
                
                sections.append("")
        
        # ─── MÉMOIRE COMPLÈTE (si demandée) ───
        if any(kw in message.lower() for kw in ["souvenir", "mémoire", "rappelle", "key_info"]):
            # Key infos
            key_infos = context.get('key_infos', [])
            if key_infos:
                sections.append("=== INFOS CLÉS UTILISATEUR ===")
                
                # Limiter à 10 corrélations les plus récentes
                correlations = [k for k in key_infos if k.get('type') == 'correlation']
                for corr in correlations[-10:]:
                    field = corr.get('field', '')
                    value = corr.get('value', '')
                    sections.append(f"  - {field}: {value}")
                
                sections.append("")
            
            # Souvenirs (10 au lieu de 3)
            memories = context.get('personal_memory', [])
            if memories:
                sections.append("=== SOUVENIRS (10 plus pertinents) ===")
                for i, mem in enumerate(memories[:10], 1):
                    text = mem.get('text', '')
                    if text:
                        sections.append(f"{i}. {text[:150]}")  # Tronquer à 150 chars
                
                sections.append("")
        
        # ─── CONTEXTE RÉCENT ───
        history = context.get('history', [])
        if history:
            sections.append("=== CONTEXTE ===")
            for interaction in history[-2:]:
                user_msg = interaction.get('input', '')
                bot_msg = interaction.get('output', '')
                if user_msg and bot_msg:
                    sections.append(f"User: {user_msg[:100]}...")
                    sections.append(f"NETY: {bot_msg[:100]}...")
            sections.append("")
        
        # ─── QUESTION ───
        sections.append(f"User: {message}")
        
        return "\n".join(sections)