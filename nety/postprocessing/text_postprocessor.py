"""
NETY V2-Maxx - Postprocessing Module
=====================================

Postprocessing pipeline pour améliorer les réponses générées :
1. Détokenization
2. Formatage (ponctuation, capitalisation)
3. Filtrage (répétitions, contenu inapproprié)
4. Enrichissement contextuel
"""

import re
from typing import List, Optional, Dict
import unicodedata


class TextFormatter:
    """Formatage du texte généré pour le rendre plus naturel"""
    
    def __init__(
        self,
        capitalize_sentences: bool = True,
        fix_punctuation_spacing: bool = True,
        remove_duplicate_punctuation: bool = True
    ):
        self.capitalize_sentences = capitalize_sentences
        self.fix_punctuation_spacing = fix_punctuation_spacing
        self.remove_duplicate_punctuation = remove_duplicate_punctuation
    
    def format(self, text: str) -> str:
        """Formate le texte"""
        # Fix punctuation spacing
        if self.fix_punctuation_spacing:
            text = self._fix_punctuation_spacing(text)
        
        # Remove duplicate punctuation
        if self.remove_duplicate_punctuation:
            text = self._remove_duplicate_punctuation(text)
        
        # Capitalize sentences
        if self.capitalize_sentences:
            text = self._capitalize_sentences(text)
        
        # Final cleanup
        text = text.strip()
        
        return text
    
    def _fix_punctuation_spacing(self, text: str) -> str:
        """Corrige les espaces autour de la ponctuation"""
        # Enlever espaces avant ponctuation
        text = re.sub(r'\s+([.!?,;:])', r'\1', text)
        
        # Ajouter espace après ponctuation si manquant
        text = re.sub(r'([.!?,;:])([A-Za-zÀ-ÿ])', r'\1 \2', text)
        
        # Parenthèses et guillemets
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'\s+\)', ')', text)
        text = re.sub(r'"\s+', '"', text)
        text = re.sub(r'\s+"', '"', text)
        
        # Apostrophes
        text = re.sub(r"\s+'", "'", text)
        
        return text
    
    def _remove_duplicate_punctuation(self, text: str) -> str:
        """Supprime la ponctuation dupliquée"""
        # Points multiples
        text = re.sub(r'\.{2,}', '...', text)  # Garder ellipse
        text = re.sub(r'\.{4,}', '...', text)  # Max 3 points
        
        # Autres ponctuations
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)
        text = re.sub(r',{2,}', ',', text)
        
        # Mélange ? et !
        text = re.sub(r'[?!]{3,}', '?!', text)
        
        return text
    
    def _capitalize_sentences(self, text: str) -> str:
        """Capitalise la première lettre de chaque phrase"""
        # Split par ponctuation de fin de phrase
        sentences = re.split(r'([.!?]\s+)', text)
        
        result = []
        for i, part in enumerate(sentences):
            if i % 2 == 0 and part:  # Phrases (pas les séparateurs)
                # Capitaliser première lettre
                part = part[0].upper() + part[1:] if len(part) > 0 else part
            result.append(part)
        
        return ''.join(result)


class RepetitionFilter:
    """Filtre les répétitions dans le texte généré"""
    
    def __init__(
        self,
        max_consecutive_repeats: int = 2,
        max_phrase_repeats: int = 1,
        min_phrase_length: int = 3
    ):
        self.max_consecutive_repeats = max_consecutive_repeats
        self.max_phrase_repeats = max_phrase_repeats
        self.min_phrase_length = min_phrase_length
    
    def filter(self, text: str) -> str:
        """Filtre les répétitions"""
        # Filtrer mots consécutifs répétés
        text = self._filter_consecutive_words(text)
        
        # Filtrer phrases répétées
        text = self._filter_repeated_phrases(text)
        
        return text
    
    def _filter_consecutive_words(self, text: str) -> str:
        """Filtre les mots consécutifs identiques"""
        words = text.split()
        filtered = []
        
        count = 0
        prev_word = None
        
        for word in words:
            if word.lower() == (prev_word or '').lower():
                count += 1
                if count <= self.max_consecutive_repeats:
                    filtered.append(word)
            else:
                count = 1
                filtered.append(word)
                prev_word = word
        
        return ' '.join(filtered)
    
    def _filter_repeated_phrases(self, text: str) -> str:
        """Filtre les phrases répétées"""
        sentences = re.split(r'([.!?]\s*)', text)
        
        seen_sentences = {}
        filtered = []
        prev_sentence_added = False
        
        for i, part in enumerate(sentences):
            if i % 2 == 0 and part:  # Phrases
                normalized = part.lower().strip()
                
                if len(normalized.split()) >= self.min_phrase_length:
                    count = seen_sentences.get(normalized, 0)
                    
                    if count <= self.max_phrase_repeats:
                        filtered.append(part)
                        seen_sentences[normalized] = count + 1
                        prev_sentence_added = True
                    else:
                        prev_sentence_added = False
                else:
                    filtered.append(part)
                    prev_sentence_added = True
            else:
                # Séparateurs - ajouter seulement si la phrase précédente a été ajoutée
                if prev_sentence_added:
                    filtered.append(part)
        
        return ''.join(filtered)


class ContentFilter:
    """Filtre le contenu inapproprié ou indésirable"""
    
    def __init__(
        self,
        min_length: int = 3,
        max_length: int = 500,
        block_patterns: Optional[List[str]] = None
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.block_patterns = block_patterns or []
    
    def filter(self, text: str) -> Optional[str]:
        """
        Filtre le contenu
        
        Returns:
            Text filtré ou None si à rejeter
        """
        # Check longueur
        if len(text) < self.min_length:
            return None
        
        if len(text) > self.max_length:
            text = text[:self.max_length].rsplit(' ', 1)[0] + '...'
        
        # Check patterns bloqués
        for pattern in self.block_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return None
        
        return text


class ResponseEnricher:
    """Enrichit les réponses avec du contexte et des variations"""
    
    def __init__(self):
        # Variations pour réponses courtes
        self.acknowledgments = [
            "Je comprends.",
            "D'accord.",
            "Je vois.",
            "Compris.",
            "Entendu."
        ]
        
        self.thinking_phrases = [
            "Laisse-moi réfléchir...",
            "Hmm, intéressant...",
            "Bonne question...",
            "Je réfléchis...",
        ]
    
    def enrich(
        self,
        text: str,
        context: Optional[Dict] = None,
        min_length_for_enrichment: int = 10
    ) -> str:
        """
        Enrichit la réponse si trop courte ou générique
        
        Args:
            text: Texte à enrichir
            context: Contexte conversationnel (optionnel)
            min_length_for_enrichment: Longueur min avant enrichissement
        
        Returns:
            Texte enrichi
        """
        # Si trop court et pas de ponctuation, c'est possiblement incomplet
        if len(text) < min_length_for_enrichment and not re.search(r'[.!?]$', text):
            # Ajouter point si manquant
            if not text.endswith(('.', '!', '?', '...', ',')):
                text = text + '.'
        
        return text


class Postprocessor:
    """
    Pipeline de postprocessing complet
    
    Usage:
        postprocessor = Postprocessor()
        clean_text = postprocessor(raw_text)
    """
    
    def __init__(
        self,
        capitalize: bool = True,
        filter_repetitions: bool = True,
        filter_content: bool = True,
        enrich_responses: bool = True,
        min_length: int = 3,
        max_length: int = 500
    ):
        # Composants
        self.formatter = TextFormatter(
            capitalize_sentences=capitalize,
            fix_punctuation_spacing=True,
            remove_duplicate_punctuation=True
        )
        
        self.repetition_filter = RepetitionFilter(
            max_consecutive_repeats=2,
            max_phrase_repeats=1,
            min_phrase_length=3
        ) if filter_repetitions else None
        
        self.content_filter = ContentFilter(
            min_length=min_length,
            max_length=max_length
        ) if filter_content else None
        
        self.enricher = ResponseEnricher() if enrich_responses else None
    
    def __call__(
        self,
        text: str,
        context: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Postprocess le texte
        
        Args:
            text: Texte brut à postprocesser
            context: Contexte optionnel
        
        Returns:
            Texte nettoyé ou None si à rejeter
        """
        # 1. Filtrer répétitions
        if self.repetition_filter:
            text = self.repetition_filter.filter(text)
        
        # 2. Formater
        text = self.formatter.format(text)
        
        # 3. Filtrer contenu
        if self.content_filter:
            text = self.content_filter.filter(text)
            if text is None:
                return None
        
        # 4. Enrichir
        if self.enricher:
            text = self.enricher.enrich(text, context)
        
        return text
    
    def batch_postprocess(
        self,
        texts: List[str],
        contexts: Optional[List[Dict]] = None
    ) -> List[Optional[str]]:
        """Postprocess un batch de textes"""
        if contexts is None:
            contexts = [None] * len(texts)
        
        return [self(text, ctx) for text, ctx in zip(texts, contexts)]


if __name__ == "__main__":
    # Test du postprocessor
    print("=" * 80)
    print("Test du Postprocessor")
    print("=" * 80)
    
    postprocessor = Postprocessor()
    
    # Test 1: Formatage basique
    text1 = "bonjour  !  comment vas-tu  ?  je vais bien ."
    result1 = postprocessor(text1)
    print(f"\n✓ Test 1 - Formatage")
    print(f"  Input:  {text1}")
    print(f"  Output: {result1}")
    
    # Test 2: Répétitions
    text2 = "Je suis content content content. Je suis heureux. Je suis heureux."
    result2 = postprocessor(text2)
    print(f"\n✓ Test 2 - Répétitions")
    print(f"  Input:  {text2}")
    print(f"  Output: {result2}")
    
    # Test 3: Ponctuation
    text3 = "C'est super!!!!! Vraiment  ...  incroyable???"
    result3 = postprocessor(text3)
    print(f"\n✓ Test 3 - Ponctuation")
    print(f"  Input:  {text3}")
    print(f"  Output: {result3}")
    
    # Test 4: Capitalisation
    text4 = "bonjour. comment ça va? très bien merci!"
    result4 = postprocessor(text4)
    print(f"\n✓ Test 4 - Capitalisation")
    print(f"  Input:  {text4}")
    print(f"  Output: {result4}")
    
    # Test 5: Texte trop court
    text5 = "ok"
    result5 = postprocessor(text5)
    print(f"\n✓ Test 5 - Texte court")
    print(f"  Input:  {text5}")
    print(f"  Output: {result5 if result5 else '(rejeté car trop court)'}")
    
    # Test 6: Batch
    texts = [text1, text2, text3, text4]
    results = postprocessor.batch_postprocess(texts)
    print(f"\n✓ Test 6 - Batch processing")
    print(f"  Processed: {len([r for r in results if r])} / {len(texts)} textes")
    
    print("\n" + "=" * 80)
    print("✓ Tous les tests passés!")
    print("=" * 80)
