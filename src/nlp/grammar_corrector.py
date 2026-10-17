"""
Enhanced Context Guesser & Grammar Engine
==========================================
Converts sequences of ASL/ISL sign glosses into fluent, grammatically correct
English sentences using contextual pattern matching, semantic infilling,
noise rejection, and interrogative/imperative structuring.

Supports the full WLASL100 vocabulary and common conversational phrases.
"""

import re
from typing import List, Union, Optional


class GrammarCorrector:
    """Intelligent context guesser and grammar reconstruction engine for sign glosses."""

    def __init__(self):
        # 1. High-frequency exact phrase & conversational idiom mappings
        self.idiom_rules = {
            # Greetings & Courtesies
            "HELLO": "Hello!",
            "HELLO HOW": "Hello, how are you?",
            "HELLO HOW YOU": "Hello, how are you doing?",
            "THANKS": "Thank you very much.",
            "THANK YOU": "Thank you very much.",
            "FINE THANKS": "I am doing fine, thank you.",
            "GOOD MORNING": "Good morning!",
            "NICE MEET": "Nice to meet you.",
            "NICE MEET YOU": "It is very nice to meet you.",
            "MEET YOU": "Pleased to meet you.",
            "I LOVE YOU": "I love you.",
            "LOVE YOU": "I love you so much.",
            "YES": "Yes, absolutely.",
            "YES CAN": "Yes, I can do that.",
            "NO": "No, not right now.",
            "NO WRONG": "No, that is incorrect.",
            "SAME": "Same here.",
            "RIGHT": "That is right.",
            "WRONG": "That is wrong.",
            "LATER": "See you later.",
            "NOW GO": "I need to go now.",
            "BIRTHDAY": "Happy Birthday!",
            "THANKSGIVING": "Happy Thanksgiving!",

            # Desires, Needs & Requests (WANT / NEED / EAT / DRINK)
            "I WANT WATER": "I want water.",
            "WANT WATER": "I would like some water, please.",
            "WATER WANT": "Could I please have some water?",
            "NEED WATER": "I need some water.",
            "WATER NEED": "I urgently need water.",
            "WANT DRINK": "I want something to drink.",
            "DRINK WANT": "Can I have a drink?",
            "WANT EAT": "I am hungry and want to eat.",
            "EAT WANT": "I want something to eat.",
            "WANT EAT PIZZA": "I would love to eat pizza.",
            "WANT PIZZA": "I want some pizza.",
            "PIZZA WANT": "Can we get pizza?",
            "EAT PIZZA": "Let's eat pizza.",
            "WANT APPLE": "I would like an apple.",
            "EAT APPLE": "I am eating an apple.",
            "WANT CANDY": "I want some candy.",
            "EAT CANDY": "Eating candy.",
            "WANT CORN": "I want corn.",
            "EAT CORN": "Eating corn.",
            "WANT FISH": "I want fish for dinner.",
            "EAT FISH": "Eating fish.",
            "NEED MEDICINE": "I need my medicine.",
            "MEDICINE NEED": "Please bring my medicine.",
            "NEED DOCTOR": "I need to see a doctor immediately.",
            "DOCTOR NEED": "Please call a doctor for me.",
            "NEED HELP": "Please help me!",
            "HELP NEED": "I need assistance right now.",
            "HELP": "Please help me!",
            "ACCIDENT": "There has been an accident!",
            "ACCIDENT HELP": "There is an accident, please help!",

            # Activities & Routine (GO / WALK / WORK / STUDY / PLAY)
            "GO BED": "I am tired and going to bed.",
            "TIME GO BED": "It is time to go to bed.",
            "GO SCHOOL": "I am heading to school.",
            "STUDY SCHOOL": "I am studying at school.",
            "GO WORK": "I am on my way to work.",
            "WORK FINISH": "I have finished my work.",
            "FINISH WORK": "My work is all done.",
            "PLAY BASKETBALL": "I want to play basketball.",
            "PLAY BOWLING": "Let's go bowling.",
            "PLAY DANCE": "Let's dance and have fun.",
            "DANCE PLAY": "We are dancing.",
            "READ BOOK": "I am reading a book.",
            "STUDY BOOK": "I am studying from the book.",
            "STUDY LANGUAGE": "I am studying sign language.",
            "LEARN LANGUAGE": "I am learning sign language.",
            "COMPUTER WORK": "I am working on the computer.",
            "COMPUTER PLAY": "Playing games on the computer.",
            "WRITE LETTER": "I am writing a letter.",
            "SEND LETTER": "Sending a letter.",
            "PAINT PAPER": "Painting on paper.",

            # Questions (WHO / WHAT / HOW / TIME)
            "WHAT TIME": "What time is it right now?",
            "TIME WHAT": "Could you tell me the time?",
            "WHO DOCTOR": "Who is the doctor here?",
            "WHO SECRETARY": "Who is the secretary?",
            "WHO WOMAN": "Who is that woman?",
            "WHO MAN": "Who is that man?",
            "WHO YOU": "Who are you?",
            "WHAT COLOR": "What color is that?",
            "WHAT NAME": "What is your name?",
            "WHAT HAPPEN": "What happened?",
            "HOW WORK": "How does this work?",
            "CAN HELP": "Can you please help me?",
            "WHERE GO": "Where are you going?",

            # People, Family & Identity
            "DEAF I": "I am deaf.",
            "I DEAF": "I am deaf.",
            "HEARING I": "I am hearing.",
            "I HEARING": "I am a hearing person.",
            "MY FAMILY": "This is my family.",
            "MEET FAMILY": "I would like you to meet my family.",
            "MEET MOTHER": "I want to meet my mother.",
            "MOTHER LOVE": "I love my mother.",
            "SON SCHOOL": "My son is currently at school.",
            "COUSIN MEET": "I am meeting my cousin.",
            "DOG PLAY": "Playing with the dog.",
            "BIRD FLY": "Look at the bird.",

            # State, Feelings & Descriptors
            "FEEL HOT": "It feels very hot here.",
            "HOT": "It is very hot.",
            "FEEL COOL": "It feels nice and cool.",
            "COOL": "That is really cool!",
            "FEEL FINE": "I feel fine.",
            "DARK": "It is getting dark outside.",
            "DARK NIGHT": "It is dark outside.",
            "FULL": "I am completely full.",
            "ALL FINISH": "Everything is completely finished.",
            "DECIDE RIGHT": "That was the right decision.",
            "DECIDE WRONG": "That was the wrong decision.",
            "FORGET TIME": "I lost track of time.",
            "FORGET": "I forgot about that.",
            "CHANGE TIME": "Can we change the time?",
            "PULL CHAIR": "Please pull up a chair.",
            "CLEAN CLOTHES": "These clothes are clean.",
            "CHANGE CLOTHES": "I need to change my clothes.",
            "BLUE SHIRT": "I like that blue shirt.",
            "BLACK JACKET": "Wearing a black jacket.",
            "WHITE HAT": "Wearing a white hat.",
        }

        # Categories for smart semantic guessing
        self.question_words = {"WHAT", "WHO", "HOW", "WHERE", "WHEN", "WHY"}
        self.verbs = {
            "WANT", "NEED", "LIKE", "EAT", "DRINK", "GO", "WALK", "COOK",
            "DANCE", "PLAY", "STUDY", "WORK", "HELP", "TELL", "GIVE", "MEET",
            "ENJOY", "KISS", "FINISH", "FORGET", "DECIDE", "CHANGE", "PULL", "PAINT"
        }
        self.foods = {"APPLE", "CANDY", "CORN", "FISH", "PIZZA", "WATER", "MEDICINE"}
        self.people = {"I", "YOU", "MOTHER", "SON", "WOMAN", "MAN", "COUSIN", "DOCTOR", "SECRETARY", "FAMILY"}
        self.clothing = {"SHIRT", "JACKET", "HAT", "CLOTHES"}
        self.colors = {"BLACK", "BLUE", "BROWN", "ORANGE", "PINK", "PURPLE", "WHITE"}
        self.objects = {"BOOK", "COMPUTER", "LETTER", "PAPER", "CHAIR", "BED", "TABLE", "BASKETBALL"}

    def deduplicate(self, glosses: List[str]) -> List[str]:
        """Collapse consecutive duplicate glosses and strip whitespace."""
        clean = []
        for g in glosses:
            if not g:
                continue
            norm = str(g).strip().upper()
            if not clean or clean[-1] != norm:
                clean.append(norm)
        return clean

    def correct(self, glosses: Union[List[str], str]) -> str:
        """
        Converts a list or string of recognized ASL glosses into a natural English sentence.

        Args:
            glosses: list of strings (e.g. ['WANT', 'WATER']) or string 'WANT WATER'

        Returns:
            Fluent English sentence (e.g. 'I want water.')
        """
        if isinstance(glosses, str):
            tokens = glosses.strip().split()
        else:
            tokens = list(glosses)

        # 1. Normalize and deduplicate consecutive repeated tokens
        clean_tokens = self.deduplicate(tokens)
        if not clean_tokens:
            return ""

        raw_key = " ".join(clean_tokens)

        # 2. Check direct idiom / common phrase match
        if raw_key in self.idiom_rules:
            return self.idiom_rules[raw_key]

        # 3. Try partial substring / prefix match if longer phrase
        if len(clean_tokens) > 2:
            sub_key = " ".join(clean_tokens[-2:])
            if sub_key in self.idiom_rules:
                return self.idiom_rules[sub_key]

        # 4. Contextual Pattern Matching & Grammar Reconstruction
        sentence = self._synthesize_grammar(clean_tokens)
        return sentence

    def _synthesize_grammar(self, tokens: List[str]) -> str:
        """Rule-based natural language synthesis for gloss sequences."""
        n = len(tokens)
        first, last = tokens[0], tokens[-1]

        # A. Questions: contains a question word
        has_question = any(q in tokens for q in self.question_words)

        if has_question:
            if "TIME" in tokens and ("WHAT" in tokens or first == "TIME"):
                return "What time is it?"
            if "COLOR" in tokens and "WHAT" in tokens:
                other_words = [t.lower() for t in tokens if t not in {"WHAT", "COLOR"}]
                if other_words:
                    return f"What color is the {' '.join(other_words)}?"
                return "What color is it?"
            if "WHO" in tokens:
                targets = [t.lower() for t in tokens if t != "WHO"]
                if targets:
                    return f"Who is that {' '.join(targets)}?"
                return "Who is that?"
            if "WHAT" in tokens:
                others = [t.lower() for t in tokens if t != "WHAT"]
                if others:
                    return f"What is {' '.join(others)}?"
                return "What is that?"
            if "HOW" in tokens:
                others = [t.lower() for t in tokens if t != "HOW"]
                if "YOU" in others:
                    return "How are you doing?"
                if others:
                    return f"How does {' '.join(others)} work?"
                return "How does this work?"

        # B. Desire / Necessity: WANT or NEED
        if "WANT" in tokens or "NEED" in tokens:
            verb = "need" if "NEED" in tokens else "want"
            obj_words = [t.lower() for t in tokens if t not in {"WANT", "NEED", "I", "YOU"}]
            if not obj_words:
                return f"I {verb} this."

            obj_phrase = " ".join(obj_words)
            # Add appropriate article if singular object
            if obj_words[0].upper() in self.foods:
                if obj_words[0].upper() in {"APPLE", "ORANGE"}:
                    return f"I {verb} an {obj_phrase}."
                if obj_words[0].upper() in {"WATER", "MEDICINE", "CANDY", "CORN", "FISH"}:
                    return f"I {verb} some {obj_phrase}."
                return f"I {verb} {obj_phrase}."
            if obj_words[0].upper() in self.verbs:
                return f"I {verb} to {obj_phrase}."
            if obj_words[0].upper() in {"DOCTOR", "SECRETARY"}:
                return f"I {verb} to see the {obj_phrase}."
            return f"I {verb} {obj_phrase}."

        # C. Actions: EAT / DRINK / PLAY / STUDY / GO
        if "EAT" in tokens:
            foods = [t.lower() for t in tokens if t != "EAT"]
            if foods:
                return f"I want to eat {' '.join(foods)}."
            return "Time to eat."

        if "DRINK" in tokens:
            drinks = [t.lower() for t in tokens if t != "DRINK"]
            if drinks:
                return f"I am drinking {' '.join(drinks)}."
            return "I would like a drink."

        if "PLAY" in tokens:
            games = [t.lower() for t in tokens if t != "PLAY"]
            if games:
                return f"Let's play {' '.join(games)}."
            return "Let's play."

        if "GO" in tokens:
            places = [t.lower() for t in tokens if t != "GO"]
            if places:
                return f"Let's go to {' '.join(places)}."
            return "Let's go."

        # D. Adjective + Noun (Color / Quality)
        if n == 2 and tokens[0] in self.colors:
            return f"A {tokens[0].lower()} {tokens[1].lower()}."
        if n == 2 and tokens[0] in {"DARK", "HOT", "COOL", "FINE", "FULL", "TALL", "SHORT", "THIN"}:
            return f"It is very {tokens[0].lower()}."

        # E. Emergency / Alert
        if "HELP" in tokens or "ACCIDENT" in tokens:
            return "Please help me, emergency!"

        # F. Single Word Enhancements
        if n == 1:
            w = tokens[0]
            if w in {"HELLO", "THANKS", "YES", "NO", "HELP", "HOT", "COOL", "DEAF", "FINE", "ACCIDENT"}:
                return self.idiom_rules.get(w, f"{w.capitalize()}.")
            return f"{w.capitalize()}."

        # G. General Fluent Fallback
        # Capitalize, insert readable spacing, and add a terminal period
        words = [t.lower() for t in tokens]
        sentence = " ".join(words).capitalize()
        if not sentence.endswith((".", "?", "!")):
            sentence += "."
        return sentence
