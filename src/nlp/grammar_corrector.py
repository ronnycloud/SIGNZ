"""
Enhanced Context Guesser & Grammar Engine
==========================================
Converts sequences of ASL/ISL sign glosses into fluent, grammatically correct
English sentences using contextual pattern matching, semantic infilling,
noise rejection, and interrogative/imperative structuring.

Supports the full 100-word ASL core vocabulary across 10 categories:
Common Signs, Home, Food, People, Questions, School, Time, Family,
Colors & Animals, and Descriptions & Conversational.
"""

import re
from typing import List, Union, Optional
from src.data.vocabulary import normalize_action_id, GLOSS_ALIASES


class GrammarCorrector:
    """Intelligent context guesser and grammar reconstruction engine for sign glosses."""

    def __init__(self):
        # 1. High-frequency exact phrase & conversational idiom mappings
        self.idiom_rules = {
            # --- Greetings & Courtesies ---
            "HELLO": "Hello!",
            "HELLO HOW": "Hello, how are you?",
            "HELLO HOW YOU": "Hello, how are you doing?",
            "HOW YOU": "How are you doing?",
            "HOW": "How are you?",
            "THANKS": "Thank you very much.",
            "THANK YOU": "Thank you very much.",
            "FINE THANKS": "I am doing fine, thank you.",
            "PLEASE": "Please.",
            "PLEASE HELP": "Please help me!",
            "HELP PLEASE": "Could you please help me?",
            "BYE": "Goodbye, see you later!",
            "HELLO BYE": "Hello and goodbye!",
            "NICE MEET": "Nice to meet you.",
            "NICE MEET YOU": "It is very nice to meet you.",
            "MEET YOU": "Pleased to meet you.",
            "I LOVE YOU": "I love you.",
            "LOVE YOU": "I love you so much.",
            "YES": "Yes, absolutely.",
            "YES CAN": "Yes, I can do that.",
            "NO": "No, not right now.",
            "NO WRONG": "No, that is incorrect.",
            "NOT": "Not at all.",
            "DONT": "Do not do that.",
            "DONT WANT": "I do not want that.",
            "NOT WANT": "I don't want this.",
            "WANT NOT": "I really do not want this.",
            "NOT YET": "Not yet.",
            "SAME": "Same here.",
            "RIGHT": "That is right.",
            "WRONG": "That is wrong.",
            "PAY ATTENTION": "Please pay attention.",
            "CHAT": "Let's chat for a while.",
            "SIGNING": "I am signing.",
            "SIGN ASL": "Do you know how to sign in ASL?",
            "LEARN ASL": "I am learning American Sign Language.",
            "STUDY ASL": "I am studying ASL.",
            "KNOW ASL": "I know sign language.",
            "SENTENCE SIGN": "Let's sign a complete sentence.",
            "LATER": "See you later.",
            "NOW GO": "I need to go now.",
            "BIRTHDAY": "Happy Birthday!",
            "THANKSGIVING": "Happy Thanksgiving!",

            # --- Feelings & States ---
            "FINE": "I am doing fine.",
            "GOOD": "That is very good!",
            "BAD": "That is bad.",
            "HAPPY": "I am so happy!",
            "SAD": "I am feeling sad.",
            "SO SO": "I am doing so-so.",
            "FEEL HAPPY": "I feel very happy today.",
            "FEEL SAD": "I am feeling down today.",
            "GOOD MORNING": "Good morning!",
            "GOOD AFTERNOON": "Good afternoon!",
            "GOOD NIGHT": "Good night, sleep well!",
            "DAY GOOD": "Have a wonderful day!",

            # --- Desires, Needs & Food Requests ---
            "WANT": "I want that.",
            "NEED": "I need this.",
            "MORE": "I would like some more.",
            "MORE PLEASE": "More, please.",
            "WANT MORE": "I want more of that.",
            "NEED MORE": "I need more assistance.",
            "I WANT WATER": "I want water.",
            "WANT WATER": "I would like some water, please.",
            "WATER WANT": "Could I please have some water?",
            "NEED WATER": "I need some water.",
            "WANT DRINK": "I want something to drink.",
            "DRINK WANT": "Can I have a drink?",
            "DRINK MILK": "I am drinking milk.",
            "WANT MILK": "I would like some milk, please.",
            "MILK WANT": "Can I have some milk?",
            "WANT EAT": "I am hungry and want to eat.",
            "EAT WANT": "I want something to eat.",
            "EAT FOOD": "Time to eat some food.",
            "FOOD EAT": "Let's eat some food.",
            "WANT HAMBURGER": "I want a hamburger.",
            "EAT HAMBURGER": "I am eating a delicious hamburger.",
            "HAMBURGER EAT": "Let's eat hamburgers.",
            "WANT SANDWICH": "I would love a sandwich.",
            "EAT SANDWICH": "I am eating a sandwich.",
            "WANT COOKIES": "I would like some cookies.",
            "EAT COOKIES": "Eating delicious cookies.",
            "WANT CHEESE": "I want some cheese.",
            "EAT CHEESE": "Eating cheese.",
            "WANT PIZZA": "I want some pizza.",
            "EAT PIZZA": "Let's eat pizza.",
            "WANT APPLE": "I would like an apple.",
            "EAT APPLE": "I am eating an apple.",
            "WANT CANDY": "I want some candy.",
            "EAT CANDY": "Eating candy.",
            "COOK FOOD": "I am cooking dinner.",
            "COOK KITCHEN": "Cooking in the kitchen.",
            "NEED MEDICINE": "I need my medicine.",
            "NEED DOCTOR": "I need to see a doctor immediately.",
            "NEED HELP": "Please help me!",
            "HELP NEED": "I need assistance right now.",
            "HELP": "Please help me!",

            # --- Home & Places ---
            "BATHROOM": "Where is the bathroom?",
            "GO BATHROOM": "I need to use the bathroom.",
            "BATHROOM GO": "May I go to the bathroom?",
            "WHERE BATHROOM": "Could you tell me where the bathroom is?",
            "GO HOME": "I am going home now.",
            "HOME GO": "Let's head home.",
            "HOME": "I am at home.",
            "KITCHEN": "In the kitchen.",
            "GO KITCHEN": "Going to the kitchen.",
            "LIVING ROOM": "In the living room.",
            "CAR": "Look at the car.",
            "GO CAR": "Let's get in the car.",
            "WHERE CAR": "Where is the car parked?",
            "PHONE": "Please check your phone.",
            "CALL PHONE": "Call me on the phone.",
            "GO SCHOOL": "I am heading to school.",
            "STUDY SCHOOL": "I am studying at school.",
            "GO WORK": "I am on my way to work.",
            "GO BED": "I am tired and going to bed.",
            "TIME GO BED": "It is time to go to bed.",

            # --- Questions ---
            "WHAT": "What is that?",
            "WHAT TIME": "What time is it right now?",
            "TIME WHAT": "Could you tell me the time?",
            "WHERE": "Where is it located?",
            "WHEN": "When is it happening?",
            "WHO": "Who is that person?",
            "WHY": "Why is that?",
            "HOW WORK": "How does this work?",
            "WHAT HAPPEN": "What happened?",
            "WHERE YOU": "Where are you?",
            "WHERE GO": "Where are you going?",
            "WHO YOU": "Who are you?",
            "WHY WRONG": "Why is that incorrect?",
            "WHAT NAME": "What is your name?",

            # --- School & Learning ---
            "STUDENT": "I am a student.",
            "TEACHER": "Here is the teacher.",
            "WHO TEACHER": "Who is the teacher for this class?",
            "CLASS": "Time for class.",
            "START CLASS": "Class is starting now.",
            "BOOK": "Look at this book.",
            "READ BOOK": "I am reading a book.",
            "STUDY BOOK": "I am studying from the book.",
            "KNOW": "I know that.",
            "LEARN": "I want to learn.",
            "KNOW NOT": "I do not know.",
            "SIGN": "Please sign that again.",
            "SENTENCE": "That is a full sentence.",

            # --- Family & People ---
            "DEAF ME": "I am deaf.",
            "ME DEAF": "I am deaf.",
            "HEARING ME": "I am hearing.",
            "ME HEARING": "I am a hearing person.",
            "MY FAMILY": "This is my family.",
            "FAMILY": "My family is here.",
            "MEET FAMILY": "I would like you to meet my family.",
            "MY MOTHER": "This is my mother.",
            "MOTHER": "My mother.",
            "MY FATHER": "This is my father.",
            "FATHER": "My father.",
            "MY BROTHER": "This is my brother.",
            "BROTHER": "My brother.",
            "MY SISTER": "This is my sister.",
            "SISTER": "My sister.",
            "MY BABY": "Look at my baby.",
            "BABY": "The sweet baby.",
            "CHILDREN": "The children are playing.",
            "MY HUSBAND": "This is my husband.",
            "HUSBAND": "My husband.",
            "MY WIFE": "This is my wife.",
            "WIFE": "My wife.",
            "GRANDFATHER": "My grandfather.",
            "GRANDMOTHER": "My grandmother.",
            "MY GRANDFATHER": "This is my grandfather.",
            "MY GRANDMOTHER": "This is my grandmother.",
            "BOY": "That boy.",
            "GIRL": "That girl.",
            "MAN": "That man over there.",
            "WOMAN": "That woman over there.",

            # --- Animals & Colors ---
            "DOG": "Look at the dog.",
            "MY DOG": "This is my dog.",
            "DOG PLAY": "The dog is playing happily.",
            "CAT": "Look at the cat.",
            "MY CAT": "This is my cat.",
            "CAT SLEEP": "The cat is sleeping.",
            "BLUE CAR": "A nice blue car.",
            "RED CAR": "A bright red car.",
            "GREEN": "The color green.",
            "YELLOW": "Bright yellow.",

            # --- Time Units ---
            "MORNING": "In the morning.",
            "AFTERNOON": "Good afternoon.",
            "NOON": "At noon.",
            "NIGHT": "At night.",
            "TIME": "Look at the time.",
            "HOUR": "One hour.",
            "MINUTE": "Just a minute.",
            "DAY": "Have a great day.",
            "WEEK": "This week.",
            "MONTH": "This month.",
            "YEAR": "This year.",
        }

        # Categories for smart semantic inference
        self.question_words = {"WHAT", "WHO", "HOW", "WHERE", "WHEN", "WHY"}
        self.verbs = {
            "WANT", "NEED", "LIKE", "EAT", "DRINK", "GO", "WALK", "COOK",
            "DANCE", "PLAY", "STUDY", "WORK", "HELP", "TELL", "GIVE", "MEET",
            "ENJOY", "KISS", "FINISH", "FORGET", "DECIDE", "CHANGE", "PULL",
            "PAINT", "KNOW", "LEARN", "SIGN", "CHAT", "SIGNING", "LOOK",
            "PAY_ATTENTION", "DONT_WANT", "NOT_WANT"
        }
        self.foods = {
            "APPLE", "CANDY", "CORN", "FISH", "PIZZA", "WATER", "MEDICINE",
            "CHEESE", "COOKIES", "HAMBURGER", "MILK", "SANDWICH", "FOOD"
        }
        self.people = {
            "ME", "I", "YOU", "YOUR", "MY", "MOTHER", "FATHER", "BROTHER",
            "SISTER", "SON", "WOMAN", "MAN", "BOY", "GIRL", "BABY", "CHILDREN",
            "GRANDFATHER", "GRANDMOTHER", "HUSBAND", "WIFE", "COUSIN",
            "DOCTOR", "SECRETARY", "FAMILY", "STUDENT", "TEACHER"
        }
        self.places = {
            "BATHROOM", "KITCHEN", "LIVING_ROOM", "HOME", "CAR", "SCHOOL",
            "WORK", "BED", "CLASS"
        }
        self.animals = {"DOG", "CAT", "BIRD", "COW", "FISH"}
        self.colors = {
            "BLACK", "BLUE", "BROWN", "GREEN", "ORANGE", "PINK", "PURPLE",
            "RED", "WHITE", "YELLOW"
        }
        self.objects = {
            "BOOK", "COMPUTER", "LETTER", "PAPER", "CHAIR", "BED", "TABLE",
            "BASKETBALL", "PHONE", "SENTENCE", "ASL"
        }
        self.times = {
            "MORNING", "AFTERNOON", "NIGHT", "NOON", "DAY", "HOUR", "MINUTE",
            "MONTH", "WEEK", "YEAR", "TIME"
        }
        self.feelings = {
            "HAPPY", "SAD", "GOOD", "BAD", "SO_SO", "FINE", "HOT", "COOL",
            "FULL", "DARK"
        }

    def deduplicate(self, glosses: List[str]) -> List[str]:
        """Collapse consecutive duplicate glosses and strip whitespace."""
        clean = []
        for g in glosses:
            if not g:
                continue
            norm = str(g).strip().upper().replace("-", "_").replace(" ", "_").replace("/", "_")
            # Map known aliases
            lower_norm = norm.lower()
            if lower_norm in GLOSS_ALIASES:
                norm = GLOSS_ALIASES[lower_norm].upper()
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
            if "WHERE" in tokens:
                places = [t.lower().replace("_", " ") for t in tokens if t not in {"WHERE", "YOU", "GO"}]
                if "BATHROOM" in tokens:
                    return "Where is the bathroom?"
                if "CAR" in tokens:
                    return "Where is the car?"
                if "GO" in tokens:
                    return "Where are you going?"
                if places:
                    return f"Where is the {' '.join(places)}?"
                return "Where is that located?"

            if "WHAT" in tokens:
                if "TIME" in tokens:
                    return "What time is it right now?"
                if "COLOR" in tokens:
                    others = [t.lower().replace("_", " ") for t in tokens if t not in {"WHAT", "COLOR"}]
                    if others:
                        return f"What color is the {' '.join(others)}?"
                    return "What color is it?"
                if "NAME" in tokens:
                    return "What is your name?"
                others = [t.lower().replace("_", " ") for t in tokens if t != "WHAT"]
                if others:
                    return f"What is {' '.join(others)}?"
                return "What is that?"

            if "WHO" in tokens:
                targets = [t.lower().replace("_", " ") for t in tokens if t != "WHO"]
                if "TEACHER" in targets:
                    return "Who is the teacher?"
                if "STUDENT" in targets:
                    return "Who is that student?"
                if targets:
                    return f"Who is that {' '.join(targets)}?"
                return "Who is that person?"

            if "HOW" in tokens:
                others = [t.lower().replace("_", " ") for t in tokens if t != "HOW"]
                if "YOU" in others or not others:
                    return "How are you doing?"
                return f"How does {' '.join(others)} work?"

            if "WHEN" in tokens:
                others = [t.lower().replace("_", " ") for t in tokens if t != "WHEN"]
                if others:
                    return f"When is {' '.join(others)} happening?"
                return "When is that happening?"

            if "WHY" in tokens:
                others = [t.lower().replace("_", " ") for t in tokens if t != "WHY"]
                if "WRONG" in tokens:
                    return "Why is that wrong?"
                if others:
                    return f"Why is {' '.join(others)}?"
                return "Why is that?"

        # B. Desire / Necessity: WANT or NEED
        if "WANT" in tokens or "NEED" in tokens or "DONT_WANT" in tokens:
            if "DONT_WANT" in tokens:
                verb = "do not want"
            elif "NEED" in tokens:
                verb = "need"
            else:
                verb = "want"

            obj_words = [t.lower().replace("_", " ") for t in tokens if t not in {"WANT", "NEED", "DONT_WANT", "ME", "I", "YOU", "MY"}]
            if not obj_words:
                return f"I {verb} this."

            obj_phrase = " ".join(obj_words)
            first_obj = tokens[-1]

            if first_obj in self.foods:
                if first_obj in {"APPLE", "ORANGE", "HAMBURGER", "SANDWICH"}:
                    return f"I {verb} a {obj_phrase}."
                if first_obj in {"WATER", "MEDICINE", "CANDY", "CORN", "FISH", "CHEESE", "MILK", "COOKIES"}:
                    return f"I {verb} some {obj_phrase}."
                return f"I {verb} {obj_phrase}."
            if first_obj in self.verbs:
                return f"I {verb} to {obj_phrase}."
            if first_obj in {"DOCTOR", "SECRETARY", "TEACHER"}:
                return f"I {verb} to see the {obj_phrase}."
            return f"I {verb} {obj_phrase}."

        # C. Actions: EAT / DRINK / PLAY / STUDY / GO / COOK
        if "EAT" in tokens:
            foods = [t.lower().replace("_", " ") for t in tokens if t not in {"EAT", "ME", "I", "YOU"}]
            if foods:
                return f"I want to eat {' '.join(foods)}."
            return "Time to eat."

        if "DRINK" in tokens:
            drinks = [t.lower().replace("_", " ") for t in tokens if t not in {"DRINK", "ME", "I", "YOU"}]
            if drinks:
                return f"I am drinking {' '.join(drinks)}."
            return "I would like a drink."

        if "COOK" in tokens:
            foods = [t.lower().replace("_", " ") for t in tokens if t not in {"COOK", "ME", "I"}]
            if foods:
                return f"I am cooking {' '.join(foods)}."
            return "I am cooking dinner."

        if "GO" in tokens:
            places = [t.lower().replace("_", " ") for t in tokens if t not in {"GO", "ME", "I", "YOU"}]
            if "BATHROOM" in tokens:
                return "I need to go to the bathroom."
            if "HOME" in tokens:
                return "I am going home."
            if places:
                return f"Let's go to the {' '.join(places)}."
            return "Let's go."

        # D. Possessive (MY / YOUR) + Noun
        if ("MY" in tokens or "ME" in tokens) and any(p in tokens for p in self.people):
            rel = [t.lower().replace("_", " ") for t in tokens if t in self.people and t not in {"ME", "MY", "I"}]
            if rel:
                return f"This is my {rel[0]}."

        if ("MY" in tokens or "ME" in tokens) and any(a in tokens for a in self.animals):
            pet = [t.lower() for t in tokens if t in self.animals]
            if pet:
                return f"This is my {pet[0]}."

        # E. Feelings: HAPPY / SAD / GOOD / BAD / SO_SO
        for f in self.feelings:
            if f in tokens:
                f_name = f.lower().replace("_", "-")
                if "ME" in tokens or "I" in tokens:
                    return f"I feel {f_name} today."
                return f"Feeling {f_name}."

        # F. Adjective + Noun (Color / Quality)
        if n == 2 and tokens[0] in self.colors:
            return f"A {tokens[0].lower()} {tokens[1].lower().replace('_', ' ')}."
        if n == 2 and tokens[0] in {"DARK", "HOT", "COOL", "FINE", "FULL", "TALL", "SHORT", "THIN"}:
            return f"It is very {tokens[0].lower()}."

        # G. Emergency / Alert
        if "HELP" in tokens or "ACCIDENT" in tokens:
            return "Please help me, emergency!"

        # H. Single Word Enhancements
        if n == 1:
            w = tokens[0]
            if w in self.idiom_rules:
                return self.idiom_rules[w]
            clean_word = w.lower().replace("_", " ").capitalize()
            return f"{clean_word}."

        # I. General Fluent Fallback
        words = [t.lower().replace("_", " ") for t in tokens]
        sentence = " ".join(words).capitalize()
        if not sentence.endswith((".", "?", "!")):
            sentence += "."
        return sentence
