"""
SignVision Vocabulary Registry
==============================
Defines the canonical 100-word vocabulary across 10 categories:
Common Signs, Home, Food, People, Questions, School, Time, Family,
Colors & Animals, and Descriptions & Conversational.
"""

from typing import Dict, List, Optional

# Structured vocabulary organized by category
VOCABULARY_CATEGORIES: Dict[str, List[Dict[str, str]]] = {
    "Common Signs": [
        {"id": "fine", "display": "Fine"},
        {"id": "finish", "display": "Finish"},
        {"id": "forget", "display": "Forget"},
        {"id": "go", "display": "Go"},
        {"id": "hello", "display": "Hello"},
        {"id": "help", "display": "Help"},
        {"id": "dont_want", "display": "Don't Want"},
        {"id": "like", "display": "Like"},
        {"id": "more", "display": "More"},
        {"id": "need", "display": "Need"},
        {"id": "wrong", "display": "Wrong"},
        {"id": "no", "display": "No"},
        {"id": "not", "display": "Not"},
        {"id": "please", "display": "Please"},
        {"id": "right", "display": "Right"},
        {"id": "thank_you", "display": "Thank-You"},
        {"id": "want", "display": "Want"},
        {"id": "yes", "display": "Yes"},
    ],
    "Home": [
        {"id": "bathroom", "display": "Bathroom"},
        {"id": "car", "display": "Car"},
        {"id": "home", "display": "Home"},
        {"id": "kitchen", "display": "Kitchen"},
        {"id": "living_room", "display": "Living Room"},
        {"id": "phone", "display": "Phone"},
    ],
    "Food": [
        {"id": "cheese", "display": "Cheese"},
        {"id": "cook", "display": "Cook"},
        {"id": "cookies", "display": "Cookies"},
        {"id": "drink", "display": "Drink"},
        {"id": "eat", "display": "Eat / Food"},
        {"id": "hamburger", "display": "Hamburger"},
        {"id": "milk", "display": "Milk"},
        {"id": "sandwich", "display": "Sandwich"},
    ],
    "People": [
        {"id": "me", "display": "I / Me"},
        {"id": "my", "display": "My / Mine"},
        {"id": "you", "display": "You"},
        {"id": "your", "display": "Your"},
        {"id": "boy", "display": "Boy"},
        {"id": "girl", "display": "Girl"},
        {"id": "man", "display": "Man"},
        {"id": "woman", "display": "Woman"},
        {"id": "deaf", "display": "Deaf"},
        {"id": "hearing", "display": "Hearing"},
    ],
    "Questions": [
        {"id": "how", "display": "How"},
        {"id": "what", "display": "What"},
        {"id": "when", "display": "When"},
        {"id": "where", "display": "Where"},
        {"id": "who", "display": "Who"},
        {"id": "why", "display": "Why"},
    ],
    "School": [
        {"id": "asl", "display": "ASL"},
        {"id": "book", "display": "Book"},
        {"id": "class", "display": "Class"},
        {"id": "know", "display": "Know"},
        {"id": "learn", "display": "Learn"},
        {"id": "sentence", "display": "Sentence"},
        {"id": "sign", "display": "Sign"},
        {"id": "student", "display": "Student"},
        {"id": "teacher", "display": "Teacher"},
    ],
    "Time": [
        {"id": "afternoon", "display": "Afternoon"},
        {"id": "day", "display": "Day"},
        {"id": "hour", "display": "Hour"},
        {"id": "minute", "display": "Minute"},
        {"id": "month", "display": "Month"},
        {"id": "morning", "display": "Morning"},
        {"id": "night", "display": "Night"},
        {"id": "noon", "display": "Noon"},
        {"id": "time", "display": "Time"},
        {"id": "week", "display": "Week"},
        {"id": "year", "display": "Year"},
    ],
    "Family": [
        {"id": "baby", "display": "Baby"},
        {"id": "brother", "display": "Brother"},
        {"id": "children", "display": "Children"},
        {"id": "family", "display": "Family"},
        {"id": "father", "display": "Father"},
        {"id": "grandfather", "display": "Grandfather"},
        {"id": "grandmother", "display": "Grandmother"},
        {"id": "husband", "display": "Husband"},
        {"id": "mother", "display": "Mother"},
        {"id": "sister", "display": "Sister"},
        {"id": "wife", "display": "Wife"},
    ],
    "Colors & Animals": [
        {"id": "dog", "display": "Dog"},
        {"id": "cat", "display": "Cat"},
        {"id": "blue", "display": "Blue"},
        {"id": "green", "display": "Green"},
        {"id": "orange", "display": "Orange"},
        {"id": "pink", "display": "Pink"},
        {"id": "purple", "display": "Purple"},
        {"id": "red", "display": "Red"},
        {"id": "yellow", "display": "Yellow"},
    ],
    "Descriptions & Conversational": [
        {"id": "so_so", "display": "So-So"},
        {"id": "good", "display": "Good"},
        {"id": "happy", "display": "Happy"},
        {"id": "sad", "display": "Sad"},
        {"id": "bad", "display": "Bad"},
        {"id": "chat", "display": "Chat"},
        {"id": "signing", "display": "Signing"},
        {"id": "look", "display": "Look / See / Watch"},
        {"id": "pay_attention", "display": "Pay Attention"},
        {"id": "dont", "display": "Not / Don't / Doesn't"},
        {"id": "not_yet", "display": "Not Yet"},
        {"id": "bye", "display": "Bye"},
    ],
}

# Flat list of all 100 action IDs in order
ACTION_IDS: List[str] = [
    item["id"]
    for cat_list in VOCABULARY_CATEGORIES.values()
    for item in cat_list
]

# Quick lookup map: id -> info dict
ACTION_MAP: Dict[str, Dict[str, str]] = {}
for category, items in VOCABULARY_CATEGORIES.items():
    for item in items:
        ACTION_MAP[item["id"]] = {
            "id": item["id"],
            "display": item["display"],
            "category": category,
        }

# Common alias mappings to tolerate alternate tokens
GLOSS_ALIASES: Dict[str, str] = {
    "eat_food": "eat",
    "food": "eat",
    "i_me": "me",
    "i": "me",
    "my_mine": "my",
    "mine": "my",
    "thanks": "thank_you",
    "thankyou": "thank_you",
    "look_see_watch": "look",
    "see": "look",
    "watch": "look",
    "not_dont_doesnt": "dont",
    "doesnt": "dont",
    "iloveyou": "hello",
    "not_want": "dont_want",
}


def get_display_name(action_id: str) -> str:
    """Returns human-readable display string for an action ID."""
    norm = action_id.lower().strip()
    norm = GLOSS_ALIASES.get(norm, norm)
    if norm in ACTION_MAP:
        return ACTION_MAP[norm]["display"]
    return action_id.replace("_", " ").title()


def get_category(action_id: str) -> str:
    """Returns the category for an action ID."""
    norm = action_id.lower().strip()
    norm = GLOSS_ALIASES.get(norm, norm)
    if norm in ACTION_MAP:
        return ACTION_MAP[norm]["category"]
    return "General"


def normalize_action_id(token: str) -> str:
    """Normalizes an input token (case-insensitive, underscores/hyphens) to standard ID."""
    t = str(token).lower().strip().replace("-", "_").replace(" ", "_").replace("/", "_")
    return GLOSS_ALIASES.get(t, t)
