"""
Project 1: Rule-Based AI Chatbot
AI Industrial Training Kit (Batch 2026)

A rule-based chatbot: continuous input loop, sanitization, a dictionary
knowledge base (5+ intents), fallback for unknown input, exit command,
context memory (remembers name + last topic), and a rock-paper-scissors
mini-game.
"""

from __future__ import annotations
import random
import time
from datetime import datetime
from typing import Dict, List, Optional


class RuleBasedChatBot:
    """A rule-based chatbot with a knowledge base, context memory,
    a rock-paper-scissors mini-game, and a lightweight personality layer."""

    RPS_CHOICES: List[str] = ["rock", "paper", "scissors"]

    def __init__(self, bot_name: str = "DecodeBot", typing_effect: bool = True) -> None:
        self.bot_name: str = bot_name
        self.typing_effect: bool = typing_effect
        self.exit_commands: set[str] = {"exit", "quit", "bye", "goodbye"}

        # ---- Context memory (nested-condition smart replies) ----
        self.user_name: Optional[str] = None
        self.last_intent: Optional[str] = None
        self.turn_count: int = 0

        # ---- Mini-game scoreboard ----
        self.user_score: int = 0
        self.bot_score: int = 0

        # ---- Knowledge Base: intent -> list of possible replies ----
        self.knowledge_base: Dict[str, List[str]] = {
            "hello": [
                "Hi there! I'm DecodeBot 🤖. How can I help you today?",
                "Hey! Ready to chat?",
                "Hello hello! What's on your mind?",
            ],
            "hi": ["Hello! Great to see you.", "Hi there! 👋"],
            "how are you": [
                "I'm just a bunch of if-else statements, but I'm doing great! How about you?",
                "Running smoothly on pure logic today. You?",
            ],
            "your name": [f"I'm {self.bot_name}, a rule-based chatbot built for DecodeLabs."],
            "what can you do": [
                "I can greet you, chat a little, remember your name, play rock-paper-scissors, "
                "and reply to basic commands. Pure logic, no machine learning!"
            ],
            "help": [
                "Try: hello, how are you, your name, what can you do, joke, weather, "
                "score, or bye. To play a game, just type ONE word: rock, paper, or scissors."
            ],
            "thanks": ["You're welcome! Happy to help.", "Anytime, glad I could help!"],
            "thank you": ["Anytime!"],
            "joke": [
                "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
                "Why did the developer go broke? They used up all their cache. 💸",
            ],
            "weather": [
                "I can't check the weather (no internet access), but I hope it's sunny wherever you are!"
            ],
            "who made you": [
                "I was built by an AI Engineering intern at DecodeLabs, as Project 1 of the training kit."
            ],
            "bored": [
                "Let's fix that! Try saying 'joke', or type 'rock', 'paper', or 'scissors' for a quick game."
            ],
        }
        self.default_response: str = (
            "I do not understand that yet. Type 'help' to see what I can do."
        )

    # ------------------------------------------------------------------
    # PHASE 1: SANITIZATION
    # ------------------------------------------------------------------
    def sanitize(self, raw_input: str) -> str:
        """Normalize user input: lowercase + strip whitespace."""
        return raw_input.lower().strip()

    def is_exit_command(self, clean_input: str) -> bool:
        """Check for the kill command that breaks the infinite loop."""
        return clean_input in self.exit_commands

    def _time_based_greeting(self) -> str:
        """Pick a greeting phrase based on the current hour (pure control flow)."""
        hour = datetime.now().hour
        if hour < 12:
            return "Good morning"
        elif hour < 18:
            return "Good afternoon"
        return "Good evening"

    def _speak(self, message: str) -> None:
        """Print the bot's message, with an optional character-by-character
        typing effect for personality."""
        print(f"{self.bot_name}: ", end="", flush=True)
        if self.typing_effect:
            for ch in message:
                print(ch, end="", flush=True)
                time.sleep(0.012)
            print()
        else:
            print(message)

    # ------------------------------------------------------------------
    # MINI-GAME: Rock-Paper-Scissors (decided entirely by if-elif logic)
    # ------------------------------------------------------------------
    def play_rps(self, user_choice: str) -> str:
        bot_choice = random.choice(self.RPS_CHOICES)

        if user_choice == bot_choice:
            result = "It's a tie!"
        elif (user_choice, bot_choice) in {
            ("rock", "scissors"),
            ("paper", "rock"),
            ("scissors", "paper"),
        }:
            result = "You win this round! 🎉"
            self.user_score += 1
        else:
            result = "I win this round! 🤖"
            self.bot_score += 1

        return (
            f"You chose {user_choice}, I chose {bot_choice}. {result} "
            f"(Score — You: {self.user_score}, Me: {self.bot_score})"
        )

    # ------------------------------------------------------------------
    # PHASE 2: DECISION LOGIC (with nested / context-aware conditions)
    # ------------------------------------------------------------------
    def match_intent(self, clean_input: str) -> str:
        """Decide a reply for the sanitized input. Checks run most-specific
        first: mini-game > score check > name capture > personalized
        greeting > context follow-up > generic keyword scan > fallback."""
        if not clean_input:
            return "Say something! I'm listening."

        # --- Mini-game trigger ---
        if clean_input in self.RPS_CHOICES:
            self.last_intent = "rps"
            return self.play_rps(clean_input)

        # --- Helpful nudge if they typed a combined/garbled game phrase ---
        rps_mentioned = any(word in clean_input for word in self.RPS_CHOICES)
        if rps_mentioned or "rps" in clean_input:
            return "Let's play! Just type ONE word — rock, paper, or scissors — and hit enter."

        # --- Score check ---
        if clean_input == "score":
            return f"Current score — You: {self.user_score}, Me: {self.bot_score}."

        # --- Nested condition A: capture the user's name (context memory) ---
        if "my name is" in clean_input:
            name_part = clean_input.split("my name is", 1)[1].strip()
            if name_part:
                self.user_name = name_part.split()[0].capitalize()
                self.last_intent = "name_capture"
                return f"Nice to meet you, {self.user_name}! I'll remember that. 😊"

        # --- Nested condition B: personality-flavored, name-aware greeting ---
        if clean_input in {"hello", "hi", "hey"}:
            self.last_intent = "greeting"
            if self.user_name:
                return f"Hey {self.user_name}! Good to see you again. 👋"
            options = self.knowledge_base.get(clean_input, self.knowledge_base["hello"])
            return random.choice(options)

        # --- Nested condition C: follow-up reply based on the LAST topic ---
        if self.last_intent == "how_are_you":
            if clean_input in {"good", "great", "fine", "well", "awesome"}:
                self.last_intent = None
                return "Awesome! That's what I like to hear. 😊"
            if clean_input in {"bad", "sad", "tired", "not good"}:
                self.last_intent = None
                return "Aw, sorry to hear that. Hope things get better soon!"

        # --- Standard keyword scan (control-flow / decision-making logic) ---
        for keyword, responses in self.knowledge_base.items():
            if keyword in clean_input:
                self.last_intent = "how_are_you" if keyword == "how are you" else keyword
                return random.choice(responses)

        # --- Atomic lookup + fallback (the "professional .get() approach") ---
        self.last_intent = None
        return random.choice(self.knowledge_base.get(clean_input, [self.default_response]))

    # ------------------------------------------------------------------
    # PHASE 3: THE HEARTBEAT — infinite loop until the kill command
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Run the chatbot in a continuous loop until an exit command is received."""
        greeting = self._time_based_greeting()
        print(
            f"{self.bot_name}: {greeting}! I'm your friendly rule-based bot. "
            "Type 'exit' or 'quit' to end the chat, or 'help' to see what I can do.\n"
        )

        while True:
            raw_input_text = input("You: ")
            self.turn_count += 1
            clean_input = self.sanitize(raw_input_text)

            if self.is_exit_command(clean_input):
                name_bit = f", {self.user_name}" if self.user_name else ""
                self._speak(
                    f"Goodbye{name_bit}! We chatted for {self.turn_count} turns today. "
                    "Thanks for stopping by! 👋"
                )
                break

            reply = self.match_intent(clean_input)
            self._speak(reply)

            if self.turn_count % 5 == 0:
                self._speak(
                    "(Psst — we've been chatting a while! Try saying 'joke', or just "
                    "type rock, paper, or scissors (one word, no quotes) to mix things up.)"
                )


if __name__ == "__main__":
    bot = RuleBasedChatBot(bot_name="DecodeBot")
    bot.run()
