# Rule-Based AI Chatbot

A deterministic, rule-based chatbot built with pure Python control flow (if-else logic and dictionary lookups) — no machine learning, no external APIs. Built as Project 1 of the DecodeLabs AI Industrial Training Kit (Batch 2026).

## Overview

Before building systems that learn from data, this project focuses on the fundamentals: control flow, decision-making logic, and deterministic response generation. Every reply is fully traceable — Input → Logic → Output — with zero hallucination risk.

## Features

**Core**
- Continuous conversation loop with a clean exit command
- Input sanitization (case-insensitive, whitespace-trimmed)
- Dictionary-based knowledge base with 12 intents
- Fallback response for unrecognized input

**Extras**
- Context memory — remembers the user's name and the topic of the previous turn, and replies accordingly
- Response variety — multiple phrasings per intent instead of repetitive replies
- Rock-Paper-Scissors mini-game with a running scoreboard, resolved entirely through if-elif logic
- Time-of-day aware greeting
- Optional typing effect for a more natural feel

## Tech Stack

- Python 3.10+
- Standard library only (`random`, `time`, `datetime`, `typing`)
- `pytest` for automated testing

## Getting Started

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# Run the chatbot
python rule_based_chatbot.py
```

## Usage

```
DecodeBot: Good afternoon! Type 'exit' or 'quit' to end the chat, or 'help' to see what I can do.

You: hello
DecodeBot: Hi there! I'm DecodeBot. How can I help you today?
You: my name is Lily
DecodeBot: Nice to meet you, Lily! I'll remember that.
You: rock
DecodeBot: You chose rock, I chose paper. I win this round! (Score — You: 0, Me: 1)
You: bye
DecodeBot: Goodbye, Lily! We chatted for 4 turns today. Thanks for stopping by!
```

Type `help` at any time to see the full list of supported commands.

## Running Tests

```bash
pip install pytest
pytest test_rule_based_chatbot.py -v
```

26 automated tests cover the core requirements (input loop, sanitization, knowledge base, fallback, exit handling) as well as the context-memory and mini-game logic.

## Project Structure

```
.
├── rule_based_chatbot.py       # Chatbot implementation
├── test_rule_based_chatbot.py  # Pytest test suite
└── README.md
```

## Architecture

| Component | Responsibility |
|---|---|
| `sanitize()` | Normalizes raw input (lowercase, strip whitespace) |
| `match_intent()` | Core decision engine — checks the mini-game, name capture, context follow-up, and knowledge base in order, then falls back to a default response |
| `play_rps()` | Resolves a Rock-Paper-Scissors round using if-elif logic |
| `run()` | Continuous input loop and exit handling |

## Author

Built by Lily as part of the AI Agent Fellowship 2026, under DecodeLabs' AI Industrial Training Kit.
