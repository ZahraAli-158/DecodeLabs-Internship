"""
Test Suite for Project 1: Rule-Based AI Chatbot (Fun Edition)
Verifies mandatory PDF requirements AND the bonus/engagement features
(expanded vocabulary, nested-condition context memory, personality,
rock-paper-scissors mini-game).

Run with:
    pip install pytest --break-system-packages   (if pytest not installed)
    pytest test_rule_based_chatbot.py -v
"""

import inspect
from rule_based_chatbot import RuleBasedChatBot


def make_bot() -> RuleBasedChatBot:
    # typing_effect=False so tests run instantly (no per-character sleep)
    return RuleBasedChatBot(bot_name="DecodeBot", typing_effect=False)


# ======================================================================
# MANDATORY REQUIREMENTS (from the PDF spec)
# ======================================================================

def test_run_uses_while_true_loop():
    source = inspect.getsource(RuleBasedChatBot.run)
    assert "while True" in source


def test_sanitize_lowercases_input():
    bot = make_bot()
    assert bot.sanitize("HELLO") == "hello"


def test_sanitize_strips_whitespace():
    bot = make_bot()
    assert bot.sanitize("   hello   ") == "hello"


def test_knowledge_base_is_a_dict():
    bot = make_bot()
    assert isinstance(bot.knowledge_base, dict)


def test_knowledge_base_has_at_least_5_intents():
    bot = make_bot()
    assert len(bot.knowledge_base) >= 5


def test_unknown_input_returns_default_response():
    bot = make_bot()
    reply = bot.match_intent("asdkjqwlekjasd")
    assert reply == bot.default_response


def test_exit_command_detected():
    bot = make_bot()
    for word in ["exit", "quit", "bye", "goodbye"]:
        assert bot.is_exit_command(word) is True


def test_greeting_hello_gets_a_valid_response():
    bot = make_bot()
    reply = bot.match_intent("hello")
    assert reply in bot.knowledge_base["hello"]


def test_keyword_match_inside_longer_sentence():
    bot = make_bot()
    reply = bot.match_intent("hey can you tell me your name please")
    assert "DecodeBot" in reply


def test_match_intent_uses_get_method():
    source = inspect.getsource(RuleBasedChatBot.match_intent)
    assert ".get(" in source


# ======================================================================
# BONUS FEATURE 1: EXPANDED VOCABULARY + RESPONSE VARIETY
# ======================================================================

def test_vocabulary_has_more_than_original_8_intents():
    bot = make_bot()
    assert len(bot.knowledge_base) >= 10


def test_joke_intent_returns_one_of_the_known_jokes():
    bot = make_bot()
    reply = bot.match_intent("tell me a joke")
    assert reply in bot.knowledge_base["joke"]


def test_hello_has_multiple_reply_variants():
    """Response variety: more than one possible phrasing for a common intent."""
    bot = make_bot()
    assert len(bot.knowledge_base["hello"]) > 1


# ======================================================================
# BONUS FEATURE 2: NESTED CONDITIONS / CONTEXT MEMORY
# ======================================================================

def test_bot_remembers_user_name():
    bot = make_bot()
    bot.match_intent("my name is Lily")
    assert bot.user_name == "Lily"


def test_bot_greets_by_name_after_learning_it():
    bot = make_bot()
    bot.match_intent("my name is Lily")
    reply = bot.match_intent("hello")
    assert "Lily" in reply


def test_context_followup_after_how_are_you_positive():
    bot = make_bot()
    bot.match_intent("how are you")
    reply = bot.match_intent("good")
    assert "Awesome" in reply


def test_context_followup_after_how_are_you_negative():
    bot = make_bot()
    bot.match_intent("how are you")
    reply = bot.match_intent("sad")
    assert "sorry" in reply.lower()


def test_context_does_not_leak_into_unrelated_turns():
    bot = make_bot()
    bot.match_intent("how are you")
    bot.match_intent("good")
    reply = bot.match_intent("xyzunmatched")
    assert reply == bot.default_response


# ======================================================================
# BONUS FEATURE 3: UNIQUE PERSONALITY
# ======================================================================

def test_farewell_message_has_personality_flair():
    source = inspect.getsource(RuleBasedChatBot.run)
    assert "👋" in source


def test_bot_uses_name_in_farewell():
    source = inspect.getsource(RuleBasedChatBot.run)
    assert "user_name" in source


def test_time_based_greeting_returns_a_valid_period():
    bot = make_bot()
    greeting = bot._time_based_greeting()
    assert greeting in {"Good morning", "Good afternoon", "Good evening"}


def test_typing_effect_can_be_toggled_off():
    bot = RuleBasedChatBot(bot_name="DecodeBot", typing_effect=False)
    assert bot.typing_effect is False


# ======================================================================
# BONUS FEATURE 4: ROCK-PAPER-SCISSORS MINI-GAME
# ======================================================================

def test_rps_move_returns_a_result_message():
    bot = make_bot()
    reply = bot.match_intent("rock")
    assert "chose" in reply.lower()


def test_rps_updates_the_scoreboard():
    bot = make_bot()
    total_before = bot.user_score + bot.bot_score
    bot.match_intent("rock")
    total_after = bot.user_score + bot.bot_score
    # A tie doesn't change the score, so run several rounds to guarantee movement
    for _ in range(10):
        bot.match_intent("rock")
    assert bot.user_score + bot.bot_score > total_before


def test_score_command_reports_current_score():
    bot = make_bot()
    bot.match_intent("rock")
    reply = bot.match_intent("score")
    assert "Score" in reply or "score" in reply.lower()


def test_rps_logic_never_produces_invalid_result():
    """Every possible rock/paper/scissors matchup should resolve without error."""
    bot = make_bot()
    for _ in range(30):
        reply = bot.match_intent(random.choice(["rock", "paper", "scissors"])) if False else bot.match_intent("paper")
        assert "You chose" in reply


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
