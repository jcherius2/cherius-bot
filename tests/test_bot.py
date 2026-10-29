"""Offline tests: the LLM and the Metaculus API are simulated, no network call is made."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecasting_tools import (  # noqa: E402
    BinaryPrediction,
    BinaryQuestion,
    DateQuestion,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    Percentile,
)
from forecasting_tools.ai_models.ai_utils.response_types import TextTokenCostResponse  # noqa: E402
from forecasting_tools.ai_models.resource_managers.hard_limit_manager import HardLimitExceededError  # noqa: E402
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import MonetaryCostManager  # noqa: E402
from forecasting_tools.data_models.questions import DiscreteQuestion, QuestionState  # noqa: E402

import main as bot_module  # noqa: E402

RESEARCH_TEXT = "Briefing: the latest value is 42 (source: https://example.org, 2026-10-01)."
PERCENTILE_VALUES = [3, 5, 10, 20, 30, 45, 60, 70]


def numeric_answer(values=PERCENTILE_VALUES) -> str:
    lines = [f"Percentile {p}: {v}" for p, v in zip(bot_module.NUMERIC_PERCENTILES, values)]
    return "Reasoning mentions 50% and 12 items.\nFINAL ANSWER\n" + "\n".join(lines)


def binary_question(**kwargs) -> BinaryQuestion:
    defaults = dict(
        question_text="Will X happen before 2027?",
        id_of_post=1,
        id_of_question=11,
        page_url="https://www.metaculus.com/questions/1/",
        resolution_criteria="Resolves Yes if X happens.",
    )
    return BinaryQuestion(**{**defaults, **kwargs})


def numeric_question(cls=NumericQuestion) -> NumericQuestion:
    return cls(
        question_text="How many Y will there be?",
        id_of_post=2,
        id_of_question=12,
        page_url="https://www.metaculus.com/questions/2/",
        upper_bound=100,
        lower_bound=0,
        open_upper_bound=True,
        open_lower_bound=False,
        unit_of_measure="units",
    )


def multiple_choice_question() -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        question_text="Which option wins?",
        id_of_post=3,
        id_of_question=13,
        page_url="https://www.metaculus.com/questions/3/",
        options=["Alpha", "Beta Gamma", "0-5"],
    )


def date_question() -> DateQuestion:
    return DateQuestion(
        question_text="When will Z happen?",
        id_of_post=4,
        id_of_question=14,
        page_url="https://www.metaculus.com/questions/4/",
        lower_bound=datetime(2026, 1, 1, tzinfo=timezone.utc),
        upper_bound=datetime(2030, 1, 1, tzinfo=timezone.utc),
        open_upper_bound=True,
        open_lower_bound=False,
    )


DATE_VALUES = ["2026-12-01", "2027-01-15", "2027-03-01", "2027-06-01", "2027-09-01", "2028-01-01", "2028-06-01", "2029-01-01"]
DATE_ANSWER = "Reasoning.\nFINAL ANSWER\n" + "\n".join(
    f"Percentile {p}: {v}" for p, v in zip(bot_module.NUMERIC_PERCENTILES, DATE_VALUES)
)
CALL_TOKENS = (1000, 500)  # prompt and completion tokens reported by the fake model, with no cost
CHEAP = bot_module.PROFILES["cheap"]  # default profile: its fallback prices price CALL_TOKENS


class FakeLlm:
    """Replaces the single network hook of GeneralLlm: canned answers, token counts, no cost."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.binary_answers = ["I doubt the 70% figure.\nFINAL ANSWER\nProbability: 35%"]
        self.multiple_choice_answer = (
            "Reasoning.\nFINAL ANSWER\nAlpha: 60%\nBeta Gamma: 30%\n0-5: 10%"
        )
        self.numeric_answer = numeric_answer()

    def install(self, monkeypatch) -> "FakeLlm":
        fake = self

        def answer(llm, prompt) -> str:
            fake.calls.append((llm.model, prompt))
            if llm.model.endswith(":online"):
                return RESEARCH_TEXT
            if "Probability: ZZ%" in prompt:
                index = len([c for c in fake.calls if "Probability: ZZ%" in c[1]]) - 1
                return fake.binary_answers[index % len(fake.binary_answers)]
            if "Percentile 5: XX" in prompt:
                return fake.numeric_answer
            if "Percentile 5: YYYY-MM-DD" in prompt:
                return DATE_ANSWER
            return fake.multiple_choice_answer

        async def direct_call(llm, prompt):
            return TextTokenCostResponse(
                data=answer(llm, str(prompt)),
                prompt_tokens_used=CALL_TOKENS[0],
                completion_tokens_used=CALL_TOKENS[1],
                total_tokens_used=sum(CALL_TOKENS),
                model=llm.model,
                cost=0,
            )

        monkeypatch.setattr(GeneralLlm, "_mockable_direct_call_to_model", direct_call)
        return self


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """Any real network call fails the test; no model override leaks in from the environment."""

    def blocked(*args, **kwargs):
        raise AssertionError("Network call attempted in an offline test")

    import litellm
    import requests

    monkeypatch.setattr(litellm, "acompletion", blocked)
    monkeypatch.setattr(requests, "post", blocked)
    monkeypatch.setattr(requests, "get", blocked)
    for name in list(os.environ):
        if name.startswith(("BOT_", "LLM_", "OPENAI_", "ANTHROPIC_")) or name in (
            "PREDICTIONS_PER_QUESTION",
            "MAX_QUESTIONS_PER_RUN",
        ):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BOT_LEDGER_PATH", str(tmp_path / "ledger.json"))


@pytest.fixture
def llm(monkeypatch) -> FakeLlm:
    return FakeLlm().install(monkeypatch)


def make_bot(publish: bool, skip: bool = True):
    client = MagicMock()
    return bot_module.build_bot(publish=publish, skip_previously_forecasted=skip, metaculus_client=client), client


def test_every_model_caps_its_output(monkeypatch):
    monkeypatch.setenv("LLM_MAX_TOKENS", "1234")
    bot, _ = make_bot(publish=False)
    for role in ("default", "researcher", "parser", "summarizer"):
        assert bot.get_llm(role, "llm").litellm_kwargs["max_tokens"] == 1234, role


def test_research_search_is_capped_by_default(monkeypatch):
    bot, _ = make_bot(publish=False)
    extra = bot.get_llm("researcher", "llm").litellm_kwargs["extra_body"]
    assert extra == {"plugins": [{"id": "web", "engine": "exa", "max_results": 3}]}
    assert "extra_body" not in bot.get_llm("default", "llm").litellm_kwargs
    monkeypatch.setenv("BOT_SEARCH_RESULTS", "5")
    assert bot_module.web_search_plugin()["max_results"] == 5
    monkeypatch.setenv("BOT_SEARCH_RESULTS", "0")
    assert bot_module.web_search_plugin()["max_results"] == 1
    monkeypatch.setenv("BOT_SEARCH_ENGINE", "native")
    assert bot_module.web_search_plugin() is None
    assert "extra_body" not in make_bot(publish=False)[0].get_llm("researcher", "llm").litellm_kwargs
    monkeypatch.delenv("BOT_SEARCH_ENGINE")
    monkeypatch.setenv("BOT_RESEARCH_MODEL", "openrouter/other/searcher")  # no ":online": not touched
    assert bot_module.web_search_plugin() is None


def published_calls(client) -> list[str]:
    return [name for name, _, _ in client.method_calls if name.startswith("post_")]


# ------------------------------- parsing -------------------------------- #
@pytest.mark.parametrize(
    "text, expected",
    [
        ("I doubt the 70% figure.\nFINAL ANSWER\nProbability: 35%", 0.35),
        ("FINAL ANSWER\nProbability: 100%", 0.98),
        ("FINAL ANSWER\nProbability: 0%", 0.02),
        ("FINAL ANSWER\nProbability: 0.4%", 0.02),
        ("FINAL ANSWER\nProbability: 99.9%", 0.98),
    ],
)
def test_binary_parsing_and_bounds(text, expected):
    assert bot_module.parse_binary(text) == pytest.approx(expected)


def test_binary_parsing_fails_without_a_number():
    with pytest.raises(ValueError):
        bot_module.parse_binary("I cannot say.")


@pytest.mark.parametrize("raw", [-1, 0, 0.001, 0.5, 0.999, 1, 7])
def test_clamp_never_returns_zero_or_one(raw):
    assert 0.02 <= bot_module.clamp_probability(raw) <= 0.98


def test_clamp_rejects_nan():
    with pytest.raises(ValueError):
        bot_module.clamp_probability(float("nan"))


def test_multiple_choice_sums_to_one():
    question = multiple_choice_question()
    prediction = bot_module.parse_multiple_choice(
        "FINAL ANSWER\nAlpha: 60%\nBeta Gamma: 30%\n0-5: 10%", question.options
    )
    probabilities = {o.option_name: o.probability for o in prediction.predicted_options}
    assert probabilities == pytest.approx({"Alpha": 0.6, "Beta Gamma": 0.3, "0-5": 0.1})
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_multiple_choice_never_gives_zero_or_one():
    question = multiple_choice_question()
    prediction = bot_module.parse_multiple_choice(
        "FINAL ANSWER\nAlpha: 100%\nBeta Gamma: 0%\n0-5: 0%", question.options
    )
    probabilities = [o.probability for o in prediction.predicted_options]
    assert all(0 < p < 1 for p in probabilities)
    assert sum(probabilities) == pytest.approx(1.0)


def test_multiple_choice_rejects_a_sum_far_from_one():
    with pytest.raises(ValueError):
        bot_module.parse_multiple_choice(
            "FINAL ANSWER\nAlpha: 50%\nBeta Gamma: 30%\n0-5: 10%", multiple_choice_question().options
        )


def test_numeric_percentiles_are_increasing():
    distribution = bot_module.parse_numeric(numeric_answer(), numeric_question())
    declared = distribution.declared_percentiles
    assert [p.percentile for p in declared] == pytest.approx([p / 100 for p in bot_module.NUMERIC_PERCENTILES])
    assert [p.value for p in declared] == PERCENTILE_VALUES
    assert all(a.value < b.value for a, b in zip(declared, declared[1:]))
    cdf = [p.percentile for p in distribution.cdf]
    assert len(cdf) == 201
    assert all(a <= b for a, b in zip(cdf, cdf[1:]))


def test_numeric_rejects_non_increasing_values():
    with pytest.raises(ValueError):
        bot_module.parse_numeric(numeric_answer([3, 5, 10, 4, 30, 45, 60, 70]), numeric_question())
    with pytest.raises(ValueError):
        bot_module.check_percentiles(
            [Percentile(percentile=0.1, value=5), Percentile(percentile=0.5, value=5), Percentile(percentile=0.9, value=9)]
        )


def test_check_percentiles_sorts_by_percentile():
    ordered = bot_module.check_percentiles(
        [Percentile(percentile=0.9, value=9), Percentile(percentile=0.1, value=1), Percentile(percentile=0.5, value=5)]
    )
    assert [p.percentile for p in ordered] == [0.1, 0.5, 0.9]


# ------------------------------- prompts -------------------------------- #
def test_prompts_give_the_date_and_never_use_the_community_prediction():
    bot = bot_module.CheriusBot
    today = bot_module.today_utc()
    prompts = [
        bot.binary_prompt(binary_question(), RESEARCH_TEXT),
        bot.multiple_choice_prompt(multiple_choice_question(), RESEARCH_TEXT),
        bot.numeric_prompt(numeric_question(), RESEARCH_TEXT),
    ]
    for prompt in prompts:
        assert today in prompt
        assert "not been resolved" in prompt
        assert "Base rate" in prompt and "status quo" in prompt
        assert RESEARCH_TEXT in prompt
        assert "community" not in prompt.lower()
    research_prompt = bot.research_prompt(binary_question())
    assert today in research_prompt and "NOT been resolved" in research_prompt
    assert "Do not look up or report the Metaculus community prediction" in research_prompt


def test_model_names_come_from_the_environment(monkeypatch):
    assert bot_module.forecast_model_name() == bot_module.DEFAULT_MODEL == bot_module.PROFILES["cheap"]["model"]
    assert bot_module.research_model_name() == bot_module.DEFAULT_MODEL + ":online"
    assert make_bot(publish=False)[0].predictions_per_research_report == 2
    monkeypatch.setenv("BOT_MODEL", "openrouter/some/model")
    assert bot_module.research_model_name() == "openrouter/some/model:online"
    monkeypatch.setenv("BOT_RESEARCH_MODEL", "openrouter/other/searcher")
    assert bot_module.research_model_name() == "openrouter/other/searcher"
    monkeypatch.setenv("PREDICTIONS_PER_QUESTION", "5")
    bot, _ = make_bot(publish=False)
    assert bot.predictions_per_research_report == 5
    assert bot.get_llm("default", "llm").model == "openrouter/some/model"


# --------------------------- forecast functions -------------------------- #
def test_forecast_on_each_question_type(llm):
    bot, _ = make_bot(publish=False)
    binary = asyncio.run(bot._run_forecast_on_binary(binary_question(), RESEARCH_TEXT))
    assert binary.prediction_value == pytest.approx(0.35)
    choice = asyncio.run(bot._run_forecast_on_multiple_choice(multiple_choice_question(), RESEARCH_TEXT))
    assert sum(o.probability for o in choice.prediction_value.predicted_options) == pytest.approx(1.0)
    numeric = asyncio.run(bot._run_forecast_on_numeric(numeric_question(), RESEARCH_TEXT))
    assert [p.value for p in numeric.prediction_value.declared_percentiles] == PERCENTILE_VALUES
    assert binary.reasoning and choice.reasoning and numeric.reasoning


def test_discrete_question_uses_the_numeric_path(llm):
    bot, _ = make_bot(publish=False)
    question = numeric_question(DiscreteQuestion)
    assert "discrete question" in bot.numeric_prompt(question, RESEARCH_TEXT)
    result = asyncio.run(bot._run_forecast_on_numeric(question, RESEARCH_TEXT))
    values = [p.value for p in result.prediction_value.declared_percentiles]
    assert values == sorted(values)


def test_parser_model_is_the_fallback_and_stays_bounded(llm, monkeypatch):
    llm.binary_answers = ["It will certainly happen."]
    seen = {}

    async def fake_structure_output(text, output_type, **kwargs):
        seen["type"] = output_type
        return BinaryPrediction(prediction_in_decimal=1.0)

    monkeypatch.setattr(bot_module, "structure_output", fake_structure_output)
    bot, _ = make_bot(publish=False)
    result = asyncio.run(bot._run_forecast_on_binary(binary_question(), RESEARCH_TEXT))
    assert seen["type"] is BinaryPrediction
    assert result.prediction_value == pytest.approx(0.98)


# ------------------------------ full pipeline ---------------------------- #
def test_dry_run_publishes_nothing(llm):
    bot, client = make_bot(publish=False)
    questions = [binary_question(), numeric_question(), multiple_choice_question()]
    client.get_all_open_questions_from_tournament.return_value = questions
    assert asyncio.run(bot_module.run(bot, "test")) == (3, 0, 0)
    assert published_calls(client) == []
    client.get_all_open_questions_from_tournament.assert_called_once_with("bot-testing-area")
    research_calls = [c for c in llm.calls if c[0].endswith(":online")]
    assert len(research_calls) == 3  # one research pass per question
    assert len(llm.calls) == 3 + 3 * 2  # plus two independent draws per question (cheap profile)


def test_publishing_posts_one_forecast_and_one_comment_per_question(llm, monkeypatch):
    monkeypatch.setenv("PREDICTIONS_PER_QUESTION", "3")
    llm.binary_answers = [f"FINAL ANSWER\nProbability: {p}%" for p in (30, 40, 90)]
    bot, client = make_bot(publish=True)
    reports = asyncio.run(bot.forecast_questions([binary_question()], return_exceptions=False))
    assert reports[0].prediction == pytest.approx(0.40)  # median of the three draws
    assert sorted(published_calls(client)) == ["post_binary_question_prediction", "post_question_comment"]
    client.post_binary_question_prediction.assert_called_once()
    assert client.post_binary_question_prediction.call_args.args[1] == pytest.approx(0.40)
    comment = client.post_question_comment.call_args.args[1]
    assert comment.startswith("#") and bot_module.AI_DISCLOSURE in comment.split("\n")[1]
    assert "CheriusBot" in comment and RESEARCH_TEXT in comment
    assert "is_private" not in client.post_question_comment.call_args.kwargs  # package default: private


def test_extreme_draws_are_published_inside_the_bounds(llm):
    llm.binary_answers = ["FINAL ANSWER\nProbability: 100%"]
    bot, client = make_bot(publish=True)
    asyncio.run(bot.forecast_questions([binary_question()]))
    assert client.post_binary_question_prediction.call_args.args[1] == pytest.approx(0.98)


def test_already_forecasted_questions_are_never_forecast_again(llm):
    bot, client = make_bot(publish=True, skip=True)
    done = binary_question(id_of_question=21, already_forecasted=True)
    fresh = binary_question(id_of_question=22, already_forecasted=False)
    client.get_all_open_questions_from_tournament.return_value = [done, fresh, fresh]
    assert asyncio.run(bot_module.collect_questions(bot, ["a"])) == [fresh]
    assert asyncio.run(bot_module.run(bot, "tournament")) == (1, 0, 0)
    listed = [call.args[0] for call in client.get_all_open_questions_from_tournament.call_args_list[1:]]
    assert listed == bot_module.TOURNAMENT_IDS == ["fall-futureeval-2026", "minibench", "market-pulse-26q4"]
    assert client.post_binary_question_prediction.call_count == 1
    assert client.post_binary_question_prediction.call_args.args[0] == 22


def test_a_failed_research_fails_the_question_without_publishing(monkeypatch):
    async def invoke(llm, prompt, *args, **kwargs):
        return ""

    monkeypatch.setattr(GeneralLlm, "invoke", invoke)
    bot, client = make_bot(publish=True)
    client.get_all_open_questions_from_tournament.return_value = [binary_question()]
    assert asyncio.run(bot_module.run(bot, "test")) == (0, 1, 0)
    assert published_calls(client) == []


class PaymentRequired(Exception):
    """Shape of the litellm error for an OpenRouter 402 (status_code attribute)."""

    status_code = 402


@pytest.mark.parametrize(
    "error, expected",
    [
        (PaymentRequired("OpenrouterException"), True),
        (ValueError('Errors: [\'{"error":{"message":"can only afford 2899 tokens","code":402}}\']'), True),
        (RuntimeError("Error code: 402 - in_flight_budget_exhausted"), True),
        (ValueError("failed for https://www.metaculus.com/questions/402/ and /40213/"), False),
        (ValueError("Error code: 429 - rate limited"), False),
    ],
)
def test_payment_required_detection(error, expected):
    assert bot_module.is_payment_required(error) is expected
    wrapped = RuntimeError("All 1 research reports/predictions failed")
    wrapped.__cause__ = error
    assert bot_module.is_payment_required(wrapped) is expected


def test_a_402_defers_every_question_and_the_run_ends_cleanly(monkeypatch, tmp_path, capsys):
    import dotenv

    calls = []

    async def invoke(llm, prompt, *args, **kwargs):
        calls.append(llm.model)
        raise PaymentRequired("OpenrouterException - This request requires more credits")

    monkeypatch.setattr(GeneralLlm, "invoke", invoke)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("METACULUS_TOKEN", "placeholder")
    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder")
    client = MagicMock()
    client.get_all_open_questions_from_tournament.return_value = [
        binary_question(id_of_question=n, id_of_post=n, page_url=f"https://www.metaculus.com/questions/{n}/")
        for n in (101, 102, 103, 104, 105)
    ]
    real_build = bot_module.build_bot
    monkeypatch.setattr(bot_module, "build_bot", lambda **kw: real_build(**kw, metaculus_client=client))
    assert bot_module.main(["--mode", "tournament"]) == 0
    assert published_calls(client) == []
    out = capsys.readouterr().out
    assert "Provider credits exhausted (HTTP 402)" in out
    assert "::notice title=CheriusBot::mode=tournament publish=True" in out and "forecasts=0 failures=0 deferred=5" in out
    ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["done"] == {} and all(count == 0 for count in ledger["attempts"].values())
    assert len(calls) <= bot_module.CheriusBot.max_concurrent_questions * 2  # no new question after the 402


# ---------------------------------- CLI ---------------------------------- #
def test_command_line_arguments():
    assert bot_module.parse_args([]).mode == "tournament"
    assert bot_module.parse_args([]).dry_run is False
    args = bot_module.parse_args(["--mode", "test", "--dry-run"])
    assert (args.mode, args.dry_run) == ("test", True)
    with pytest.raises(SystemExit):
        bot_module.parse_args(["--mode", "other"])


def test_main_stops_without_credentials(monkeypatch, capsys):
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("METACULUS_TOKEN", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert bot_module.main(["--mode", "test", "--dry-run"]) == 2
    assert "METACULUS_TOKEN" in capsys.readouterr().err


def test_main_dry_run_does_not_publish(monkeypatch, llm):
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("METACULUS_TOKEN", "placeholder")
    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder")
    client = MagicMock()
    client.get_all_open_questions_from_tournament.return_value = [binary_question()]
    real_build = bot_module.build_bot
    built = {}

    def build(**kwargs):
        built.update(kwargs)
        return real_build(**kwargs, metaculus_client=client)

    monkeypatch.setattr(bot_module, "build_bot", build)
    assert bot_module.main(["--mode", "test", "--dry-run"]) == 0
    assert built["publish"] is False
    assert published_calls(client) == []
    assert bot_module.main(["--mode", "tournament"]) == 0
    assert built == {"publish": True, "skip_previously_forecasted": True}
    assert "post_binary_question_prediction" in published_calls(client)


# ----------------------- defects found by the review ---------------------- #
def test_listing_works_inside_the_event_loop_with_the_real_client(llm, monkeypatch):
    """The real client lists with its own asyncio.run(): check that it works when called from run()."""
    monkeypatch.setattr(
        MetaculusClient, "_get_questions_from_api", lambda self, params, group_mode: [binary_question(state=QuestionState.OPEN)]
    )
    client = MetaculusClient(token="placeholder", sleep_seconds_between_requests=0, sleep_jitter_seconds=0)
    bot = bot_module.build_bot(publish=False, skip_previously_forecasted=False, metaculus_client=client)
    assert asyncio.run(bot_module.run(bot, "test")) == (1, 0, 0)


def test_a_tournament_that_cannot_be_listed_is_a_failure(llm):
    bot, client = make_bot(publish=True)
    client.get_all_open_questions_from_tournament.side_effect = RuntimeError("503")
    assert asyncio.run(bot_module.run(bot, "tournament")) == (0, 2, 0)
    assert llm.calls == []


def test_unsupported_types_are_skipped_not_failed(llm):
    bot, client = make_bot(publish=False, skip=False)
    other = MetaculusQuestion(question_text="Conditional?", id_of_question=99, page_url="https://x/99/")
    client.get_all_open_questions_from_tournament.return_value = [other, binary_question()]
    assert asyncio.run(bot_module.run(bot, "test")) == (1, 0, 0)


def test_date_questions_are_forecast(llm):
    percentiles = bot_module.parse_date_percentiles(DATE_ANSWER)
    assert [p.percentile for p in percentiles] == pytest.approx([p / 100 for p in bot_module.NUMERIC_PERCENTILES])
    assert percentiles[0].value == datetime(2026, 12, 1, tzinfo=timezone.utc).timestamp()
    with pytest.raises(ValueError):
        bot_module.parse_date_percentiles(DATE_ANSWER.replace("2027-01-15", "2026-01-15"))
    bot, _ = make_bot(publish=False)
    prompt = bot.date_prompt(date_question(), RESEARCH_TEXT)
    assert bot_module.today_utc() in prompt and "2030-01-01" in prompt and "community" not in prompt.lower()
    result = asyncio.run(bot._run_forecast_on_date(date_question(), RESEARCH_TEXT))
    cdf = [p.percentile for p in result.prediction_value.cdf]
    assert len(cdf) == 201 and all(a <= b for a, b in zip(cdf, cdf[1:])) and 0 <= cdf[0] and cdf[-1] <= 1


def test_test_mode_takes_few_questions_of_varied_types(llm):
    bot, client = make_bot(publish=False, skip=False)
    binaries = [binary_question(id_of_question=100 + i) for i in range(6)]
    client.get_all_open_questions_from_tournament.return_value = binaries + [numeric_question(), date_question()]
    picked = asyncio.run(bot_module.collect_questions(bot, ["bot-testing-area"], "test"))
    assert len(picked) == 4
    assert {type(q) for q in picked} == {BinaryQuestion, NumericQuestion, DateQuestion}


# --------------------------------- budget -------------------------------- #
def test_profiles_and_fallback_prices(monkeypatch):
    cheap, full = bot_module.PROFILES["cheap"], bot_module.PROFILES["full"]
    assert bot_module.fallback_prices() == (cheap["usd_in"], cheap["usd_out"])
    monkeypatch.setenv("BOT_PROFILE", "full")
    assert bot_module.forecast_model_name() == bot_module.PROFILES["full"]["model"]
    assert make_bot(publish=False)[0].predictions_per_research_report == 3
    monkeypatch.setenv("BOT_PROFILE", "nonsense")
    assert bot_module.forecast_model_name() == bot_module.DEFAULT_MODEL
    monkeypatch.setenv("BOT_MODEL", "openrouter/some/model")  # unknown price: assume the expensive class
    assert bot_module.fallback_prices() == (full["usd_in"], full["usd_out"])
    monkeypatch.setenv("BOT_USD_PER_MTOK_INPUT", "0.3")
    monkeypatch.setenv("BOT_USD_PER_MTOK_OUTPUT", "2.5")
    assert bot_module.estimate_cost("m", 1_000_000, 1_000_000) == pytest.approx(2.8)
    assert bot_module.estimate_cost("m:online", 0, 0) == pytest.approx(0.02)


def test_run_allowance(monkeypatch):
    assert bot_module.run_allowance(0) == pytest.approx(1.0)
    assert bot_module.run_allowance(54.5) == pytest.approx(0.5)
    assert bot_module.run_allowance(60) < 0
    monkeypatch.setenv("BOT_MAX_USD_PER_RUN", "0.25")
    monkeypatch.setenv("BOT_MAX_USD_TOTAL", "0")
    assert bot_module.run_allowance(999) == pytest.approx(0.25)
    monkeypatch.setenv("BOT_MAX_USD_PER_RUN", "0")
    assert bot_module.run_allowance(999) is None


def test_unreported_costs_are_estimated_and_written_to_the_ledger(llm, tmp_path):
    bot, client = make_bot(publish=True)
    client.get_all_open_questions_from_tournament.return_value = [binary_question()]
    ledger = bot_module.Ledger(str(tmp_path / "l.json"))
    ledger.total_usd = 2.0
    assert asyncio.run(bot_module.run(bot, "test", ledger, 1.0)) == (1, 0, 0)
    one_call = CALL_TOKENS[0] / 1e6 * CHEAP["usd_in"] + CALL_TOKENS[1] / 1e6 * CHEAP["usd_out"]
    assert ledger.run_usd == pytest.approx(3 * one_call + 0.02)  # research (with search fee) + 2 draws
    saved = json.loads((tmp_path / "l.json").read_text())
    assert saved["total_usd"] == pytest.approx(2.0 + ledger.run_usd)
    assert saved["done"] == {"11": bot_module.today_utc()} and saved["attempts"] == {"11": 1} and saved["runs"] == 1
    reloaded = bot_module.Ledger.load(str(tmp_path / "l.json"))
    assert reloaded.total_usd == pytest.approx(saved["total_usd"]) and reloaded.done == saved["done"]


def test_the_run_allowance_stops_the_spending(llm):
    bot, client = make_bot(publish=True)
    bot.max_concurrent_questions = 1
    questions = [binary_question(id_of_question=30 + i) for i in range(3)]
    client.get_all_open_questions_from_tournament.return_value = questions
    ledger = bot_module.Ledger(None)
    assert asyncio.run(bot_module.run(bot, "test", ledger, 0.001)) == (0, 0, 3)
    assert len(llm.calls) == 1  # the first research call spends the allowance, nothing else is called
    assert published_calls(client) == []
    assert ledger.attempts == {"30": 0} and ledger.done == {}


def test_llm_refuses_to_call_over_the_limit(llm):
    model = bot_module.BudgetedLlm(model="openrouter/some/model", allowed_tries=1)

    async def spend() -> float:
        with MonetaryCostManager(0.0001) as meter:
            await model._mockable_direct_call_to_model("Probability: ZZ%")
            assert meter.current_usage > 0.0001
            with pytest.raises(HardLimitExceededError):
                await model._mockable_direct_call_to_model("Probability: ZZ%")
            return meter.current_usage

    assert asyncio.run(spend()) == pytest.approx(CALL_TOKENS[0] / 1e6 * CHEAP["usd_in"] + CALL_TOKENS[1] / 1e6 * CHEAP["usd_out"])
    assert bot_module.is_budget_error(RuntimeError("x: exceeding the hard limit of 1"))
    assert not bot_module.is_budget_error(ValueError("parse error"))


def test_ledger_blocks_published_questions_and_repeated_failures(llm, tmp_path):
    bot, client = make_bot(publish=True, skip=True)
    done, failing, fresh = (binary_question(id_of_question=n) for n in (41, 42, 43))
    client.get_all_open_questions_from_tournament.return_value = [done, failing, fresh]
    ledger = bot_module.Ledger(None)
    ledger.done, ledger.attempts = {"41": "2026-10-01"}, {"42": 3}
    assert asyncio.run(bot_module.collect_questions(bot, ["a"], "tournament", ledger)) == [fresh]
    (tmp_path / "bad.json").write_text("{not json")
    assert bot_module.Ledger.load(str(tmp_path / "bad.json")).total_usd == 0.0


def test_failed_attempts_are_counted(monkeypatch):
    async def invoke(llm, prompt, *args, **kwargs):
        return ""

    monkeypatch.setattr(GeneralLlm, "invoke", invoke)
    bot, client = make_bot(publish=True)
    client.get_all_open_questions_from_tournament.return_value = [binary_question()]
    ledger = bot_module.Ledger(None)
    for expected in (1, 2):
        assert asyncio.run(bot_module.run(bot, "tournament", ledger)) == (0, 1, 0)
        assert ledger.attempts == {"11": expected} and ledger.done == {}


def test_no_question_starts_after_the_deadline(llm):
    bot, _ = make_bot(publish=False)
    assert bot._reason_to_defer() is None
    bot.deadline = time.monotonic() - 1
    assert "time" in bot._reason_to_defer()
    with pytest.raises(bot_module.Deferred):
        asyncio.run(bot._run_individual_question(binary_question()))
    assert llm.calls == []


def test_main_does_nothing_once_the_season_budget_is_spent(monkeypatch, tmp_path, capsys):
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("METACULUS_TOKEN", "placeholder")
    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder")
    monkeypatch.setenv("BOT_MAX_USD_TOTAL", "5")
    (tmp_path / "ledger.json").write_text(json.dumps({"total_usd": 5.0}))
    monkeypatch.setattr(bot_module, "run", lambda *a, **k: pytest.fail("must not run"))
    assert bot_module.main(["--mode", "tournament"]) == 0
    assert "::warning" in capsys.readouterr().out


def test_openrouter_usage_raises_the_ledger_total(monkeypatch):
    import requests

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-placeholder")
    seen = {}

    def get(url, headers=None, timeout=None):
        seen["url"] = url
        response = MagicMock()
        response.json.return_value = {"data": {"usage": 12.5, "limit": 55}}
        return response

    monkeypatch.setattr(requests, "get", get)
    assert bot_module.openrouter_usage_usd() == pytest.approx(12.5)
    assert seen["url"] == bot_module.OPENROUTER_KEY_URL
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    assert bot_module.openrouter_usage_usd() is None
    monkeypatch.setenv("BOT_MODEL", "openai/some-model")
    assert bot_module.openrouter_usage_usd() is None


def test_the_provider_is_configurable(monkeypatch):
    assert bot_module.required_env() == ["METACULUS_TOKEN", "OPENROUTER_API_KEY"]
    monkeypatch.setenv("BOT_MODEL", "openai/some-model")
    assert bot_module.parser_model_name() == bot_module.research_model_name() == "openai/some-model"
    assert bot_module.required_env() == ["METACULUS_TOKEN", "OPENAI_API_KEY"]
    monkeypatch.setenv("BOT_MODEL", "metaculus/anthropic/some-model")
    assert bot_module.required_env() == ["METACULUS_TOKEN"]


def test_secrets_never_reach_the_log(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-abcdef123456")
    monkeypatch.setenv("METACULUS_TOKEN", "tok-9876543210")
    record = logging.LogRecord("x", logging.ERROR, "f", 1, "401 for key %s token %s", ("sk-or-v1-abcdef123456", "tok-9876543210"), None)
    assert bot_module.SecretFilter().filter(record)
    assert record.getMessage() == "401 for key *** token ***"


def test_repository_files_hold_no_secret_and_pin_dependencies():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workflow = open(os.path.join(root, ".github", "workflows", "forecast.yml")).read()
    assert "secrets.METACULUS_TOKEN" in workflow and "vars.BOT_ENABLED == 'true'" in workflow
    assert "requirements.lock" in workflow and "ledger.json" in workflow
    for name in ("requirements.txt", "requirements.lock"):
        lines = [l for l in open(os.path.join(root, name)).read().splitlines() if l.strip() and not l.startswith("#")]
        assert lines and all("==" in line for line in lines), name
    for line in open(os.path.join(root, ".env.example")).read().splitlines():
        if line and not line.startswith("#"):
            assert line.endswith("="), line
