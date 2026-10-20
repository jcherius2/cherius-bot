"""CheriusBot - a simple, fully autonomous forecasting bot for the Metaculus FutureEval tournament.

Pipeline per question: one web research pass, then several independent forecasts
from a frontier model, aggregated by forecasting-tools (median for binary and
numeric, mean for multiple choice), then published with a comment.

Spending is capped per run and for the season (see Budget below); the model
provider is any litellm provider (OpenRouter by default, or an OpenAI-compatible API).

All code in this repository was written by Claude, an AI model made by Anthropic,
under the supervision of the account holder. No human is in the forecasting loop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from forecasting_tools import (
    BinaryPrediction,
    BinaryQuestion,
    DatePercentile,
    DateQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOptionList,
    PredictionExtractor,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)
from forecasting_tools.ai_models.resource_managers.hard_limit_manager import HardLimitExceededError
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import MonetaryCostManager

logger = logging.getLogger("cherius_bot")

# --------------------------------------------------------------------------- #
# Configuration (names only - secrets are read from the environment, never stored)
# --------------------------------------------------------------------------- #
# market-pulse-26q4: its own page admits bot accounts for prizes ("You are welcome to enter this
# tournament and compete for prizes using a forecasting bot account, but you can only enter the
# competition once, either as a bot or as a normal user." - read 2026-10-11). The owner of this
# bot must therefore NOT enter it with a personal account.
TOURNAMENT_IDS: list[str] = ["fall-futureeval-2026", "minibench", "market-pulse-26q4"]
# Tournaments whose listing may be refused to a bot token (unverified): a listing error there is
# a warning, not a failed run.
OPTIONAL_TOURNAMENT_IDS: frozenset[str] = frozenset({"market-pulse-26q4"})
TEST_TOURNAMENT_ID = "bot-testing-area"

# BOT_PROFILE picks the defaults. "cheap" is the default because the season budget may be
# the starting 55 EUR only; "full" is for when model credits are granted.
# usd_in / usd_out: fallback prices per million tokens, used when the provider reports no cost.
PROFILES: dict[str, dict] = {
    "cheap": {"model": "openrouter/anthropic/claude-haiku-5.5", "draws": 2, "usd_in": 0.1, "usd_out": 0.5},
    "full": {"model": "openrouter/anthropic/claude-opus-5.5", "draws": 3, "usd_in": 4.0, "usd_out": 20.0},
}
DEFAULT_PROFILE = "cheap"
DEFAULT_MODEL = PROFILES[DEFAULT_PROFILE]["model"]
DEFAULT_MAX_USD_PER_RUN = 1.0
DEFAULT_MAX_USD_TOTAL = 55.0  # dollars for the whole season; 0 disables the cap
MIN_RUN_USD = 0.01
DEFAULT_LEDGER_PATH = ".budget/ledger.json"
# Environment variable that must hold the key, by litellm model prefix.
PROVIDER_KEYS = {
    "openrouter/": "OPENROUTER_API_KEY",
    "openai/": "OPENAI_API_KEY",  # any OpenAI-compatible API: also set OPENAI_API_BASE
    "anthropic/": "ANTHROPIC_API_KEY",
    "metaculus/": "METACULUS_TOKEN",  # Metaculus proxy, when Metaculus grants credits that way
}
OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"
DEFAULT_PARSER_MODEL = "openrouter/anthropic/claude-haiku-5.5"
ONLINE_SUFFIX = ":online"  # OpenRouter web-search variant of a model

MIN_BINARY_PROBABILITY = 0.02
MAX_BINARY_PROBABILITY = 0.98
FINAL_ANSWER_MARKER = "FINAL ANSWER"
NUMERIC_PERCENTILES = [5, 10, 20, 40, 60, 80, 90, 95]

AI_DISCLOSURE = (
    "*This forecast and this comment were generated automatically by CheriusBot, "
    "an AI forecasting bot, with no human in the loop.*"
)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(1, value)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError:
        value = default
    return default if value != value or value < 0 else value


def profile() -> dict:
    name = os.getenv("BOT_PROFILE", "").strip().lower() or DEFAULT_PROFILE
    if name not in PROFILES:
        logger.warning("Unknown BOT_PROFILE %r, using %r", name, DEFAULT_PROFILE)
        name = DEFAULT_PROFILE
    return PROFILES[name]


def forecast_model_name() -> str:
    return os.getenv("BOT_MODEL", "").strip() or profile()["model"]


def fallback_prices() -> tuple[float, float]:
    """USD per million tokens (input, output) when the provider reports no cost.

    With a BOT_MODEL override and no explicit price, assume the expensive class: an
    over-estimate stops the bot early, an under-estimate overspends.
    """
    base = PROFILES["full"] if os.getenv("BOT_MODEL", "").strip() else profile()
    return (
        _env_float("BOT_USD_PER_MTOK_INPUT", base["usd_in"]),
        _env_float("BOT_USD_PER_MTOK_OUTPUT", base["usd_out"]),
    )


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    usd_in, usd_out = fallback_prices()
    cost = (prompt_tokens or 0) / 1e6 * usd_in + (completion_tokens or 0) / 1e6 * usd_out
    if ONLINE_SUFFIX in model:
        cost += _env_float("BOT_USD_PER_SEARCH", 0.02)
    return cost


def required_env() -> list[str]:
    names = ["METACULUS_TOKEN"]
    for model in (forecast_model_name(), research_model_name(), parser_model_name()):
        for prefix, key in PROVIDER_KEYS.items():
            if model.startswith(prefix) and key not in names:
                names.append(key)
    return names


def secret_values() -> list[str]:
    return [
        value
        for name, value in os.environ.items()
        if name.endswith(("_API_KEY", "_TOKEN", "_SECRET")) and len(value) >= 8
    ]


class SecretFilter(logging.Filter):
    """Last line of defence: never let a key or token reach the log."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for value in secret_values():
            message = message.replace(value, "***")
        record.msg, record.args = message, None
        return True


# --------------------------------------------------------------------------- #
# Budget: a ledger file kept between runs (Actions cache) and a per-run allowance
# --------------------------------------------------------------------------- #
class Deferred(Exception):
    """The question was not attempted in this run (budget or time); the next run takes it."""


class Ledger:
    """Estimated spending, attempts and published questions, kept between runs."""

    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path) if path else None
        self.total_usd = 0.0  # spent before this run
        self.run_usd = 0.0  # spent by this run so far
        self.runs = 0
        self.attempts: dict[str, int] = {}
        self.done: dict[str, str] = {}

    @classmethod
    def load(cls, path: str | None) -> "Ledger":
        ledger = cls(path)
        if ledger.path is None or not ledger.path.exists():
            return ledger
        try:
            data = json.loads(ledger.path.read_text(encoding="utf-8"))
            ledger.total_usd = max(0.0, float(data.get("total_usd", 0.0)))
            ledger.runs = int(data.get("runs", 0))
            ledger.attempts = {str(k): int(v) for k, v in dict(data.get("attempts", {})).items()}
            ledger.done = {str(k): str(v) for k, v in dict(data.get("done", {})).items()}
        except (OSError, ValueError, TypeError, AttributeError) as error:
            logger.error("Unreadable ledger %s (%s): starting from zero", ledger.path, type(error).__name__)
            return cls(path)
        return ledger

    def save(self) -> None:
        if self.path is None:
            return
        data = {
            "total_usd": round(self.total_usd + self.run_usd, 6),
            "runs": self.runs,
            "attempts": self.attempts,
            "done": self.done,
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(self.path.name + ".tmp")
            temporary.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
            temporary.replace(self.path)
        except OSError as error:
            logger.error("Could not write the ledger %s: %s", self.path, error)


def run_allowance(spent_total: float) -> float | None:
    """Dollars this run may spend; None means no cap at all."""
    per_run = _env_float("BOT_MAX_USD_PER_RUN", DEFAULT_MAX_USD_PER_RUN)
    total = _env_float("BOT_MAX_USD_TOTAL", DEFAULT_MAX_USD_TOTAL)
    caps = ([per_run] if per_run > 0 else []) + ([total - spent_total] if total > 0 else [])
    return min(caps) if caps else None


def openrouter_usage_usd() -> float | None:
    """Usage reported by OpenRouter for this key: survives a lost ledger. None if unavailable."""
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key or not forecast_model_name().startswith("openrouter/"):
        return None
    try:
        response = requests.get(OPENROUTER_KEY_URL, headers={"Authorization": f"Bearer {key}"}, timeout=20)
        response.raise_for_status()
        return max(0.0, float(response.json()["data"]["usage"]))
    except Exception as error:  # never block forecasting on this check
        logger.warning("OpenRouter key usage unavailable (%s); using the local ledger only", type(error).__name__)
        return None


def is_budget_error(error: BaseException | None, depth: int = 0) -> bool:
    if error is None or depth > 6:
        return False
    if isinstance(error, HardLimitExceededError) or "exceeding the hard limit" in str(error):
        return True
    children = list(getattr(error, "exceptions", ())) + [error.__cause__, error.__context__]
    return any(is_budget_error(child, depth + 1) for child in children)


# OpenRouter answers 402 when the key lacks credits (or the in-flight reservation fails).
# litellm may wrap it, and forecasting-tools may stringify it: match the status and the
# known messages, never a bare "402" (a question URL can contain it).
PAYMENT_REQUIRED = re.compile(
    r"(error code|status[_ ]code|\"code\")\s*[:=]?\s*402\b|402 payment required|payment required"
    r"|insufficient credits|requires more credits|can only afford|in_flight_budget_exhausted",
    re.IGNORECASE,
)


def is_payment_required(error: BaseException | None, depth: int = 0) -> bool:
    """True if the error, or one it wraps, is a provider refusal for lack of credits (HTTP 402)."""
    if error is None or depth > 6:
        return False
    if getattr(error, "status_code", None) == 402 or PAYMENT_REQUIRED.search(str(error)):
        return True
    children = list(getattr(error, "exceptions", ())) + [error.__cause__, error.__context__]
    return any(is_payment_required(child, depth + 1) for child in children)


class BudgetedLlm(GeneralLlm):
    """GeneralLlm that refuses to call once the run allowance is spent, and that books an
    estimated cost when neither litellm nor the provider reports one."""

    async def _mockable_direct_call_to_model(self, prompt):
        MonetaryCostManager.raise_error_if_limit_would_be_reached()
        with MonetaryCostManager() as probe:
            response = await super()._mockable_direct_call_to_model(prompt)
            tracked = probe.current_usage
        if tracked <= 0:
            MonetaryCostManager.increase_current_usage_in_parent_managers(
                estimate_cost(self.model, response.prompt_tokens_used, response.completion_tokens_used)
            )
        return response


def question_key(question: MetaculusQuestion) -> str:
    return str(question.id_of_question or question.page_url)


def research_model_name() -> str:
    """Model used for the research step: BOT_RESEARCH_MODEL, else BOT_MODEL with web search."""
    explicit = os.getenv("BOT_RESEARCH_MODEL", "").strip()
    if explicit:
        return explicit
    model = forecast_model_name()
    if model.startswith("openrouter/") and not model.endswith(ONLINE_SUFFIX):
        return model + ONLINE_SUFFIX
    return model


def parser_model_name() -> str:
    """BOT_PARSER_MODEL, else a small OpenRouter model, else (other provider) the forecast model."""
    explicit = os.getenv("BOT_PARSER_MODEL", "").strip()
    if explicit:
        return explicit
    model = forecast_model_name()
    return DEFAULT_PARSER_MODEL if model.startswith("openrouter/") else model


# --------------------------------------------------------------------------- #
# Pure helpers (parsing and guard rails) - covered by the offline tests
# --------------------------------------------------------------------------- #
def clamp_probability(probability: float) -> float:
    """Never publish 0 or 1: keep binary probabilities inside [0.02, 0.98]."""
    if probability != probability:  # NaN
        raise ValueError("Probability is NaN")
    return max(MIN_BINARY_PROBABILITY, min(MAX_BINARY_PROBABILITY, float(probability)))


def final_section(text: str) -> str:
    """Return what follows the last FINAL ANSWER marker (whole text if absent)."""
    index = text.rfind(FINAL_ANSWER_MARKER)
    return text if index < 0 else text[index + len(FINAL_ANSWER_MARKER) :]


def parse_binary(text: str) -> float:
    value = PredictionExtractor.extract_last_percentage_value(
        final_section(text),
        max_prediction=MAX_BINARY_PROBABILITY,
        min_prediction=MIN_BINARY_PROBABILITY,
    )
    return clamp_probability(value)


def parse_multiple_choice(text: str, options: list[str]) -> PredictedOptionList:
    prediction = PredictionExtractor.extract_option_list_with_percentage_afterwards(
        final_section(text), options
    )
    check_option_list(prediction, options)
    return prediction


def check_option_list(prediction: PredictedOptionList, options: list[str]) -> None:
    names = [option.option_name for option in prediction.predicted_options]
    if sorted(names) != sorted(options):
        raise ValueError(f"Parsed options {names} do not match question options {options}")
    total = sum(option.probability for option in prediction.predicted_options)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Option probabilities sum to {total}, not 1")
    if any(not 0 < option.probability < 1 for option in prediction.predicted_options):
        raise ValueError("Option probabilities must be strictly between 0 and 1")


def check_percentiles(percentiles: list[Percentile]) -> list[Percentile]:
    """Percentiles must be sorted and their values strictly increasing; otherwise discard the draw."""
    ordered = sorted(percentiles, key=lambda p: p.percentile)
    if len(ordered) < 3:
        raise ValueError(f"Need at least 3 percentiles, got {len(ordered)}")
    for previous, current in zip(ordered, ordered[1:]):
        if current.percentile <= previous.percentile:
            raise ValueError("Duplicate percentile in the answer")
        if current.value <= previous.value:
            raise ValueError(
                f"Percentile values are not increasing: {previous.value} then {current.value}"
            )
    return ordered


def parse_numeric(text: str, question: NumericQuestion) -> NumericDistribution:
    distribution = (
        PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
            final_section(text), question
        )
    )
    ordered = check_percentiles(distribution.declared_percentiles)
    return NumericDistribution.from_question(ordered, question)


DATE_LINE = re.compile(r"Percentile\s+(\d{1,2})\s*:\s*\**\s*(\d{4}-\d{2}-\d{2}(?:T[0-9:.]+(?:Z|[+-]\d{2}:\d{2})?)?)")


def as_utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment


def parse_date_percentiles(text: str) -> list[Percentile]:
    """'Percentile 10: 2026-11-03' lines after the marker -> percentiles valued in Unix seconds."""
    found = {
        int(rank): as_utc(datetime.fromisoformat(value.replace("Z", "+00:00"))).timestamp()
        for rank, value in DATE_LINE.findall(final_section(text))
    }
    return check_percentiles([Percentile(percentile=rank / 100, value=value) for rank, value in found.items()])


def bound_messages(question: NumericQuestion | DateQuestion) -> tuple[str, str]:
    if isinstance(question, DateQuestion):
        unit = ""
        upper = question.upper_bound.date().isoformat()
        lower = question.lower_bound.date().isoformat()
    else:
        unit = question.unit_of_measure or ""
        upper = question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
        lower = question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound
    if question.open_upper_bound:
        upper_message = f"The question creator thinks the number is likely not higher than {upper} {unit}."
    else:
        upper_message = f"The outcome can not be higher than {upper} {unit}."
    if question.open_lower_bound:
        lower_message = f"The question creator thinks the number is likely not lower than {lower} {unit}."
    else:
        lower_message = f"The outcome can not be lower than {lower} {unit}."
    return upper_message, lower_message


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def question_block(question: MetaculusQuestion) -> str:
    return clean_indents(
        f"""
        Question:
        {question.question_text}

        Background:
        {question.background_info or "(none given)"}

        Resolution criteria (these alone decide the outcome):
        {question.resolution_criteria or "(none given)"}

        Fine print:
        {question.fine_print or "(none)"}
        """
    )


REASONING_STEPS = """
Before answering, write your reasoning in this order:
(a) The time left until the question resolves.
(b) Base rate: how often outcomes like this happened in comparable past situations (name the reference class).
(c) The strongest arguments for a higher value / a Yes, and the strongest arguments for a lower value / a No.
(d) The status quo scenario: what happens if nothing changes. The world changes slowly most of the time, so give the status quo extra weight.
(e) A scenario that would produce a surprising outcome, and how likely it is.
Stay calibrated: be humble about unknowns, and do not go to extremes unless the evidence is overwhelming.
""".strip()


# --------------------------------------------------------------------------- #
# The bot
# --------------------------------------------------------------------------- #
class CheriusBot(ForecastBot):
    """Research once, forecast several times, aggregate, publish with a comment."""

    max_concurrent_questions = 3  # bounds the rate of calls and the overshoot of the run allowance
    ledger: Ledger | None = None
    meter: MonetaryCostManager | None = None  # set by run(): the allowance of this run
    deadline: float | None = None  # time.monotonic() after which no new question starts
    credits_exhausted: bool = False  # set on the first 402: no new question starts in this run
    _slots: asyncio.Semaphore | None = None
    _slots_loop: object | None = None

    def _question_slots(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._slots is None or self._slots_loop is not loop:
            self._slots, self._slots_loop = asyncio.Semaphore(self.max_concurrent_questions), loop
        return self._slots

    def _checkpoint(self) -> None:
        if self.ledger is not None:
            if self.meter is not None:
                self.ledger.run_usd = self.meter.current_usage
            self.ledger.save()

    def _reason_to_defer(self) -> str | None:
        if self.meter is not None and self.meter.hard_limit > 0 and self.meter.amount_left <= 0:
            return "run allowance spent"
        if self.deadline is not None and time.monotonic() > self.deadline:
            return "run time budget spent"
        if self.credits_exhausted:
            return "provider credits exhausted (HTTP 402 earlier in this run)"
        return None

    async def _run_individual_question(self, question: MetaculusQuestion):
        """One question at a time per slot; nothing starts once the money or the time is spent."""
        if not hasattr(self, "deferred"):
            self.deferred: dict[str, str] = {}
        key = question_key(question)
        async with self._question_slots():
            reason = self._reason_to_defer()
            if reason:
                self.deferred[key] = reason
                raise Deferred(reason)
            if self.ledger is not None:
                self.ledger.attempts[key] = self.ledger.attempts.get(key, 0) + 1
            try:
                report = await super()._run_individual_question(question)
            except Exception as error:
                payment_required = is_payment_required(error)
                if payment_required:
                    if not self.credits_exhausted:
                        logger.warning("HTTP 402 from the provider (no credits): remaining questions are deferred")
                    self.credits_exhausted = True
                    self.deferred[key] = "provider refused the call for lack of credits (HTTP 402)"
                elif is_budget_error(error):
                    self.deferred[key] = "run allowance reached during the question"
                if payment_required or is_budget_error(error):  # not the question's fault: do not count it
                    if self.ledger is not None:
                        self.ledger.attempts[key] = max(0, self.ledger.attempts.get(key, 1) - 1)
                raise
            else:
                if self.ledger is not None and self.publish_reports_to_metaculus:
                    self.ledger.done[key] = today_utc()
                return report
            finally:
                self._checkpoint()

    # ------------------------------ research ------------------------------ #
    async def run_research(self, question: MetaculusQuestion) -> str:
        research = await self.get_llm("researcher", "llm").invoke(self.research_prompt(question))
        if not research or not research.strip():
            # Fail this question now; the next scheduled run retries it while it is still open.
            raise ValueError(f"Empty research for {question.page_url}")
        return research

    @staticmethod
    def research_prompt(question: MetaculusQuestion) -> str:
        return clean_indents(
            f"""
            You are a research assistant to a professional forecaster. Today is {today_utc()} (UTC).
            Search the web and write a factual briefing for the question below. You do not produce a forecast yourself.

            {question_block(question)}

            The question has NOT been resolved yet: its outcome is still unknown today. Treat any source that
            claims a final outcome with suspicion and check its date.

            Cover, in this order, with dates and sources (URLs) for every fact:
            1. Current state: where things stand today on exactly what the resolution criteria measure, with the latest known value or status.
            2. Recent developments: the most relevant news from the last weeks, newest first.
            3. Base rates: how often similar events happened in the past, with historical figures or a relevant reference class.
            4. Scheduled events before the resolution date that could change the outcome.
            5. What is unknown or disputed.

            Do not look up or report the Metaculus community prediction or other Metaculus forecasts on this question.
            Say plainly when you could not find something. Do not invent facts, figures or sources.
            """
        )

    # ------------------------------- binary ------------------------------- #
    @staticmethod
    def binary_prompt(question: BinaryQuestion, research: str) -> str:
        return clean_indents(
            f"""
            You are a professional forecaster. Today is {today_utc()} (UTC).

            {question_block(question)}

            Research briefing from your assistant:
            {research}

            The question has not been resolved yet.

            {REASONING_STEPS}

            Then end your answer with exactly these two lines, where ZZ is a number between 2 and 98:
            {FINAL_ANSWER_MARKER}
            Probability: ZZ%
            """
        )

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        reasoning = await self.get_llm("default", "llm").invoke(
            self.binary_prompt(question, research)
        )
        try:
            probability = parse_binary(reasoning)
        except Exception as error:
            logger.warning("Direct binary parsing failed (%s); using the parser model", error)
            parsed: BinaryPrediction = await structure_output(
                reasoning,
                BinaryPrediction,
                model=self.get_llm("parser", "llm"),
                additional_instructions="The text gives a probability forecast for a binary question. Extract the final probability.",
            )
            probability = clamp_probability(parsed.prediction_in_decimal)
        return ReasonedPrediction(prediction_value=probability, reasoning=reasoning)

    # --------------------------- multiple choice --------------------------- #
    @staticmethod
    def multiple_choice_prompt(question: MultipleChoiceQuestion, research: str) -> str:
        option_lines = "\n".join(f"{option}: NN%" for option in question.options)
        return clean_indents(
            f"""
            You are a professional forecaster. Today is {today_utc()} (UTC).

            {question_block(question)}

            The options are: {question.options}

            Research briefing from your assistant:
            {research}

            The question has not been resolved yet.

            {REASONING_STEPS}

            Leave some probability on every option (at least 1%), since unexpected outcomes happen.
            Then end your answer with the marker line and one line per option, in the order given,
            with the option names written exactly as above and percentages that sum to 100:
            {FINAL_ANSWER_MARKER}
            {option_lines}
            """
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        reasoning = await self.get_llm("default", "llm").invoke(
            self.multiple_choice_prompt(question, research)
        )
        try:
            prediction = parse_multiple_choice(reasoning, question.options)
        except Exception as error:
            logger.warning("Direct option parsing failed (%s); using the parser model", error)
            prediction = await structure_output(
                reasoning,
                PredictedOptionList,
                model=self.get_llm("parser", "llm"),
                additional_instructions=(
                    f"Option names must be exactly these: {question.options}. "
                    "Keep every option, including those with a very low probability."
                ),
            )
            check_option_list(prediction, question.options)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    # ------------------------- numeric and discrete ------------------------ #
    @staticmethod
    def numeric_prompt(question: NumericQuestion, research: str) -> str:
        upper_message, lower_message = bound_messages(question)
        percentile_lines = "\n".join(f"Percentile {p}: XX" for p in NUMERIC_PERCENTILES)
        discrete_note = (
            "This is a discrete question: the outcome is one of a fixed set of evenly spaced values, usually integers."
            if question.question_type == "discrete"
            else ""
        )
        return clean_indents(
            f"""
            You are a professional forecaster. Today is {today_utc()} (UTC).

            {question_block(question)}

            Units for the answer: {question.unit_of_measure or "not stated (infer from the question)"}
            {lower_message}
            {upper_message}
            {discrete_note}

            Research briefing from your assistant:
            {research}

            The question has not been resolved yet.

            {REASONING_STEPS}
            Also state the outcome implied by the current trend, and what experts or markets expect.
            Set wide 90/10 intervals to account for unknown unknowns.

            Formatting rules: give values in the units above, as plain numbers (no scientific notation, no
            thousands separators, no unit after the number). Values must strictly increase from one percentile to the next.
            Then end your answer with the marker line and exactly these lines:
            {FINAL_ANSWER_MARKER}
            {percentile_lines}
            """
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(
            self.numeric_prompt(question, research)
        )
        try:
            prediction = parse_numeric(reasoning, question)
        except Exception as error:
            logger.warning("Direct percentile parsing failed (%s); using the parser model", error)
            percentiles: list[Percentile] = await structure_output(
                reasoning,
                list[Percentile],
                model=self.get_llm("parser", "llm"),
                additional_instructions=(
                    f"The text gives a percentile forecast for: {question.question_text!r}. "
                    f"Units: {question.unit_of_measure}. Return plain numbers in these units. "
                    "If no percentiles are given explicitly, do not invent any."
                ),
            )
            prediction = NumericDistribution.from_question(check_percentiles(percentiles), question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    # -------------------------------- date -------------------------------- #
    @staticmethod
    def date_prompt(question: DateQuestion, research: str) -> str:
        upper_message, lower_message = bound_messages(question)
        percentile_lines = "\n".join(f"Percentile {p}: YYYY-MM-DD" for p in NUMERIC_PERCENTILES)
        return clean_indents(
            f"""
            You are a professional forecaster. Today is {today_utc()} (UTC).

            {question_block(question)}

            The answer is a date.
            {lower_message}
            {upper_message}

            Research briefing from your assistant:
            {research}

            The question has not been resolved yet.

            {REASONING_STEPS}
            Also state the date implied by the current trend, and what experts or markets expect.
            Set wide 90/10 intervals to account for unknown unknowns.

            Formatting rules: dates as YYYY-MM-DD (UTC), nothing after the date on the line. Dates must
            strictly increase from one percentile to the next, starting with the earliest.
            Then end your answer with the marker line and exactly these lines:
            {FINAL_ANSWER_MARKER}
            {percentile_lines}
            """
        )

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(self.date_prompt(question, research))
        try:
            percentiles = parse_date_percentiles(reasoning)
        except Exception as error:
            logger.warning("Direct date parsing failed (%s); using the parser model", error)
            parsed: list[DatePercentile] = await structure_output(
                reasoning,
                list[DatePercentile],
                model=self.get_llm("parser", "llm"),
                additional_instructions=(
                    f"The text gives a percentile forecast, in dates, for: {question.question_text!r}. "
                    "Return valid datetimes (midnight UTC if no hour is given). "
                    "If no percentiles are given explicitly, do not invent any."
                ),
            )
            percentiles = check_percentiles(
                [Percentile(percentile=p.percentile, value=as_utc(p.value).timestamp()) for p in parsed]
            )
        prediction = NumericDistribution.from_question(percentiles, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    # ------------------------------- comment ------------------------------ #
    def _create_comment(self, *args, **kwargs) -> str:
        # forecasting-tools requires the explanation to start with a '#' heading,
        # so the AI disclosure goes right under the first heading.
        heading, _, rest = super()._create_comment(*args, **kwargs).lstrip().partition("\n")
        return f"{heading}\n{AI_DISCLOSURE}\n\n{rest}"


# --------------------------------------------------------------------------- #
# Wiring and command line
# --------------------------------------------------------------------------- #
SUPPORTED_TYPES = (BinaryQuestion, MultipleChoiceQuestion, NumericQuestion, DateQuestion)
PAGE_SIZE = 100  # the package reads one page of the tournament listing


def build_bot(
    *,
    publish: bool,
    skip_previously_forecasted: bool = True,
    metaculus_client: MetaculusClient | None = None,
) -> CheriusBot:
    timeout = _env_int("LLM_TIMEOUT_SECONDS", 150)
    max_tokens = _env_int("LLM_MAX_TOKENS", 4096)
    forecaster = BudgetedLlm(
        model=forecast_model_name(), timeout=timeout, allowed_tries=2, max_tokens=max_tokens
    )
    researcher = BudgetedLlm(
        model=research_model_name(), timeout=timeout, allowed_tries=2, max_tokens=max_tokens
    )
    # Every model gets max_tokens: without it OpenRouter reserves the model's maximum output
    # (64,000 tokens for Haiku 4.5) and refuses the call (402) when credits run low.
    parser = BudgetedLlm(
        model=parser_model_name(), timeout=60, allowed_tries=2, temperature=0, max_tokens=max_tokens
    )
    return CheriusBot(
        research_reports_per_question=1,
        predictions_per_research_report=_env_int("PREDICTIONS_PER_QUESTION", profile()["draws"]),
        use_research_summary_to_forecast=False,
        enable_summarize_research=False,
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=skip_previously_forecasted,
        extra_metadata_in_explanation=True,
        llms={
            "default": forecaster,
            "researcher": researcher,
            "parser": parser,
            "summarizer": parser,
        },
        metaculus_client=metaculus_client,
    )


def tournaments_for_mode(mode: str) -> list[str]:
    return list(TOURNAMENT_IDS) if mode == "tournament" else [TEST_TOURNAMENT_ID]


def order_questions(questions: list[MetaculusQuestion], mode: str) -> list[MetaculusQuestion]:
    """Test mode: one question of each type first. Tournament: the soonest to close first."""
    if mode == "test":
        groups: dict[str, list[MetaculusQuestion]] = {}
        for question in questions:
            groups.setdefault(type(question).__name__, []).append(question)
        ordered: list[MetaculusQuestion] = []
        while any(groups.values()):
            ordered.extend(group.pop(0) for group in groups.values() if group)
        return ordered
    last = datetime.max.replace(tzinfo=timezone.utc)

    def closes(question: MetaculusQuestion) -> datetime:
        moment = getattr(question, "close_time", None)
        return as_utc(moment) if isinstance(moment, datetime) else last

    return sorted(questions, key=closes)


async def collect_questions(
    bot: CheriusBot, tournament_ids: list[str], mode: str = "tournament", ledger: Ledger | None = None
) -> list[MetaculusQuestion]:
    """Open questions of the tournaments, deduplicated, minus what must not be forecast (again).

    Sets bot.listing_errors to the number of tournaments that could not be listed.
    """
    questions: list[MetaculusQuestion] = []
    seen: set[str] = set()
    bot.listing_errors = 0
    skip_done = bot.skip_previously_forecasted_questions
    max_attempts = _env_int("BOT_MAX_ATTEMPTS_PER_QUESTION", 3)
    for tournament_id in tournament_ids:
        try:
            # The package method is blocking (requests, sleeps, its own asyncio.run): keep it off this loop.
            found = await asyncio.to_thread(
                bot.metaculus_client.get_all_open_questions_from_tournament, tournament_id
            )
        except Exception as error:  # one tournament failing must not block the other
            if tournament_id in OPTIONAL_TOURNAMENT_IDS:
                logger.warning(
                    "Optional tournament %s could not be listed: %s: %s", tournament_id, type(error).__name__, error
                )
                continue
            bot.listing_errors += 1
            logger.error("Could not list questions of %s: %s: %s", tournament_id, type(error).__name__, error)
            continue
        if len(found) >= PAGE_SIZE:
            logger.warning("%s returned %d questions: the listing may be truncated", tournament_id, len(found))
        for question in found:
            key = question_key(question)
            if key in seen:
                continue
            seen.add(key)
            if not isinstance(question, SUPPORTED_TYPES):
                logger.info("Skipping %s: type %s is not handled", question.page_url, type(question).__name__)
            elif skip_done and (question.already_forecasted or (ledger and key in ledger.done)):
                continue
            elif skip_done and ledger and ledger.attempts.get(key, 0) >= max_attempts:
                logger.warning("Giving up on %s after %d failed attempts", question.page_url, max_attempts)
            else:
                questions.append(question)
    questions = order_questions(questions, mode)
    limit = _env_int("MAX_QUESTIONS_PER_RUN", 15 if mode == "tournament" else 4)
    if len(questions) > limit:
        logger.warning("%d questions to forecast, keeping %d for this run", len(questions), limit)
        questions = questions[:limit]
    return questions


async def run(
    bot: CheriusBot, mode: str, ledger: Ledger | None = None, allowance: float | None = None
) -> tuple[int, int, int]:
    """Forecast on the tournaments of the mode. Returns (successes, failures, deferred).

    allowance: dollars this run may spend (None: no cap). A tournament that cannot be
    listed counts as a failure, so that a broken API never looks like "nothing to do".
    """
    ledger = ledger if ledger is not None else Ledger(None)
    questions = await collect_questions(bot, tournaments_for_mode(mode), mode, ledger)
    failures = bot.listing_errors
    logger.info("%d question(s) to forecast in mode %s", len(questions), mode)
    if not questions:
        return 0, failures, 0
    bot.ledger, bot.deferred, bot.credits_exhausted = ledger, {}, False
    bot.deadline = time.monotonic() + _env_int("BOT_RUN_SECONDS", 900)
    with MonetaryCostManager(allowance or 0) as meter:  # 0 means "no limit" for the package
        bot.meter = meter
        try:
            reports = await bot.forecast_questions(questions, return_exceptions=True)
        finally:
            ledger.runs += 1
            bot._checkpoint()
    successes = deferred = 0
    for question, report in zip(questions, reports):
        if not isinstance(report, BaseException):
            successes += 1
            logger.info(
                "OK %s -> %s", question.page_url, type(report).make_readable_prediction(report.prediction)
            )
        elif question_key(question) in bot.deferred:
            deferred += 1
            logger.warning("DEFERRED %s: %s", question.page_url, bot.deferred[question_key(question)])
        else:
            failures += 1
            logger.error("FAILED %s: %s: %s", question.page_url, type(report).__name__, str(report)[:600])
    return successes, failures, deferred


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CheriusBot - Metaculus FutureEval forecasting bot")
    parser.add_argument(
        "--mode",
        choices=["tournament", "test"],
        default="tournament",
        help=f"tournament: {TOURNAMENT_IDS}; test: {TEST_TOURNAMENT_ID} (default: tournament)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run research and forecasts but publish nothing to Metaculus",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import dotenv

        dotenv.load_dotenv()
    except ImportError:
        pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for handler in logging.getLogger().handlers:
        handler.addFilter(SecretFilter())
    for name, value in list(os.environ.items()):  # GitHub passes unset variables as empty strings
        if value == "" and name.startswith(("OPENAI_", "OPENROUTER_", "ANTHROPIC_")):
            del os.environ[name]
    missing = [name for name in required_env() if not os.getenv(name)]
    if missing:
        print(f"Missing environment variable(s): {', '.join(missing)}", file=sys.stderr)
        return 2
    if ONLINE_SUFFIX not in research_model_name():
        logger.warning("Research model %s has no web search: briefings rest on model memory", research_model_name())

    ledger = Ledger.load(os.getenv("BOT_LEDGER_PATH", "").strip() or DEFAULT_LEDGER_PATH)
    ledger.total_usd = max(ledger.total_usd, openrouter_usage_usd() or 0.0)
    ledger.save()  # the file must exist for the Actions cache, even if the run stops here
    allowance = run_allowance(ledger.total_usd)
    # In the tournament a question is forecast once and never again. The test area may be re-forecast.
    bot = build_bot(publish=not args.dry_run, skip_previously_forecasted=(args.mode == "tournament"))
    logger.info(
        "mode=%s publish=%s forecaster=%s researcher=%s draws=%d spent_so_far=$%.2f allowance_this_run=%s",
        args.mode,
        not args.dry_run,
        forecast_model_name(),
        research_model_name(),
        bot.predictions_per_research_report,
        ledger.total_usd,
        "unlimited" if allowance is None else f"${allowance:.2f}",
    )
    if allowance is not None and allowance < MIN_RUN_USD:
        print("::warning title=CheriusBot::Season budget spent (BOT_MAX_USD_TOTAL): nothing was forecast.")
        logger.error("BUDGET EXHAUSTED: $%.2f spent, cap BOT_MAX_USD_TOTAL reached. No forecast.", ledger.total_usd)
        return 0
    successes, failures, deferred = asyncio.run(run(bot, args.mode, ledger, allowance))
    logger.info(
        "Done: %d forecast(s), %d failure(s), %d deferred. Estimated cost: run $%.4f, season $%.2f",
        successes,
        failures,
        deferred,
        ledger.run_usd,
        ledger.total_usd + ledger.run_usd,
    )
    # Run summary as an annotation: shown on the public run page, where the logs need a sign-in.
    print(
        f"::notice title=CheriusBot::mode={args.mode} publish={not args.dry_run} model={forecast_model_name()} "
        f"forecasts={successes} failures={failures} deferred={deferred} "
        f"run_usd={ledger.run_usd:.4f} season_usd={ledger.total_usd + ledger.run_usd:.2f}"
    )
    if bot.credits_exhausted:
        print("::warning title=CheriusBot::Provider credits exhausted (HTTP 402): questions deferred, top up the key.")
        logger.warning("CREDITS EXHAUSTED: the provider answered 402; %d question(s) deferred to a later run", deferred)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
