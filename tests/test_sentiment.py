"""Tests for the news-sentiment scaffold (no network — fetcher is injected)."""

from __future__ import annotations

from typing import Any

import numpy as np

from etf_intel.ingestion.sentiment import LexiconSentimentScorer, NewsSentimentProvider


def test_lexicon_scorer_signs() -> None:
    scorer = LexiconSentimentScorer()
    assert scorer.score("Stock surges after earnings beat and record profit") > 0
    assert scorer.score("Shares plunge on downgrade and weak guidance") < 0
    assert scorer.score("The fund holds a basket of securities") == 0.0
    # Score stays within bounds.
    assert -1.0 <= scorer.score("miss loss cut weak fall") <= 1.0


def test_news_provider_aggregates_with_fake_fetcher() -> None:
    fake = {
        "AAA": [{"title": "AAA surges on strong growth"}, {"title": "AAA hits record high"}],
        "BBB": [{"content": {"title": "BBB plunges on downgrade"}}],  # new-schema item
        "CCC": [],  # no news
    }

    def fetcher(ticker: str) -> list[dict[str, Any]]:
        return fake.get(ticker, [])

    out = NewsSentimentProvider(news_fetcher=fetcher).fetch(["AAA", "BBB", "CCC"])
    out = out.set_index("ticker")
    assert out.loc["AAA", "sentiment"] > 0
    assert out.loc["BBB", "sentiment"] < 0
    assert np.isnan(out.loc["CCC", "sentiment"])
    assert out.loc["AAA", "n_headlines"] == 2
    assert out.loc["CCC", "n_headlines"] == 0
