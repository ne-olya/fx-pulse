"""Cache point-in-time NBK currency-market announcements and extract stated plans."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd


BASE = "https://nationalbank.kz"
YEAR_RUBRICS = {
    2018: 394, 2019: 395, 2020: 396, 2021: 1583, 2022: 1700,
    2023: 1844, 2024: 2099, 2025: 2274, 2026: 2313,
}
SEARCH = "валютному рынку"
ARTICLE_PATTERN = re.compile(r"^/ru/news/informacionnye-soobshcheniya/\d+$")
MONTHS = {
    "январе": 1, "феврале": 2, "марте": 3, "апреле": 4,
    "мае": 5, "июне": 6, "июле": 7, "августе": 8,
    "сентябре": 9, "октябре": 10, "ноябре": 11, "декабре": 12,
}


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: set[str] = set()
        self.pages: set[int] = {1}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        href = html.unescape(values.get("href") or "")
        if ARTICLE_PATTERN.match(href):
            self.links.add(href)
        match = re.search(r"[?&]page=(\d+)", href)
        if match:
            self.pages.add(int(match.group(1)))


class _ArticleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.published_at = ""
        self._content_depth = 0
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "meta" and values.get("property") == "og:title":
            self.title = values.get("content") or ""
        if tag == "meta" and values.get("property") == "og:article:published_time":
            self.published_at = values.get("content") or ""
        classes = set((values.get("class") or "").split())
        if self._content_depth:
            self._content_depth += 1
        elif tag == "div" and "post__content-text" in classes:
            self._content_depth = 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if self._content_depth:
            self._content_depth -= 1

    def handle_endtag(self, tag: str) -> None:
        if self._content_depth:
            self._content_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._content_depth and data.strip():
            self._text.append(data.strip())

    @property
    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._text)).strip()


def _get(url: str, retries: int = 4) -> str:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "fx-pulse-research/0.1"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as error:
            last = error
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"NBK request failed: {url}: {last}")


def discover_articles() -> list[str]:
    links: set[str] = set()
    for year, rubric in YEAR_RUBRICS.items():
        query = urllib.parse.urlencode({"search": SEARCH})
        root = f"{BASE}/ru/news/informacionnye-soobshcheniya/rubrics/{rubric}?{query}"
        parser = _PageParser()
        parser.feed(_get(root))
        links.update(parser.links)
        for page in range(2, max(parser.pages) + 1):
            page_parser = _PageParser()
            page_parser.feed(_get(f"{root}&page={page}"))
            links.update(page_parser.links)
        print(f"NBK {year}: cumulative {len(links)} article links", flush=True)
    return sorted(f"{BASE}{link}" for link in links)


def _parse_article(url: str) -> dict[str, str]:
    parser = _ArticleParser()
    parser.feed(_get(url))
    return {"url": url, "title": parser.title, "published_at": parser.published_at, "text": parser.text}


def _number(value: str, unit: str | None) -> float:
    number = float(value.replace(" ", "").replace(",", "."))
    return number * 1000 if unit == "млрд" else number


def _extract_range(sentence: str) -> tuple[float, float] | None:
    pattern = re.compile(
        r"от\s+([0-9][0-9\s]*(?:[,.][0-9]+)?)\s*(млн|млрд)?\s+до\s+"
        r"([0-9][0-9\s]*(?:[,.][0-9]+)?)\s*(млн|млрд)\s+(?:доллар|долл)",
        re.IGNORECASE,
    )
    match = pattern.search(sentence)
    if not match:
        return None
    final_unit = match.group(4).lower()
    return _number(match.group(1), (match.group(2) or final_unit).lower()), _number(match.group(3), final_unit)


def extract_plan(article: dict[str, str]) -> dict[str, object] | None:
    text = article["text"]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sale_sentence = next(
        (
            sentence for sentence in sentences
            if "ожидается продажа валюты из Национального фонда" in sentence
            or "ожидается продажа иностранной валюты из Национального фонда" in sentence
        ),
        None,
    )
    if sale_sentence is None:
        return None
    amount = _extract_range(sale_sentence)
    month_match = re.search(
        r"в\s+(" + "|".join(MONTHS) + r")(?:\s+(\d{4})\s+года)?", sale_sentence, re.IGNORECASE
    )
    if amount is None or month_match is None:
        return None
    month = MONTHS[month_match.group(1).lower()]
    publication = pd.Timestamp(article["published_at"])
    year = int(month_match.group(2)) if month_match.group(2) else publication.year
    if month < publication.month and not month_match.group(2):
        year += 1
    purchases = []
    explicit_no_purchase = False
    for sentence in sentences:
        lowered = sentence.lower()
        if "покупка валюты" in lowered and "не планируется" in lowered:
            explicit_no_purchase = True
        if "покупка валюты" in lowered and ("ожидается" in lowered or "планируется" in lowered):
            parsed = _extract_range(sentence)
            if parsed:
                purchases.append(sum(parsed) / 2)
    low, high = amount
    purchase_mid = sum(purchases) if purchases else (0.0 if explicit_no_purchase else np.nan)
    return {
        "plan_month": f"{year:04d}-{month:02d}-01",
        "published_at": article["published_at"],
        "planned_sale_low_usd_mn": low,
        "planned_sale_high_usd_mn": high,
        "planned_sale_mid_usd_mn": (low + high) / 2,
        "planned_purchase_mid_usd_mn": purchase_mid,
        "planned_net_sale_mid_usd_mn": (low + high) / 2 - purchase_mid,
        "url": article["url"],
        "source_sentence": sale_sentence,
    }


def run(
    *, raw_output: Path, processed_output: Path, meta_output: Path, refresh: bool = False
) -> dict[str, object]:
    if raw_output.exists() and not refresh:
        articles = [json.loads(line) for line in raw_output.read_text(encoding="utf-8").splitlines() if line]
    else:
        urls = discover_articles()
        with ThreadPoolExecutor(max_workers=8) as pool:
            articles = list(pool.map(_parse_article, urls))
        articles = sorted(articles, key=lambda item: (item["published_at"], item["url"]))
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        raw_output.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in articles), encoding="utf-8"
        )
    plans = [plan for article in articles if (plan := extract_plan(article)) is not None]
    frame = pd.DataFrame(plans).sort_values("plan_month", kind="mergesort") if plans else pd.DataFrame()
    processed_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(processed_output, index=False)
    meta = {
        "source": "National Bank of Kazakhstan official announcements",
        "source_root": f"{BASE}/ru/news/informacionnye-soobshcheniya",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "search": SEARCH,
        "articles": len(articles),
        "plans_extracted": len(frame),
        "first_plan_month": frame["plan_month"].min() if len(frame) else None,
        "last_plan_month": frame["plan_month"].max() if len(frame) else None,
        "raw_sha256": hashlib.sha256(raw_output.read_bytes()).hexdigest(),
        "processed_sha256": hashlib.sha256(processed_output.read_bytes()).hexdigest(),
        "limitation": "Regex extraction; extracted source sentences require manual audit before production use.",
    }
    meta_output.parent.mkdir(parents=True, exist_ok=True)
    meta_output.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-output", type=Path, default=Path("data/raw/nbk_fx_plan_announcements.jsonl"))
    parser.add_argument("--processed-output", type=Path, default=Path("data/processed/nbk_fx_plans.csv"))
    parser.add_argument("--meta-output", type=Path, default=Path("data/raw/nbk_fx_plan_announcements.meta.json"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(
        raw_output=args.raw_output,
        processed_output=args.processed_output,
        meta_output=args.meta_output,
        refresh=args.refresh,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
