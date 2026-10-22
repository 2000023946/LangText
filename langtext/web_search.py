
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
        }


class SearchProvider(ABC):
    @abstractmethod
    def search(
        self, query: str, max_results: int = 5
    ) -> list[SearchResult]:
        raise NotImplementedError

    @staticmethod
    def require_env(name: str) -> str:
        value = os.getenv(name)
        if not value:
            raise KeyError(f"Missing environment variable: {name}")
        return value


class TavilySearchProvider(SearchProvider):
    def __init__(self):
        self.api_key = self.require_env("TAVILY_API_KEY")

    def search(
        self, query: str, max_results: int = 5
    ) -> list[SearchResult]:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "max_results": max(1, min(max_results, 20)),
            },
            timeout=30,
        )
        response.raise_for_status()

        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
            )
            for item in response.json().get("results", [])
        ]


class BraveSearchProvider(SearchProvider):
    def __init__(self):
        self.api_key = self.require_env("BRAVE_SEARCH_API_KEY")

    def search(
        self, query: str, max_results: int = 5
    ) -> list[SearchResult]:
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
            },
            params={
                "q": query,
                "count": max(1, min(max_results, 20)),
            },
            timeout=30,
        )
        response.raise_for_status()

        items = response.json().get("web", {}).get("results", [])
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
            )
            for item in items
        ]


class GoogleSearchProvider(SearchProvider):
    """
    Adapter for Google's Custom Search JSON API.
    Requires existing API access and a Programmable Search Engine ID.
    """

    def __init__(self):
        self.api_key = self.require_env("GOOGLE_SEARCH_API_KEY")
        self.engine_id = self.require_env("GOOGLE_SEARCH_ENGINE_ID")

    def search(
        self, query: str, max_results: int = 5
    ) -> list[SearchResult]:
        response = requests.get(
            "https://customsearch.googleapis.com/customsearch/v1",
            params={
                "key": self.api_key,
                "cx": self.engine_id,
                "q": query,
                "num": max(1, min(max_results, 10)),
            },
            timeout=30,
        )
        response.raise_for_status()

        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
            )
            for item in response.json().get("items", [])
        ]


class WebSearch:
    """Provider-independent facade for web search."""

    def __init__(self, provider: SearchProvider):
        self._provider = provider

    def search(
        self, query: str, max_results: int = 5
    ) -> list[SearchResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must be a non-empty string.")

        if not 1 <= max_results <= 20:
            raise ValueError("max_results must be between 1 and 20.")

        return self._provider.search(query.strip(), max_results)


class WebSearchFactory:
    """Construct a search facade for a named provider."""

    _providers = {
        "tavily": TavilySearchProvider,
        "brave": BraveSearchProvider,
        "google": GoogleSearchProvider,
    }

    @classmethod
    def create(cls, provider: str = "tavily") -> WebSearch:
        name = provider.strip().lower()

        try:
            provider_class = cls._providers[name]
        except KeyError:
            supported = ", ".join(sorted(cls._providers))
            raise ValueError(
                f"Unknown search provider '{provider}'. "
                f"Supported providers: {supported}"
            ) from None

        return WebSearch(provider_class())
