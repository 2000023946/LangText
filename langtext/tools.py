
import inspect
import os
from pathlib import Path
from typing import Any, Callable, get_type_hints

from pydantic import create_model

from langtext.web_search import WebSearchFactory


class Tool:
    """A callable capability with a name, description, schema, and executor."""

    def __init__(
        self,
        name: str,
        description: str,
        arguments_model: type,
        executor: Callable[..., Any],
    ):
        self.name = name
        self.description = description
        self.args_model = arguments_model
        self._executor = executor

    def execute(self, arguments: dict[str, Any]) -> Any:
        validated = self.args_model.model_validate(arguments)
        return self._executor(**validated.model_dump())

    @classmethod
    def from_function(cls, func: Callable) -> "Tool":
        signature = inspect.signature(func)
        hints = get_type_hints(func)
        fields = {}

        for name, parameter in signature.parameters.items():
            annotation = hints.get(name, parameter.annotation)

            if annotation is inspect.Parameter.empty:
                raise TypeError(
                    f"Tool parameter '{name}' needs a type annotation."
                )

            default = (
                ...
                if parameter.default is inspect.Parameter.empty
                else parameter.default
            )
            fields[name] = (annotation, default)

        arguments_model = create_model(
            f"{func.__name__.title()}Arguments",
            **fields,
        )

        return cls(
            name=func.__name__,
            description=inspect.getdoc(func) or func.__name__,
            arguments_model=arguments_model,
            executor=func,
        )

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


def web_search(provider: str = "tavily") -> Tool:
    """
    Create a built-in web search tool.

    The provider is selected during configuration. The model supplies
    the actual query when it calls this tool.
    """
    selected_provider = provider.strip().lower()

    # Defer provider construction and API-key checks until tool execution.
    def execute_search(
        query: str,
        max_results: int = 5,
    ) -> list[dict[str, str]]:
        search = WebSearchFactory.create(selected_provider)
        results = search.search(query, max_results=max_results)
        return [result.to_dict() for result in results]

    execute_search.__name__ = "web_search"
    execute_search.__doc__ = (
        f"Search the public web using {selected_provider}. "
        "Returns titles, URLs, and snippets for relevant pages."
    )

    return Tool.from_function(execute_search)


def _filesystem_root() -> Path:
    return Path(os.getenv("LANGTEXT_FILES_ROOT", ".")).expanduser().resolve()


def _safe_path(path: str) -> Path:
    root = _filesystem_root()
    candidate = (root / path).resolve()

    if not candidate.is_relative_to(root):
        raise ValueError("Path is outside the permitted filesystem root.")

    return candidate


def list_files(path: str = ".") -> list[str]:
    """List files and directories inside the permitted filesystem root."""
    directory = _safe_path(path)

    if not directory.is_dir():
        raise NotADirectoryError(path)

    return sorted(item.name for item in directory.iterdir())


def read_file(path: str) -> str:
    """Read a UTF-8 text file inside the permitted filesystem root."""
    file_path = _safe_path(path)

    if not file_path.is_file():
        raise FileNotFoundError(path)

    if file_path.stat().st_size > 1_000_000:
        raise ValueError("File exceeds the 1 MB read limit.")

    return file_path.read_text(encoding="utf-8")


def search_files(query: str, path: str = ".") -> list[dict[str, Any]]:
    """Search for text in files inside the permitted filesystem root."""
    directory = _safe_path(path)

    if not directory.is_dir():
        raise NotADirectoryError(path)

    root = _filesystem_root()
    matches = []

    excluded = {
        ".git", ".venv", "venv", "node_modules", "__pycache__"
    }

    for file_path in directory.rglob("*"):
        if not file_path.is_file():
            continue

        if any(part in excluded for part in file_path.parts):
            continue

        try:
            if file_path.stat().st_size > 1_000_000:
                continue

            lines = file_path.read_text(encoding="utf-8").splitlines()

            for line_number, line in enumerate(lines, start=1):
                if query.casefold() in line.casefold():
                    matches.append({
                        "path": str(file_path.relative_to(root)),
                        "line": line_number,
                        "text": line.strip(),
                    })

                    if len(matches) >= 100:
                        return matches

        except (UnicodeDecodeError, PermissionError, OSError):
            continue

    return matches


# Export filesystem functions as Tool objects for ai_agent.
list_files = Tool.from_function(list_files)
read_file = Tool.from_function(read_file)
search_files = Tool.from_function(search_files)
