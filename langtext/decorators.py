
import inspect
from functools import wraps
from typing import Any, Callable, get_type_hints

from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import BaseModel

from langtext.llm import (
    LLMClientFactory,
    execute_required_tools,
)
from langtext.tools import Tool

load_dotenv()

_mcp_server = FastMCP("LangText")


def ai_agent(
    model: str = "gemini-2.5-flash",
    max_retries: int = 3,
    tools: list[Tool | Callable[..., Any]] | None = None,
    max_tool_calls: int = 10,
    tool_policy: str = "auto",
) -> Callable:
    """Turn a documented Python function into an LLM-powered function."""

    if max_retries < 1:
        raise ValueError("max_retries must be at least 1.")

    if max_tool_calls < 1:
        raise ValueError("max_tool_calls must be at least 1.")

    if tool_policy not in ("auto", "required"):
        raise ValueError(
            "tool_policy must be 'auto' or 'required'."
        )

    allowed_tools: list[Tool] = []

    for item in tools or []:
        if isinstance(item, Tool):
            allowed_tools.append(item)
        elif callable(item):
            allowed_tools.append(Tool.from_function(item))
        else:
            raise TypeError(
                "Each tool must be a LangText Tool or a Python function."
            )

    tool_names = [item.name for item in allowed_tools]

    if len(tool_names) != len(set(tool_names)):
        raise ValueError("Tool names must be unique.")

    if tool_policy == "required" and not allowed_tools:
        raise ValueError(
            "tool_policy='required' needs at least one configured tool."
        )

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(func)
        hints = get_type_hints(func)
        return_type = hints.get(
            "return", signature.return_annotation
        )

        if (
            return_type is inspect.Signature.empty
            or not isinstance(return_type, type)
            or not issubclass(return_type, BaseModel)
        ):
            raise TypeError(
                f"Function '{func.__name__}' must return a Pydantic "
                "BaseModel type, for example -> ResearchReport."
            )

        instructions = inspect.getdoc(func)
        if not instructions:
            raise ValueError(
                f"Function '{func.__name__}' needs a descriptive docstring."
            )

        adapter = LLMClientFactory.get_adapter(model)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> BaseModel:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()

            try:
                formatted_prompt = instructions.format(
                    **bound.arguments
                )
            except (KeyError, IndexError, ValueError) as exc:
                raise ValueError(
                    f"Could not format instructions for "
                    f"'{func.__name__}': {exc}. "
                    "Use {{ and }} for literal braces."
                ) from exc

            # Execute required tools exactly once per wrapper invocation.
            # Retrying the LLM response must not rerun these tools.
            required_results = None

            if tool_policy == "required":
                required_results = execute_required_tools(
                    tools=allowed_tools,
                    input_arguments=dict(bound.arguments),
                    max_tool_calls=max_tool_calls,
                )

            last_error: Exception | None = None

            for attempt in range(1, max_retries + 1):
                try:
                    raw_json = adapter.generate_structured_json(
                        prompt=formatted_prompt,
                        schema=return_type,
                        tools=allowed_tools,
                        max_tool_calls=max_tool_calls,
                        tool_policy=tool_policy,
                        tool_results=required_results,
                    )

                    return return_type.model_validate_json(raw_json)

                except Exception as exc:
                    last_error = exc

                    if attempt < max_retries:
                        formatted_prompt = (
                            f"{instructions.format(**bound.arguments)}"
                            "\n\nYour previous attempt failed. Try again "
                            "and return a valid response matching the "
                            f"schema. Previous error: {exc}"
                        )

            raise RuntimeError(
                f"Agent '{func.__name__}' failed after "
                f"{max_retries} attempts."
            ) from last_error

        @_mcp_server.tool(name=func.__name__)
        def mcp_tool_wrapper(**kwargs: Any) -> dict:
            result = wrapper(**kwargs)
            return result.model_dump()

        mcp_tool_wrapper.__doc__ = (
            f"Runs the LangText agent '{func.__name__}'."
        )

        return wrapper

    return decorator


def run_mcp_server(transport: str = "stdio") -> None:
    """Start the LangText MCP server."""

    if transport == "http":
        _mcp_server.run(transport="http", port=8000)
    elif transport == "stdio":
        _mcp_server.run(transport="stdio")
    else:
        raise ValueError(
            "Unsupported transport. Use 'stdio' or 'http'."
        )
