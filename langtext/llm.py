
import json
import os
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from langtext.tools import Tool


class LLMAdapter(ABC):
    @abstractmethod
    def generate_structured_json(
        self,
        prompt: str,
        schema: type[BaseModel],
        tools: list[Tool],
        max_tool_calls: int = 10,
    ) -> str:
        raise NotImplementedError


class OpenAIAdapter(LLMAdapter):
    def __init__(self, model_name: str):
        from openai import OpenAI

        self.client = OpenAI()
        self.model_name = model_name

    def generate_structured_json(
        self,
        prompt: str,
        schema: type[BaseModel],
        tools: list[Tool],
        max_tool_calls: int = 10,
    ) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt}
        ]
        tools_by_name = {item.name: item for item in tools}
        tool_schemas = [item.openai_schema() for item in tools]
        calls_made = 0

        if tools:
            while True:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tool_schemas,
                    tool_choice="auto",
                )

                message = response.choices[0].message

                if not message.tool_calls:
                    if message.content:
                        messages.append({
                            "role": "assistant",
                            "content": message.content,
                        })
                    break

                # Preserve the assistant tool-call message in conversation history.
                messages.append(
                    message.model_dump(exclude_none=True)
                )

                for call in message.tool_calls:
                    if calls_made >= max_tool_calls:
                        raise RuntimeError(
                            "Maximum tool-call limit exceeded."
                        )

                    tool = tools_by_name.get(call.function.name)
                    if tool is None:
                        raise ValueError(
                            f"Model requested unavailable tool: "
                            f"{call.function.name}"
                        )

                    arguments = json.loads(call.function.arguments)
                    result = tool.execute(arguments)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, default=str),
                    })
                    calls_made += 1

        response = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            response_format=schema,
        )

        parsed = response.choices[0].message.parsed
        if parsed is not None:
            return parsed.model_dump_json()

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("OpenAI returned no structured output.")

        return content


class GeminiAdapter(LLMAdapter):
    def __init__(self, model_name: str):
        from google import genai
        from google.genai import types

        self.client = genai.Client()
        self.model_name = model_name
        self.types = types

    def generate_structured_json(
        self,
        prompt: str,
        schema: type[BaseModel],
        tools: list[Tool],
        max_tool_calls: int = 10,
    ) -> str:
        types = self.types
        tools_by_name = {item.name: item for item in tools}

        history: list[Any] = [prompt]
        calls_made = 0

        declarations = [
            types.FunctionDeclaration(
                name=item.name,
                description=item.description,
                parameters_json_schema=item.args_model.model_json_schema(),
            )
            for item in tools
        ]

        if tools:
            tool_config = [
                types.Tool(function_declarations=declarations)
            ]

            while True:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=history,
                    config=types.GenerateContentConfig(
                        tools=tool_config,
                    ),
                )

                if not response.candidates:
                    raise RuntimeError("Gemini returned no candidates.")

                content = response.candidates[0].content
                history.append(content)

                function_calls = [
                    part.function_call
                    for part in (content.parts or [])
                    if part.function_call is not None
                ]

                if not function_calls:
                    break

                function_responses = []

                for call in function_calls:
                    if calls_made >= max_tool_calls:
                        raise RuntimeError(
                            "Maximum tool-call limit exceeded."
                        )

                    tool = tools_by_name.get(call.name)
                    if tool is None:
                        raise ValueError(
                            f"Model requested unavailable tool: {call.name}"
                        )

                    arguments = dict(call.args or {})
                    result = tool.execute(arguments)

                    function_responses.append(
                        types.Part.from_function_response(
                            name=call.name,
                            response={"result": result},
                        )
                    )
                    calls_made += 1

                history.append(
                    types.Content(
                        role="user",
                        parts=function_responses,
                    )
                )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=history,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.1,
            ),
        )

        if not response.text:
            raise RuntimeError("Gemini returned no structured output.")

        return response.text


class LLMClientFactory:
    @staticmethod
    def get_adapter(model_name: str) -> LLMAdapter:
        name = model_name.lower()

        if "gemini" in name:
            if not os.getenv("GEMINI_API_KEY"):
                raise KeyError("Missing GEMINI_API_KEY.")
            return GeminiAdapter(model_name)

        if any(token in name for token in ("gpt", "o1", "o3")):
            if not os.getenv("OPENAI_API_KEY"):
                raise KeyError("Missing OPENAI_API_KEY.")
            return OpenAIAdapter(model_name)

        raise ValueError(f"Unsupported model: '{model_name}'.")
