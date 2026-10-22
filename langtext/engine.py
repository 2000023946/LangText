import os
import inspect
from abc import ABC, abstractmethod
from functools import wraps
from typing import Type, Callable, Any
from pydantic import BaseModel
from fastmcp import FastMCP
from dotenv import load_dotenv

# Automatically load a local .env file when the framework imports
load_dotenv()

# Initialize the global, zero-boilerplate MCP Server infrastructure
_mcp_server = FastMCP("Prompt-Language-Engine")

# =====================================================================
# 1. THE WRAPPER / ADAPTER PATTERN (Unified API Responses)
# =====================================================================
class LLMAdapter(ABC):
    """Abstract base class unifying diverse AI provider APIs into clean strings."""
    @abstractmethod
    def generate_structured_json(self, prompt: str, schema: Type[BaseModel]) -> str:
        pass

class GeminiAdapter(LLMAdapter):
    def __init__(self, model_name: str):
        from google import genai
        from google.genai import types
        self.client = genai.Client()
        self.model_name = model_name
        self.types = types

    def generate_structured_json(self, prompt: str, schema: Type[BaseModel]) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.1
            )
        )
        return response.text

class OpenAIAdapter(LLMAdapter):
    def __init__(self, model_name: str):
        from openai import OpenAI
        self.client = OpenAI()
        self.model_name = model_name

    def generate_structured_json(self, prompt: str, schema: Type[BaseModel]) -> str:
        response = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format=schema,
            temperature=0.1
        )
        return response.choices.message.content

# =====================================================================
# 2. THE FACTORY METHOD PATTERN (Environment Variable Guards)
# =====================================================================
class LLMClientFactory:
    """Dynamically resolves adapters and strictly enforces API Key presence."""
    @staticmethod
    def get_adapter(model_name: str) -> LLMAdapter:
        name_lower = model_name.lower()
        
        if "gemini" in name_lower:
            if "GEMINI_API_KEY" not in os.environ:
                raise KeyError(
                    "\n❌ [PromptLang Error]: Missing required environment variable 'GEMINI_API_KEY'.\n"
                    "👉 To fix this, add your key to a local .env file or run: export GEMINI_API_KEY='your_key'"
                )
            return GeminiAdapter(model_name)
            
        elif "gpt" in name_lower or "o1" in name_lower or "o3" in name_lower:
            if "OPENAI_API_KEY" not in os.environ:
                raise KeyError(
                    "\n❌ [PromptLang Error]: Missing required environment variable 'OPENAI_API_KEY'.\n"
                    "👉 To fix this, add your key to a local .env file or run: export OPENAI_API_KEY='your_key'"
                )
            return OpenAIAdapter(model_name)
            
        else:
            raise ValueError(f"Unsupported model target: '{model_name}'. Implement an Adapter to expand.")

# =====================================================================
# 3. THE FACADE DECORATOR & RUNTIME
# =====================================================================
def ai_agent(model: str = "gemini-2.5-flash", max_retries: int = 3) -> Callable:
    """
    The main developer Facade. Hides token logic, model switching, 
    JSON parsing, and structural loop retries completely out of sight.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Inspect function schema constraints
        sig = inspect.signature(func)
        return_type: Type[BaseModel] = sig.return_annotation
        
        if not issubclass(return_type, BaseModel):
            raise TypeError(f"Function '{func.__name__}' must have a Pydantic BaseModel return type hint.")

        # Grab raw multi-line English docstring block
        english_algorithm = inspect.getdoc(func)
        if not english_algorithm:
            raise ValueError(f"Function '{func.__name__}' requires an English algorithm body docstring.")

        # Instantiate our adapter using our Factory pattern right at compilation time
        adapter = LLMClientFactory.get_adapter(model)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> BaseModel:
            # Bind parameters seamlessly to their named variables
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Heavy Lifting: Injects data values into matching English text slots
            formatted_prompt = english_algorithm.format(**bound_args.arguments)
            current_prompt = formatted_prompt
            
            # CPU Control Flow: Manage deterministic retries and error self-healing loops
            last_error = None
            for attempt in range(max_retries):
                try:
                    raw_json_text = adapter.generate_structured_json(current_prompt, return_type)
                    return return_type.model_validate_json(raw_json_text)
                except Exception as e:
                    last_error = e
                    print(f"🔄 [Retry {attempt + 1}/{max_retries}] Auto-healing schema error in '{func.__name__}': {e}")
                    current_prompt = f"{formatted_prompt}\n\nCRITICAL: Your previous response caused a validation failure.\nError context: {str(e)}\nFix your schema generation."

            raise RuntimeError(f"Engine failed execution paths for '{func.__name__}' after {max_retries} retries. Source: {last_error}")

        # 4. AUTOMATIC NO-CODE MCP EXPORT
        @_mcp_server.tool(name=func.__name__)
        def mcp_tool_wrapper(**kwargs: Any) -> dict:
            result = wrapper(**kwargs)
            return result.model_dump()

        mcp_tool_wrapper.__doc__ = f"Executes agent tool: {func.__name__} via promptlang architecture."
        
        return wrapper
    return decorator

def run_mcp_server(transport: str = "stdio"):
    """Exposes all registered English functions instantly over a terminal stream or local network."""
    if transport == "http":
        print("🌐 PromptLang Engine running as a Network Service on port 8000...")
        _mcp_server.run(transport="http", port=8000)
    else:
        print("💻 PromptLang Engine running over Terminal STDIO...")
        _mcp_server.run(transport="stdio")
