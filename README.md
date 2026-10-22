# 🧠 LangText

**Write AI agents using Python on the outside and natural-language logic on the inside.**

LangText is an open-source domain-specific language (DSL) for building AI-powered applications. It combines Python's explicit control flow and structured data contracts with LLM-driven natural-language processing.

With LangText, function signatures define input and output contracts, while multiline docstrings describe the instructions an LLM executes. The goal is to make AI logic modular, testable, and easier to maintain without scattering prompt templates, parsing code, and provider-specific logic throughout your application.

---

## ✨ Three Core Ideas

### 1. 🛡️ AI Encapsulation

LangText aims to separate application logic from model-provider details. Instead of embedding API calls and prompt-processing code throughout your application, you can define an AI-powered function using a decorator.

For example, switching from `gemini-2.5-flash` to another supported model can be as simple as changing the model parameter. Actual provider compatibility depends on the adapters implemented by LangText.

### 2. 🔗 Composable Functions and Python Control Flow

Break complex AI tasks into smaller, focused functions rather than relying on one enormous prompt.

Pydantic models can define structured inputs and outputs, allowing compatible functions to pass validated data between steps. Python loops, conditionals, and exception handling remain in control of application execution, while the LLM handles natural-language reasoning.

### 3. 🔌 Model Context Protocol (MCP) Integration

LangText is designed to make AI functions available as tools through the [Model Context Protocol (MCP)](https://modelcontextprotocol.io).

By inspecting function signatures, type hints, and docstrings, an MCP integration can expose compatible functions to supported agent clients. The exact tool-generation and server behavior depends on the implementation.

---

## 🆚 The Traditional Approach vs. LangText

Consider a function that generates a corporate strategy.

### ❌ Traditional approach

A conventional implementation may require separate prompt templates, model configuration, structured-output parsing, and error handling.

```python
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

class Strategy(BaseModel):
    threat_level: str
    pivot: str

model = ChatOpenAI(model="gpt-4o-mini")
parser = PydanticOutputParser(pydantic_object=Strategy)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Analyze this company: {data}\n{format_instructions}"),
    ("user", "Generate a strategic report.")
]).partial(
    format_instructions=parser.get_format_instructions()
)

chain = prompt | model | parser
```

### ✅ The LangText approach

LangText aims to express the same task as a Python function whose docstring contains the natural-language instructions.

```python
from pydantic import BaseModel
from langtext import ai_agent

class CompanyData(BaseModel):
    name: str
    industry: str

class Strategy(BaseModel):
    threat_level: str
    pivot: str

@ai_agent(model="gemini-2.5-flash", max_retries=3)
def generate_strategy(company: CompanyData) -> Strategy:
    """
    Analyze the target company using the provided information.

    Company name: {company.name}
    Industry: {company.industry}

    Determine the competitive threat level facing the company.
    Recommend a strategic pivot that could help protect its
    long-term market position.
    """
```

The intended benefit is a smaller application-facing interface, with provider calls and structured-output handling delegated to the library.

---

## 🚀 Quick Start

### 1. Installation

Install LangText from PyPI:

```bash
python -m pip install langtext
```

If the package has not yet been published, install it from your local repository instead:

```bash
python -m pip install -e .
```

### 2. Configure API credentials

Set the environment variable for your chosen provider. For example:

```bash
export GEMINI_API_KEY="your-api-key"
```

Or create a `.env` file in your application directory:

```text
GEMINI_API_KEY=your-api-key
OPENAI_API_KEY=your-api-key
```

Use environment variables or a local `.env` file to keep credentials out of source control. Add `.env` to `.gitignore`.

### 3. Define an AI function

Create `app.py`:

```python
from pydantic import BaseModel
from langtext import ai_agent

class RawText(BaseModel):
    content: str

class AnalyzedData(BaseModel):
    sentiment: str
    summary: str

@ai_agent(model="gemini-2.5-flash")
def triage_text(text: RawText) -> AnalyzedData:
    """
    Analyze the supplied text.

    Classify its sentiment as POSITIVE, NEGATIVE, or NEUTRAL.
    Summarize its main point in one sentence.
    """
```

The example assumes that LangText supports the shown decorator, input interpolation, and Pydantic output conversion. Consult the implementation for the exact supported behavior.

### 4. Expose functions through MCP

If your installed LangText version implements `run_mcp_server`, you can use the following application entry point:

```python
from langtext import run_mcp_server

if __name__ == "__main__":
    run_mcp_server(transport="stdio")
```

Run the application:

```bash
python app.py
```

For stdio-based MCP integrations, configure your MCP client to launch this script using the appropriate Python interpreter and working directory.

---

## 🧩 Design Philosophy

LangText is built around a simple separation of responsibilities:

- **Python** handles application structure, data flow, control flow, and integration.
- **Pydantic** defines structured data contracts.
- **Natural-language instructions** describe the work performed by an LLM.
- **Provider adapters** isolate supported model APIs from application code.
- **MCP integration** can expose compatible functions as tools to external clients.

This approach is intended to encourage modular AI functions that are easier to compose, inspect, and test.

---

## 🤝 Contributing

Contributions are welcome, especially for new model providers, improved validation, testing, and MCP support.

To get started:

1. Clone the repository.
2. Install the development dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Install LangText in editable mode:

   ```bash
   python -m pip install -e .
   ```

4. Run the project's tests before submitting a pull request.

Provider implementations should follow the project's existing adapter interface and include tests for successful responses, invalid outputs, API failures, and retry behavior.

---

## 📜 License

LangText is open-source software distributed under the MIT License. See [LICENSE](LICENSE) for details.