# 🧠 LangText

**Write AI agents using Python on the outside and natural-language logic on the inside.**

LangText is an open-source Python library for building AI-powered applications with function signatures, Pydantic data contracts, and natural-language instructions.

Instead of scattering prompt templates, model calls, and output parsing throughout your application, define an AI-powered function with a decorator. Python specifies the interface, the docstring describes the task, and the library handles the model interaction.

## ✨ Core Ideas

### 1. 🛡️ AI Encapsulation

Keep application code separate from provider-specific model logic.

```python
from pydantic import BaseModel
from langtext import ai_agent


class Summary(BaseModel):
    summary: str
    key_points: list[str]


@ai_agent(model="gemini-2.5-flash")
def summarize(text: str) -> Summary:
    """
    Summarize the following text: {text}

    Return a concise summary and its key points.
    """
```

The function signature defines the input and output contract. The docstring describes the task, and the return value is validated against the Pydantic model.

Supported models and their capabilities depend on the adapters implemented by LangText.

### 2. 🔗 Composable Functions and Python Control Flow

Build larger applications from smaller, focused AI functions.

- **Python** controls application flow, loops, conditionals, and integration.
- **Pydantic** defines structured input and output contracts.
- **Natural-language instructions** describe the task assigned to the model.
- **LangText** connects these pieces through a consistent function interface.

This design makes AI functionality easier to isolate, reuse, and test.

### 3. 🔧 Tools and Web Search

Give an AI function access to explicitly configured tools. LangText can expose tool schemas to supported models and execute tool calls requested by the model.

For example, configure a web search tool with Tavily:

```python
from pydantic import BaseModel
from langtext import ai_agent
from langtext.tools import web_search


class ResearchReport(BaseModel):
    summary: str
    sources: list[str]


@ai_agent(tools=[web_search(provider="tavily")])
def research(topic: str) -> ResearchReport:
    """
    Research {topic} using @web_search.

    Summarize the evidence and include relevant source URLs.
    """
```

The `web_search()` call configures a tool; it does not execute a search immediately. At runtime, the model can request searches using the tool's defined arguments.

Supported search providers in the current implementation include:

- `tavily`
- `brave`
- `google`

Each provider requires its own credentials and applicable API access. Google's adapter uses its Custom Search API, which may not be available to every new customer.

## 🆚 Traditional Approach vs. LangText

A conventional implementation may require separate prompt templates, model configuration, structured-output parsing, and error handling.

With LangText, the application-facing function can express the same task more directly.

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
    Analyze the target company.

    Company name: {company.name}
    Industry: {company.industry}

    Determine the company's competitive threat level.
    Recommend a strategic pivot that could protect its
    long-term market position.
    """
```

The intended benefit is a compact application-facing interface. Provider compatibility, structured-output support, and error behavior depend on the selected adapter and installed version.

## 🚀 Quick Start

### 1. Install LangText

```bash
python -m pip install langtext
```

To work on the source code locally instead:

```bash
git clone https://github.com/2000023946/promptlang-project.git
cd promptlang-project
python -m pip install -e .
```

Replace the repository URL above if your published repository uses a different URL.

### 2. Configure API credentials

For Gemini:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

For OpenAI:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

For Tavily web search:

```bash
export TAVILY_API_KEY="your-tavily-api-key"
```

Brave and Google search use their respective credentials. Google also requires a Search Engine ID.

Alternatively, create a `.env` file in your application directory:

```text
GEMINI_API_KEY=your-gemini-api-key
OPENAI_API_KEY=your-openai-api-key
TAVILY_API_KEY=your-tavily-api-key
```

Keep credentials out of source control. Add `.env` to `.gitignore`, and never commit real API keys.

### 3. Define an AI function

Create `app.py`:

```python
from pydantic import BaseModel
from langtext import ai_agent


class AnalyzedData(BaseModel):
    sentiment: str
    summary: str


@ai_agent(model="gemini-2.5-flash")
def triage_text(text: str) -> AnalyzedData:
    """
    Analyze the supplied text.

    Classify its sentiment as POSITIVE, NEGATIVE, or NEUTRAL.
    Summarize its main point in one sentence.
    """


if __name__ == "__main__":
    result = triage_text(
        "The new release is faster and much easier to use."
    )
    print(result.model_dump_json(indent=2))
```

Run it:

```bash
python app.py
```

The example assumes the configured model supports the required structured-output behavior and that your API credentials are configured correctly.

### 4. Compose functions with Python

AI functions remain ordinary Python-callable functions, so you can use their validated results in application logic.

```python
analysis = triage_text("The service is slow and frequently crashes.")

if analysis.sentiment == "NEGATIVE":
    print("Flag this text for review.")
```

The conditional is ordinary Python control flow; the classification itself is produced by the AI function.

## 🔌 Model Context Protocol (MCP)

LangText includes an MCP server integration intended to expose compatible AI functions as tools for external MCP clients.

If your installed version supports `run_mcp_server`, an application entry point can look like this:

```python
from langtext import run_mcp_server


if __name__ == "__main__":
    run_mcp_server(transport="stdio")
```

Configure your MCP client to launch the script using the appropriate Python interpreter and working directory.

MCP registration, argument-schema generation, and transport behavior depend on the installed FastMCP version and LangText implementation.

## 🧩 Design Philosophy

LangText separates responsibilities:

- **Python:** application structure, integration, and control flow.
- **Pydantic:** structured data contracts and validation.
- **Natural-language instructions:** descriptions of work performed by an LLM.
- **Provider adapters:** isolation of model-specific APIs.
- **Tools:** explicitly configured capabilities, such as web search.
- **MCP:** integration with compatible external tool clients.

The goal is not to replace Python with natural language. It is to make the AI-dependent parts of a Python application easier to express and compose.

## 🤝 Contributing

Contributions are welcome, especially for additional model providers, tools, testing, validation, and MCP support.

To get started:

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install the package and development dependencies.
4. Run the project's tests.
5. Submit changes through a pull request.

For example:

```bash
git clone <your-repository-url>
cd <your-repository-directory>

python -m venv .venv
source .venv/bin/activate

python -m pip install -e .
```

Provider implementations should follow the existing adapter interface and include tests for successful responses, malformed outputs, tool execution, API failures, and retry behavior.

## 📜 License

LangText is open-source software distributed under the MIT License. See [LICENSE](LICENSE) for details.
