# Ghi chú kiến thức & luồng hoạt động — Homework + Seminar

> Nguồn: `homework/week10_hw.ipynb`, `homework/ysda_mcp/*`, `seminar/ysda_agents_sem.ipynb`, `seminar/ysda_tools.py`
> Chủ đề: Week 10 — AI Agents thực hành

---

## PHẦN A. KIẾN THỨC CỐT LÕI

### A1. MCP (Model Context Protocol)

**MCP là gì?**
- Giao thức tiêu chuẩn cho phép model truy cập tools bên ngoài.
- Do Anthropic phát triển, được gọi qua **JSON RPC** trên các transport: `stdio`, `SSE`, `Streamable HTTP`.

**Ba "primitive" của server MCP:**
| Primitive | Ai điều khiển | Ví dụ |
|-----------|--------------|-------|
| **Prompts** | Người dùng | Slash commands, menu |
| **Resources** | Ứng dụng/client | File contents, git history |
| **Tools** | Model (LLM) | Gọi API, ghi file |

**Cách viết MCP server bằng `FastMCP` (Python):**
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Calculator")

@mcp.tool()
def add(a: float, b: float) -> float:
    return a + b

if __name__ == "__main__":
    mcp.run(transport="stdio")   # chạy qua stdio
```

**MCP client (từ seminar/homework):**
```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

class MCPClient:
    def __init__(self):
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.tools = []

    async def connect_to_server(self, server_script_path: str):
        server_params = StdioServerParameters(command="python", args=[server_script_path], env=None)
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        response = await self.session.list_tools()
        self.tools = response.tools

    def list_tools(self):
        print("Connected to server with tools:", [t.name for t in self.tools])
        return self.tools

    async def call_tool(self, tool_name, args):
        return await self.session.call_tool(tool_name, args)
```

**Kết nối MCP từ OpenAI Agents SDK:**
```python
from agents.mcp import MCPServerStdio

async with MCPServerStdio(
    name="Server name",
    params={"command": "python", "args": ["ysda_mcp/mcp_tool.py"]},
) as server:
    agent = Agent(..., mcp_servers=[server])
```

---

### A2. LLM tool calling (Function calling)

- Model được cung cấp danh sách tool dạng JSON schema:
  ```python
  tools = [{
      "type": "function",
      "function": {
          "name": "get_youtube_captions",
          "description": "...",
          "parameters": {
              "type": "object",
              "properties": {"video_id": {"type": "string"}},
              "required": ["video_id"]
          }
      }
  }]
  ```
- Model trả về `function_call`, chương trình thực thi rồi trả `function_call_output` lại cho model.

**Tự động chuyển hàm Python → JSON schema tool:**
```python
import inspect
PYTHON_TO_JSON = {str: "string", int: "integer", float: "number",
                  bool: "boolean", list: "array", dict: "object"}

def create_tool_description(func):
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    params_schema = {"type": "object", "properties": {}, "required": []}
    for name, param in sig.parameters.items():
        annotation = param.annotation
        json_type = PYTHON_TO_JSON.get(annotation, "string")  # fallback
        entry = {"type": json_type}
        if param.default is not inspect.Parameter.empty:
            entry["default"] = param.default
        else:
            params_schema["required"].append(name)
        params_schema["properties"][name] = entry
    return {"type": "function", "function": {
        "name": func.__name__, "description": doc, "parameters": params_schema}}
```

---

### A3. ReAct Agent (Reasoning + Acting)

**Ý tưởng:** Lặp `Thought → Action → Observation` rồi đưa ra `Final Answer`.

Các action trong seminar:
1. `Search[query]` — tìm web, trả url/title/snippet cho 5 trang.
2. `Visit web-page[link]` — lấy nội dung trang.
3. `Finish[answer]` — trả lời và kết thúc.

**Vòng lặp cơ bản:**
```python
def call_react(question, prompt=final_prompt, to_print=True):
    prompt += question + "\n"
    n_calls, n_badcalls = 0, 0
    for i in range(1, 8):
        n_calls += 1
        thought_action = <YOUR CODE: gọi llm>          # sinh Thought + Action
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except Exception as e:
            # xử lý lỗi parse
            pass
        obs = execute_action(action)                     # chạy action
        obs = obs.replace('\\n', '')
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str                               # thêm vào lịch sử
    return r, info
```

---

### A4. Multi-Agent với smolagents

**Tạo tool:**
```python
from smolagents import tool

@tool
def visit_webpage(link: str) -> str:
    """Visit web-page, returns content.
    Args: link: link to the webpage
    Returns: content of the webpage"""
    return get_webpage_content(link)
```

**Tạo agent:**
```python
from smolagents import CodeAgent, OpenAIModel, ToolCallingAgent, WebSearchTool

model = OpenAIModel(model_id=MODEL, api_base=URL, api_key=KEY)

web_agent = CodeAgent(
    tools=[WebSearchTool(), visit_webpage],
    model=model, max_steps=5,
    name="web_search_agent",
    description="Runs web searches for you.",
)

manager_agent = CodeAgent(  # manager điều phối các agent con
    tools=[], model=model, managed_agents=[web_agent],
)
answer = manager_agent.run("How does ReAct agent works? ...")
```

**Khái niệm quan trọng:**
- `CodeAgent`: agent có thể viết & chạy code.
- `managed_agents`: agent con do manager điều phối (multi-agent).
- `handoffs` (trong OpenAI Agents SDK): chuyển quyền điều khiển sang agent khác.

---

### A5. OpenAI Agents SDK

```python
from agents import Agent, ModelSettings, Runner
from agents.extensions.models.litellm_model import LitellmModel

no_reasoning = ModelSettings(extra_body={"reasoning": {"enabled": True}})

litellm_model = LitellmModel(model=MODEL_NAME, base_url=BASE_URL, api_key=OPENROUTER_KEY)

smart_agent = Agent(
    name="Smart Rewriter Agent",
    instructions="...",    # prompt hướng dẫn
    model=litellm_model,
    model_settings=no_reasoning,
)

result = await Runner.run(smart_agent, "Task: ...")
print(result.final_output)
```

**Multi-agent với handoffs:**
```python
ysda_agent = Agent(
    name="Ysda Interface Agent",
    handoff_description="Handles interactions with the data school lms system",
    instructions="...",
    mcp_servers=[server],
    model=litellm_model,
)

manager_agent = Agent(
    name="Manager Agent",
    model=litellm_model,
    instructions="You are a manager agent...",
    handoffs=[ysda_agent, smart_agent],
)
result = await Runner.run(manager_agent, "user query")
```

**Các thuộc tính agent:**
- `instructions`: prompt hệ thống.
- `handoffs`: danh sách agent có thể chuyển tới.
- `handoff_description`: mô tả để manager quyết định chuyển khi nào.
- `mcp_servers`: các MCP server agent dùng.
- `output_guardrails` / `input_guardrails`: bộ bảo vệ đầu ra/đầu vào.

---

### A6. Guardrails (Bộ bảo vệ)

**Input guardrail** — kiểm tra câu hỏi đầu vào của user:
```python
from agents import input_guardrail, GuardrailFunctionOutput

@input_guardrail
async def nlp_guardrail(ctx, agent, input):
    # kiểm tra user có hỏi giải bài NLP assignment không
    # trả về GuardrailFunctionOutput(tripwire_triggered=bool, output_info=...)
    ...
```

**Output guardrail** — dùng `transformers` classifier chống output toxic (KHÔNG dùng LLM):
```python
from transformers import pipeline
from pydantic import BaseModel
from agents import (Agent, GuardrailFunctionOutput, OutputGuardrailTripwireTriggered,
                    RunContextWrapper, Runner, output_guardrail)

toxicity_classifier = pipeline(...)   # pretrained text classifier (HF)

class ToxicityResult(BaseModel):
    toxic_score: float
    label: str

class MessageOutput(BaseModel):
    response: str

@output_guardrail
async def toxicity_guardrail(ctx, agent, output: MessageOutput) -> GuardrailFunctionOutput:
    # chạy classifier trên output.response
    # nếu toxic_score vượt ngưỡng -> tripwire_triggered = True
    ...

moderated_agent = Agent(
    name="Test agent",
    instructions="You answer extremely unpolitely and rudely.",
    output_guardrails=[toxicity_guardrail],
    model=litellm_model,
)
```

**Cách demo guardrail trip:**
```python
try:
    await Runner.run(moderated_agent, "You are an idiot. Write something rude please...")
    print("Guardrail didn't trip, but should have")
except OutputGuardrailTripwireTriggered:
    print("Toxicity guardrail tripped correctly")
```

---

### A7. Mô hình & LiteLLM

- Dùng **OpenRouter** (BASE_URL=`https://openrouter.ai/api/v1/`) để chọn model free có hỗ trợ tools.
- `LitellmModel` từ `agents.extensions.models.litellm_model` cho phép dùng model qua LiteLLM.
- Với model free, tắt `reasoning` để tiết kiệm token:
  ```python
  ModelSettings(extra_body={"reasoning": {"enabled": False}})
  ```
- Seminar dùng **vLLM** local server (Qwen3-4B) hoặc OpenRouter (grok-4.1).

---

### A8. Observability với CometML Opik

```python
import opik
opik.configure()

import os
from agents import set_trace_processors
from opik.integrations.openai.agents import OpikTracingProcessor

os.environ["OPIK_PROJECT_NAME"] = "openai-agents-demo"
set_trace_processors(processors=[OpikTracingProcessor()])
```
- Cấu hình API key Opik trong `.env`.
- Sau đó mọi lần chạy agent đều được **trace** để gỡ lỗi và đánh giá.

---

### A9. Parsing HTML với BeautifulSoup (LMS parser)

```python
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
import os

def parse_tasks():
    load_dotenv()
    lk_cookie = os.getenv("LK_SESSION_COOKIE")
    url = "https://lk.dataschool.yandex.ru/learning/assignments/"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; parser/1.0)"}
    session = requests.Session()
    if lk_cookie:
        session.cookies.set("sessionid", lk_cookie, domain="lk.dataschool.yandex.ru")
    resp = session.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    open_tasks_table = soup.find("h3", string="Открытые задания").find_next("table")
    tasks = []
    for row in open_tasks_table.find_all("tr", class_="noop"):
        date_block = row.find("div", class_="assignment-date")
        # ... trích deadline, assignment name, course name
        tasks.append({"course": course_name, "assignment": assignment_name, "deadline": deadline})
    return tasks
```

**Điểm mấu chốt:** cần cookie `sessionid` từ LMS (lấy từ DevTools → Network → request `timetable/`) để xác thực.

---

## PHẦN B. LUỒNG HOẠT ĐỘNG CÔNG VIỆC (HOME WORK)

> Mục tiêu homework: Xây dựng **Personal Project Manager** — một Multi-Agent System hoàn chỉnh.

### Luồng tổng quát (thứ tự thực hiện từng bước):

```
[Part 0] Setup model (OpenRouter + key)
    ↓
[Part 1] First agent — SMART Rewriter
    ↓
[Part 2] Tooling
    ├── YSDA LMS Tools (parse_tasks, parse_lectures)
    └── Google Calendar MCP Server
    ↓
[Part 3] Refined MAS — Multi-Agent System
    └── Breakdown agent, LMS agent, Calendar manager, Main manager
    ↓
[Part 4] Safety & Guardrails
    ├── Input guardrail (chặn yêu cầu giải NLP)
    └── Output guardrail (chống toxic, dùng HF classifier)
    ↓
[Part 5] Observability — Opik tracing
```

### Chi tiết từng bước:

**Part 0 — Model setup [0.5 điểm]**
1. Tạo tài khoản `openrouter.ai`.
2. Tạo key, cho vào `.env` (`OPENROUTER_KEY`).
3. Chọn model free có tools support, gán vào `MODEL_NAME`.
4. Test model với câu `"How many r's are in 'strawberry'?"`.

**Part 1 — First agent [1 điểm]**
- Viết prompt tốt nhất cho `smart_agent`.
- Yêu cầu: hỏi làm rõ nếu cần, biết ngày hiện tại, giữ phong cách viết.
- Áp dụng **SMART principle** (Specific, Measurable, Achievable, Relevant, Time-bound) để viết lại task.

**Part 2 — Tooling [3.5 điểm]**
- **YSDA LMS Tools [1.5 điểm]:** `parse_tasks` đã có sẵn; cần tự viết `parse_lectures()` (lấy lịch học từ `learning/timetable/`). Cần cookie `sessionid` trong `.env`.
- **YSDA MCP server [0.5 điểm]:** viết các wrapper MCP trong `ysda_mcp/mcp_tool.py` (dùng FastMCP) cho `parse_tasks` và `parse_lectures`, test với MCP client từ seminar.
- **Google Calendar MCP [1.5 điểm]:** tìm và cấu hình MCP server Google Calendar (vd `https://github.com/deciduus/calendar-mcp`), test các tool.

**Part 3 — Refined MAS [2 điểm]**
- Xây dựng Multi-Agent System hoàn chỉnh. Gợi ý cấu trúc tối thiểu:
  - **Breakdown agent:** phân rã task thành to-do list.
  - **YSDA LMS agent:** gọi YSDA tools (assignments, lectures).
  - **Calendar Manager:** tương tác với gcal tools.
  - **Main manager agent:** phân tích user query rồi route tới agent phù hợp.
- Dùng `handoffs` + `MCP servers`.

**Part 4 — Safety & Guardrails [2.5 điểm]**
- **Input guardrail [0.5 điểm]:** chặn user hỏi giải bài NLP assignment.
- **Output guardrail [2 điểm]:** dùng pretrained text classifier từ HuggingFace (KHÔNG dùng LLM) để chặn output toxic. Test bằng agent có system prompt toxic.

**Part 5 — Observability [0.5 điểm]**
- Set up Opik cloud (free tier), lấy API key cho vào `.env`.
- Cấu hình `OpikTracingProcessor`, chạy MAS, dán screenshot trace vào notebook.

---

## PHẦN C. LUỒNG HOẠT ĐỘNG CÔNG VIỆC (SEMINAR)

> Mục tiêu seminar: Làm quen với cách viết tools, MCP, ReAct từ đầu, và multi-agent với smolagents.

### Luồng tổng quát:

```
[1] Setup model (vLLM local / OpenRouter)
    ↓
[2] Viết tools như hàm Python
    ├── duckduckgo_search (web search)
    └── get_webpage_content (lấy nội dung trang)
    ↓
[3] Chạy MCP server + kết nối MCP client
    ↓
[4] Tool call / Function calling (JSON schema)
    ↓
[5] ReAct Agent từ scratch (Thought → Action → Observation)
    ↓
[6] Multi-agent với smolagents (CodeAgent + managed_agents)
```

### Chi tiết từng bước:

**1. Setup model**
- Dùng vLLM local (`Qwen/Qwen3-4B-Instruct-2507`) hoặc OpenRouter (`x-ai/grok-4.1-fast`).
- Đối với vLLM: khởi động server bằng `subprocess.Popen` với `python -m vllm.entrypoints.openai.api_server ...`.

**2. Viết tools như hàm Python**
- `duckduckgo_search(query, max_results)` — dùng thư viện `ddgs` để tìm kiếm web.
- `get_webpage_content(url)` — dùng `requests` + `markdownify` để lấy HTML thành Markdown, loại bỏ nhiều dòng trống.
- **TODO:** `websearch_full_text(query, top_k)` — lấy top-k kết quả kèm đầy đủ text.

**3. MCP server + client**
- MCP server: `ysda_tools.py` (FastMCP Calculator với add/subtract/multiply/divide, vector ops, matrix ops).
  - Vector: add, subtract, dot, elementwise_multiply.
  - Matrix: add, subtract, multiply, transpose.
- MCP client: lớp `MCPClient` (connect_to_server → initialize → list_tools → call_tool).

**4. Tool calling / Function calling**
- Tự động chuyển hàm Python → JSON schema tool bằng `create_tool_description` + `inspect`.

**5. ReAct Agent từ scratch**
- Định nghĩa `instruction` với 3 loại action: `Search`, `Visit web-page`, `Finish`.
- Viết `execute_action(action)` để phân tích và chạy action.
- Vòng lặp `call_react`: sinh Thought+Action, parse, execute, gắn Observation, lặp tối đa 8 vòng.

**6. Multi-agent với smolagents**
- Định nghĩa tool `visit_webpage` bằng decorator `@tool`.
- Tạo `web_agent` (CodeAgent) với `WebSearchTool()` + `visit_webpage`.
- Tạo `manager_agent` với `managed_agents=[web_agent]`, rồi `manager_agent.run(...)`.

---

## PHẦN D. TỔNG KẾT CÁC KIẾN THỨC CẦN NẮM

| Kiến thức | Công cụ / Thư viện | Homework | Seminar |
|-----------|--------------------|:--------:|:-------:|
| Tool calling | OpenAI API, inspect | ✅ | ✅ |
| MCP server | `mcp.server.fastmcp.FastMCP` | ✅ | ✅ |
| MCP client | `mcp.ClientSession`, `stdio_client` | ✅ | ✅ |
| ReAct | Tự viết vòng lặp | - | ✅ |
| Multi-agent | smolagents / OpenAI Agents SDK | ✅ | ✅ |
| Handoffs | OpenAI Agents SDK | ✅ | - |
| Guardrails | `agents` + `transformers` | ✅ | - |
| Observability / Tracing | CometML Opik | ✅ | - |
| Parsing HTML | BeautifulSoup, requests | ✅ | - |
| Web tool | ddgs, requests, markdownify | - | ✅ |

---

## PHẦN E. NHỮNG ĐIỂM CẦN LƯU Ý

1. **Homework chạy local (không dùng Colab)** — cần file `.env` với `OPENROUTER_KEY`, `LK_SESSION_COOKIE`, Opik key.
2. **Chỉ nộp file `.ipynb`** nhưng phải có đầy đủ outputs.
3. **Output guardrail không được dùng LLM** — bắt buộc dùng pretrained text classifier từ HuggingFace (`transformers.pipeline`).
4. **Model free thường phải tắt reasoning** để tránh cháy token.
5. **MCP server chạy qua `stdio`** — client dùng `StdioServerParameters(command="python", args=[path])`.
6. **Handoff vs Agent-as-tool:** handoff chuyển toàn bộ control+context; agent-as-tool trả response ngay, không truyền context.
7. Guardrail hoạt động bằng cơ chế **tripwire** — nếu vượt ngưỡng → ném `OutputGuardrailTripwireTriggered`.
