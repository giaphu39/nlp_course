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

### A8. Observability & Tracing trong AI Agents (CometML Opik)

#### 1. Observability (Khả năng quan sát) là gì?
Trong phần mềm truyền thống, ta debug bằng log/breakpoint vì code chạy **tất định (deterministic)**. Nhưng trong hệ thống AI Agent:
- LLM mang tính **bất định (non-deterministic)**, phản hồi có thể khác nhau ở mỗi lần chạy.
- Luồng thực thi phức tạp: một câu hỏi đơn giản có thể kích hoạt nhiều bước suy luận (ReAct), gọi nhiều Tool, handoff qua lại giữa nhiều Sub-agent.
- **Observability** là khả năng theo dõi, ghi vết (trace), đo lường chi phí (tokens, latency, USD), và gỡ lỗi toàn bộ quá trình Agent suy nghĩ và hành động theo thời gian thực.

#### 2. Các khái niệm cốt lõi trong LLM Observability:

```
[TRACE] User Query: "Lấy deadline và lên kế hoạch học tập"
   ├── [SPAN 1] Input Guardrail: Check Anti-Cheat NLP (Latency: 45ms)
   ├── [SPAN 2] Manager Agent: Reason & Decide Handoff (Tokens: 320, Latency: 620ms)
   ├── [SPAN 3] YSDA LMS Agent: Call Tool (Latency: 850ms)
   │      └── [SUB-SPAN] HTTP GET lk.dataschool.yandex.ru/assignments/ (Status: 200)
   ├── [SPAN 4] Smart Breakdown Agent: Prompt SMART decomposition (Tokens: 950, Latency: 1.4s)
   └── [SPAN 5] Output Guardrail: Toxic Classifier (HuggingFace BERT) (Latency: 110ms)
```

| Thuật ngữ | Định nghĩa & Ý nghĩa trong Agent |
|-----------|---------------------------------|
| **Trace** (Vết thực thi) | Đại diện cho **toàn bộ một vòng đời** xử lý một request từ lúc user gửi câu hỏi đến khi nhận câu trả lời cuối cùng. Mỗi lần `Runner.run()` là 1 Trace. |
| **Span** (Khoảng thực thi con) | Một đơn vị công việc cụ thể bên trong Trace (VD: 1 lần gọi LLM, 1 lần thực thi Tool, 1 lần chạy Guardrail, 1 bước Handoff). Các Span lồng nhau tạo thành cây phân cấp (*Span Tree*). |
| **Tokens & Cost Tracking** | Theo dõi chi tiết số lượng `prompt_tokens`, `completion_tokens`, `total_tokens` và ước tính chi phí tiền mặt (USD) phát sinh ở từng bước. |
| **Latency / Duration** | Đo thời gian chạy từng Span để phát hiện "điểm nghẽn" (bottleneck) — ví dụ tool nào chạy chậm, model nào phản hồi lâu. |
| **Metadata & Tags** | Các thông tin bổ sung đính kèm vào Trace: `user_id`, `session_id`, `model_name`, `temperature`, `git_commit` để lọc và phân tích sau này. |
| **Evaluation / Feedback** | Gán nhãn đánh giá chất lượng phản hồi (Thumbs up/down, Toxicity score, LLM-as-a-Judge) trực tiếp lên Trace. |

#### 3. Các công cụ Observability phổ biến:
- **CometML Opik**: Nền tảng Open-source + Cloud chuyên sâu cho LLM/Agent Tracing, tích hợp sẵn với `openai-agents`, `langchain`, `litellm`.
- **LangSmith**: Do LangChain phát triển, mạnh về debug chuỗi LCEL và Agentic flows.
- **Arize Phoenix / Helicone / Lunary / OpenTelemetry**: Các giải pháp tracing mã nguồn mở và chuẩn hóa OpenTelemetry cho AI.

#### 4. Cấu hình Opik với OpenAI Agents SDK trong Code:
```python
import os
import opik
from agents import set_trace_processors
from opik.integrations.openai.agents import OpikTracingProcessor

# 1. Cấu hình API Key (hoặc tự động đọc từ .env: OPIK_API_KEY)
opik.configure()

# 2. Đặt tên project trên Dashboard
os.environ["OPIK_PROJECT_NAME"] = "ysda-nlp-agent-project"

# 3. Gắn Tracing Processor vào hệ thống Agent
# Mọi lời gọi Runner.run() sau dòng này sẽ tự động capture Spans và đẩy lên Dashboard
set_trace_processors(processors=[OpikTracingProcessor()])
```

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

### A10. Bộ từ điển Thuật ngữ & Khái niệm cốt lõi về AI Agents

Để làm chủ và phỏng vấn về hệ thống AI Agent, bạn cần nắm vững các thuật ngữ quan trọng sau:

#### 1. Các mô hình kiến trúc điều phối Agent (Agent Orchestration Patterns)
- **Router / Dispatcher Pattern**: Một LLM phân loại ý định (Intent Classification) và định tuyến câu hỏi tới Agent chuyên biệt phù hợp.
- **Supervisor / Manager Pattern**: Agent quản lý cấp cao nhận request, lập kế hoạch (Planning), phân chia nhiệm vụ cho các Worker Agent và tổng hợp kết quả cuối cùng.
- **Hierarchical Multi-Agent (HMA)**: Cấu trúc phân cấp nhiều tầng (Manager -> Sub-Manager -> Specialist Workers).
- **Sequential / Pipeline Pattern**: Output của Agent này trở thành Input của Agent tiếp theo (A -> B -> C).
- **Swarm / Peer-to-Peer (Collaborative)**: Các Agent ngang hàng trao đổi thông điệp trực tiếp với nhau thông qua cơ chế Handoff hoặc Message Bus.

#### 2. So sánh Handoffs vs Agent-as-a-Tool (Sub-Agent)
| Tiêu chí | **Handoffs (Chuyển giao quyền)** | **Agent-as-a-Tool (Sub-Agent)** |
|----------|---------------------------------|--------------------------------|
| **Cơ chế** | Agent A nhường hoàn toàn quyền điều khiển cuộc trò chuyện và ngữ cảnh cho Agent B. | Agent A xem Agent B như một hàm bình thường, gọi Agent B và chờ nhận lại chuỗi kết quả (string). |
| **Context** | Giữ nguyên lịch sử hội thoại xuyên suốt giữa các agent. | Context bị cô lập (Isolated), chỉ gửi prompt con và nhận text trả về. |
| **Phù hợp khi** | Chuyển đổi trạng thái nghiệp vụ (VD: Chuyển từ Bot chào đón sang Bot thanh toán). | Thực hiện tác vụ độc lập, chuyên sâu rồi quay lại luồng chính (VD: Web Search, Code Interpreter). |

#### 3. Bộ nhớ của Agent (Agent Memory)
- **Short-term Memory (Bộ nhớ ngắn hạn)**: Lịch sử hội thoại (Message history) và bộ nhớ cào tạm thời (Scratchpad / Working Memory trong vòng lặp ReAct). Giới hạn bởi Context Window của LLM.
- **Long-term Memory (Bộ nhớ dài hạn)**: Lưu trữ lâu dài thông tin người dùng, sở thích, tài liệu kiến thức qua Vector Database (RAG) hoặc cơ sở dữ liệu SQL/NoSQL.
- **Entity Memory**: Trích xuất và theo dõi các thực thể quan trọng (tên người, ngày tháng, sự kiện, sở thích) xuyên suốt các phiên chat.

#### 4. An toàn và Kiểm soát (Safety, Guardrails & Jailbreaking)
- **Prompt Injection / Jailbreak**: Kỹ thuật tấn công lừa LLM bỏ qua các chỉ dẫn an toàn hệ thống (System Prompt) để thực hiện hành vi xấu.
- **Input Guardrail**: Bộ lọc kiểm tra tính hợp lệ, an toàn và đúng phạm vi của câu hỏi trước khi gửi vào LLM (ngăn chặn Prompt Injection, từ chối câu hỏi ngoài phạm vi, chống gian lận).
- **Output Guardrail**: Bộ kiểm tra câu trả lời của LLM trước khi gửi cho người dùng (ngăn chặn Hallucination, phát hiện nội dung độc hại/Toxic/PII leak).
- **Tripwire (Dây bẫy an toàn)**: Cơ chế ngắt luồng ngay lập tức (`tripwire_triggered=True`) và ném Exception khi phát hiện vi phạm bảo mật, không cho phép Agent tiếp tục chạy.

#### 5. Đánh giá chất lượng Agent (Agent Evaluations & Evals)
- **LLM-as-a-Judge**: Dùng một LLM mạnh (như GPT-4o, Claude 3.5 Sonnet) để chấm điểm và đánh giá độ chính xác, mức độ bám sát yêu cầu của một Agent khác theo thang điểm (Rubric).
- **Tool Selection Accuracy**: Tỷ lệ phần trăm Agent chọn đúng Tool cần thiết cho một bài toán.
- **Argument Correctness**: Tỷ lệ Agent trích xuất và truyền đúng các tham số (JSON Arguments) cho Tool.
- **Step Count / Efficiency**: Số bước trung bình Agent cần lặp lại (Thought-Action loop) để hoàn thành một mục tiêu. Càng ít bước thừa thì Agent càng tối ưu.

---

## PHẦN B. HƯỚNG DẪN CHI TIẾT HOMEWORK (WEEK 10)

> **Mục tiêu tối thượng của Homework:** Xây dựng một **Personal Project Manager** — Hệ thống Multi-Agent hoàn chỉnh (MAS) giúp sinh viên tự động hóa việc quản lý học tập: lấy bài tập/lịch học từ LMS của YSDA, phân rã công việc theo chuẩn SMART, đồng bộ lịch Google Calendar, được bảo vệ bởi Input/Output Guardrails và được giám sát qua Opik Tracing.

```
                  ┌──────────────────────────────────────────────────┐
                  │                 USER (Sinh viên)                 │
                  └─────────────────────────┬────────────────────────┘
                                            │ Query / Lệnh
                                            ▼
                  ┌──────────────────────────────────────────────────┐
                  │              Main Manager Agent                  │
                  │   (Phân tích câu hỏi, điều phối & Handoffs)      │
                  │   [Bảo vệ: Input Guardrail chặn giải hộ NLP]    │
                  └──────┬───────────────────┬───────────────────┬───┘
                         │ Handoff           │ Handoff           │ Handoff
                         ▼                   ▼                   ▼
      ┌─────────────────────────┐ ┌────────────────────┐ ┌────────────────────┐
      │  YSDA Interface Agent   │ │ Smart Breakdown Ag │ │  Calendar Agent    │
      │ (Truy vấn LMS qua MCP)  │ │ (Phân rã SMART)    │ │ (Đồng bộ Google Cal│
      └────────────┬────────────┘ └─────────┬──────────┘ └─────────┬──────────┘
                   │ MCP stdio              │ Prompt/Logic         │ MCP stdio/API
                   ▼                        ▼                      ▼
      ┌─────────────────────────┐           │            ┌────────────────────┐
      │  ysda_mcp/mcp_tool.py   │           │            │ Calendar MCP Server│
      │  - parse_tasks_tool     │           │            │ - list_events      │
      │  - parse_lectures_tool  │           │            │ - create_event     │
      └────────────┬────────────┘           │            └────────────────────┘
                   │ Gọi nội bộ             │
                   ▼                        │
      ┌─────────────────────────┐           │
      │  ysda_mcp/parser.py     │           │
      │  (BeautifulSoup + Req)  │           │
      └────────────┬────────────┘           │
                   │ HTTP + Cookie          │
                   ▼                        │
      ┌─────────────────────────┐           │
      │ LMS Dataschool Yandex   │           │
      └─────────────────────────┘           │
                                            │
                                            ▼
                  ┌──────────────────────────────────────────────────┐
                  │       Output Guardrail (HuggingFace BERT)        │
                  │        (Quét độc hại / Toxic -> Tripwire)        │
                  └─────────────────────────┬────────────────────────┘
                                            │ Traced by Opik
                                            ▼
                  ┌──────────────────────────────────────────────────┐
                  │            FINAL ANSWER CHO SINH VIÊN            │
                  └──────────────────────────────────────────────────┘
```

---

### B1. Cấu trúc thư mục & Quan hệ giữa các File

| File / Thư mục | Vai trò trong hệ thống | Liên kết với file khác |
|----------------|------------------------|------------------------|
| `homework/week10_hw.ipynb` | **Notebook chính**: Nơi tích hợp toàn bộ code, chạy thực nghiệm, định nghĩa các Agent, Guardrails và hiển thị output để nộp bài. | Đọc `.env`, import từ `ysda_mcp/parser.py`, gọi subprocess `ysda_mcp/mcp_tool.py`. |
| `homework/.env` | **Lưu trữ Secret/Key**: Chứa API Key và Cookie xác thực môi trường cục bộ. | `week10_hw.ipynb`, `parser.py`, `opik` cùng đọc các biến môi trường này. |
| `homework/ysda_mcp/parser.py` | **Scraper Module**: Chứa logic cào dữ liệu HTML từ trang LMS bằng `requests` + `BeautifulSoup` kèm session cookie. | Được gọi trực tiếp bởi `mcp_tool.py` hoặc import test trong notebook. |
| `homework/ysda_mcp/mcp_tool.py` | **MCP Server Độc lập**: Dùng `FastMCP` để đóng gói các hàm scraper thành các MCP Tools chuẩn giao tiếp qua `stdio`. | `week10_hw.ipynb` (hoặc `MCPServerStdio`) sẽ khởi chạy file này như một tiến trình con. |
| `seminar/ysda_tools.py` | **Tham khảo mẫu MCP**: Mẫu viết FastMCP với các phép toán Math, Vector, Matrix. | Dùng để đối chiếu cú pháp FastMCP và `transport="stdio"`. |

---

### B2. Luồng dữ liệu chi tiết giữa các tầng (End-to-End Execution Flow)

1. **Khởi tạo môi trường (`.env` & LiteLLM):**
   - Notebook đọc `OPENROUTER_KEY`, `LK_SESSION_COOKIE`, `OPIK_KEY` từ `.env`.
   - Cấu hình `LitellmModel` kết nối qua OpenRouter API.
2. **Khởi động MCP Server (`mcp_tool.py`):**
   - Agent SDK khởi chạy `python ysda_mcp/mcp_tool.py` dưới dạng tiến trình con (subprocess).
   - Hai tiến trình bắt tay qua JSON-RPC trên kênh `stdin`/`stdout`.
   - Server gửi danh sách tool schema (`parse_tasks_tool`, `parse_lectures_tool`) cho Agent.
3. **Tiếp nhận User Query & Input Guardrail:**
   - User đặt câu hỏi: *"Kiểm tra deadline sắp tới trên LMS và lên kế hoạch to-do list cho tôi"*.
   - `nlp_guardrail` kiểm tra: User có đang gian lận nhờ LLM giải bài tập NLP không? Nếu có -> kích hoạt tripwire chặn ngay; nếu không -> chuyển tiếp tới `manager_agent`.
4. **Handoffs & Tool Execution:**
   - `manager_agent` nhận thấy cần lấy dữ liệu LMS -> Handoff sang `ysda_agent`.
   - `ysda_agent` gọi tool `parse_tasks_tool` -> MCP Server thực thi `parser.py` -> gửi HTTP request có `LK_SESSION_COOKIE` lên LMS Yandex -> cào danh sách deadline và trả kết quả JSON về cho `ysda_agent`.
   - `ysda_agent` trả kết quả lại hoặc `manager_agent` handoff tiếp sang `smart_agent` (Breakdown Agent).
   - `smart_agent` áp dụng nguyên tắc SMART để phân tích bài tập thành các đầu việc nhỏ có thời hạn cụ thể.
   - (Tùy chọn) `calendar_agent` nhận task và gọi Google Calendar MCP để thêm sự kiện vào lịch.
5. **Output Guardrail & Tracing:**
   - Phản hồi cuối cùng trước khi trả về được đưa qua `toxicity_guardrail` (chạy mô hình Transformer local). Nếu không toxic -> cho phép xuất ra.
   - Toàn bộ hành trình (prompts, tool calls, latency, tokens, handoffs) được `OpikTracingProcessor` ghi lại và đẩy lên CometML Opik Dashboard.

---

### B3. Danh sách CHI TIẾT TẤT CẢ CÁC NHIỆM VỤ CẦN ĐIỀN CODE (TODOs)

Dưới đây là bảng tổng hợp tất cả 6 nhiệm vụ cần hoàn thiện trong Homework kèm số điểm và vị trí code:

| STT | Nhiệm vụ (TODO) | Vị trí file / Cell | Điểm | Trọng tâm kỹ thuật |
|:---:|-----------------|-------------------|:----:|-------------------|
| **1** | Cấu hình `.env` & Model Setup | `homework/.env` + `week10_hw.ipynb` (Cell 3) | **0.5** | Chọn model OpenRouter free có hỗ trợ tool calling |
| **2** | Prompt Engineering cho SMART Agent | `week10_hw.ipynb` (Cell 8) | **1.0** | Viết system instruction chuyển hóa task thô thành SMART |
| **3** | Viết scraper `parse_lectures()` | `homework/ysda_mcp/parser.py` (Dòng 59) | **1.5** | BeautifulSoup cào bảng thời khóa biểu `learning/timetable/` |
| **4** | Đóng gói YSDA MCP Server | `homework/ysda_mcp/mcp_tool.py` | **0.5** | FastMCP decorators `@mcp.tool()` và `transport="stdio"` |
| **5** | Kết nối Google Calendar MCP | `week10_hw.ipynb` (Cell 22) | **1.5** | Tích hợp MCP Server quản lý Google Calendar |
| **6** | Xây dựng hệ thống hoàn chỉnh (MAS) | `week10_hw.ipynb` (Cell 26) | **2.0** | Multi-Agent orchestration với Handoffs và MCP |
| **7** | Viết Input Guardrail (Anti-Cheat NLP) | `week10_hw.ipynb` (Cell 28) | **0.5** | Decorator `@input_guardrail` phát hiện request giải hộ bài |
| **8** | Viết Output Guardrail (Anti-Toxic HF) | `week10_hw.ipynb` (Cell 30) | **2.0** | Dùng Pretrained HuggingFace Classifier (KHÔNG dùng LLM) |
| **9** | Cấu hình Observability Tracing | `week10_hw.ipynb` (Cell 36-40) | **0.5** | Opik configuration + Chèn ảnh trace kết quả |
| | **TỔNG ĐIỂM** | | **10.0** | |

---

### B4. Hướng dẫn chi tiết & Code mẫu cho từng TODO

#### 🎯 TODO 1: Cấu hình Môi trường & Model (`.env` + Notebook Cell 3) — [0.5 điểm]

**Nhiệm vụ:**
1. Điền key vào file `homework/.env`:
   ```env
   OPIK_KEY=opik_api_key_cua_ban
   OPENROUTER_KEY=sk-or-v1-key_cua_ban
   LK_SESSION_COOKIE=session_id_lay_tu_browser_network_tab
   ```
2. Trong `week10_hw.ipynb` Cell 3: Điền tên mô hình OpenRouter có hỗ trợ **Function Calling / Tools**.
   - Gợi ý model free tốt: `"meta-llama/llama-3.3-70b-instruct:free"`, `"google/gemini-2.0-flash-thinking-exp:free"`, `"qwen/qwen-2.5-72b-instruct:free"`, hoặc `"mistralai/mistral-small-24b-instruct-2501:free"`.

```python
# [Cell 3] CODE ĐIỀN:
from dotenv import load_dotenv
import os
load_dotenv()

OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
MODEL_NAME = "meta-llama/llama-3.3-70b-instruct:free"  # Chọn model hỗ trợ tools
BASE_URL = "https://openrouter.ai/api/v1/"
```

---

#### 🎯 TODO 2: Prompt Engineering cho SMART Rewriter Agent (Notebook Cell 8) — [1.0 điểm]

**Nhiệm vụ:**
Viết `instructions` chi tiết cho `smart_agent` sao cho khi nhận một task chung chung (ví dụ: *"Complete the project report"*), agent sẽ chuyển đổi nó thành một kế hoạch hành động chuẩn **SMART**:
- **S (Specific):** Rõ ràng từng bước làm gì, đối tượng là gì.
- **M (Measurable):** Có tiêu chí đo lường hoàn thành (độ dài trang, số liệu, mục tiêu).
- **A (Achievable):** Khả thi với nguồn lực hiện có.
- **R (Relevant):** Phù hợp với mục tiêu dự án lớn.
- **T (Time-bound):** Có hạn chót hoặc mốc thời gian rõ ràng (biết ngày hiện tại).

```python
# [Cell 8] CODE ĐIỀN:
SMART_INSTRUCTIONS = """
You are an expert Project Management AI Agent specializing in task decomposition using the SMART framework (Specific, Measurable, Achievable, Relevant, Time-bound).

Your responsibilities:
1. Analyze the input task from the user or manager.
2. If the task is underspecified, make reasonable contextual assumptions or formulate clarifying questions.
3. Reformulate the task into a structured SMART goal breakdown:
   - **Specific**: Clearly define what needs to be done, who is involved, and what the key deliverable is.
   - **Measurable**: Provide concrete success criteria, quantitative milestones, and validation checks.
   - **Achievable**: Ensure the action steps are realistic and broken down into actionable sub-tasks.
   - **Relevant**: Explain how this task contributes to the broader objective.
   - **Time-bound**: Assign explicit deadlines, phased milestones, and time estimates.
4. Output format: Use clean, professional Markdown with bullet points, checklist boxes (`- [ ]`), and clear headings.
"""

smart_agent = Agent(
    name="Smart Rewriter Agent",
    instructions=SMART_INSTRUCTIONS,
    model=litellm_model,
    model_settings=no_reasoning,
)
```

---

#### 🎯 TODO 3: Cào dữ liệu bài giảng `parse_lectures()` (`ysda_mcp/parser.py`) — [1.5 điểm]

**Nhiệm vụ:**
Viết hàm `parse_lectures()` trong `homework/ysda_mcp/parser.py` để cào thời khóa biểu bài giảng sắp tới từ URL `https://lk.dataschool.yandex.ru/learning/timetable/`.

**Cách lấy Cookie `sessionid`:**
1. Mở trình duyệt, đăng nhập vào `https://lk.dataschool.yandex.ru/`.
2. Bấm F12 mở DevTools -> Chuyển sang tab **Network**.
3. Tải lại trang hoặc vào mục Timetable.
4. Tìm request `timetable/` -> Xem Header `Cookie` -> Copy giá trị `sessionid=...` dán vào `.env` (`LK_SESSION_COOKIE`).

**Code hoàn thiện trong `homework/ysda_mcp/parser.py`:**
```python
def parse_lectures() -> list[dict[str, str]]:
    load_dotenv()
    lk_cookie = os.getenv("LK_SESSION_COOKIE")

    url = "https://lk.dataschool.yandex.ru/learning/timetable/"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; parser/1.0)"
    }

    session = requests.Session()
    if lk_cookie:
        session.cookies.set("sessionid", lk_cookie, domain="lk.dataschool.yandex.ru")

    resp = session.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    html = resp.text

    soup = BeautifulSoup(html, "html.parser")
    lectures = []

    # Parse bảng thời khóa biểu (các sự kiện/bài giảng sắp diễn ra)
    # Tìm các thẻ chứa thông tin buổi học (tùy cấu trúc HTML thực tế của LMS)
    timetable_events = soup.find_all("div", class_="timetable-event") or soup.find_all("tr", class_="event-row")
    
    if not timetable_events:
        # Fallback: tìm qua table chung hoặc class tương đương
        table = soup.find("table")
        if table:
            rows = table.find_all("tr")
            for row in rows:
                cols = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
                if len(cols) >= 3:
                    lectures.append({
                        "date": cols[0],
                        "time": cols[1],
                        "course": cols[2],
                        "lecture": cols[3] if len(cols) > 3 else "Lecture"
                    })
    else:
        for event in timetable_events:
            date_tag = event.find("span", class_="date") or event.find("div", class_="date")
            title_tag = event.find("a", class_="title") or event.find("span", class_="title")
            course_tag = event.find("span", class_="course")
            lectures.append({
                "date": date_tag.get_text(strip=True) if date_tag else "Upcoming",
                "course": course_tag.get_text(strip=True) if course_tag else "Data School",
                "lecture": title_tag.get_text(strip=True) if title_tag else "Lecture Event",
            })

    for l in lectures:
        print(f"[{l.get('date')}] {l.get('course')}: {l.get('lecture')}")

    return lectures
```

---

#### 🎯 TODO 4: Xây dựng YSDA MCP Server (`ysda_mcp/mcp_tool.py`) — [0.5 điểm]

**Nhiệm vụ:**
Tạo MCP Server bằng `FastMCP` để xuất 2 tool `parse_tasks_tool` và `parse_lectures_tool` qua giao thức `stdio`.

**Code hoàn thiện trong `homework/ysda_mcp/mcp_tool.py`:**
```python
from mcp.server.fastmcp import FastMCP
from parser import parse_lectures, parse_tasks

mcp = FastMCP("DataSchoolParser")

@mcp.tool()
def parse_tasks_tool() -> list[dict[str, str]]:
    """Fetches open assignments, courses, and upcoming deadlines from YSDA Data School LMS."""
    return parse_tasks()

@mcp.tool()
def parse_lectures_tool() -> list[dict[str, str]]:
    """Fetches upcoming lectures and timetable events from YSDA Data School LMS."""
    return parse_lectures()

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

---

#### 🎯 TODO 5: Kết nối Google Calendar MCP Server (Notebook Cell 22) — [1.5 điểm]

**Nhiệm vụ:**
Cấu hình và test MCP Server Google Calendar (hoặc dùng package npx / python calendar mcp như `calendar-mcp` hoặc mock wrapper).

```python
# [Cell 22] CODE MẪU:
# Kết nối qua MCPServerStdio (npx hoặc python)
from agents.mcp import MCPServerStdio

# Ví dụ nếu sử dụng node/npx calendar mcp hoặc python calendar server:
gcal_server_params = {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-google-calendar"],
}
# Hoặc nếu dùng python server local:
# gcal_server_params = {"command": "python", "args": ["path/to/calendar_mcp.py"]}

print("Google Calendar MCP configured.")
```

---

#### 🎯 TODO 6: Tích hợp Hệ thống Multi-Agent Hoàn chỉnh (MAS) (Notebook Cell 26) — [2.0 điểm]

**Nhiệm vụ:**
Kết nối `manager_agent`, `ysda_agent`, `smart_agent` (Breakdown), và `calendar_agent` với cơ chế Handoffs và MCP tools.

```python
# [Cell 26] CODE ĐIỀN:
from agents import Agent, Runner
from agents.mcp import MCPServerStdio

async def run_full_mas(user_prompt: str):
    async with MCPServerStdio(
        name="YSDA LMS Server",
        params={"command": "python", "args": ["ysda_mcp/mcp_tool.py"]},
    ) as ysda_server:
        
        # 1. YSDA LMS Agent: Có quyền gọi tool cào LMS
        ysda_agent = Agent(
            name="YSDA LMS Agent",
            handoff_description="Interacts with YSDA Data School LMS to fetch assignments, deadlines, and upcoming lectures.",
            instructions="""You are a specialized agent for the YSDA LMS system.
Use your tools (`parse_tasks_tool`, `parse_lectures_tool`) to query assignments and schedule.
Summarize the retrieved data clearly with course name, task title, and deadline.""",
            mcp_servers=[ysda_server],
            model=litellm_model,
            model_settings=no_reasoning,
        )

        # 2. Smart Breakdown Agent: Phân rã công việc SMART
        breakdown_agent = Agent(
            name="Breakdown Agent",
            handoff_description="Decomposes tasks and assignments into actionable, detailed SMART to-do lists.",
            instructions=SMART_INSTRUCTIONS,
            model=litellm_model,
            model_settings=no_reasoning,
        )

        # 3. Main Manager Agent: Điều phối toàn bộ luồng
        manager_agent = Agent(
            name="Personal Project Manager",
            instructions="""You are the Head Project Manager AI.
Your goal is to assist students in managing their coursework, deadlines, and schedules.
Workflow:
1. When a user asks about coursework or deadlines, hand off to `YSDA LMS Agent` to fetch the real data.
2. Once assignments are retrieved, hand off to `Breakdown Agent` to decompose key tasks into a detailed SMART plan.
3. Synthesize the final response clearly for the user.""",
            handoffs=[ysda_agent, breakdown_agent],
            model=litellm_model,
            model_settings=no_reasoning,
        )

        result = await Runner.run(manager_agent, user_prompt)
        return result

# Chạy thử nghiệm hệ thống MAS:
query = "What upcoming deadlines do I have in the data school LMS? Decompose the first task and write a full to-do list for me."
mas_result = await run_full_mas(query)
display(Markdown(mas_result.final_output))
```

---

#### 🎯 TODO 7: Input Guardrail — Chống gian lận NLP Assignment (Notebook Cell 28) — [0.5 điểm]

**Nhiệm vụ:**
Tạo `@input_guardrail` để kiểm tra input từ người dùng. Nếu người dùng yêu cầu giải hộ bài tập NLP -> kích hoạt **Tripwire** ngăn chặn ngay lập tức.

```python
# [Cell 28] CODE ĐIỀN:
from agents import input_guardrail, GuardrailFunctionOutput, RunContextWrapper, TResponseInputItem

@input_guardrail
async def nlp_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    # Trích xuất nội dung text từ input
    if isinstance(input, list):
        text_content = " ".join([str(item) for item in input])
    else:
        text_content = str(input)
    
    # Danh sách từ khóa/dấu hiệu gian lận giải bài tập NLP
    cheat_keywords = [
        "solve this nlp assignment", "solve my nlp homework", "write solution for week",
        "do my nlp homework", "giải bài tập nlp", "giải hộ homework nlp",
        "give me code for week10_hw", "solve assignment"
    ]
    
    is_cheat = any(kw in text_content.lower() for kw in cheat_keywords)
    
    if is_cheat:
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info="Academic Integrity Violation: You cannot ask the agent to solve NLP course assignments directly."
        )
    
    return GuardrailFunctionOutput(tripwire_triggered=False)
```

---

#### 🎯 TODO 8: Output Guardrail — Chống Toxic bằng Pretrained HuggingFace Classifier (Notebook Cell 30) — [2.0 điểm]

**Nhiệm vụ:**
Sử dụng thư viện `transformers` với mô hình text-classification tải từ HuggingFace (ví dụ `unitary/toxic-bert` hoặc `martin-ha/toxic-comment-model`) để lọc output toxic. **TUYỆT ĐỐI KHÔNG DÙNG LLM CHO BƯỚC NÀY**.

```python
# [Cell 30] CODE ĐIỀN:
from pydantic import BaseModel
from transformers import pipeline
from agents import (
    Agent,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    output_guardrail,
)

# 1. Khởi tạo pipeline phân loại văn bản (chạy local trên CPU/GPU)
toxicity_classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert", # hoặc "martin-ha/toxic-comment-model"
    top_k=None
)

class ToxicityResult(BaseModel):
    toxic_score: float
    label: str

class MessageOutput(BaseModel): 
    response: str

# 2. Định nghĩa hàm Output Guardrail
@output_guardrail
async def toxicity_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    output: MessageOutput,
) -> GuardrailFunctionOutput:
    text_to_check = output.response
    predictions = toxicity_classifier(text_to_check)[0] # List of dicts: [{'label': 'toxic', 'score': 0.98}, ...]
    
    toxic_score = 0.0
    detected_label = "non-toxic"
    
    for item in predictions:
        # Nếu nhãn là toxic/insult/threat/obscene và điểm số cao
        if item["label"].lower() in ["toxic", "severe_toxic", "insult", "identity_hate", "threat"]:
            if item["score"] > toxic_score:
                toxic_score = item["score"]
                detected_label = item["label"]
    
    # Đặt ngưỡng kích hoạt (Threshold = 0.5)
    THRESHOLD = 0.5
    if toxic_score >= THRESHOLD:
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info=ToxicityResult(toxic_score=toxic_score, label=detected_label)
        )
    
    return GuardrailFunctionOutput(tripwire_triggered=False)
```

---

#### 🎯 TODO 9: Observability & Tracing với CometML Opik (Notebook Cell 36-40) — [0.5 điểm]

**Nhiệm vụ:**
1. Đăng ký tài khoản free tại `https://www.comet.com/opik/`.
2. Tạo API Key và cấu hình `OPIK_KEY` trong `.env`.
3. Chạy `opik.configure()` và thiết lập `set_trace_processors([OpikTracingProcessor()])`.
4. Thực thi MAS để Opik ghi lại toàn bộ Trace.
5. Chụp ảnh màn hình Dashboard Opik thể hiện các Trace và chèn vào Cell 39 của notebook.

---

### B5. Checklist Hoàn thành & Nộp bài

- [ ] File `.env` đã điền đầy đủ `OPENROUTER_KEY`, `LK_SESSION_COOKIE`, `OPIK_KEY`.
- [ ] `MODEL_NAME` chọn model hỗ trợ tools và chạy test strawberry thành công.
- [ ] `smart_agent` được viết prompt chuẩn SMART chi tiết.
- [ ] `parse_lectures()` trong `ysda_mcp/parser.py` chạy trả về list các buổi học.
- [ ] `ysda_mcp/mcp_tool.py` chạy qua MCP Client lấy được 2 tool `parse_tasks_tool` và `parse_lectures_tool`.
- [ ] MAS hoàn chỉnh thực thi mượt mà với Handoff giữa Manager, YSDA Agent và Breakdown Agent.
- [ ] Test Input Guardrail kích hoạt khi hỏi giải bài tập NLP.
- [ ] Test Output Guardrail (Cell 32-33) kích hoạt tripwire thành công khi gặp phản hồi toxic.
- [ ] Trace từ Opik đã được chụp và nhúng ảnh vào Markdown cell 39.
- [ ] Toàn bộ các cell trong `week10_hw.ipynb` đã được chạy tuần tự từ trên xuống dưới và **có đầy đủ output hiển thị**.

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
