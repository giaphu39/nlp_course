# Cẩm Nang Chi Tiết Module `UIT_MCP` (UIT Schedule & Room Management)

> **Mục tiêu tài liệu:** Giới thiệu chi tiết về module **`uit_mcp`** — giải pháp thay thế cho `ysda_mcp` trong bài tập Week 10. Module này cho phép AI Agent cào dữ liệu lịch học, thời khóa biểu và lịch phòng học thực tế từ Cổng thông tin trường **Đại học Công nghệ Thông tin (UIT - ĐHQG-HCM)** tại [https://portal.uit.edu.vn/lich-phong](https://portal.uit.edu.vn/lich-phong) mà **không cần tài khoản đăng nhập hay cookie**.

---

## 📌 MỤC LỤC
1. [Tổng Quan & Bối Cảnh](#1-tổng-quan--bối-cảnh)
2. [Cấu Trúc Thư Mục & Vai Trò Từng File](#2-cấu-trúc-thư-mục--vai-trò-từng-file)
3. [Phân Tích Chi Tiết Từng File](#3-phân-tích-chi-tiết-từng-file)
   - 3.1. `uit_mcp/parser.py` (Bộ trích xuất dữ liệu)
   - 3.2. `uit_mcp/mcp_tool.py` (MCP Server & Tool Registration)
4. [Danh Sách Các MCP Tools & Nhiệm Vụ](#4-danh-sách-các-mcp-tools--nhiệm-vụ)
5. [Hướng Dẫn Kết Nối Trong Notebook `week10_hw.ipynb`](#5-hướng-dẫn-kết-nối-trong-notebook-week10_hwipynb)
6. [Kịch Bản Tích Hợp Hệ Thống Multi-Agent (MAS)](#6-kịch-bản-tích-hợp-hệ-thống-multi-agent-mas)
7. [So Sánh `ysda_mcp` vs `uit_mcp`](#7-so-sánh-ysda_mcp-vs-uit_mcp)
8. [Troubleshooting & Những Lưu Ý Quan Trọng](#8-troubleshooting--những-lưu-ý-quan-trọng)

---

## 1. TỔNG QUAN & BỐI CẢNH

Trong bài tập gốc của khóa học YSDA NLP, học viên được yêu cầu xây dựng MCP Server (`ysda_mcp`) để cào dữ liệu bài tập và lịch học từ cổng LMS nội bộ của YSDA (`https://lk.dataschool.yandex.ru/`). Hệ thống đó yêu cầu phải có cookie xác thực `sessionid` của sinh viên YSDA.

**Vấn đề:** Đối với người tự học hoặc không phải sinh viên YSDA, ta không có tài khoản LMS này.

**Giải pháp `uit_mcp`:**
- Chuyển hướng bài toán sang cổng thông tin công khai của **ĐH Công nghệ Thông tin (UIT)**: `https://portal.uit.edu.vn/lich-phong`.
- **Ưu điểm vượt trội:**
  1. Dữ liệu công khai, **không cần đăng nhập, không cần Cookie**.
  2. Dữ liệu thời khóa biểu thực tế, phong phú (hơn 400 lớp học/tuần, đầy đủ các tòa nhà A, B, C, E...).
  3. Cung cấp thêm tính năng tìm phòng học trống rất thiết thực cho sinh viên tự học hoặc họp nhóm.

```
[User Prompt]
      │
      ▼
[Main Manager Agent] ──(Handoff)──► [UIT Academic Assistant]
                                              │
                                              ▼ (MCP Protocol - stdio)
                                      [uit_mcp/mcp_tool.py]
                                              │
                                              ▼
                                      [uit_mcp/parser.py]
                                              │ (HTTP GET)
                                              ▼
                                  https://portal.uit.edu.vn/lich-phong
```

---

## 2. CẤU TRÚC THƯ MỤC & VAI TRÒ TỪNG FILE

```text
week10_agents/homework/
├── uit_mcp/
│   ├── parser.py       # Module cào và chuẩn hóa dữ liệu từ UIT Portal
│   └── mcp_tool.py     # MCP Server đăng ký các công cụ chuẩn JSON-RPC
├── ysda_mcp/           # Thư mục gốc của YSDA (giữ lại để đối chiếu)
└── week10_hw.ipynb     # Notebook thực hành chính
```

| Tên File | Vai Trò | Nhiệm Vụ Cốt Lõi |
| :--- | :--- | :--- |
| **`uit_mcp/parser.py`** | Data Scraper & Parser | Gửi HTTP request đến UIT Portal, bóc tách JSON payload từ HTML Next.js và trả về danh sách Python dict sạch sẽ. |
| **`uit_mcp/mcp_tool.py`** | MCP Server | Sử dụng thư viện `mcp` (`FastMCP`/`MCPServer`) để bọc các hàm trong `parser.py` thành các **MCP Tools** có docstring rõ ràng để LLM tự động nhận diện và gọi khi cần. |

---

## 3. PHÂN TÍCH CHI TIẾT TỪNG FILE

### 3.1. `uit_mcp/parser.py`

File này chịu trách nhiệm thu thập và xử lý dữ liệu thô:

1. **`fetch_uit_portal_data(date, building)`**:
   - Gửi request `GET https://portal.uit.edu.vn/lich-phong?view=week&building=B&date=YYYY-MM-DD`.
   - Trích xuất trường `initialData` chứa `rooms`, `periods`, `weekdays`, `bookings` từ Next.js RSC payload.
2. **`parse_uit_schedule(date, building, limit)`**:
   - Ghép nối thông tin lớp học (`bookings`) với thông tin phòng học (`rooms`) và bảng tiết học (`periods`).
   - Chuẩn hóa thời gian sang định dạng dễ hiểu: ví dụ `Tiết 1-4 (07:30 - 10:45)`.
3. **`get_classes_by_date(date, building)`**:
   - Lọc danh sách các lớp học chỉ diễn ra trong đúng ngày được yêu cầu (ví dụ: `2026-09-07`).
4. **`get_empty_rooms(date, period_code, building)`**:
   - So sánh danh sách toàn bộ phòng học với danh sách phòng đang có lịch học trong tiết được chỉ định để tìm ra các **phòng trống**.

### 3.2. `uit_mcp/mcp_tool.py`

File này khởi chạy một tiến trình con (subprocess) hoạt động như một **MCP Server**:

```python
try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP

from parser import parse_uit_schedule, get_classes_by_date, get_empty_rooms

mcp = FastMCP("UITSchedulerParser")

@mcp.tool()
def parse_schedule_tool(date: str = "2026-09-07", building: str = "B", limit: int = 30) -> list[dict]:
    """Cào danh sách thời khóa biểu lớp học theo tuần và tòa nhà từ UIT Portal."""
    return parse_uit_schedule(date=date, building=building, limit=limit)

@mcp.tool()
def get_classes_by_date_tool(date: str = "2026-09-07", building: str = "B") -> list[dict]:
    """Lấy danh sách các môn học / lớp học diễn ra trong một ngày cụ thể tại tòa nhà."""
    return get_classes_by_date(date=date, building=building)

@mcp.tool()
def get_empty_rooms_tool(date: str = "2026-09-07", period_code: str = "first", building: str = "B") -> list[dict]:
    """Tìm danh sách các phòng học đang còn trống trong một tiết cụ thể để tự học hoặc họp nhóm."""
    return get_empty_rooms(date=date, period_code=period_code, building=building)

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

---

## 4. DANH SÁCH CÁC MCP TOOLS & NHIỆM VỤ

| Tool Name | Tham Số Đầu Vào | Mô Tả & Nhiệm Vụ |
| :--- | :--- | :--- |
| **`parse_schedule_tool`** | - `date` (str): Ngày trong tuần (`YYYY-MM-DD`)<br>- `building` (str): Tòa nhà (`B`, `A`, `C`, `E`)<br>- `limit` (int): Giới hạn kết quả (mặc định: 30) | Cào toàn bộ lịch học trong tuần. Dùng khi user hỏi: *"Cho tôi xem toàn bộ thời khóa biểu tuần này ở tòa B"* |
| **`get_classes_by_date_tool`** | - `date` (str): Ngày cụ thể (`YYYY-MM-DD`)<br>- `building` (str): Tòa nhà (`B`, `A`, `C`, `E`) | Lấy danh sách lớp học của một ngày nhất định. Dùng khi user hỏi: *"Ngày 2026-09-07 có những lớp nào học ở tòa B?"* |
| **`get_empty_rooms_tool`** | - `date` (str): Ngày kiểm tra (`YYYY-MM-DD`)<br>- `period_code` (str): Mã tiết (`first` = Tiết 1, `sixth` = Tiết 6...)<br>- `building` (str): Tòa nhà (`B`, `A`, `C`, `E`) | Tìm phòng học còn trống. Dùng khi user hỏi: *"Tiết 1 sáng mai ở tòa B phòng nào còn trống để nhóm mình học bài?"* |

---

## 5. HƯỚNG DẪN KẾT NỐI TRONG NOTEBOOK `week10_hw.ipynb`

Trong [week10_hw.ipynb](file:///d:/01_Study/03_SelfLearning/NLP/YSDA%20Natural%20Language%20Processing%20course/nlp_course/week10_agents/homework/week10_hw.ipynb), tại phần **Tooling (Cell kết nối MCP)**:

### Bước 1: Khởi tạo và kết nối MCP Client
Thay thế đường dẫn kết nối cũ `./ysda_mcp/mcp_tool.py` bằng `./uit_mcp/mcp_tool.py`:

```python
client = MCPClient()
await client.connect_to_server('./uit_mcp/mcp_tool.py')
tools = client.list_tools()
```

Output hiển thị:
```text
Connected to server with tools: ['parse_schedule_tool', 'get_classes_by_date_tool', 'get_empty_rooms_tool']
```

### Bước 2: Thử nghiệm gọi tool
```python
# Gọi thử tool lấy danh sách lớp học ngày 2026-09-07
tool_result = await client.call_tool('get_classes_by_date_tool', {'date': '2026-09-07', 'building': 'B'})
print(tool_result)

# Gọi thử tool tìm phòng trống Tiết 1 sáng ngày 2026-09-07
empty_rooms = await client.call_tool('get_empty_rooms_tool', {'date': '2026-09-07', 'period_code': 'first', 'building': 'B'})
print(empty_rooms)
```

---

## 6. KỊCH BẢN TÍCH HỢP HỆ THỐNG MULTI-AGENT (MAS)

Tại phần **Refined MAS architecture** trong Notebook, bạn xây dựng hệ thống đa tác tử như sau:

```python
from agents import Agent, Runner

# 1. UIT Assistant Agent: Chuyên gia tra cứu lịch học UIT
uit_agent = Agent(
    name="UIT Academic Assistant",
    instructions="""You are an expert Academic Assistant for students at University of Information Technology (UIT).
Your role is to assist students in checking class schedules, finding empty classrooms for self-study/group meetings, and checking upcoming lectures.
Always call the provided UIT tools (get_classes_by_date_tool, get_empty_rooms_tool, parse_schedule_tool) to fetch accurate data.
Format your responses using clean Markdown tables.""",
    model=litellm_model,
    tools=tools, # Gắn danh sách tools từ uit_mcp
)

# 2. Smart Rewriter Agent: Chuyên phân rã mục tiêu theo chuẩn SMART (từ Phần 1)
smart_agent = Agent(
    name="Smart Rewriter Agent",
    instructions="You are an expert in SMART goal breakdown for academic and project tasks.",
    model=litellm_model,
)

# 3. Main Coordinator Agent: Tiếp nhận câu hỏi và điều phối (Handoff)
manager_agent = Agent(
    name="Main Coordinator Agent",
    instructions="""You are the central coordinator.
- If the user asks about UIT schedules, courses, rooms, or study spaces, hand off to 'UIT Academic Assistant'.
- If the user wants to break down a project or study goal, hand off to 'Smart Rewriter Agent'.""",
    model=litellm_model,
    handoffs=[uit_agent, smart_agent],
)
```

### Ví dụ chạy thử nghiệm hệ thống:
```python
# Test 1: Tra cứu lịch học qua Agent
result1 = await Runner.run(
    manager_agent,
    "Thứ Hai ngày 2026-09-07 ở tòa B trường UIT có những môn học nào diễn ra?"
)
print(result1.final_output)

# Test 2: Tìm phòng học trống
result2 = await Runner.run(
    manager_agent,
    "Sáng ngày 2026-09-07 vào Tiết 1, có phòng nào ở tòa B còn trống để nhóm mình họp không?"
)
print(result2.final_output)
```

---

## 7. SO SÁNH `ysda_mcp` VS `uit_mcp`

| Tiêu chí | `ysda_mcp` (Bản gốc YSDA) | `uit_mcp` (Bản UIT Portal) |
| :--- | :--- | :--- |
| **Nguồn dữ liệu** | LMS Yandex Data School (`lk.dataschool.yandex.ru`) | Cổng thông tin UIT (`portal.uit.edu.vn/lich-phong`) |
| **Xác thực / Cookie** | Bắt buộc phải có `LK_SESSION_COOKIE` | **Không cần xác thực (Công khai)** |
| **Độ ổn định** | Dễ bị lỗi `403/401` nếu cookie hết hạn | **Rất ổn định, không lo hết hạn cookie** |
| **Khả năng tùy biến ngày** | Cố định theo trang LMS hiện tại | **Tự do đổi tham số `date` (YYYY-MM-DD)** |
| **Số lượng Tools** | 2 tools (`parse_tasks`, `parse_lectures`) | **3 tools (`parse_schedule`, `get_classes_by_date`, `get_empty_rooms`)** |

---

## 8. TROUBLESHOOTING & NHỮNG LƯU Ý QUAN TRỌNG

1. **Lỗi `ModuleNotFoundError: No module named 'mcp.server.fastmcp'`:**
   - Phiên bản `mcp >= 2.0` đã đổi tên module `FastMCP` thành `MCPServer`. Trong file [uit_mcp/mcp_tool.py](file:///d:/01_Study/03_SelfLearning/NLP/YSDA%20Natural%20Language%20Processing%20course/nlp_course/week10_agents/homework/uit_mcp/mcp_tool.py), mã nguồn đã được tích hợp lớp bọc tương thích ngược `try...except` để chạy mượt trên cả `mcp 1.x` và `mcp 2.x`.
2. **Định dạng ngày:**
   - Luôn sử dụng định dạng chuẩn `YYYY-MM-DD` (ví dụ: `2026-09-07`) khi truyền tham số `date` cho các tool.
3. **Mã tiết học:**
   - Các mã tiết hợp lệ: `first` (tiết 1), `second` (tiết 2), ..., `sixth` (tiết 6), `tenth` (tiết 10).
