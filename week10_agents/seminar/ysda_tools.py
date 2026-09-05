import numpy as np
from mcp.server.fastmcp import FastMCP

# ==============================================================================
# 1. KHỞI TẠO MCP SERVER
# ==============================================================================
# FastMCP là lớp tiện ích cấp cao (từ thư viện mcp của Anthropic) giúp biến các hàm
# Python thành công cụ (tools) theo chuẩn giao thức Model Context Protocol (MCP).
mcp = FastMCP("Calculator")


# ==============================================================================
# 2. CÁC CÔNG CỤ TÍNH TOÁN SỐ HỌC CƠ BẢN
# ==============================================================================
# Decorator `@mcp.tool()`:
# - Tự động đăng ký hàm này vào danh sách công cụ mà LLM có thể gọi.
# - Tự động đọc Type Hints (ví dụ: a: float, b: float -> float) và docstring
#   để tạo thành JSON Schema gửi cho mô hình LLM hiểu cách truyền tham số.

@mcp.tool()
def add(a: float, b: float) -> float:
    """Cộng hai số thực: a + b"""
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Trừ hai số thực: a - b"""
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Nhân hai số thực: a * b"""
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Chia hai số thực: a / b (báo lỗi nếu chia cho 0)"""
    if b == 0:
        raise ValueError("Division by zero")
    return a / b


# ==============================================================================
# 3. CÁC HÀM TIỆN ÍCH CHUYỂN ĐỔI KIỂU DỮ LIỆU (Helper functions)
# ==============================================================================
# Các hàm bắt đầu bằng dấu gạch dưới `_` không có decorator `@mcp.tool()`,
# nghĩa là đây là hàm nội bộ của Python, LLM sẽ KHÔNG nhìn thấy hay gọi được.

def _to_vector(x):
    """Ép kiểu dữ liệu danh sách (list) về vector 1 chiều của NumPy"""
    arr = np.array(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Input must be a 1D list representing a vector.")
    return arr


def _to_matrix(x):
    """Ép kiểu dữ liệu danh sách lồng (nested list) về ma trận 2 chiều của NumPy"""
    arr = np.array(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Input must be a 2D list representing a matrix.")
    return arr


# ==============================================================================
# 4. CÁC CÔNG CỤ TÍNH TOÁN ĐẠI SỐ TUYẾN TÍNH (Vector & Matrix)
# ==============================================================================

@mcp.tool()
def vector_add(a: list, b: list) -> list:
    """Cộng 2 vector cùng kích thước"""
    va, vb = _to_vector(a), _to_vector(b)
    if va.shape != vb.shape:
        raise ValueError("Vectors must have the same size.")
    # .tolist(): chuyển numpy array về dạng list thuần của Python để gửi qua JSON
    return (va + vb).tolist()


@mcp.tool()
def vector_subtract(a: list, b: list) -> list:
    """Trừ 2 vector cùng kích thước"""
    va, vb = _to_vector(a), _to_vector(b)
    if va.shape != vb.shape:
        raise ValueError("Vectors must have the same size.")
    return (va - vb).tolist()


@mcp.tool()
def vector_dot(a: list, b: list) -> float:
    """Tích vô hướng (dot product) của 2 vector: trả về 1 số thực đơn lẻ"""
    va, vb = _to_vector(a), _to_vector(b)
    if va.shape != vb.shape:
        raise ValueError("Vectors must have the same size.")
    return float(np.dot(va, vb))


@mcp.tool()
def vector_elementwise_multiply(a: list, b: list) -> list:
    """Nhân từng phần tử tương ứng (Hadamard product) giữa 2 vector"""
    va, vb = _to_vector(a), _to_vector(b)
    if va.shape != vb.shape:
        raise ValueError("Vectors must have the same size.")
    return (va * vb).tolist()


@mcp.tool()
def matrix_add(a: list, b: list) -> list:
    """Cộng 2 ma trận cùng kích thước hàng và cột"""
    ma, mb = _to_matrix(a), _to_matrix(b)
    if ma.shape != mb.shape:
        raise ValueError("Matrices must have the same shape.")
    return (ma + mb).tolist()


@mcp.tool()
def matrix_subtract(a: list, b: list) -> list:
    """Trừ 2 ma trận cùng kích thước hàng và cột"""
    ma, mb = _to_matrix(a), _to_matrix(b)
    if ma.shape != mb.shape:
        raise ValueError("Matrices must have the same shape.")
    return (ma - mb).tolist()


@mcp.tool()
def matrix_multiply(a: list, b: list) -> list:
    """Nhân 2 ma trận theo quy tắc đại số tuyến tính"""
    ma, mb = _to_matrix(a), _to_matrix(b)
    try:
        # Cấu trúc lạ: Toán tử `@` trong Python dùng riêng cho phép nhân ma trận (Matrix Multiplication).
        # Tương đương với np.matmul(ma, mb) hoặc ma.dot(mb)
        result = ma @ mb
    except ValueError:
        raise ValueError(
            f"Incompatible matrix shapes for multiplication: {ma.shape} x {mb.shape}"
        )
    return result.tolist()


@mcp.tool()
def matrix_transpose(a: list) -> list:
    """Chuyển vị ma trận: đổi hàng thành cột"""
    ma = _to_matrix(a)
    # Cấu trúc: `.T` là thuộc tính chuyển vị ma trận (Transpose) trong NumPy
    return ma.T.tolist()


# ==============================================================================
# 5. CHẠY SERVER LẮNG NGHE YÊU CẦU
# ==============================================================================
if __name__ == "__main__":
    # Giao thức transport="stdio":
    # Server chạy ngầm như một tiến trình con, giao tiếp với tiến trình cha (Client)
    # thông qua Standard Input (stdin) và Standard Output (stdout) bằng các gói tin JSON-RPC.
    mcp.run(transport="stdio")
