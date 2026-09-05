try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP

from parser import parse_uit_schedule, get_classes_by_date, get_empty_rooms

mcp = FastMCP("UITSchedulerParser")


@mcp.tool()
def parse_schedule_tool(date: str = "2026-09-07", building: str = "B", limit: int = 30) -> list[dict]:
    """
    Cào danh sách thời khóa biểu lớp học theo tuần và tòa nhà từ UIT Portal (https://portal.uit.edu.vn/lich-phong).
    
    Args:
        date (str): Ngày trong tuần cần tra cứu theo định dạng 'YYYY-MM-DD' (Ví dụ: '2026-09-07').
        building (str): Tòa nhà cần tra cứu ('B', 'A', 'C', 'E'). Mặc định là 'B'.
        limit (int): Giới hạn số lượng lớp trả về (mặc định 30).
        
    Returns:
        list[dict]: Danh sách các lớp học gồm tên môn, mã lớp, phòng, thứ, ngày, tiết, thời gian.
    """
    return parse_uit_schedule(date=date, building=building, limit=limit)


@mcp.tool()
def get_classes_by_date_tool(date: str = "2026-09-07", building: str = "B") -> list[dict]:
    """
    Lấy danh sách các môn học / lớp học diễn ra trong một ngày cụ thể tại tòa nhà.
    
    Args:
        date (str): Ngày cụ thể theo định dạng 'YYYY-MM-DD' (Ví dụ: '2026-09-07').
        building (str): Tòa nhà cần tra cứu ('B', 'A', 'C', 'E'). Mặc định là 'B'.
        
    Returns:
        list[dict]: Danh sách lớp học diễn ra trong ngày.
    """
    return get_classes_by_date(date=date, building=building)


@mcp.tool()
def get_empty_rooms_tool(date: str = "2026-09-07", period_code: str = "first", building: str = "B") -> list[dict]:
    """
    Tìm danh sách các phòng học đang còn trống trong một tiết cụ thể để tự học hoặc họp nhóm.
    
    Args:
        date (str): Ngày cần tìm phòng trống ('YYYY-MM-DD').
        period_code (str): Mã tiết ('first': tiết 1, 'second': tiết 2, 'third': tiết 3, 'fourth': tiết 4, 'fifth': tiết 5, 'sixth': tiết 6, 'seventh': tiết 7, 'eighth': tiết 8, 'ninth': tiết 9, 'tenth': tiết 10).
        building (str): Tòa nhà ('B', 'A', 'C', 'E'). Mặc định là 'B'.
        
    Returns:
        list[dict]: Danh sách phòng trống kèm số phòng, tên phòng, sức chứa, tầng, thời gian.
    """
    return get_empty_rooms(date=date, period_code=period_code, building=building)


if __name__ == "__main__":
    mcp.run(transport="stdio")
