import re
import json
import sys
import requests
from datetime import datetime

# Đảm bảo UTF-8 cho Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

UIT_PORTAL_URL = "https://portal.uit.edu.vn/lich-phong"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

PERIOD_MAPPING = {
    "first": {"order": 1, "name": "Tiết 1", "time": "07:30 - 08:15", "session": "Sáng"},
    "second": {"order": 2, "name": "Tiết 2", "time": "08:15 - 09:00", "session": "Sáng"},
    "third": {"order": 3, "name": "Tiết 3", "time": "09:00 - 09:45", "session": "Sáng"},
    "fourth": {"order": 4, "name": "Tiết 4", "time": "10:00 - 10:45", "session": "Sáng"},
    "fifth": {"order": 5, "name": "Tiết 5", "time": "10:45 - 11:30", "session": "Sáng"},
    "sixth": {"order": 6, "name": "Tiết 6", "time": "13:00 - 13:45", "session": "Chiều"},
    "seventh": {"order": 7, "name": "Tiết 7", "time": "13:45 - 14:30", "session": "Chiều"},
    "eighth": {"order": 8, "name": "Tiết 8", "time": "14:30 - 15:15", "session": "Chiều"},
    "ninth": {"order": 9, "name": "Tiết 9", "time": "15:30 - 16:15", "session": "Chiều"},
    "tenth": {"order": 10, "name": "Tiết 10", "time": "16:15 - 17:00", "session": "Chiều"},
    "eleventh": {"order": 11, "name": "Tiết 11", "time": "17:00 - 17:45", "session": "Chiều"},
    "twelfth": {"order": 12, "name": "Tiết 12", "time": "17:45 - 18:30", "session": "Chiều"},
    "thirteenth": {"order": 13, "name": "Tiết 13", "time": "18:30 - 19:15", "session": "Tối"},
    "fourteenth": {"order": 14, "name": "Tiết 14", "time": "19:15 - 20:00", "session": "Tối"},
    "fifteenth": {"order": 15, "name": "Tiết 15", "time": "20:00 - 20:45", "session": "Tối"},
}

WEEKDAY_VN = {
    "monday": "Thứ Hai",
    "tuesday": "Thứ Ba",
    "wednesday": "Thứ Tư",
    "thursday": "Thứ Năm",
    "friday": "Thứ Sáu",
    "saturday": "Thứ Bảy",
    "sunday": "Chủ Nhật",
}


def _extract_initial_data(html: str) -> dict:
    """Trích xuất dictionary initialData từ HTML Next.js của UIT Portal."""
    # Tìm đoạn JSON chứa "initialData":{...}
    # Do Next.js escape dấu ngoặc kép dạng \" trong self.__next_f.push
    # Ta chuẩn hóa bằng cách unescape hoặc tìm bằng regex
    clean_html = html.replace('\\"', '"').replace('\\\\', '\\')
    
    # Tìm khối JSON initialData
    pattern = r'"initialData":(\{.*?"bookings":\s*\[.*?\]\s*,\s*"weekdays":\s*\[.*?\]\s*\})'
    match = re.search(pattern, clean_html, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
            
    # Pattern 2: Tìm theo cấu trúc rooms, periods, weekdays, bookings
    match2 = re.search(r'(\{"rooms":\[.*?\],"periods":\[.*?\],"weekdays":\[.*?\],"bookings":\[.*?\]\})', clean_html, re.DOTALL)
    if match2:
        try:
            return json.loads(match2.group(1))
        except Exception:
            pass
            
    raise RuntimeError("Không tìm thấy dữ liệu thời khóa biểu (initialData) trên trang UIT Portal.")


def fetch_uit_portal_data(date: str = "2026-09-07", building: str = "B") -> dict:
    """Gửi GET request tới portal.uit.edu.vn và trả về raw data dict."""
    params = {
        "view": "week",
        "building": building,
        "date": date
    }
    
    response = requests.get(UIT_PORTAL_URL, params=params, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return _extract_initial_data(response.text)


def parse_uit_schedule(date: str = "2026-09-07", building: str = "B", limit: int = 50) -> list[dict]:
    """
    Cào toàn bộ lịch học / lịch phòng tại UIT trong tuần chứa ngày `date` và tòa nhà `building`.
    
    Args:
        date (str): Ngày cần tra cứu dạng 'YYYY-MM-DD' (Ví dụ: '2026-09-07').
        building (str): Tòa nhà tại UIT ('B', 'A', 'C', 'E', ...). Mặc định là 'B'.
        limit (int): Giới hạn số lượng kết quả trả về.
        
    Returns:
        list[dict]: Danh sách các lớp học / sự kiện với thông tin chi tiết.
    """
    raw_data = fetch_uit_portal_data(date=date, building=building)
    
    rooms_by_id = {r["id"]: r for r in raw_data.get("rooms", [])}
    
    results = []
    for b in raw_data.get("bookings", []):
        room_info = rooms_by_id.get(b.get("room_id"), {})
        room_code = room_info.get("code", "N/A")
        room_name = room_info.get("name", room_code)
        
        # Chuyển đổi mã tiết sang thời gian
        period_codes = b.get("period_codes", [])
        period_names = [PERIOD_MAPPING.get(p, {}).get("name", p) for p in period_codes]
        
        start_time = PERIOD_MAPPING.get(period_codes[0], {}).get("time", "").split(" - ")[0] if period_codes else "N/A"
        end_time = PERIOD_MAPPING.get(period_codes[-1], {}).get("time", "").split(" - ")[-1] if period_codes else "N/A"
        time_range = f"{start_time} - {end_time}" if start_time != "N/A" else "N/A"
        
        weekday_code = b.get("weekday_code", "")
        weekday_str = WEEKDAY_VN.get(weekday_code, weekday_code)
        
        results.append({
            "course": b.get("title") or "Sự kiện / Họp",
            "class_code": b.get("unit_name", "N/A"),
            "room": room_code,
            "room_name": room_name,
            "building": room_info.get("place_name", f"Nhà {building}"),
            "capacity": room_info.get("capacity", 0),
            "date": b.get("date"),
            "weekday": weekday_str,
            "periods": ", ".join(period_names),
            "time": time_range,
            "student_count": b.get("quantity", 0),
            "type": b.get("regist_type", "timetable")
        })
        
    return results[:limit]


def get_classes_by_date(date: str = "2026-09-07", building: str = "B") -> list[dict]:
    """
    Lấy danh sách các lớp học diễn ra đúng vào ngày `date` tại tòa nhà `building`.
    
    Args:
        date (str): Ngày cụ thể (Ví dụ: '2026-09-07').
        building (str): Tòa nhà ('B', 'A', 'C', 'E').
        
    Returns:
        list[dict]: Danh sách các lớp học trong ngày.
    """
    all_schedule = parse_uit_schedule(date=date, building=building, limit=500)
    filtered = [item for item in all_schedule if item.get("date") == date]
    return filtered


def get_empty_rooms(date: str = "2026-09-07", period_code: str = "first", building: str = "B") -> list[dict]:
    """
    Tìm danh sách các phòng học đang còn trống trong một tiết cụ thể vào ngày `date`.
    
    Args:
        date (str): Ngày cần kiểm tra ('YYYY-MM-DD').
        period_code (str): Mã tiết ('first', 'second', 'third', 'fourth', 'fifth', 'sixth', ...).
        building (str): Tòa nhà ('B', 'A', ...).
        
    Returns:
        list[dict]: Danh sách các phòng trống kèm sức chứa và tầng.
    """
    raw_data = fetch_uit_portal_data(date=date, building=building)
    
    rooms = raw_data.get("rooms", [])
    bookings = raw_data.get("bookings", [])
    
    # Tập hợp các room_id có lịch trong ngày và tiết đó
    busy_room_ids = set()
    for b in bookings:
        if b.get("date") == date and period_code in b.get("period_codes", []):
            busy_room_ids.add(b.get("room_id"))
            
    empty_rooms = []
    for r in rooms:
        if r.get("id") not in busy_room_ids:
            empty_rooms.append({
                "room": r.get("code"),
                "name": r.get("name"),
                "floor": r.get("floor"),
                "capacity": r.get("capacity"),
                "building": r.get("place_name", f"Nhà {building}"),
                "date": date,
                "period": PERIOD_MAPPING.get(period_code, {}).get("name", period_code),
                "time": PERIOD_MAPPING.get(period_code, {}).get("time", "")
            })
            
    return empty_rooms


if __name__ == "__main__":
    print("--- TEST 1: Cào lịch tuần 2026-09-07 tòa B ---")
    schedule = parse_uit_schedule("2026-09-07", "B", limit=5)
    for s in schedule:
        print(f"[{s['weekday']} {s['date']}] {s['time']} | Phòng {s['room']} | Môn: {s['course']} ({s['class_code']})")
        
    print(f"\nTổng số lớp cào được trong tuần: {len(parse_uit_schedule('2026-09-07', 'B', limit=500))}")
    
    print("\n--- TEST 2: Lớp học ngày 2026-09-07 ---")
    today_classes = get_classes_by_date("2026-09-07", "B")
    print(f"Số lớp học ngày 2026-09-07: {len(today_classes)}")
    
    print("\n--- TEST 3: Phòng trống Tiết 1 sáng 2026-09-07 ---")
    free_rooms = get_empty_rooms("2026-09-07", "first", "B")
    print(f"Số phòng trống Tiết 1: {len(free_rooms)}")
    if free_rooms:
        print("Ví dụ phòng trống:", free_rooms[:3])
