"""ดึงรายการเกมและชั่วโมงเล่นของผู้ใช้จาก Steam Web API (ใช้ได้เฉพาะโปรไฟล์ที่ตั้งเป็น Public)

ต้องมี Steam Web API key (ขอฟรีที่ https://steamcommunity.com/dev/apikey) อย่าใส่ key ลงในโค้ดหรือ commit ขึ้น GitHub
"""
import json
import re
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.steampowered.com"
TIMEOUT = 15


class SteamError(Exception):
    """ข้อความผิดพลาดที่ตั้งใจให้ผู้ใช้อ่านได้ (ภาษาไทย)"""


def _get(path: str, params: dict):
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise SteamError("API key ไม่ถูกต้อง หรือไม่มีสิทธิ์ใช้งาน") from None
        if e.code == 429:
            raise SteamError("เรียก Steam ถี่เกินไป ลองใหม่ภายหลัง") from None
        raise SteamError(f"Steam ตอบกลับผิดพลาด (HTTP {e.code})") from None
    except (urllib.error.URLError, TimeoutError):
        raise SteamError("เชื่อมต่อ Steam ไม่ได้ ตรวจอินเทอร์เน็ตแล้วลองใหม่") from None


def parse_steam_input(text: str):
    """คืน ('id', steamid64) หรือ ('vanity', ชื่อ) จากเลข SteamID64 / ชื่อ vanity / ลิงก์โปรไฟล์"""
    t = (text or "").strip().rstrip("/")
    m = re.search(r"steamcommunity\.com/profiles/(\d{17})", t)
    if m:
        return "id", m.group(1)
    m = re.search(r"steamcommunity\.com/id/([A-Za-z0-9_-]+)", t)
    if m:
        return "vanity", m.group(1)
    if re.fullmatch(r"\d{17}", t):
        return "id", t
    if re.fullmatch(r"[A-Za-z0-9_-]{2,32}", t):
        return "vanity", t
    raise SteamError("รูปแบบไม่ถูกต้อง: ใส่ SteamID64 (เลข 17 หลัก) ชื่อ vanity หรือลิงก์โปรไฟล์")


def resolve_steam_id(text: str, api_key: str) -> str:
    kind, value = parse_steam_input(text)
    if kind == "id":
        return value
    data = _get("ISteamUser/ResolveVanityURL/v1/", {"key": api_key, "vanityurl": value}).get("response", {})
    if data.get("success") != 1:
        raise SteamError("ไม่พบโปรไฟล์นี้ ลองใช้เลข SteamID64 แทน")
    return data["steamid"]


def fetch_owned_games(steam_id: str, api_key: str) -> list[dict]:
    """คืน [{'appid', 'name', 'hours'}] เฉพาะเกมที่เคยเล่น (ชั่วโมง > 0)"""
    data = _get("IPlayerService/GetOwnedGames/v1/",
                {"key": api_key, "steamid": steam_id, "include_appinfo": 1, "format": "json"}).get("response", {})
    if "games" not in data:
        raise SteamError("อ่านรายการเกมไม่ได้: โปรไฟล์อาจเป็น Private "
                         "(ตั้ง Game details เป็น Public ที่ Steam > Edit Profile > Privacy Settings)")
    out = []
    for g in data["games"]:
        hours = g.get("playtime_forever", 0) / 60.0
        if hours > 0:
            out.append({"appid": int(g["appid"]), "name": g.get("name", ""), "hours": round(hours, 1)})
    return out
