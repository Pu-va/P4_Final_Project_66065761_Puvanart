"""ตัวช่วยลดเกมซีรีส์เดียวกันซ้ำในรายการแนะนำ (ใช้ร่วมกันระหว่างโน้ตบุ๊กกับ app.py)

หลักการ: เกม 2 เกมถือว่าเป็นซีรีส์เดียวกันเมื่อ "คำนำของชื่อ" ตรงกัน (ตัดเลขภาค เลขโรมัน และคำเชื่อมออกก่อน)
เช่น Deus Ex Human Revolution / Deus Ex The Fall, Counter-Strike Source / Counter-Strike Global Offensive
เป็นกฎประมาณ (heuristic) ไม่ใช่ข้อมูลซีรีส์จริง อาจพลาดบางคู่ เช่นซีรีส์ที่ชื่อภาคต่อไม่ขึ้นต้นเหมือนกัน
"""
import re

_STOP = {"the", "of", "and", "a", "an"}
_ROMAN = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"}
# คำนำแบรนด์ที่ใช้ร่วมกันหลายซีรีส์ ตัดออกก่อนเทียบ (เช่น Tom Clancy's Splinter Cell กับ Tom Clancy's Rainbow Six คนละซีรีส์)
_BRAND_PREFIXES = [("tom", "clancy", "s"), ("sid", "meier", "s"), ("lego",), ("star", "wars"),
                   ("warhammer", "40", "000"), ("warhammer", "40k"), ("warhammer",)]


def series_tokens(title: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", str(title).lower())
    for p in _BRAND_PREFIXES:
        if len(toks) > len(p) and tuple(toks[:len(p)]) == p:
            toks = toks[len(p):]
            break
    return [t for t in toks if t not in _STOP and t not in _ROMAN and not t.isdigit()]


def same_series(a: str, b: str) -> bool:
    ta, tb = series_tokens(a), series_tokens(b)
    n = min(len(ta), len(tb), 2)
    return n > 0 and ta[:n] == tb[:n]


def limit_series(titles, max_per_series: int = 1, k: int | None = None) -> list[str]:
    """คงลำดับเดิม แต่เก็บแต่ละซีรีส์ไม่เกิน max_per_series เกม แล้วตัดเหลือ k เกม"""
    kept: list[str] = []
    for t in titles:
        if sum(same_series(t, x) for x in kept) < max_per_series:
            kept.append(t)
            if k is not None and len(kept) == k:
                break
    return kept
