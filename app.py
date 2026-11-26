"""GameMatch AI: เว็บเดโมแนะนำเกม (Streamlit)  ·  รัน: python -m streamlit run app.py
ไฟล์ที่ต้องอยู่โฟลเดอร์เดียวกัน: app.py, steam_client.py, recommender.py, gamematch_model.joblib
(ถ้ามีโฟลเดอร์ .streamlit/config.toml ธีมสีเข้มจะสมบูรณ์ขึ้น แต่ไม่บังคับ)
"""
import os
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

import steam_client
from series_utils import limit_series
from recommender import GameMatchRecommender

try:
    from recommender import DEFAULT_HOURS
except ImportError:
    DEFAULT_HOURS = 10.0
from recommender import TAG_COLD_START_WEIGHTS

HOURS_CAP = 908.0  # ตัดชั่วโมงที่ percentile 99 เหมือนตอนฝึกโมเดล
MODEL_PATH = Path(__file__).with_name("gamematch_model.joblib")
MODES = ["🎮 เลือกเกมที่เคยเล่น", "🔗 เชื่อม Steam", "🆕 ผู้เล่นใหม่: เลือกแนวที่ชอบ"]

st.set_page_config(page_title="GameMatch AI", page_icon="🎮", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=Prompt:wght@400;600&display=swap');
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #1b1040 0%, transparent 60%),
                     radial-gradient(900px 500px at 100% 0%, #06283a 0%, transparent 55%), #0b0f1a; }
html, body, [class*="css"] { font-family: 'Prompt', sans-serif; }
[data-testid="stSidebar"] { background: #0f1424; border-right: 1px solid #1f2a44; }
.hero { padding: 28px 32px; border-radius: 18px; margin-bottom: 18px;
        background: linear-gradient(120deg, rgba(0,229,255,.14), rgba(255,46,136,.14));
        border: 1px solid rgba(0,229,255,.35); box-shadow: 0 0 28px rgba(0,229,255,.12); }
.hero h1 { font-family: 'Orbitron', sans-serif; font-weight: 800; font-size: 44px; margin: 0; letter-spacing: 2px;
           background: linear-gradient(90deg, #00e5ff, #ff2e88); -webkit-background-clip: text;
           -webkit-text-fill-color: transparent; }
.hero p { margin: 8px 0 0; color: #9fb0d0; font-size: 16px; }
.sec { font-family: 'Orbitron', sans-serif; color: #00e5ff; letter-spacing: 2px; font-size: 20px; margin: 22px 0 10px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 16px; }
.card { position: relative; padding: 18px 18px 16px 18px; border-radius: 16px; background: #131a2e;
        border: 1px solid #25314f; transition: all .18s ease; }
.card:hover { transform: translateY(-3px); border-color: #00e5ff; box-shadow: 0 6px 26px rgba(0,229,255,.18); }
.rank { position: absolute; top: -12px; left: 14px; font-family: 'Orbitron', sans-serif; font-weight: 800;
        padding: 2px 12px; border-radius: 10px; background: linear-gradient(90deg, #ff2e88, #7c4dff); color: #fff; font-size: 15px; }
.title { font-weight: 600; font-size: 19px; color: #f2f6ff; margin: 8px 0 6px; line-height: 1.3; }
.why { display: inline-block; font-size: 13px; color: #00e5ff; background: rgba(0,229,255,.1);
       border: 1px solid rgba(0,229,255,.3); border-radius: 999px; padding: 2px 10px; margin-bottom: 8px; }
.tags { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; min-height: 26px; }
.tag { font-size: 12.5px; color: #c7d3ee; background: #1c2540; border-radius: 6px; padding: 2px 8px; }
.row { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.price { font-weight: 600; font-size: 15px; padding: 3px 12px; border-radius: 8px; background: #1c2540; color: #e8ecf8; }
.price.free { background: rgba(46,230,128,.15); color: #2ee680; border: 1px solid rgba(46,230,128,.4); }
.rate { font-size: 13px; margin-top: 12px; color: #9fb0d0; }
.bar { height: 6px; border-radius: 4px; background: #1c2540; margin-top: 5px; overflow: hidden; }
.bar > span { display: block; height: 100%; border-radius: 4px; }
.btn { display: inline-block; text-decoration: none !important; font-size: 13.5px; font-weight: 600; color: #0b0f1a !important;
       background: linear-gradient(90deg, #00e5ff, #7c4dff); padding: 5px 14px; border-radius: 8px; }
.btn:hover { filter: brightness(1.15); }
.stButton > button[kind="primary"] { background: linear-gradient(90deg, #ff2e88, #7c4dff); border: none; font-weight: 600; }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
.logo { font-family: 'Orbitron', sans-serif; font-weight: 800; font-size: 24px; letter-spacing: 2px; margin: 0;
        background: linear-gradient(90deg, #00e5ff, #ff2e88); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.logo-sub { color: #7c8db5; font-size: 13px; margin: 2px 0 14px; }
.side-h { font-family: 'Orbitron', sans-serif; color: #00e5ff; font-size: 13px; letter-spacing: 2px; margin: 16px 0 6px; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: 8px; }
[data-testid="stSidebar"] [role="radiogroup"] > label { background: #151d35; border: 1px solid #25314f; border-radius: 12px;
        padding: 10px 12px; width: 100%; transition: all .15s ease; }
[data-testid="stSidebar"] [role="radiogroup"] > label:hover { border-color: #00e5ff; background: #19233f; }
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) { border-color: #ff2e88;
        background: linear-gradient(90deg, rgba(0,229,255,.12), rgba(255,46,136,.16)); box-shadow: 0 0 14px rgba(255,46,136,.25); }
.stat { background: #151d35; border: 1px solid #25314f; border-radius: 12px; padding: 12px 14px; margin-top: 8px; }
.stat b { font-family: 'Orbitron', sans-serif; font-size: 22px; color: #2ee680; }
.stat span { display: block; color: #9fb0d0; font-size: 12.5px; }
.style-panel { display: flex; gap: 22px; align-items: center; flex-wrap: wrap; padding: 20px 24px; border-radius: 18px;
        background: linear-gradient(120deg, rgba(124,77,255,.20), rgba(0,229,255,.10)); border: 1px solid rgba(124,77,255,.5); }
.style-emoji { font-size: 58px; line-height: 1; filter: drop-shadow(0 0 14px rgba(124,77,255,.8)); }
.style-label { font-size: 12.5px; letter-spacing: 2px; color: #9fb0d0; font-family: 'Orbitron', sans-serif; }
.style-name { font-size: 28px; font-weight: 600; color: #fff; margin: 2px 0; }
.style-desc { color: #b8c5e3; font-size: 14.5px; max-width: 520px; }
.style-bars { flex: 1; min-width: 260px; }
.sb { display: flex; align-items: center; gap: 10px; margin: 6px 0; font-size: 13.5px; color: #c7d3ee; }
.sb .n { width: 120px; } .sb .t { flex: 1; height: 8px; background: #1c2540; border-radius: 5px; overflow: hidden; }
.sb .t > span { display: block; height: 100%; background: linear-gradient(90deg, #00e5ff, #ff2e88); border-radius: 5px; }
.sb .p { width: 42px; text-align: right; color: #9fb0d0; }
.podium { display: flex; justify-content: center; align-items: flex-end; gap: 16px; flex-wrap: wrap; margin: 26px 0 8px; }
.pcol { width: 310px; display: flex; flex-direction: column; align-items: stretch; }
.pcard { text-align: center; padding: 16px 14px; border-radius: 16px 16px 0 0; background: #131a2e; border: 1px solid #25314f; border-bottom: none; }
.pcard .medal { font-size: 40px; line-height: 1; }
.pcard .title { margin: 6px 0; }
.pcard .tags { justify-content: center; min-height: 0; margin-bottom: 10px; }
.ped { display: flex; align-items: center; justify-content: center; font-family: 'Orbitron', sans-serif; font-weight: 800;
       font-size: 44px; color: rgba(255,255,255,.92); border-radius: 0 0 10px 10px; }
.p1 .pcard { border-color: #ffd24d; box-shadow: 0 0 30px rgba(255,210,77,.28); }
.p1 .ped { height: 120px; background: linear-gradient(180deg, #ffd24d, #b8860b); }
.p2 .pcard { border-color: #c5ccd8; } .p2 .ped { height: 84px; background: linear-gradient(180deg, #c5ccd8, #6b7487); }
.p3 .pcard { border-color: #d98a4e; } .p3 .ped { height: 56px; background: linear-gradient(180deg, #d98a4e, #7a4720); }
</style>
"""


@st.cache_resource
def load_model():
    return GameMatchRecommender.load(MODEL_PATH)


@st.cache_resource
def appid_to_title(_model):
    ids = pd.to_numeric(_model.game_info["app_id"], errors="coerce")
    return {int(a): t for a, t in zip(ids, _model.game_titles) if pd.notna(a)}


def fmt_price(p):
    if p is None or pd.isna(p):
        return "-"
    return "ฟรี" if float(p) == 0 else f"${float(p):,.2f}"


def steam_link(a):
    return f"https://store.steampowered.com/app/{int(a)}" if a is not None and pd.notna(a) else None


def rating_color(label):
    s = str(label).lower()
    if "negative" in s:
        return "#ff4d6d"
    if "mixed" in s:
        return "#ffc247"
    if "positive" in s:
        return "#2ee680"
    return "#7c8db5"


def card_html(r) -> str:
    tags = "".join(f'<span class="tag">{escape(t.strip())}</span>'
                   for t in str(r["tags"]).split(",") if t.strip() and not t.strip().startswith("("))
    free = pd.notna(r["price_final"]) and float(r["price_final"]) == 0
    pct = r["positive_ratio"]
    pct_val = int(pct) if pd.notna(pct) else 0
    label = r["rating"] if pd.notna(r["rating"]) else "ไม่มีข้อมูลรีวิว"
    color = rating_color(label)
    link = steam_link(r["app_id"])
    btn = f'<a class="btn" href="{link}" target="_blank" rel="noopener">เปิดบน Steam ↗</a>' if link else ""
    pct_txt = f" · {pct_val}% บวก" if pd.notna(pct) else ""
    return (
        f'<div class="card"><div class="rank">#{int(r["rank"])}</div>'
        f'<div class="title">{escape(str(r["game_title"]))}</div>'
        f'<div class="why">{escape(str(r["reason"]))}</div>'
        f'<div class="tags">{tags or "<span class=tag>ไม่มีข้อมูลแท็ก</span>"}</div>'
        f'<div class="row"><span class="price{" free" if free else ""}">{escape(fmt_price(r["price_final"]))}</span>{btn}</div>'
        f'<div class="rate"><span style="color:{color}">{escape(str(label))}</span>{pct_txt}'
        f'<div class="bar"><span style="width:{pct_val}%;background:{color}"></span></div></div></div>'
    )


def podium_item(r, place: int) -> str:
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}[place]
    tags = "".join(f'<span class="tag">{escape(t.strip())}</span>'
                   for t in str(r["tags"]).split(",")[:3] if t.strip() and not t.strip().startswith("("))
    free = pd.notna(r["price_final"]) and float(r["price_final"]) == 0
    pct = r["positive_ratio"]
    label = r["rating"] if pd.notna(r["rating"]) else "ไม่มีข้อมูลรีวิว"
    link = steam_link(r["app_id"])
    btn = f'<a class="btn" href="{link}" target="_blank" rel="noopener">เปิดบน Steam ↗</a>' if link else ""
    pct_txt = f" · {int(pct)}% บวก" if pd.notna(pct) else ""
    return (
        f'<div class="pcol p{place}"><div class="pcard"><div class="medal">{medal}</div>'
        f'<div class="title">{escape(str(r["game_title"]))}</div>'
        f'<div class="why">{escape(str(r["reason"]))}</div>'
        f'<div class="tags">{tags}</div>'
        f'<div class="row" style="justify-content:center"><span class="price{" free" if free else ""}">'
        f'{escape(fmt_price(r["price_final"]))}</span>{btn}</div>'
        f'<div class="rate"><span style="color:{rating_color(label)}">{escape(str(label))}</span>{pct_txt}</div></div>'
        f'<div class="ped">{place}</div></div>'
    )


def show_cards(df: pd.DataFrame):
    top, rest = df.iloc[:3], df.iloc[3:]
    if len(top) == 3:  # วางแท่นแบบ 2-1-3 โดยอันดับ 1 อยู่กลาง
        html = "".join(podium_item(top.iloc[i], i + 1) for i in (1, 0, 2))
        st.markdown(f'<div class="podium">{html}</div>', unsafe_allow_html=True)
    if len(rest):
        st.markdown('<div class="sec">MORE PICKS</div><div class="grid">'
                    + "".join(card_html(r) for _, r in rest.iterrows()) + "</div>", unsafe_allow_html=True)


# ---- สไตล์ผู้เล่น: จัดกลุ่มจาก tags ของเกมที่เคยเล่น (ถ่วงน้ำหนักด้วย log ชั่วโมงเล่น) ----
STYLES = [
    ("นักล่าสายยิง", "🎯", "ชอบเกมยิงและการประลองฝีมือ เน้นรีเฟล็กซ์และความแม่นยำ",
     ["fps", "shooter", "tactical", "battle royale"]),
    ("นักผจญภัย", "🗺️", "ชอบโลกกว้าง เนื้อเรื่อง และการสำรวจ",
     ["rpg", "adventure", "open world", "story", "fantasy", "exploration"]),
    ("นักวางแผน", "♟️", "ชอบคิดเกมยาว บริหารทรัพยากร และวางกลยุทธ์",
     ["strategy", "rts", "turn-based", "tower defense", "city builder", "4x", "tactics"]),
    ("นักเอาตัวรอด", "🏕️", "ชอบเก็บของ สร้างฐาน และต่อสู้กับความอยู่รอด",
     ["survival", "crafting", "zombie", "sandbox"]),
    ("สายปาร์ตี้ทีมเวิร์ก", "🤝", "ชอบเล่นกับเพื่อนและแข่งกับผู้เล่นคนอื่น",
     ["multiplayer", "co-op", "pvp", "moba", "team-based", "online"]),
    ("นักสร้างสรรค์", "🏗️", "ชอบจำลอง บริหาร และสร้างสิ่งต่าง ๆ ตามใจ",
     ["simulation", "building", "management", "farming", "design"]),
    ("สายสยองขวัญ", "👻", "ชอบความตึงเครียด บรรยากาศมืด และความหลอน",
     ["horror", "gore", "dark", "psychological"]),
    ("สายสปีด", "🏎️", "ชอบความเร็ว การแข่งขัน และกีฬา",
     ["racing", "sports", "driving", "football", "soccer"]),
    ("สายชิลอินดี้", "🍃", "ชอบเกมเบา ๆ ผ่อนคลาย ศิลปะน่ารักและไอเดียแปลกใหม่",
     ["indie", "casual", "relaxing", "cute", "puzzle", "pixel", "atmospheric"]),
    ("สายแอ็กชันสายบู๊", "⚔️", "ชอบเกมต่อสู้ จังหวะไว และบอสสุดโหด",
     ["hack and slash", "fighting", "platformer", "souls", "beat", "roguelike"]),
]


def _style_hits(tag_tokens):
    toks = [str(t).lower().replace("_", " ") for t in tag_tokens]
    return [sum(any(kw in t for kw in kws) for t in toks) for _, _, _, kws in STYLES]


def player_style(model, history=None, tag_choices=None):
    """คืน (ชื่อสไตล์หลัก, emoji, คำอธิบาย, [(ชื่อ, สัดส่วน%)...]) หรือ None ถ้าไม่มีข้อมูลแท็กพอ"""
    import math
    score = [0.0] * len(STYLES)
    if history:
        for title, hours in history.items():
            if title not in model.title_to_idx:
                continue
            tags = model.game_info.loc[title, "tags"]
            if isinstance(tags, pd.Series):
                tags = tags.iloc[0]
            if not isinstance(tags, str) or not tags.strip():
                continue
            toks = tags.split()
            hits = _style_hits(toks)
            w = math.log1p(float(hours or DEFAULT_HOURS))
            for i, h in enumerate(hits):
                score[i] += w * h / max(1, len(toks))
    elif tag_choices:
        for i, h in enumerate(_style_hits(tag_choices)):
            score[i] += h
    total = sum(score)
    if total <= 0:
        return None
    ranked = sorted(((STYLES[i][0], 100 * s / total) for i, s in enumerate(score) if s > 0), key=lambda x: -x[1])
    main = next(s for s in STYLES if s[0] == ranked[0][0])
    return main[0], main[1], main[2], ranked[:4]


def show_style(info):
    name, emoji, desc, ranked = info
    bars = "".join(f'<div class="sb"><span class="n">{escape(n)}</span><span class="t"><span style="width:{p:.0f}%"></span></span>'
                   f'<span class="p">{p:.0f}%</span></div>' for n, p in ranked)
    st.markdown(
        f'<div class="sec">YOUR PLAYER TYPE</div><div class="style-panel"><div class="style-emoji">{emoji}</div>'
        f'<div><div class="style-label">คุณคือผู้เล่นสไตล์</div><div class="style-name">{escape(name)}</div>'
        f'<div class="style-desc">{escape(desc)}</div></div><div class="style-bars">{bars}</div></div>',
        unsafe_allow_html=True)


def show_table(df: pd.DataFrame):
    view = pd.DataFrame({
        "อันดับ": df["rank"], "เกม": df["game_title"], "เหตุผลที่แนะนำ": df["reason"],
        "แท็ก": df["tags"].replace("", "(ไม่มีข้อมูลแท็ก)"),
        "ราคา": df["price_final"].map(fmt_price), "เรตติ้ง": df["rating"].fillna("-"),
        "% รีวิวบวก": df["positive_ratio"], "Steam": df["app_id"].map(steam_link),
    })
    st.dataframe(view, hide_index=True, use_container_width=True, column_config={
        "Steam": st.column_config.LinkColumn("Steam", display_text="เปิดหน้าเกม"),
        "% รีวิวบวก": st.column_config.NumberColumn(format="%d"),
    })


def get_api_key() -> str:
    try:
        key = st.secrets.get("STEAM_API_KEY", "")
    except Exception:  # ไม่มีไฟล์ secrets
        key = ""
    key = key or os.environ.get("STEAM_API_KEY", "")
    if key:
        st.sidebar.caption("ใช้ Steam API key จาก secrets / ตัวแปรสภาพแวดล้อม")
        return key
    return st.sidebar.text_input("Steam API key", type="password",
                                 help="ขอฟรีที่ steamcommunity.com/dev/apikey ไม่ถูกบันทึกลงไฟล์") or ""


st.markdown(CSS, unsafe_allow_html=True)
model = load_model()

st.markdown('<div class="hero"><h1>GAMEMATCH AI</h1>'
            '<p>หาเกมที่ใช่สำหรับคุณบน Steam · Hybrid = Popularity + Content-based + Collaborative (SVD)</p></div>',
            unsafe_allow_html=True)

st.sidebar.markdown('<p class="logo">GAMEMATCH AI</p><p class="logo-sub">Steam game recommender</p>'
                    '<div class="side-h">MODE</div>', unsafe_allow_html=True)
mode = st.sidebar.radio("โหมด", MODES, label_visibility="collapsed")
st.sidebar.markdown('<div class="side-h">SETTINGS</div>', unsafe_allow_html=True)
k = st.sidebar.slider("จำนวนเกมที่แนะนำ", 5, 20, 10)
as_table = st.sidebar.toggle("แสดงเป็นตาราง", value=False)
st.sidebar.markdown('<div class="side-h">FILTERS</div>', unsafe_allow_html=True)
free_only = st.sidebar.checkbox("เฉพาะเกมฟรี")
max_price = st.sidebar.slider("ราคาสูงสุด ($)", 0, 70, 70, help="70 = ไม่จำกัด", disabled=free_only)
min_pos = st.sidebar.slider("รีวิวบวกอย่างน้อย (%)", 0, 100, 0, step=5)
price_filter = free_only or max_price < 70
filters_active = price_filter or min_pos > 0
st.sidebar.markdown('<div class="side-h">VARIETY</div>', unsafe_allow_html=True)
series_cap = st.sidebar.select_slider("เกมซีรีส์เดียวกันได้ไม่เกิน", options=[1, 2, 3, "ไม่จำกัด"], value=2,
                                      help="กันรายการที่เต็มไปด้วยภาคต่อของเกมเดียวกัน (วัดผลแล้ว: ไม่เกิน 2 ทำให้ Hit Rate ลด 0.0075, "
                                           "ไม่เกิน 1 ลด 0.028 ยิ่งจำกัดมาก ความหลากหลายยิ่งขึ้นแต่ความแม่นยิ่งลด)")
series_on = series_cap != "ไม่จำกัด"
st.sidebar.markdown('<div class="side-h">REVIEW & PRICE</div>', unsafe_allow_html=True)
w_review = st.sidebar.slider("น้ำหนักรีวิวบวก", 0.0, 0.5, 0.0, 0.05,
                             help="ยิ่งมาก ยิ่งดันเกมที่ % รีวิวบวกสูงขึ้นมาก่อน (0 = ไม่ใช้) · เกมที่ไม่มีรีวิวได้ค่ากลาง")
w_price = st.sidebar.slider("น้ำหนักราคาถูก", 0.0, 0.5, 0.0, 0.05,
                            help="ยิ่งมาก ยิ่งดันเกมราคาถูก/ฟรีขึ้นมาก่อน (0 = ไม่ใช้) · ต่างจากตัวกรองราคาที่ตัดเกมทิ้งทั้งเกม")
if w_review > 0 or w_price > 0:
    st.sidebar.caption("ตัวเลขใต้หัวข้อ MODEL เป็นผลของสูตรเดิม (ยังไม่รวมน้ำหนักรีวิว/ราคา)")
score_w = tuple(model.weights[:3]) + (w_review, w_price)
tag_w = tuple(TAG_COLD_START_WEIGHTS[:3]) + (w_review, w_price)
FETCH = 300 if filters_active else (80 if series_on else k)   # ดึงผู้สมัครมากขึ้นก่อน แล้วค่อยกรอง/ตัดซีรีส์ซ้ำ ให้ได้ครบ k เกม


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if not filters_active and not series_on:
        return df
    keep = pd.Series(True, index=df.index)
    price = pd.to_numeric(df["price_final"], errors="coerce")
    if free_only:
        keep &= price.eq(0)
    elif max_price < 70:
        keep &= price.le(max_price)       # เกมที่ไม่มีข้อมูลราคาจะถูกซ่อนเมื่อใช้ตัวกรองราคา
    if min_pos > 0:
        keep &= pd.to_numeric(df["positive_ratio"], errors="coerce").ge(min_pos)
    out = df[keep]
    if series_on:
        out = out[out["game_title"].isin(limit_series(out["game_title"], int(series_cap)))]
    out = out.head(k).copy()
    out["rank"] = range(1, len(out) + 1)
    return out

st.sidebar.markdown('<div class="side-h">MODEL</div>'
                    '<div class="stat"><b>0.3217</b><span>Hit Rate@10 (Hybrid)</span></div>'
                    '<div class="stat"><b>+13.2%</b><span>ดีกว่า Popularity (0.2841)</span></div>'
                    f'<div class="stat"><b>{len(model.game_titles):,}</b><span>เกมที่โมเดลรู้จัก</span></div>',
                    unsafe_allow_html=True)

result = None
style_info = None

if mode == MODES[0]:
    picked = st.multiselect("เกมที่คุณเคยเล่น", model.game_titles)
    history = {}
    if picked:
        with st.expander(f"ระบุชั่วโมงเล่น (ไม่บังคับ ถ้าไม่ระบุจะใช้ {DEFAULT_HOURS:.0f} ชม. ต่อเกม)", expanded=True):
            for t in picked:
                history[t] = st.number_input(t, min_value=0.0, value=DEFAULT_HOURS, step=1.0, key=f"h_{t}")
    result = model.recommend(history or None, k=FETCH, weights=score_w)
    style_info = player_style(model, history)

elif mode == MODES[1]:
    st.markdown('<div class="sec">STEAM LINK</div>', unsafe_allow_html=True)
    st.write("ดึงเกมและชั่วโมงเล่นจากโปรไฟล์ของคุณ แล้วให้โมเดลแนะนำเกมใหม่ "
             "(ต้องตั้ง **Game details** ของโปรไฟล์เป็น Public)")
    key = get_api_key()
    who = st.text_input("SteamID64 / ชื่อ vanity / ลิงก์โปรไฟล์", placeholder="เช่น 76561198000000000")
    if st.button("ดึงเกมจาก Steam", type="primary"):
        if not key or not who:
            st.warning("กรอก API key (แถบข้างซ้าย) และโปรไฟล์ให้ครบ")
        else:
            try:
                with st.spinner("กำลังดึงข้อมูลจาก Steam..."):
                    sid = steam_client.resolve_steam_id(who, key)
                    st.session_state["steam_games"] = steam_client.fetch_owned_games(sid, key)
            except steam_client.SteamError as e:
                st.session_state.pop("steam_games", None)
                st.error(str(e))

    games = st.session_state.get("steam_games")
    if games is not None:
        lookup = appid_to_title(model)
        history, missing = {}, []
        for g in games:
            title = lookup.get(g["appid"])
            if title:
                history[title] = min(g["hours"], HOURS_CAP)
            else:
                missing.append(g)
        st.info(f"พบเกมที่เคยเล่น {len(games)} เกม · ตรงกับเกมที่โมเดลรู้จัก {len(history)} เกม "
                f"(อีก {len(missing)} เกมไม่อยู่ในข้อมูลของโมเดล จึงไม่ถูกใช้)")
        if games:
            with st.expander("ดูรายการเกมที่ดึงมา"):
                st.dataframe(pd.DataFrame(
                    [{"เกม": g["name"], "ชั่วโมงเล่น": g["hours"], "ใช้ในโมเดล": g["appid"] in lookup}
                     for g in games]).sort_values("ชั่วโมงเล่น", ascending=False),
                    hide_index=True, use_container_width=True)
        if history:
            result = model.recommend(history, k=FETCH, weights=score_w)
            style_info = player_style(model, history)
        else:
            st.warning("ไม่มีเกมของคุณที่ตรงกับโมเดลเลย จึงยังแนะนำแบบเฉพาะตัวไม่ได้ "
                       "(โมเดลรู้จักเฉพาะเกมในชุดข้อมูลที่ใช้ฝึก)")

else:
    tags = st.multiselect("แนวเกมที่ชอบ", model.popular_tags(80))
    result = model.recommend_from_tags(tags, k=FETCH, weights=tag_w)
    style_info = player_style(model, tag_choices=tags)
    st.caption("ผลทดสอบ: การเลือกแนวให้ผลดีกว่าสุ่มราว 3.6 เท่า (Hit Rate@10 0.089 เทียบ 0.024) "
               "แต่ยังไม่แม่นกว่าการแนะนำเกมยอดนิยมให้ทุกคน (0.234) "
               "จึงเหมาะใช้ค้นเกมตามแนวที่สนใจ มากกว่าใช้ทำนายความชอบของผู้เล่นใหม่")

if style_info:
    show_style(style_info)
if result is not None:
    result = apply_filters(result)
    if filters_active:
        st.caption(f"ตัวกรองที่ใช้: " + " · ".join(filter(None, [
            "เฉพาะเกมฟรี" if free_only else (f"ราคา ≤ ${max_price}" if max_price < 70 else ""),
            f"รีวิวบวก ≥ {min_pos}%" if min_pos > 0 else ""])) + f" → เหลือ {len(result)} เกม")
if result is not None and len(result) == 0:
    st.warning("ไม่มีเกมที่ผ่านตัวกรองนี้ ลองผ่อนเงื่อนไขลง")
elif result is not None:
    st.markdown('<div class="sec">TOP PICKS FOR YOU</div>', unsafe_allow_html=True)
    (show_table if as_table else show_cards)(result)

with st.expander("เกี่ยวกับโมเดล"):
    st.markdown(
        f"- น้ำหนัก Popularity / Content / Collaborative / Review / Price = {model.weights} "
        f"(ตอนนี้ที่ใช้อยู่: {score_w})\n"
        "- Review = % รีวิวบวกของเกม · Price = ยิ่งถูกยิ่งได้คะแนน (ฟรี = สูงสุด) · ทั้งสองยังไม่ผ่านการวัด Hit Rate ในเว็บนี้ "
        "ดูผลทดลองในโน้ตบุ๊กหัวข้อ 23\n"
        "- Hit Rate@10 (เฉลี่ย 10 seed): Hybrid 0.3217±0.0098 เทียบ Popularity 0.2841±0.0087\n"
        "- NDCG@10: Hybrid 0.1995 เทียบ Popularity 0.1775 · แนะนำเกมหลากหลายกว่า (Coverage 39.3% เทียบ 0.9%)\n"
        "- ผู้เล่นใหม่ (เลือกแนว): ทดสอบ 13 สูตรแล้ว ไม่มีสูตรใดชนะการแนะนำเกมยอดนิยมล้วน (0.234)\n"
        "- ตัวเลือกลดเกมซีรีส์ซ้ำ (ไม่รวมในตัวเลขข้างบน): ไม่เกิน 2 เกมต่อซีรีส์ Hit Rate ลด 0.0075 · ไม่เกิน 1 ลด 0.028\n"
        "- ข้อจำกัด: เกม 24.8% ไม่มี tags (รวมเกมดังบางเกม เช่น Dota 2) และโมเดลรู้จักเกมราว 3,600 เกม "
        "รายการแนะนำของคนเดียวกันอาจมีเกมซีรีส์เดียวกันซ้ำ"
    )
