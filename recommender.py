"""
GameMatch AI - Hybrid Recommender (Popularity + Content-based + Collaborative SVD)

ไฟล์เดียวที่ใช้ได้ทั้งใน Colab และเว็บแอป (app.py) เพื่อให้โค้ดที่ใช้ประเมินผลกับโค้ดที่ใช้งานจริงเป็นตัวเดียวกัน

ใช้งานแบบสั้น ๆ:
    model = GameMatchRecommender.load("gamematch_model.joblib")
    model.recommend({"Dota 2": 120, "Portal 2": 15}, k=10)       # ผู้เล่นที่มีประวัติ (ระบุชั่วโมงหรือไม่ก็ได้)
    model.recommend(["Dota 2", "Portal 2"], k=10)                # ไม่ระบุชั่วโมง -> ใช้ DEFAULT_HOURS
    model.recommend(None, k=10)                                  # ผู้เล่นใหม่ ไม่มีประวัติ -> เกมยอดนิยม
    model.recommend(hist, k=10, weights=(0.5, 0.25, 0.25, 0.1, 0.0))  # เพิ่มน้ำหนักรีวิว (ตัวที่ 4) / ราคา (ตัวที่ 5)
    model.recommend_from_tags(["action", "co-op"], k=10)         # ผู้เล่นใหม่ เลือกแนวที่ชอบ (heuristic ยังไม่ผ่านการวัดผล)
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_HOURS = 10.0                    # ผู้เล่นไม่ระบุชั่วโมงเล่น -> สมมติว่าเล่นเกมละ 10 ชม.
# น้ำหนัก 5 ตัว = (popularity, content, collaborative, review, price)
# 3 ตัวแรกคือสูตร Pop-heavy ที่ validation เลือก ส่วน review/price เริ่มที่ 0 (= ผลเท่าเดิมเป๊ะ) จนกว่าจะวัดผลหัวข้อ 23 แล้วเลือกค่า
DEFAULT_WEIGHTS = (0.5, 0.25, 0.25, 0.0, 0.0)
TAG_COLD_START_WEIGHTS = (0.5, 0.5, 0.0, 0.0, 0.0)  # ผู้เล่นใหม่เลือกแนวเอง: (ความนิยมแบบเปอร์เซ็นไทล์, tags ตรง, ไม่มี collaborative, review, price)
REASONS = ("เป็นเกมยอดนิยม", "tags คล้ายเกมที่เลือก", "ผู้เล่นรสนิยมคล้ายกันชอบ",
           "รีวิวจากผู้เล่นดี", "ราคาเข้าถึงง่าย")
PRICE_CAP = 70.0                        # ราคา >= ค่านี้ถือว่าแพงสุด (ตรงกับสไลเดอร์ราคาสูงสุดในเว็บ)


def pad_weights(w) -> tuple:
    """รับน้ำหนัก 3 ตัว (โมเดลเก่า) หรือ 5 ตัว -> คืน 5 ตัวเสมอ (ที่ขาดเติม 0)"""
    w = tuple(float(x) for x in w)
    if not 3 <= len(w) <= 5:
        raise ValueError("weights ต้องมี 3-5 ค่า: (pop, content, collab[, review, price])")
    return w + (0.0,) * (5 - len(w))


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + 1e-12)


class GameMatchRecommender:
    def __init__(self, game_titles, pop_vec, item_matrix, item_factors, tag_vocab, tag_idf,
                 game_info: pd.DataFrame, weights=DEFAULT_WEIGHTS):
        self.game_titles = list(game_titles)
        self.title_to_idx = {t: i for i, t in enumerate(self.game_titles)}
        self.pop_vec = np.asarray(pop_vec, dtype=float)
        self.pop_pct = rankdata(self.pop_vec) / len(self.pop_vec)  # อันดับความนิยมเป็นเปอร์เซ็นไทล์ 0-1 (ใช้โหมดเลือกแนว)
        self.item_matrix = item_matrix          # sparse n_games x n_tags (TF-IDF)
        self.item_factors = np.asarray(item_factors, dtype=float)  # n_games x k (จาก SVD)
        self.tag_vocab = list(tag_vocab)
        self.tag_idf = np.asarray(tag_idf, dtype=float)
        self.weights = pad_weights(weights)
        info = game_info.drop_duplicates("game_title").set_index("game_title")
        self.game_info = info.reindex(self.game_titles)
        self.quality_vec, self.value_vec = self._meta_scores(self.game_info)

    @staticmethod
    def _meta_scores(info: pd.DataFrame):
        """คะแนน 0-1 จากข้อมูลเกม (ไม่ต้องฝึกใหม่ คำนวณจาก game_info ที่เก็บในโมเดลอยู่แล้ว)
        - quality: % รีวิวบวก (positive_ratio) ปรับเป็น 0-1 · เกมไม่มีรีวิวใช้ค่ามัธยฐาน = ไม่ได้ทั้งโบนัสและโทษ
        - value:   ยิ่งถูกยิ่งได้คะแนนสูง (ฟรี = 1, ราคา >= PRICE_CAP = 0) · เกมไม่มีราคาใช้ค่ามัธยฐาน
        หมายเหตุ: ไม่มีจำนวนรีวิว จึงแยกไม่ได้ว่า 100% จากรีวิว 10 คนหรือ 100,000 คน"""
        pos = pd.to_numeric(info["positive_ratio"], errors="coerce")
        pos = pos.fillna(pos.median() if pos.notna().any() else 50.0).to_numpy(dtype=float)
        quality = _minmax(pos)
        price = pd.to_numeric(info["price_final"], errors="coerce")
        price = price.fillna(price.median() if price.notna().any() else 0.0).clip(0, PRICE_CAP).to_numpy(dtype=float)
        value = 1.0 - price / PRICE_CAP
        return quality, value

    # ------------------------------------------------------------------ train / save / load
    @classmethod
    def fit(cls, plays: pd.DataFrame, game_titles, item_matrix, tag_vocab, tag_idf,
            game_info: pd.DataFrame, n_components: int = 50, weights=DEFAULT_WEIGHTS, seed: int = 42):
        """plays: คอลัมน์ user_id, game_title, hours_played_capped (1 แถว = 1 คู่ user-game)"""
        from scipy.sparse import csr_matrix
        from sklearn.decomposition import TruncatedSVD

        game_titles = list(game_titles)
        t2i = {t: i for i, t in enumerate(game_titles)}
        sub = plays[plays["game_title"].isin(t2i)]

        u_cat = sub["user_id"].astype("category")
        rows = u_cat.cat.codes.to_numpy()
        cols = sub["game_title"].map(t2i).to_numpy()
        hrs = sub["hours_played_capped"].to_numpy(dtype=float)
        M = csr_matrix((np.log1p(hrs), (rows, cols)), shape=(len(u_cat.cat.categories), len(game_titles)))

        k = min(n_components, min(M.shape) - 1)
        svd = TruncatedSVD(n_components=k, random_state=seed).fit(M)

        st = (sub.groupby("game_title")
                 .agg(n_players=("user_id", "nunique"), total_hours=("hours_played_capped", "sum"))
                 .reindex(game_titles).fillna(0))
        pop = 0.6 * st["n_players"] / max(st["n_players"].max(), 1e-12) \
            + 0.4 * st["total_hours"] / max(st["total_hours"].max(), 1e-12)
        pop_vec = _minmax(pop.to_numpy())

        return cls(game_titles, pop_vec, item_matrix, svd.components_.T, tag_vocab, tag_idf, game_info, weights)

    def save(self, path) -> None:
        joblib.dump({
            "game_titles": self.game_titles, "pop_vec": self.pop_vec, "item_matrix": self.item_matrix,
            "item_factors": self.item_factors, "tag_vocab": self.tag_vocab, "tag_idf": self.tag_idf,
            "game_info": self.game_info.reset_index(), "weights": self.weights,
        }, path, compress=3)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        return cls(d["game_titles"], d["pop_vec"], d["item_matrix"], d["item_factors"], d["tag_vocab"],
                   d["tag_idf"], d["game_info"], d["weights"])

    # ------------------------------------------------------------------ scoring
    def _parse_history(self, history):
        if history is None:
            items = {}
        elif isinstance(history, dict):
            items = dict(history)
        else:
            items = {t: None for t in history}
        known, unknown = {}, []
        for title, hours in items.items():
            if title in self.title_to_idx:
                ok = hours is not None and hours == hours and hours > 0   # กัน None / NaN / 0
                known[self.title_to_idx[title]] = float(hours) if ok else DEFAULT_HOURS
            else:
                unknown.append(title)
        return known, unknown

    def _components(self, known: dict):
        n = len(self.game_titles)
        if not known:
            return self.pop_vec, np.zeros(n), np.zeros(n)
        idx = np.array(list(known.keys()))
        hrs = np.array(list(known.values()), dtype=float)

        # content: โปรไฟล์ = ค่าเฉลี่ยเวกเตอร์ tags ถ่วงน้ำหนักด้วยชั่วโมงเล่น -> cosine similarity
        profile = np.asarray(self.item_matrix[idx].T.dot(hrs / hrs.sum())).ravel()
        content = (_minmax(cosine_similarity(profile.reshape(1, -1), self.item_matrix).ravel())
                   if profile.any() else np.zeros(n))

        # collaborative: ฉาย (fold-in) ประวัติของผู้เล่นลงปริภูมิ SVD เหมือนที่ทำตอนฝึก
        x = np.zeros(n)
        x[idx] = np.log1p(hrs)
        collab = _minmax(self.item_factors @ (x @ self.item_factors))
        return self.pop_vec, content, collab

    def _parts(self, known: dict, w) -> np.ndarray:
        """เมทริกซ์ 5 x n_games = น้ำหนัก x คะแนนของแต่ละองค์ประกอบ (ผลรวมแต่ละคอลัมน์ = คะแนนรวมของเกมนั้น)"""
        pop, content, collab = self._components(known)
        return np.vstack([w[0] * pop, w[1] * content, w[2] * collab,
                          w[3] * self.quality_vec, w[4] * self.value_vec])

    def _rank(self, parts: np.ndarray, owned_idx, k: int):
        score = parts.sum(axis=0).astype(float)
        if len(owned_idx):
            score[list(owned_idx)] = -np.inf
        top = np.argsort(-score)[:k]
        return top, score

    def _frame(self, top, score, parts) -> pd.DataFrame:
        rows = []
        for r, i in enumerate(top, start=1):
            title = self.game_titles[i]
            info = self.game_info.loc[title]
            tags = info.get("tags")
            tag_list = [t.replace("_", " ") for t in tags.split()[:6]] if isinstance(tags, str) else []
            tags = ", ".join(tag_list) if tag_list else "(ไม่มีข้อมูลแท็ก)"
            reason = REASONS[int(np.argmax(parts[:, i]))] if parts[:, i].max() > 0 else REASONS[0]
            rows.append({
                "rank": r, "game_title": title, "score": round(float(score[i]), 4), "reason": reason, "tags": tags,
                "price_final": info.get("price_final"), "rating": info.get("rating"),
                "positive_ratio": info.get("positive_ratio"), "app_id": info.get("app_id"),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ public API
    def recommend(self, history=None, k: int = 10, weights=None, exclude_owned: bool = True) -> pd.DataFrame:
        """history: dict {ชื่อเกม: ชั่วโมงเล่น|None} หรือ list ชื่อเกม หรือ None (ผู้เล่นใหม่ -> เกมยอดนิยม)
        ชื่อเกมที่ไม่รู้จักจะถูกข้าม และเก็บไว้ใน result.attrs['unknown']"""
        known, unknown = self._parse_history(history)
        w = pad_weights(weights or self.weights)
        if not known:
            w = (1.0, 0.0, 0.0, w[3], w[4])  # ไม่มีประวัติที่รู้จัก -> Popularity (ยังใช้น้ำหนักรีวิว/ราคาได้)
        parts = self._parts(known, w)
        top, score = self._rank(parts, known.keys() if exclude_owned else [], k)
        out = self._frame(top, score, parts)
        out.attrs["unknown"] = unknown
        return out

    def recommend_titles(self, history=None, k: int = 10, weights=None) -> list[str]:
        """เหมือน recommend แต่คืนแค่ชื่อเกม (เร็วกว่า ใช้ตอนประเมินผลเป็นพัน ๆ คน)"""
        known, _ = self._parse_history(history)
        w = pad_weights(weights or self.weights)
        if not known:
            w = (1.0, 0.0, 0.0, w[3], w[4])
        parts = self._parts(known, w)
        top, _ = self._rank(parts, known.keys(), k)
        return [self.game_titles[i] for i in top]

    def popular_tags(self, n: int = 60) -> list[str]:
        """tags ที่พบในเกมเยอะที่สุด (ไว้ให้ผู้เล่นใหม่เลือกแนวที่ชอบ)"""
        df = np.asarray((self.item_matrix > 0).sum(axis=0)).ravel()
        order = np.argsort(-df)[:n]
        return [self.tag_vocab[i].replace("_", " ") for i in order]

    def recommend_from_tags(self, tags, k: int = 10, weights=None) -> pd.DataFrame:
        """ผู้เล่นใหม่ที่ไม่มีประวัติ: เลือกแนว/tags ที่ชอบแทน (สูตรนี้เป็น heuristic ยังไม่ได้ผ่านการวัด Hit Rate)"""
        vocab = {t: i for i, t in enumerate(self.tag_vocab)}
        profile = np.zeros(len(self.tag_vocab))
        unknown = []
        for t in tags:
            key = str(t).strip().lower().replace(" ", "_")
            if key in vocab:
                profile[vocab[key]] = self.tag_idf[vocab[key]]
            else:
                unknown.append(t)
        n = len(self.game_titles)
        # ใช้ cosine ดิบ (ไม่ min-max) และความนิยมแบบเปอร์เซ็นไทล์ ไม่งั้นเกมไร้ชื่อเสียงที่มี tags ตรงเป๊ะจะชนะทุกครั้ง
        content = (cosine_similarity(profile.reshape(1, -1), self.item_matrix).ravel()
                   if profile.any() else np.zeros(n))
        w = pad_weights(weights or TAG_COLD_START_WEIGHTS)
        if not profile.any():
            w = (1.0, 0.0, 0.0, w[3], w[4])
        parts = np.vstack([w[0] * (self.pop_pct if profile.any() else self.pop_vec), w[1] * content, np.zeros(n),
                           w[3] * self.quality_vec, w[4] * self.value_vec])
        top, score = self._rank(parts, [], k)
        out = self._frame(top, score, parts)
        out["reason"] = ["ตรงกับแนวที่เลือก และเป็นที่นิยม" if content[i] > 0 else REASONS[0] for i in top]
        out.attrs["unknown"] = unknown
        return out
