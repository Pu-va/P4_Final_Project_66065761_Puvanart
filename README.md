# 🎮 GameMatch AI

เว็บเดโมแนะนำเกมจาก Steam ด้วยโมเดล Hybrid Recommender สร้างด้วย Streamlit
โปรเจกต์ P4 Final Project (66065761 Puvanart)

**🔗 ลองใช้งานได้ที่:** https://p4finalproject66065761puvanart-jbgft8ztzkymgjwazigdk5.streamlit.app/

---

## ฟีเจอร์

แอปมี 3 โหมดให้เลือกที่แถบด้านข้าง

| โหมด | ใช้เมื่อ |
|------|---------|
| 🎮 เลือกเกมที่เคยเล่น | เลือกเกมที่เคยเล่นเอง ระบุชั่วโมงเล่นหรือไม่ก็ได้ (ถ้าไม่ระบุใช้ค่าเริ่มต้น 10 ชม./เกม) |
| 🔗 เชื่อม Steam | ดึงรายการเกมและชั่วโมงเล่นจาก Steam โดยใส่ SteamID64, ชื่อ vanity หรือลิงก์โปรไฟล์ |
| 🆕 ผู้เล่นใหม่ | ยังไม่มีประวัติ เลือกแนวเกมที่ชอบแล้วระบบแนะนำให้ |

ตัวเลือกปรับแต่งผลลัพธ์:

- จำนวนเกมที่แนะนำ (5–20) และแสดงผลเป็นการ์ดหรือตาราง
- ตัวกรอง: เฉพาะเกมฟรี, ราคาสูงสุด, % รีวิวบวกขั้นต่ำ
- จำกัดจำนวนเกมซีรีส์เดียวกันในรายการ (เพิ่มความหลากหลาย)
- น้ำหนักรีวิวบวก / น้ำหนักราคาถูก (ปรับให้ดันเกมรีวิวดีหรือราคาถูกขึ้นมาก่อน)

## โมเดลทำงานอย่างไร

`GameMatchRecommender` เป็นโมเดลแบบ Hybrid ที่รวมคะแนน 3 ส่วน (บวกรีวิวและราคาแบบเลือกได้)

1. **Popularity**: ความนิยมของเกม
2. **Content-based**: ความคล้ายของ tags เกม (TF-IDF + cosine similarity)
3. **Collaborative filtering**: ผู้เล่นที่มีรสนิยมคล้ายกัน (SVD)

น้ำหนักตั้งต้นของ 3 ส่วนแรกคือ `0.5 / 0.25 / 0.25` (เลือกจากผล validation) ส่วนโหมดผู้เล่นใหม่ใช้ Popularity + Content เท่านั้น ซึ่งเป็น heuristic ที่ยังไม่ได้วัดผลอย่างเป็นทางการ

โมเดลที่ฝึกแล้วเก็บไว้ในไฟล์ `gamematch_model.joblib`

## โครงสร้างไฟล์

```
.
├── app.py                  # หน้าเว็บ Streamlit
├── recommender.py          # คลาสโมเดล GameMatchRecommender
├── steam_client.py         # ดึงข้อมูลผู้ใช้จาก Steam Web API
├── series_utils.py         # ตัวช่วยลดเกมซีรีส์เดียวกันซ้ำ
├── gamematch_model.joblib  # โมเดลที่ฝึกแล้ว
├── requirements.txt        # แพ็กเกจที่ต้องใช้
└── .streamlit/config.toml  # ธีมสีเข้ม
```

## วิธีรันในเครื่อง

```bash
git clone https://github.com/Pu-va/P4_Final_Project_66065761_Puvanart.git
cd P4_Final_Project_66065761_Puvanart
pip install -r requirements.txt
python -m streamlit run app.py
```

ทดสอบกับ Python 3.13 (Streamlit Cloud ใช้ 3.13)

> `scikit-learn` และ `joblib` ควรเป็นเวอร์ชันเดียวกับใน `requirements.txt` เพราะโมเดลถูกเซฟด้วยเวอร์ชันนั้น

## การใช้โหมดเชื่อม Steam

1. ขอ Steam Web API key ฟรีที่ https://steamcommunity.com/dev/apikey
2. กรอก key ในช่องที่แถบด้านข้างของแอป (ไม่ถูกบันทึกลงไฟล์)
   - หรือตั้งค่าเป็น `STEAM_API_KEY` ใน Streamlit Secrets / ตัวแปรสภาพแวดล้อม
3. ตั้งโปรไฟล์ Steam เป็น **Public** โดยเฉพาะส่วน **Game details** ไม่เช่นนั้นจะดึงรายการเกมไม่ได้

⚠️ ห้ามใส่ API key ลงในโค้ดหรือ commit ขึ้น GitHub

## ข้อจำกัด

- การจับซีรีส์เดียวกันใช้กฎประมาณจากชื่อเกม อาจพลาดบางคู่
- ข้อมูลรีวิวไม่มีจำนวนรีวิว จึงแยกไม่ได้ว่า 100% มาจากผู้รีวิว 10 คนหรือ 100,000 คน
- โหมดผู้เล่นใหม่เป็น heuristic ยังไม่ผ่านการวัดผล

## เทคโนโลยีที่ใช้

Python · Streamlit · pandas · NumPy · SciPy · scikit-learn · joblib · Steam Web API
