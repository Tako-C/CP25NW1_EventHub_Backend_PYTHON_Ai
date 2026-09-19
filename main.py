
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List
from openai import AsyncOpenAI
from openai import AsyncOpenAI
from dotenv import load_dotenv
import json
import asyncio
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# ตั้งค่า OpenAI API Client
client = AsyncOpenAI(api_key=api_key)

app = FastAPI()

class FeedbackDetail(BaseModel):
    category: str
    event_role: str # "Staff", "Exhibitor", "Visitor"
    count: int = Field(..., gt=-1)
    example_text: str
    sentiment: str # "Positive" หรือ "Negative"

class EventKPI(BaseModel):
    event_name: str
    event_type: str = "งานจัดแสดงสินค้าและนวัตกรรม"
    location: str
    event_detail: str
    total_registered: int = Field(..., gt=0)
    total_checked_in: int
    total_feedback: int
    total_pre_feedback: int
    total_pos_feedback: int
    total_submit_pre_v_feedback: int
    total_submit_pos_v_feedback: int
    total_submit_pre_e_feedback: int
    total_submit_pos_e_feedback: int
    occupations: Dict[str, int]
    role_distribution: Dict[str, int] = {}
    gender_reach: Dict[str, int] = {}
    visitor_score: float = 0.0
    exhibitor_score: float = 0.0
    top_issues: List[FeedbackDetail] # ข้อมูลเชิงลบ
    top_good: List[FeedbackDetail]   # ข้อมูลเชิงบวก (ที่เพิ่มเข้ามา)
    # returning_visitor_rate: float = 0.0
    # returning_exhibitor_rate: float = 0.0

# --- Models ใหม่สำหรับการวิเคราะห์รายข้อ ---
class SuggestionInput(BaseModel):
    rs_id: str
    suggestion: str

class SuggestionAnalysisResponse(BaseModel):
    data: List[Dict[str, str]]


STANDARD_KEYWORDS = [
    # Venue & Facilities
    "สถานที่ (Venue)", "การเดินทาง (Accessibility)", "ที่จอดรถ (Parking)", 
    "ห้องน้ำ (Restroom)", "แอร์/อุณหภูมิ (Temperature)", "ความสะอาด (Cleanliness)", "ป้ายบอกทาง (Signage)",
    
    # Technology & Infrastructure
    "อินเทอร์เน็ต (WiFi)", "ระบบลงทะเบียน (Registration)", "แอปพลิเคชัน (Mobile App)", 
    "ระบบจองคิว (Queue System)", "ระบบเสียง/ภาพ (AV System)",
    
    # Staff & Service
    "เจ้าหน้าที่ (Staff)", "วิทยากร (Speaker)", "พิธีกร (MC)", 
    "การบริการ (Service)", "ความรวดเร็ว (Efficiency)",
    
    # Content & Activities
    "เนื้อหา (Content)", "เวิร์กชอป (Workshop)", "บูธแสดงสินค้า (Exhibitor)", 
    "ของสมนาคุณ (Giveaway)", "ระยะเวลา (Timing)",
    
    # Catering
    "อาหาร (Food)", "เครื่องดื่ม (Beverage)", "ความหลากหลาย (Variety)"
]

# --- Endpoint ใหม่: วิเคราะห์ Keyword & Sentiment ---
@app.post("/analyze-suggestions")
async def analyze_suggestions(inputs: List[SuggestionInput]):
    try:
        # เตรียมข้อมูล Input
        raw_data = [{"rs_id": item.rs_id, "text": item.suggestion} for item in inputs]
        
        # สร้าง String ของ Keyword มาตรฐานเพื่อส่งให้ AI
        keywords_str = ", ".join(STANDARD_KEYWORDS)

        prompt = f"""
        วิเคราะห์ข้อเสนอแนะจากงานอีเวนต์ โดยมีกฎเหล็กดังนี้:
        1. **Keyword**: ต้องเลือกใช้คำจากรายการ "STANDARD_KEYWORDS" ที่กำหนดให้เท่านั้น ห้ามคิดคำใหม่เองเด็ดขาด
        2. **Sentiment**: ระบุว่าเป็น Positive, Negative หรือ Neutral
        3. **Example Text**: ใช้ข้อความต้นฉบับที่ส่งมา

        รายการ STANDARD_KEYWORDS:
        [{keywords_str}, อื่นๆ (Other)]

        ข้อมูลที่ต้องวิเคราะห์:
        {json.dumps(raw_data, ensure_ascii=False)}

        ตอบกลับในรูปแบบ JSON:
        {{
          "data": [
            {{
              "rs_id": "string",
              "keyword": "ต้องตรงกับใน List เท่านั้น",
              "example_text": "string",
              "sentiment": "Positive/Negative/Neutral"
            }}
          ]
        }}
        """

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "คุณคือ AI ผู้เชี่ยวชาญด้านการจัดกลุ่มข้อมูล (Data Classification) ที่ทำงานได้อย่างแม่นยำและตอบเป็น JSON เท่านั้น"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0 # ตั้งเป็น 0 เพื่อให้ AI ทำตามกฎ Keyword อย่างเคร่งครัด
        )

        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# --- CONFIGURATION ---
# 🔴 เปลี่ยนเป็น Model ID ที่คุณจูนเสร็จแล้ว
# FINE_TUNED_TOPIC_MODEL = "ft:gpt-4o-mini-2024-07-18:personal:event-topic-v1:DO3BaCnX" 
# SENTIMENT_MODEL = "gpt-4o-mini"

# class SuggestionInput(BaseModel):
#     rs_id: str
#     suggestion: str

# # --- Helper Functions ---

# async def get_topic_classification(rs_id: str, text: str):
#     """ใช้ Fine-tuned Model เพื่อคัดแยกหมวดหมู่โดยเฉพาะ"""
#     response = await client.chat.completions.create(
#         model=FINE_TUNED_TOPIC_MODEL,
#         messages=[
#             {"role": "system", "content": "Event Topic Classifier. Return the output in JSON format."},
#             {"role": "user", "content": f"rs_id: {rs_id}, content: '{text}'"}
#         ],
#         response_format={"type": "json_object"},
#         temperature=0
#     )
#     return json.loads(response.choices[0].message.content)

# async def get_sentiment_analysis(text: str):
#     """ใช้ Standard Model เพื่อวิเคราะห์ความรู้สึก"""
#     response = await client.chat.completions.create(
#         model=SENTIMENT_MODEL,
#         messages=[
#             {"role": "system", "content": "You are a sentiment analyzer. Answer only in JSON format."},
#             {"role": "user", "content": f"Analyze sentiment (Positive, Negative, Neutral) for this text: '{text}'"}
#         ],
#         response_format={"type": "json_object"},
#         temperature=0
#     )
#     # คาดหวัง JSON: {"sentiment": "Positive"}
#     return json.loads(response.choices[0].message.content)

# @app.post("/analyze-suggestions")
# async def analyze_suggestions(inputs: List[SuggestionInput]):
#     try:
#         final_results = []

#         for item in inputs:
#             # 1. เรียกฟังก์ชัน Topic (Fine-tuned)
#             topic_data = await get_topic_classification(item.rs_id, item.suggestion)
            
#             # 2. เรียกฟังก์ชัน Sentiment (Standard)
#             sentiment_data = await get_sentiment_analysis(item.suggestion)

#             # 3. รวมร่างข้อมูล
#             final_results.append({
#                 "rs_id": item.rs_id,
#                 "keyword": topic_data.get("keyword", "อื่นๆ (Other)"),
#                 "example_text": item.suggestion,
#                 "sentiment": sentiment_data.get("sentiment", "Neutral")
#             })

#         return {"data": final_results}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/analyze-event-performance")
async def analyze_event(kpi: EventKPI):
    try:
        if kpi.total_checked_in > kpi.total_registered:
            raise ValueError("Checked-in count cannot exceed registered count")

        # --- การคำนวณพื้นฐาน ---
        check_in_rate = (kpi.total_checked_in / kpi.total_registered) * 100
        # survey_rate = (kpi.total_feedback / kpi.total_checked_in * 100) if kpi.total_checked_in > 0 else 0
        
        survey_rate = (kpi.total_submit_pos_e_feedback + kpi.total_submit_pre_e_feedback + kpi.total_submit_pos_v_feedback + kpi.total_submit_pre_v_feedback) / kpi.total_feedback * 100 if kpi.total_checked_in > 0 else 0
        overall_avg = (kpi.visitor_score + kpi.exhibitor_score) / 2
        top_occ = max(kpi.occupations, key=kpi.occupations.get) if kpi.occupations else "ไม่ระบุ"
        satisfaction_gap = abs(kpi.visitor_score - kpi.exhibitor_score)
        is_imbalanced = satisfaction_gap > 0.5
        status_label = "สภาวะขาดสมดุลเชิงกลยุทธ์ (Strategic Imbalance)" if is_imbalanced else "สภาวะสมดุลเชิงกลยุทธ์ (Strategic Alignment)"

        # --- เตรียม Data Input สำหรับ AI ---
        data_input = {
            "event_context": {
                "name": kpi.event_name,
                "type": kpi.event_type,
                "location": kpi.location,
                "detail": kpi.event_detail
            },
            "metrics": {
                "registered": kpi.total_registered,
                "checked_in": kpi.total_checked_in,
                "check_in_rate": f"{round(check_in_rate, 2)}%",
                "top_occupation": top_occ
                # "returning_rate": f"{kpi.returning_visitor_rate}%"
            },
            "satisfaction": {
                "visitor": kpi.visitor_score,
                "exhibitor": kpi.exhibitor_score,
                "average": round(overall_avg, 2)
            },
            "stakeholder_analysis": {
                "gap_score": satisfaction_gap,
                "status": "Critical Imbalance" if is_imbalanced else "Balanced",
                "status_label": status_label
            },
            "strengths": [
                {"topic": g.category, "mentions": g.count, "sample": g.example_text} 
                for g in kpi.top_good
            ],
            "weaknesses": [
                {"topic": i.category, "mentions": i.count, "sample": i.example_text} 
                for i in kpi.top_issues
            ]
        }
        # prompt = f"""
        # คำสั่ง: "ในฐานะที่ปรึกษาด้านการบริหารจัดการอีเวนต์ระดับมืออาชีพ และนักตรวจทานเอกสาร (Proofreader) ช่วยวิเคราะห์ข้อมูลสถิติจากระบบ EventHub และจัดทำ 'รายงานสรุปผลการดำเนินงานหลังจบงาน (Post-Event Executive Report)'

        # กฎเหล็กที่ต้องปฏิบัติอย่างเคร่งครัด (Strict Execution Rules):
        # 1. **ความถูกต้องของภาษา**: ใช้ภาษาไทยระดับทางการ ห้ามมีตัวอักษรภาษาอื่น (เช่น จีน, อังกฤษ) ปนมาในประโยคเด็ดขาด และห้ามจบประโยคค้างไว้ ต้องสรุปให้จบกระบวนความ
        # 2. **ห้ามใช้คำทับศัพท์ที่ผิดเพี้ยน**: เปลี่ยนคำทับศัพท์เป็นภาษาไทยที่สละสลวย (เช่น ห้ามใช้ 'ซัมซิเปล', 'แชมเปิล', 'ถนอมนโยบาย' ให้ใช้ 'ตัวอย่างความเห็น', 'กรณีที่พบ', 'รักษามาตรฐานแนวทาง' ตามลำดับ)
        # 3. **ตรรกะข้อมูลและความแม่นยำ**: 
        #    - ตรวจสอบจำนวน 'ผู้ลงทะเบียน (Registered)' และ 'ผู้เช็กอิน (Checked-in)' ให้ถูกต้อง ห้ามใช้สลับกัน [ข้อมูลจริง: {kpi.total_registered} ลงทะเบียน, {kpi.total_checked_in} เช็กอิน]
        #    - วิเคราะห์ความย้อนแย้ง: หากหมวดหมู่ใดมีทั้งคนชมและคนบ่น (เช่น Registration) ให้วิเคราะห์ว่าเป็นปัญหาจากช่วงเวลา Peak Load หรือความไม่สม่ำเสมอของระบบ
        # 4. **วิเคราะห์ตามบริบท (Context-Aware)**: เชื่อมโยงปัญหาที่พบเข้ากับกลุ่มเป้าหมายหลัก (เช่น ปัญหา WiFi กระทบต่อกลุ่มนักศึกษาและนักพัฒนาที่ต้องใช้งานอินเทอร์เน็ตใน Workshop AI โดยตรง)
        # 5. การประยุกต์ใช้ทฤษฎีผู้มีส่วนได้ส่วนเสีย (Stakeholder Theory Implementation): ห้ามรายงานเพียงตัวเลขลอยๆ แต่ต้องวิเคราะห์ว่าความพึงพอใจที่ต่างกันระหว่าง Visitor และ Exhibitor ส่งผลต่อระบบนิเวศ (Ecosystem) ของงานอย่างไร โดยใช้หลักการสร้างสมดุล (Balance of Interests) เพื่อชี้ให้เห็นว่าความล้มเหลวในการตอบสนองความต้องการของกลุ่มหนึ่ง (เช่น Exhibitor) จะส่งผลกระทบลูกโซ่ต่อความยั่งยืนของงานในระยะยาว
        # 6. **ห้ามจบประโยคค้าง**: ตรวจสอบว่าประโยคสุดท้ายของรายงานสรุปจบอย่างสมบูรณ์และได้ใจความ
        # 7. **ตัวเลขคู่ขนาน**: ในบทสรุปผู้บริหาร ต้องระบุทั้งตัวเลขจำนวนคน (ลงทะเบียน/เช็กอิน) ควบคู่ไปกับค่าร้อยละ (%) เสมอ
        # 8. **วิเคราะห์ความขัดแย้งเชิงบวกและลบ**: หากหมวดหมู่ใดมีทั้งคนชมและคนบ่น ให้สรุปว่าเป็นปัญหาเฉพาะช่วงเวลา (เช่น ช่วงคนหนาแน่น) เพื่อความแม่นยำของข้อมูล
        # 9. **การสะกดคำ**: ตรวจสอบว่าไม่มีคำที่สะกดผิดหรือพยัญชนะหล่นหายแม้แต่ตัวเดียว
        # 10. **เจาะลึก Stakeholder Gap**: ต้องวิเคราะห์เปรียบเทียบความพึงพอใจระหว่าง Visitor และ Exhibitor อย่างชัดเจนจากคะแนนที่มีและสรุปว่าความแตกต่างนี้ส่งผลต่อความยั่งยืนของงานอย่างไร
        # 12. **สรุปความสัมพันธ์ 1-ต่อ-1**: ในแผนกลยุทธ์ (Future Action Plan) ต้องระบุวิธีแก้ปัญหาที่ล้อตาม Critical Issues ในข้อ 3 แบบเป็นข้อๆ ให้ครบถ้วน
        # 13. **วิเคราะห์ความคาดหวังกลุ่มเป้าหมาย**: เชื่อมโยงว่าทำไม WiFi ถึงสำคัญต่อนักศึกษาใน Workshop AI เพื่อเพิ่มน้ำหนักให้กับการวิเคราะห์ปัญหา
        # 14. **การวิเคราะห์ Gap**: ปัจจุบันคือ {satisfaction_gap} ดังนั้นสถานะคือ {status_label} (ห้าม AI เปลี่ยนเอง)
        #    - หาก Gap <= 0.5: ให้ชื่นชมความสำเร็จในการรักษาสมดุล และวิเคราะห์ว่าปัจจัยใด (เช่น WiFi หรือการลงทะเบียน) ที่ช่วยสร้างความพึงพอใจร่วมกัน
        #    - หาก Gap > 0.5: ให้ระบุว่าเป็น '{status_label}' และและวิเคราะห์ถึงความเสี่ยงของการเสียผู้สนับสนุน (Exhibitor) ในอนาคต และวิเคราะห์ว่าปัญหาที่พบ ว่าเป็นตัวฉุดคะแนนของกลุ่มใดมากกว่ากัน
        # 15. **การตีความในตาราง**: 
        #    - ช่อง 'สถานะ' ของ Exhibitor: หากคะแนน 4.0 ให้ใส่ ✅ (ตามกฎ 4.0 ขึ้นไป)
        #    - ช่อง 'การตีความข้อมูล' ในแถว Exhibitor: ให้ระบุความสัมพันธ์กับคะแนน Visitor เสมอ เช่น "สอดคล้องกับความพึงพอใจของผู้เข้าชม"
        # 16. **อ้างอิงสถานะ**: ให้ยึดสถานะตามค่า 'status_label' ใน Data Input เป็นหลัก ห้าม AI เปลี่ยนสถานะเอง

        # กฎเหล็กที่ต้องย้ำ (The Final Guardrails):
        # 1. **ห้ามลืมตัวเลขดิบ**: ในข้อ 1 (Executive Summary) ต้องเขียนว่า "มีผู้ลงทะเบียนจำนวน {kpi.total_registered} คน และเข้างานจริง {kpi.total_checked_in} คน คิดเป็น {round(check_in_rate, 2)}%" เสมอ
        # 2. **ความถูกต้องของคำศัพท์**: ตรวจสอบการใช้คำว่า "เช็กอิน" (ใช้ ก ไก่) และ "เสถียรภาพ" ให้ถูกต้องตามหลักภาษาไทยทางการ
        # 3. **ความต่อเนื่องของตาราง**: ในตาราง Dashboard ช่อง 'สถานะ' สำหรับคะแนนความพึงพอใจ: 4.0 ขึ้นไปให้ ✅, 3.5-3.9 ให้ ⚠️, ต่ำกว่า 3.5 ให้ 🚨 และหากเกิด Strategic Imbalance (Gap > 0.5) ให้ติด ⚠️ ในช่อง Interpretation กำกับด้วย
        # 4. **ห้ามมีคำภาษาอังกฤษหลุดรอด**: หากต้องใช้คำทับศัพท์ เช่น WiFi หรือ AI ให้เขียนด้วยตัวพิมพ์ใหญ่ตามมาตรฐานสากล แต่เนื้อหาแวดล้อมต้องเป็นไทย 100%
        # 5. การปิดจบรายงาน: ต้องสรุปปิดท้ายด้วยประโยคที่แสดงถึงความมุ่งมั่นในการพัฒนาโครงการให้ดียิ่งขึ้นในอนาคต และตรวจสอบให้แน่ใจว่าไม่มีอักขระตัวสุดท้ายตัวใดขาดหายไป
       
        # [ข้อมูลสรุปจากระบบ (Data Input)]
        # {json.dumps(data_input, ensure_ascii=False, indent=2)}

        # ---
        # โครงสร้างรายงาน (ใช้ Markdown):

        # # รายงานสรุปผลการดำเนินงาน: {kpi.event_name}

        # ### ตารางสรุปประสิทธิภาพงาน (KPI Performance Dashboard)
        # | ตัวชี้วัด (KPI) | ผลลัพธ์ | สถานะ | การตีความข้อมูล |
        # | :--- | :--- | :--- | :--- |
        # | อัตราการเช็กอิน (Check-in Rate) | {round(check_in_rate, 2)}% | | (เทียบกับเกณฑ์ 70%) |
        # | ความพึงพอใจผู้เข้าชม (Visitor) | {kpi.visitor_score} | | (คะแนนเต็ม 5.0) |
        # | ความพึงพอใจผู้แสดงงาน (Exhibitor) | {kpi.exhibitor_score} | | (คะแนนเต็ม 5.0) |
        # | อัตราการทำแบบสอบถาม (Survey Rate) | {round(survey_rate, 2)}% | | (ความร่วมมือในการให้ข้อมูล) |

        # ## 1. บทสรุปผู้บริหาร (Executive Summary)
        # - วิเคราะห์ความสำเร็จเชิงปริมาณผ่านอัตราการเช็กอิน {round(check_in_rate, 2)}% เทียบกับเกณฑ์มาตรฐานอุตสาหกรรม (70%)
        # - วิเคราะห์นัยสำคัญของกลุ่มเป้าหมายหลักคือ {max(kpi.occupations, key=kpi.occupations.get)} และผลกระทบต่อภาพรวมงาน

        # ## 2. จุดแข็งและปัจจัยความสำเร็จ (Core Strengths)
        # - สรุปสิ่งที่ทำได้ดีเยี่ยม (Top Good) โดยเชื่อมโยงคะแนน Rating กับคำชมใน Feedback (เช่น ระบบลงทะเบียนที่รวดเร็วช่วยสร้างความประทับใจแรกพบ)
        # - ระบุแนวทางการรักษามาตรฐานนี้ไว้สำหรับงานในอนาคต

        # ## 3. ประเด็นที่ต้องปรับปรุงเร่งด่วน (Critical Issues)
        # - วิเคราะห์ช่องว่างความพึงพอใจระหว่าง Visitor และ Exhibitor (ถ้ามี)
        # - สรุปประเด็นเชิงลบที่วิกฤตที่สุด ระบุจำนวนการกล่าวถึง (Mentions) และตัวอย่างปัญหาที่ส่งผลเสียต่อประสบการณ์ผู้ใช้
        # - วิเคราะห์ช่องว่าง (Gap Analysis): คำนวณส่วนต่างระหว่างคะแนน Visitor และ Exhibitor
        # - ผลกระทบเชิงระบบ: อธิบายว่าปัญหาที่พบ กระทบต่อความคาดหวังเฉพาะด้านของแต่ละกลุ่มอย่างไร
        # - การวิเคราะห์ความย้อนแย้ง: เช่น หากระบบลงทะเบียนดี (Visitor ชม) แต่ WiFi แย่ (Exhibitor บ่น) ให้ชี้ว่าเป็นปัญหาของการจัดสรรทรัพยากรที่ให้น้ำหนักกับ "ปริมาณผู้เข้าชม" มากกว่า "คุณภาพการปฏิบัติงาน"

        # ## 4. ข้อเสนอแนะเชิงกลยุทธ์ (Future Action Plan)
        # - เสนอแนวทางแก้ไขปัญหาเชิงเทคนิคและโลจิสติกส์แบบ 1-ต่อ-1 (Actionable Steps) เพื่อปิดช่องโหว่ที่พบในข้อ 3
        # - กลยุทธ์การรักษาฐานผู้เข้าร่วมเดิม (Retention) และการขยายผลจากจุดแข็งเพื่อดึงดูดกลุ่มเป้าหมายใหม่
        # - แนวทางแบบ Win-Win: เสนอทางออกที่ตอบโจทย์ Stakeholders ทุกฝ่ายพร้อมกัน (เช่น การปรับปรุง WiFi ไม่ใช่แค่เพื่อลดคำบ่น แต่เพื่อเพิ่มอัตราการตอบแบบสอบถาม (Survey Rate) และช่วยให้ Exhibitor ปิดการขายได้ดีขึ้น)
        # - กลยุทธ์การรักษาความสัมพันธ์ (Retention Strategy): ระบุแผนการกู้คืนความเชื่อมั่นของกลุ่มที่ได้รับผลกระทบหากเกิดสภาวะขาดสมดุลเชิงกลยุทธ์ เพื่อป้องกันการสูญเสียผู้สนับสนุนหลักในอนาคต

        # สไตล์การเขียน: เฉียบคม, ภาษาทางการสละสลวย, ข้อมูลแม่นยำ 100%, และสรุปจบทุกประเด็น"
        # """

        # --- เตรียมตัวแปรให้พร้อมก่อนใส่ใน Prompt ---
        c_rate = round((kpi.total_checked_in / kpi.total_registered) * 100, 2)
        s_rate = round((kpi.total_submit_pos_e_feedback + kpi.total_submit_pre_e_feedback + kpi.total_submit_pos_v_feedback + kpi.total_submit_pre_v_feedback) / kpi.total_feedback * 100, 2) if kpi.total_feedback > 0 else 0
        gap_value = round(abs(kpi.visitor_score - kpi.exhibitor_score), 2)
        top_occ = max(kpi.occupations, key=kpi.occupations.get) if kpi.occupations else "ไม่ระบุ"

        # กำหนดสถานะเชิงกลยุทธ์ (Strategic Labeling)
        if gap_value <= 0.5:
            status_label = "สภาวะสมดุลเชิงกลยุทธ์ (Strategic Alignment)"
            status_desc = "การบริหารจัดการตอบโจทย์ความคาดหวังของทุกกลุ่มส่วนงานได้อย่างสอดคล้อง"
        else:
            status_label = "สภาวะขาดสมดุลเชิงกลยุทธ์ (Strategic Imbalance)"
            status_desc = "พบความเหลื่อมล้ำในการตอบสนองความต้องการระหว่างกลุ่มเป้าหมาย (Stakeholders)"

        # main_issue = kpi.top_issues[0].category if kpi.top_issues else "การปฏิบัติงานทั่วไป"
        # example_issue_text = kpi.top_issues[0].example_text if kpi.top_issues else "ไม่มีข้อมูลตัวอย่าง"

        main_issue = kpi.top_issues[0].category if kpi.top_issues else "ภาพรวมการดำเนินงาน"
        example_issue_text = kpi.top_issues[0].example_text if kpi.top_issues else "ไม่มีข้อมูลตัวอย่าง"
        
        # สร้าง Prompt
        # ใช้ปีกกาคู่ {{ }} สำหรับส่วนที่เป็น Markdown Table เพื่อป้องกัน f-string error ในบางจุด
        prompt = f"""
        คำสั่ง: "ในฐานะที่ปรึกษาด้านกลยุทธ์การจัดอีเวนต์ระดับสากล (Strategic Event Consultant) จงจัดทำ 'รายงานสรุปผลการดำเนินงานหลังจบงาน (Post-Event Executive Summary)' โดยวิเคราะห์ข้อมูลเชิงลึกจากระบบ EventHub

        กฎเหล็กเพื่อความถูกต้องแม่นยำ 100% (Strict Intelligence Rules):
        1. **Root Cause Analysis**: ห้ามเสนอทางแก้ปัญหาแบบกว้างๆ (Generic) ให้วิเคราะห์จาก Feedback จริงในหมวด 'Top Issues' เท่านั้น หากบ่นเรื่องคิวนาน ให้เสนอการกระจายจุดบริการหรือเทคโนโลยีลดคิว ไม่ใช่แค่บอกว่า 'ปรับปรุงระบบ'
        2. **Context-Driven Insight**: วิเคราะห์ตามประเภทงาน (Event Type: {kpi.event_type}) 
        - งานธุรกิจ: เน้นคุณภาพเครือข่าย (Networking) และความคุ้มค่าเชิงธุรกิจ (ROI)
        - งาน Lifestyle/Fashion: เน้นประสบการณ์ทางกายภาพ (Physical Experience), แสงสีเสียง และความลื่นไหล (Flow)
        3. **Double-Entry Logic**: ต้องระบุ 'จำนวนคน' และ 'ร้อยละ' ควบคู่กันเสมอในทุกการอ้างอิงเชิงปริมาณ
        4. **Professional Thai Lexicon**: ใช้ภาษาไทยระดับทางการ ตรวจสอบคำทับศัพท์มาตรฐาน: 'เช็กอิน' (ก ไก่), 'ฟีดแบ็ก', 'ดิจิทัล', 'อินเทอร์เน็ต', 'ซอฟต์แวร์' ห้ามจบประโยคค้าง และห้ามใช้ภาษาอังกฤษปนยกเว้นชื่อเฉพาะ
        5. **Stakeholder Ecosystem Analysis**: 
        - อ้างอิง Gap {gap_value} และสถานะ {status_label} ห้ามเปลี่ยนเอง
        - ต้องติด ⚠️ ใน Dashboard หากกลุ่มใดกลุ่มหนึ่งมีคะแนนต่ำกว่า 3.8 แม้ค่า Gap จะน้อยก็ตาม (เพื่อระบุจุดเปราะบาง)

        [ข้อมูลดิบจากระบบ (Data Input)]
        {json.dumps(data_input, ensure_ascii=False, indent=2)}

        ---
        โครงสร้างรายงาน (Markdown Standard):

        # รายงานสรุปผลการดำเนินงาน: {kpi.event_name}

        ### 📊 ตารางสรุปประสิทธิภาพงาน (KPI Performance Dashboard)
        | ตัวชี้วัดหลัก (KPI) | ผลลัพธ์ | สถานะ | การตีความเชิงบริหาร |
        | :--- | :--- | :--- | :--- |
        | อัตราการเช็กอิน (Check-in Rate) | {c_rate}% | {"✅" if c_rate >= 70 else "⚠️"} | (เกณฑ์ความสำเร็จขั้นต่ำ 70%) |
        | ความพึงพอใจผู้เข้าชม (Visitor) | {kpi.visitor_score} / 5.0 | {"✅" if kpi.visitor_score >= 3.8 else "⚠️"} | กลุ่มเป้าหมายหลัก: {top_occ} |
        | ความพึงพอใจผู้แสดงงาน (Exhibitor) | {kpi.exhibitor_score} / 5.0 | {"✅" if kpi.exhibitor_score >= 3.8 else "🚨" if kpi.exhibitor_score < 3.5 else "⚠️"} | สถานะคู่ค้า (Partnership Health) |
        | อัตราการตอบกลับ (Survey Rate) | {s_rate}% | {"✅" if s_rate >= 20 else "⚠️"} | นัยสำคัญทางสถิติของข้อมูลฟีดแบ็ก |

        ## 1. บทสรุปผู้บริหาร (Executive Summary)
        - วิเคราะห์ประสิทธิผลเชิงปริมาณ: สรุปยอดผู้ลงทะเบียนจำนวน {kpi.total_registered} คน และเข้างานจริง {kpi.total_checked_in} คนมีอัตราการการเช็กอินเข้างาน ({c_rate}%) ซึ่งสะท้อนถึงความสามารถในการดึงดูดกลุ่มเป้าหมายของกลุ่ม {top_occ} ในบริบทของงาน {kpi.event_type} ลองสรุปส่วนอื่นๆ ที่น่าสนใจจากข้อมูลดิบที่มี เช่น การกระจายตัวของอาชีพอื่นๆ หรือการมีส่วนร่วมของเพศต่างๆ เพื่อเพิ่มมิติในการวิเคราะห์
        - วิเคราะห์นัยสำคัญของกลุ่มเป้าหมายหลัก: เน้นว่ากลุ่ม {top_occ} มีบทบาทอย่างไรในความสำเร็จของงาน และวิเคราะห์ว่าความพึงพอใจของกลุ่มนี้ส่งผลต่อภาพรวมงานอย่างไร

        ## 2. การวิเคราะห์จุดแข็ง (Operational Excellence)
        - สรุปปัจจัยความสำเร็จ (Top Good) โดยระบุจำนวนครั้งที่ถูกกล่าวถึง (Mentions) และยกตัวอย่างคำพูดจริง
        - ระบุแนวทางมาตรฐาน (SOP) ที่ควรส่งต่อ (Scalability) ในการจัดงานครั้งถัดไป

        ## 3. การวิเคราะห์จุดวิกฤตและช่องว่างความพึงพอใจ (Strategic Gap Analysis)
        - **Gap Analysis**: ส่วนต่าง {gap_value} คะแนน อยู่ในสถานะ {status_label} ซึ่งหมายความว่า {status_desc} หากเป็น 'Strategic Imbalance' ให้วิเคราะห์ว่าความแตกต่างนี้เกิดจากปัจจัยใดมากที่สุด และชี้ให้เห็นว่าปัญหานี้อาจนำไปสู่การสูญเสียผู้สนับสนุนหลักในอนาคตได้อย่างไร
        - **Pain Point Depth**: ให้เจาะลึกปัญหาหลัก (เช่น {main_issue}) จากข้อความจริง "{example_issue_text}" และวิเคราะห์ผลกระทบลูกโซ่ (Domino Effect) ที่เกิดขึ้นจากปัญหานี้ต่อประสบการณ์ของผู้เข้าร่วมและความสัมพันธ์กับคู่ค้า
        
        ## 4. ข้อเสนอแนะเชิงกลยุทธ์ (Strategic Recommendations)
        - **Targeted Action Plan**: เสนอแผนแก้ไขปัญหาที่พบในบทที่ 3 แบบเป็นรูปธรรม (Actionable) 
        - **Shared Value Creation**: วิธีการปรับปรุงที่จะทำให้ทั้งสองฝ่าย (Visitor/Exhibitor) ได้ประโยชน์ร่วมกัน (Win-Win Outcome)
        - **Retention & Growth Strategy**: กลยุทธ์การรักษาฐานกลุ่ม {top_occ} และการยกระดับความพึงพอใจของคู่ค้า (Exhibitors) เพื่อความยั่งยืนในระยะยาว

        ปิดท้ายด้วยประโยคที่แสดงถึงความมุ่งมั่นในการยกระดับมาตรฐานการจัดการอีเวนต์สู่ระดับสากล"
        """

        async def stream_generator():
            yield (json.dumps({"check_in_rate": round(check_in_rate, 2), "survey_rate": round(survey_rate, 2)}, ensure_ascii=False) + "\n---\n").encode('utf-8')

            response = await client.chat.completions.create(
                model="gpt-4o", 
                messages=[
                    {"role": "system", "content": "คุณคือที่ปรึกษาด้านอีเวนต์ที่วิเคราะห์สถิติได้อย่างแม่นยำและใช้ภาษาไทยระดับทางการที่ยอดเยี่ยม"},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
                max_tokens=2000, # เผื่อไว้ให้เขียนรายงานจนจบ (ป้องกันประโยคค้าง)
                temperature=0.2  # ปรับให้ต่ำเพื่อให้ AI ทำตามกฎเหล็กได้แม่นยำขึ้น ไม่ฟุ้งซ่าน
            )

            async for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content.encode('utf-8')
                    await asyncio.sleep(0.01)

        return StreamingResponse(stream_generator(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))