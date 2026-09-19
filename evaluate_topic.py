import json
import os
from time import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
from sklearn.metrics import classification_report, accuracy_score, f1_score
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 🔴 Model ID ของคุณ
MODEL_ID = "ft:gpt-4o-mini-2024-07-18:personal:event-topic-v1:DO3BaCnX"

# def create_visual_dashboard(y_true, y_pred, labels):
#     print("\n--- 🎨 กำลังสร้าง Dashboard... ---")
    
#     # คำนวณ Metrics หลัก
#     acc = accuracy_score(y_true, y_pred)
#     f1_macro = f1_score(y_true, y_pred, average='macro')
    
#     # 1. เตรียมข้อมูลรายงานรายหมวด
#     data = []
#     for label in labels:
#         correct = sum((at == pt == label) for at, pt in zip(y_true, y_pred))
#         total = y_true.count(label)
#         cat_acc = (correct / total * 100) if total > 0 else 0
#         data.append({"Category": label, "Accuracy": cat_acc, "Count": total})
    
#     df = pd.DataFrame(data)

#     # 2. กราฟแท่งสรุปผล (Bar Chart)
#     fig_bar = px.bar(
#         df, x='Category', y='Accuracy', color='Accuracy',
#         text_auto='.1f',
#         title="<b>Accuracy แยกตามหมวดหมู่ (%)</b>",
#         color_continuous_scale='RdYlGn', range_y=[0, 105],
#         labels={'Accuracy': 'ความแม่นยำ (%)', 'Category': 'หมวดหมู่'}
#     )

#     # 3. Confusion Matrix
#     matrix_df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
#     matrix_pivot = pd.crosstab(matrix_df['Actual'], matrix_df['Predicted'])
#     fig_matrix = px.imshow(
#         matrix_pivot, text_auto=True,
#         title="<b>Confusion Matrix (วิเคราะห์จุดที่ทายสลับกัน)</b>",
#         labels=dict(x="AI ทายว่าเป็น", y="ความจริงคือ", color="จำนวนข้อ"),
#         color_continuous_scale='Blues'
#     )

#     # 4. สร้างการ์ดแสดงผลตัวเลข (Indicator Cards)
#     fig_metrics = go.Figure()
#     fig_metrics.add_trace(go.Indicator(
#         mode = "gauge+number",
#         value = acc * 100,
#         title = {'text': "Overall Accuracy (%)"},
#         domain = {'x': [0, 0.45], 'y': [0, 1]},
#         gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "darkblue"}}
#     ))
#     fig_metrics.add_trace(go.Indicator(
#         mode = "gauge+number",
#         value = f1_macro * 100,
#         title = {'text': "Macro F1-Score (%)"},
#         domain = {'x': [0.55, 1], 'y': [0, 1]},
#         gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "darkgreen"}}
#     ))
#     fig_metrics.update_layout(height=300)

#     # 5. รวมร่างเป็น HTML
#     timestamp = int(time())
#     dashboard_name = f"ai_topic_dashboard_{timestamp}.html"
#     with open(dashboard_name, 'w', encoding='utf-8') as f:
#         f.write("<html><head><meta charset='utf-8'><title>AI Evaluation Dashboard</title>")
#         f.write("<style>body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f9; }")
#         f.write(".container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }</style></head><body>")
#         f.write("<div class='container'>")
#         f.write("<h1 style='text-align:center;'>📊 AI Model Evaluation Report</h1>")
#         f.write(f"<p style='text-align:center; color:gray;'>Model ID: {MODEL_ID}</p>")
#         f.write(fig_metrics.to_html(full_html=False, include_plotlyjs='cdn'))
#         f.write(fig_bar.to_html(full_html=False, include_plotlyjs='cdn'))
#         f.write("<br><hr><br>")
#         f.write(fig_matrix.to_html(full_html=False, include_plotlyjs='cdn'))
#         f.write("</div></body></html>")
    
#     print(f"✅ Dashboard เสร็จแล้ว! เปิดไฟล์ '{dashboard_name}' ได้เลย")

# เพิ่มรายการ STANDARD_KEYWORDS ไว้ด้านบนสุดหรือในส่วน Configuration
STANDARD_KEYWORDS = [
    "สถานที่ (Venue)", "การเดินทาง (Accessibility)", "ที่จอดรถ (Parking)", 
    "ห้องน้ำ (Restroom)", "แอร์/อุณหภูมิ (Temperature)", "ความสะอาด (Cleanliness)", "ป้ายบอกทาง (Signage)",
    "อินเทอร์เน็ต (WiFi)", "ระบบลงทะเบียน (Registration)", "แอปพลิเคชัน (Mobile App)", 
    "ระบบจองคิว (Queue System)", "ระบบเสียง/ภาพ (AV System)",
    "เจ้าหน้าที่ (Staff)", "วิทยากร (Speaker)", "พิธีกร (MC)", 
    "การบริการ (Service)", "ความรวดเร็ว (Efficiency)",
    "เนื้อหา (Content)", "เวิร์กชอป (Workshop)", "บูธแสดงสินค้า (Exhibitor)", 
    "ของสมนาคุณ (Giveaway)", "ระยะเวลา (Timing)",
    "อาหาร (Food)", "เครื่องดื่ม (Beverage)", "ความหลากหลาย (Variety)"
]

def create_visual_dashboard(y_true, y_pred, labels):
    print("\n--- 🎨 กำลังสร้าง Dashboard (Locked Columns)... ---")
    
    # ใช้ STANDARD_KEYWORDS เป็นลำดับหลักในการแสดงผล
    # กรองเอาเฉพาะ Keyword ที่มีอยู่จริงในผลลัพธ์ (y_true หรือ y_pred) เพื่อไม่ให้กราฟว่างเกินไป
    # หรือจะใช้ STANDARD_KEYWORDS ทั้งหมดเลยก็ได้ถ้าต้องการเห็นภาพรวมทุกหมวด
    display_labels = [k for k in STANDARD_KEYWORDS if k in y_true or k in y_pred]
    
    # หากมีหมวดหมู่นอกเหนือจาก List (เผื่อ AI ตอบอย่างอื่น) ให้เอาไปไว้ท้ายสุด
    extra_labels = sorted(list(set(y_true + y_pred) - set(STANDARD_KEYWORDS)))
    final_labels = display_labels + extra_labels

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # 1. เตรียมข้อมูลรายงานรายหมวด (เรียงลำดับตาม final_labels)
    data = []
    for label in final_labels:
        correct = sum((at == pt == label) for at, pt in zip(y_true, y_pred))
        total = y_true.count(label)
        cat_acc = (correct / total * 100) if total > 0 else 0
        data.append({"Category": label, "Accuracy": cat_acc, "Count": total})
    
    df = pd.DataFrame(data)

    # 2. กราฟแท่ง (ล็อคแกน X ตามลำดับที่กำหนด)
    fig_bar = px.bar(
        df, x='Category', y='Accuracy', color='Accuracy',
        text_auto='.1f',
        title="<b>Accuracy แยกตามหมวดหมู่ (%) - เรียงตามกลุ่มมาตรฐาน</b>",
        color_continuous_scale='RdYlGn', range_y=[0, 105],
        category_orders={"Category": final_labels} # ล็อคลำดับที่นี่
    )

    # 3. Confusion Matrix (ล็อคทั้งแกน X และ Y)
    matrix_df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
    # สร้างตารางโดยระบุหมวดหมู่ทั้งหมด เพื่อให้ลำดับถูกต้อง
    matrix_pivot = pd.crosstab(
        pd.Categorical(matrix_df['Actual'], categories=final_labels),
        pd.Categorical(matrix_df['Predicted'], categories=final_labels),
        dropna=False
    )
    
    fig_matrix = px.imshow(
        matrix_pivot, text_auto=True,
        title="<b>Confusion Matrix (Locked Categories)</b>",
        labels=dict(x="AI ทายว่าเป็น", y="ความจริงคือ", color="จำนวนข้อ"),
        color_continuous_scale='Blues',
        x=final_labels,
        y=final_labels
    )

    # --- ส่วนที่เหลือ (Indicator Cards & HTML) ใช้โค้ดเดิมได้เลย ---
    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Indicator(
        mode = "gauge+number", value = acc * 100,
        title = {'text': "Overall Accuracy (%)"},
        domain = {'x': [0, 0.45], 'y': [0, 1]},
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "darkblue"}}
    ))
    fig_metrics.add_trace(go.Indicator(
        mode = "gauge+number", value = f1_macro * 100,
        title = {'text': "Macro F1-Score (%)"},
        domain = {'x': [0.55, 1], 'y': [0, 1]},
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "darkgreen"}}
    ))
    fig_metrics.update_layout(height=300)

    timestamp = int(time())
    dashboard_name = f"ai_topic_dashboard_{timestamp}.html"
    with open(dashboard_name, 'w', encoding='utf-8') as f:
        f.write("<html><head><meta charset='utf-8'><title>AI Evaluation Dashboard</title>")
        f.write("<style>body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f9; }")
        f.write(".container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }</style></head><body>")
        f.write("<div class='container'>")
        f.write("<h1 style='text-align:center;'>📊 AI Model Evaluation Report</h1>")
        f.write(fig_metrics.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig_bar.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("<br><hr><br>")
        f.write(fig_matrix.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("</div></body></html>")
    
    print(f"✅ Dashboard เสร็จแล้ว! ลำดับหมวดหมู่ถูกล็อคตามมาตรฐานเรียบร้อย")

def run_evaluation(ground_truth_path):
    if not os.path.exists(ground_truth_path):
        print(f"❌ ไม่พบไฟล์: {ground_truth_path}")
        return

    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    y_true = []
    y_pred = []

    print(f"🚀 เริ่มการทดสอบข้อมูลจำนวน {len(test_cases)} รายการ...")

    for case in test_cases:
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": "Event Topic Classifier. Return the output in JSON format."},
                    {"role": "user", "content": f"rs_id: {case['rs_id']}, content: '{case['content']}'"}
                ],
                response_format={"type": "json_object"},
                temperature=0
            )

            ai_output = json.loads(response.choices[0].message.content)
            predicted_topic = ai_output.get("keyword")

            y_true.append(case['expected_keyword'])
            y_pred.append(predicted_topic)
            
            status = "✅" if predicted_topic == case['expected_keyword'] else "❌"
            print(f"{status} rs_id: {case['rs_id']} | AI: {predicted_topic}")

        except Exception as e:
            print(f"❌ Error ที่ rs_id {case['rs_id']}: {e}")

    if not y_true:
        print("\n⚠️ ไม่มีข้อมูลการทดสอบถูกเก็บมาได้เลย!")
        return

    # --- การคำนวณและแสดงผลใน Terminal ---
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')

    print("\n" + "="*60)
    print("📈 สรุปผลประสิทธิภาพโมเดล")
    print("-"*60)
    print(f"Overall Accuracy : {acc:.2%}")
    print(f"Macro F1-Score   : {f1_macro:.2%}")
    print("="*60 + "\n")

    # สร้าง Dashboard
    unique_labels = sorted(list(set(y_true)))
    create_visual_dashboard(y_true, y_pred, unique_labels)

if __name__ == "__main__":
    run_evaluation("train_data/ground_truth_5.json")