import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# โหลด API Key จากไฟล์ .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def start_finetuning(file_path):
    print(f"--- Step 1: Uploading file '{file_path}' ---")
    
    # 1. อัปโหลดไฟล์ Training Data
    response_file = client.files.create(
        file=open(file_path, "rb"),
        purpose="fine-tune"
    )
    file_id = response_file.id
    print(f"File uploaded successfully. File ID: {file_id}")

    # รอสักครู่เพื่อให้ OpenAI ตรวจสอบโครงสร้างไฟล์ (Processing)
    print("Waiting for file to be processed...")
    time.sleep(15) 

    print(f"--- Step 2: Creating Fine-tuning Job ---")
    
    # 2. เริ่มสร้าง Job สำหรับ Fine-tune
    # เราเลือกใช้ gpt-4o-mini-2024-07-18 เป็นฐาน (Base Model)
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model="gpt-4o-mini-2024-07-18",
        suffix="event-topic-v1" # ตั้งชื่อต่อท้ายโมเดลเพื่อให้จำง่าย
    )
    
    job_id = job.id
    print(f"Fine-tuning job started!")
    print(f"Job ID: {job_id}")
    print(f"Status: {job.status}")
    print("-" * 30)
    print("คุณสามารถนำ Job ID ไปเช็คสถานะได้ที่ https://platform.openai.com/finetune")
    
    return job_id
if __name__ == "__main__":
    # ระบุชื่อไฟล์ .jsonl 100 ข้อที่คุณเตรียมไว้
    file_name = "train_data/train_data_2.jsonl" 
    
    if os.path.exists(file_name):
        start_finetuning(file_name)
    else:
        print(f"Error: ไม่พบไฟล์ {file_name} กรุณาตรวจสอบชื่อไฟล์อีกครั้ง")