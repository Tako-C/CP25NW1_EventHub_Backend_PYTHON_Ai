# check_status.py
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI()

# ใส่ Job ID ของคุณลงไป
JOB_ID = "ftjob-6YKNKOf114MPYHr3l3mGa1F5"

job = client.fine_tuning.jobs.retrieve(JOB_ID)
print(f"Status: {job.status}")
if job.fine_tuned_model:
    print(f"Model ID ที่ต้องนำไปใช้: {job.fine_tuned_model}")
else:
    print("ระบบกำลังดำเนินการจูน... กรุณารอประมาณ 10-20 นาที")