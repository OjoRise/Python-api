import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from openai import OpenAI
from datetime import datetime
import json
import os
from dotenv import load_dotenv

load_dotenv()

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    prefer_grpc=False,
    timeout=10.0
)

collection_name = "plan_collection"

if not qdrant.collection_exists(collection_name):
    qdrant.create_collection(
        collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

class Plan(BaseModel):
    planId: int
    name: str
    baseDataGb: str
    dailyDataGb: str
    sharingDataGb: str
    monthlyFee: int
    voiceCallPrice: str
    sms: str
    throttleSpeedKbps: int
    eligibility: str
    mobileType: str
    isOnline: int
    planUrl: str
    telecomProvider: str
    description: str

class UserProfile(BaseModel):
    birthdate: str
    telecomProvider: str
    planName: str
    familyBundle: str
    tongResult: str

@app.post("/vectorize")
def vectorize_plans(plans: List[Plan]):
    points = []
    for plan in plans:
        text = (
            f"{plan.name} 요금제, 기본 데이터 {plan.baseDataGb}GB, "
            f"일일 {plan.dailyDataGb}GB, "
            f"공유 {plan.sharingDataGb}GB, "
            f"월 {plan.monthlyFee}원, 통화 {plan.voiceCallPrice}분, "
            f"SMS {plan.sms}건, "
            f"속도제한 {plan.throttleSpeedKbps}Kbps, "
            f"대상 {plan.eligibility}, 망 {plan.mobileType}, "
            f"데이터 {plan.isOnline}, 설명 {plan.description}"
        )
        points.append(
            PointStruct(id=plan.planId, vector=model.encode(text).tolist(), payload=plan.dict())
        )

    qdrant.upsert(collection_name, points)
    return {"status": "ok", "inserted": len(points)}

@app.post("/search")
async def search_and_recommend(request: Request):
    body = await request.json()
    query = body.get("query")
    user_profile_raw = body.get("userProfile")
    ambiguous_count = body.get("ambiguousCount")
    history = body.get("history")

    eligibilityList = ["ALL"]
    birthday_str = user_profile_raw.get("birthdate")
    if birthday_str:
        birthday = datetime.strptime(birthday_str, "%Y-%m-%d")
        today = datetime.today()
        age = today.year - birthday.year - ((today.month, today.day) < (birthday.month, birthday.day))

        if age <= 12:
            eligibilityList.append("KID")
        elif age <= 18:
            eligibilityList.append("BOY")
        elif age <= 34:
            eligibilityList.append("YOUTH")
        elif age >= 65:
            eligibilityList.append("OLD")

    if not query or not user_profile_raw:
        return JSONResponse(content={"status": False, "message": "query and userProfile are required"})

    query_vec = model.encode(query).tolist()
    hits = qdrant.search(collection_name, query_vec, limit=3)
    similar = [h.payload for h in hits]
    plans_json = json.dumps(similar, ensure_ascii=False, indent=2)

    print(history)

    prompt = f"""
당신은 LG U+ 통신 요금제 추천 전문가입니다.

🧾 아래 사용자 정보와 LG U+ 요금제 리스트를 바탕으로, LG U+로 이동할 경우 가장 적합한 요금제 2~3개를 추천해주세요.
📦 추천 가능한 LG U+ 요금제 리스트: {plans_json}

❗ 반드시 다음 기준을 따르세요
1. 사용자 연령, 데이터 사용 패턴, 가족 결합 여부, 약정 상태, 통신사 이동 사유(요금/데이터/통화 등)를 고려합니다.  
2. 연령 특화 요금제(청소년·청년·시니어 등)가 있다면 최우선으로 고려합니다.
(
나이가 12살 이하(ex. 아이)인 요금제는 "KID",
나이가 18세 이하(ex. 청소년)인 요금제는 "BOY"
나이가 34세 이하(ex. 청년)인 요금제는 "YOUTH"
나이가 65세 이상(ex. 노인, 시니어)인 요금제는 "OLD"
)
3. 다음과 같은 메시지가 들어오거나 요금제를 추천할 수 없다면  **절대 요금제를 추천하지 말고** 아래 지침에 따라 응답합니다. (status가 false라면 반드시)  
   - 단순 인삿말: “ㅎㅇ”, “하이”, “hello”, “hi”, “안녕”, “^^”, “ㅋㅋ”, “ㅇㅇ”, “테스트”  
   - 감탄사·추임새만 포함된 말: “헐”, “하”, “후”, “ㅋㅋㅋ”, “ㅎㅎㅎ”, “몰라요”  
   - 일상 대화·요금제와 무관한 문장: “배고파요”, “날씨 좋다”, “졸려요”, “심심해”, “점심 뭐 먹지”, “피곤하다”, “뭐해”
   - 아무런 의미가 없는 말 : "fh", "gd", 'rimeqwe'

   **응답 형식**  
   • {ambiguous_count} ≥ 3 →  
   {{
     "status": false,
     "item": [],
     "message": "\n\n질문을 잘 알아듣지 못했어요. 고객센터로 연결해드리겠습니다."
   }}  

   • {ambiguous_count} < 3 →  
   {{
     "status": false,
     "item": [],
     "message": "\n\n질문을 잘 알아듣지 못했어요."
   }}

4. 추천이 가능할 때 출력은 반드시 아래 JSON 형식을 따릅니다.
(link는 반드시 {plans_json} 안에 있는 planUrl로 출력하세요.)  
{{
  "status": true,
  "item": [
    {{ "name": "요금제명1", "link": "https://..." }},
    {{ "name": "요금제명2", "link": "https://..." }},
    {{ "name": "요금제명3", "link": "https://..." }}
  ],
  "message": "<아래 형식으로 작성한 설명>"
}}

5. **"message" 필드 형식 (모두 지킬 것, 메세지 시작과 끝에 \n을 무조건 2번 붙여서 출력)**  
\n\n
1. 요금제 이름\n
(월 요금 - 정확한 숫자 / 데이터 제공량 / 음성통화 / SMS / 주요 혜택)\n
- (추천 사유는 간결하고 명확하게 한 줄)\n
\n
2. 요금제 이름\n  
(월 요금 - 정확한 숫자 / 데이터 제공량 / 음성통화 / SMS / 주요 혜택)\n
- (추천 사유는 간결하고 명확하게 한 줄)\n
\n
3. 요금제 이름\n
(월 요금 - 정확한 숫자 / 데이터 제공량 / 음성통화 / SMS / 주요 혜택)\n
- (추천 사유는 간결하고 명확하게 한 줄)\n
\n\n

‣ 괄호 안 정보는 **순서**대로: 월 요금 → 데이터 → 음성통화 → SMS → 혜택  
‣ 버튼 태그 안에는 대응하는 item[].link 값을 삽입합니다.  
‣ 추천 사유는 한 줄로 짧게 작성합니다.

6. 사용자 표현 해석 기준  
| 입력 표현          | 해석 |  
|--------------------|------|  
| 유튜브를 자주 봐요  | 데이터 사용량 많음 |  
| 게임 자주 해요      | 데이터 사용량 많음 |  
| 웹서핑만 해요       | 데이터 사용량 적음 |  
| 영상을 조금만 봐요  | 데이터 사용량 적음 |

7. **사용자의 현재 요금제와 이름이 같은 LG U+ 요금제는 절대 추천하지 않습니다.**

8. 반드시 다음에 작성할 이전 사용자와의 대화를 기반으로 대답해주세요.
{history}

────────────────────────

🧾 사용자 정보  
- 대상: {eligibilityList}  
- 현재 통신사: {user_profile_raw.get('telecomProvider')}  
- 현재 사용 요금제: {user_profile_raw.get('planName')}  
- 가족 결합 여부: {user_profile_raw.get('familyBundle')}  
- 통BTI 성향: {user_profile_raw.get('tongResult')}  
- 사용자의 입력 메시지: {query}
"""

    async def get_response():
        stream = await asyncio.to_thread(lambda: openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            stream=True,
        ))

        buffer = ""
        first_line_sent = False

        try:
            for chunk in stream:
                if chunk.choices[0].delta and chunk.choices[0].delta.content:
                    buffer += chunk.choices[0].delta.content
                    if not first_line_sent:
                        try:
                            parsed = json.loads(buffer)
                            yield json.dumps({"status": bool(parsed.get("item")), "item": parsed.get("item", [])}) + "\n"
                            for ch in parsed.get("message", ""):
                                yield ch
                                await asyncio.sleep(0.005)
                            first_line_sent = True
                            break
                        except Exception:
                            continue
        except Exception as e:
            print("GPT stream error:", e)

        if not first_line_sent:
            yield json.dumps({"status": False, "item": []}) + "\n"
            yield "질문을 잘 알아듣지 못했어요.\n"

    response = StreamingResponse(get_response(), media_type="text/plain")
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
    return response
