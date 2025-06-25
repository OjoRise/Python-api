import os
import asyncio
import json
from datetime import datetime
from typing import List
from dateutil.parser import parse
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, VectorParams, Distance,
    PayloadSchemaType, Filter, FieldCondition, MatchAny
)
from openai import OpenAI

load_dotenv()

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedding_model = "text-embedding-3-small"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "https://yople.vercel.app", "https://backend-ojorise.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    prefer_grpc=False,
    timeout=30.0,
)

collection_name = "plan_collection"
if not qdrant.collection_exists(collection_name):
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
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
    tongName: str

@app.post("/vectorize")
def vectorize_plans(plans: List[Plan]):
    if collection_name in qdrant.get_collections().collections:
        qdrant.delete_collection(collection_name=collection_name)
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    for field in ["eligibility", "mobileType"]:
        qdrant.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD
        )
    points = []
    for plan in plans:
        text = (
            f"{plan.name} 요금제, 기본 데이터 {plan.baseDataGb}GB, 일일 {plan.dailyDataGb}GB, 공유 {plan.sharingDataGb}GB, "
            f"월 {plan.monthlyFee}원, 통화 {plan.voiceCallPrice}분, SMS {plan.sms}건, 속도제한 {plan.throttleSpeedKbps}Kbps, "
            f"대상 {plan.eligibility}, 망 {plan.mobileType}, 데이터 {plan.isOnline}, 설명 {plan.description}"
        )
        embedding = openai.embeddings.create(model=embedding_model, input=text).data[0].embedding
        points.append(PointStruct(id=plan.planId, vector=embedding, payload=plan.dict()))
    qdrant.upsert(collection_name=collection_name, points=points)
    return {"status": "ok", "inserted": len(points)}

@app.post("/search")
async def search_and_recommend(request: Request):
    body = await request.json()
    query = body.get("query")
    user_profile_raw = body.get("userProfile")
    ambiguous_count = body.get("ambiguousCount")
    history = body.get("history")

    formatted_history = "\n".join(f"사용자: {msg}" for msg in history)
    eligibilityList = ["ALL"]

    def get_age(birthdate):
        birthday = parse(birthdate)
        today = datetime.today()
        return today.year - birthday.year - ((today.month, today.day) < (birthday.month, birthday.day))

    if birth := user_profile_raw.get("birthdate"):
        age = get_age(birth)
        if age <= 12:
            eligibilityList.append("KID")
        elif age <= 18:
            eligibilityList.append("BOY")
        elif age <= 34:
            eligibilityList.append("YOUTH")
        elif age >= 65:
            eligibilityList.append("OLD")

    print(formatted_history, query)
    system_prompt = f"""
당신은 통신 요금제 추천을 위한 사용자 프로필 보정 도우미입니다.

다음 4가지 정보를 바탕으로 동작합니다:
① 현재 질문
② 대화 히스토리
③ 초기 userProfile
④ 초기 eligibilityList

🎯 먼저 다음을 판단하세요:
- 입력이 **요금제 추천 요청**인지
- 아니면 **인삿말 / 의미 없는 말 / 설명 요청**인지
- 그 외에는 모두 의미 없는 말로 처리하세요.

────────────────────────

📌 아래 조건 중 하나라도 만족한다면, 반드시 해당 JSON만 출력하세요:

**요금제 추천 요청이 아닌, 요금제 설명만 요청한 경우**
- 예시: "5G 시그니처 요금제 설명해줘", "청년 요금제 어떤 거 있어?", "시니어 요금제가 뭐야?" 등
→ 설명만 출력하고 추천은 하지 마세요.

**인삿말 또는 자기소개가 포함된 경우**
- 예시: "안녕", "하이", "ㅎㅇ", "반가워", "방가", "헬로", "너는 누구니", "소개해줘" 등
→ 아래 응답 고정 출력:
{{
  "status": false,
  "item": [],
  "message": "\\n\\n안녕하세요, 여러분들을 도와줄 AI 챗봇 홀맨입니다."
}}

**의미 없는 단어 / 감탄사 / 테스트 입력일 경우**
- 예시: ㅇㅇ, ㅋㅋ, ㅁㄴㅇㄹ, 테스트, asdf, ??? 등
→ ambiguous_count ≥ 3일 경우:
{{
  "status": false,
  "item": [],
  "message": "\\n\\n질문을 잘 이해하지 못했어요. LG U+ 고객센터에 문의해주시길 바랍니다."
}}

→ ambiguous_count < 3일 경우:
{{
  "status": false,
  "item": [],
  "message": "\\n\\n질문을 잘 이해하지 못했어요."
}}

────────────────────────

📦 요금제 **추천 요청**이라면 다음을 수행하세요:

1. userProfile에는 반드시 다음 항목 포함:
   - birthdate, telecomProvider, planName, familyBundle, tongName

2. birthdate가 있다면 나이를 계산해 eligibilityList 보정:
   - 나이 ≤ 12세: 'KID'
   - 나이 ≤ 18세: 'BOY'
   - 나이 ≤ 34세: 'YOUTH'
   - 나이 ≥ 65세: 'OLD'

3. 질문/대화 히스토리에 나이에 대한 언급이 있다면 해당 eligibility 추가

4. eligibilityList에는 항상 'ALL'을 포함하세요

5. 기존 userProfile이 틀려도 문맥상 확실하다면 수정하세요

🔐 최종 출력 형식은 아래와 같이 고정하세요:
{{
  "userProfile": {{ ... }},
  "eligibilityList": ["ALL", "YOUTH"]
}}
"""

    user_prompt = f"""
### 현재 질문
{query}

### 대화 히스토리
{formatted_history}

### 초기 userProfile
{json.dumps(user_profile_raw, ensure_ascii=False)}

### 초기 eligibilityList
{json.dumps(eligibilityList, ensure_ascii=False)}

### ambiguous_count
{ambiguous_count}
"""

    gpt_response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    content = gpt_response.choices[0].message.content
    print(content)
    try:
        parsed = json.loads(content)
        print(parsed)
        if parsed.get("status") is False:
            async def stream_json_message():
                json_str = json.dumps(parsed, ensure_ascii=False)
                for ch in json_str:
                    yield ch
                    await asyncio.sleep(0.002)
                yield "\n"
                for ch in parsed.get("message", ""):
                    yield ch
                    await asyncio.sleep(0.005)
            return StreamingResponse(stream_json_message(), media_type="text/plain")
        new_user_profile = parsed.get("userProfile", user_profile_raw)
        new_eligibilityList = parsed.get("eligibilityList", eligibilityList)
    except json.JSONDecodeError:
        new_user_profile = user_profile_raw
        new_eligibilityList = eligibilityList

    query_vec = openai.embeddings.create(model=embedding_model, input=query).data[0].embedding
    hits = qdrant.search(
        collection_name=collection_name,
        query_vector=query_vec,
        limit=10,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="eligibility",
                    match=MatchAny(any=new_eligibilityList)
                )
            ]
        )
    )
    plans = [h.payload for h in hits]
    plans_json = json.dumps(plans, ensure_ascii=False)

    prompt = f"""
당신은 LG U+ 통신 요금제 추천 전문가입니다.

🧾 아래 사용자 정보와 LG U+ 요금제 리스트를 바탕으로, LG U+로 이동할 경우 가장 적합한 요금제 최대 3개 추천해주세요.
📦 추천 가능한 LG U+ 요금제 리스트: {plans_json} // 반드시 이 리스트 안에 있는 요금제만 추천하세요.

────────────────────────

❗ 반드시 다음 기준을 따르세요:

1. 사용자 연령, 데이터 사용 패턴, 가족 결합 여부, 약정 상태, 통신사 이동 사유(요금/데이터/통화 등)를 고려하세요.

2. 연령 특화 요금제(청소년·청년·시니어 등)가 있다면 최우선으로 고려하세요.  
**사용자에게 해당되는 요금제 유형은 다음과 같습니다: {new_eligibilityList}**  

※ 아래 문장은 모두 명확한 요금제 추천 요청입니다. 절대 인삿말로 오인하지 말고 반드시 요금제를 추천해야 합니다:
- "시니어 요금제 추천해줘"
- "노인 요금제 추천해줘"
- "아이 요금제 추천해줘" → 여기서 '아이'는 어린이 요금제 의미입니다.
- "초등학생 요금제 뭐가 있어?"
- "어린이 요금제 알려줘"

3. 인삿말(“ㅎㅇ”, “하이”, “hello”, “hi”, “안녕”, “^^”)이나 자기소개 요청 메시지는 **절대 요금제를 추천하지 말고** 아래 형식으로 응답하세요:
{{
  "status": false,
  "item": [],
  "message": "\\n\\n안녕하세요, 여러분들을 도와줄 AI 챗봇 홀맨입니다."
}}

4. 다음과 같은 메시지가 들어오거나 요금제를 추천할 수 없는 경우, 반드시 아래 조건에 따라 정확한 JSON 응답을 출력하세요:

- 의미 없는 말, 잡담, 감탄사, 일상 표현 예시:
  “헐”, “하”, “후”, “ㅋㅋㅋ”, “ㅎㅎㅎ”, “몰라요”, “배고파요”, “심심해”, “피곤하다”, “테스트”, “gd”, “ㅇㅇ” 등

🔒 **ambiguous_counts는 현재 {ambiguous_count}입니다. 반드시 다음과 같은 조건으로 출력합니다:**

- ambiguous_count >= 3인 경우 아래 응답을 **무조건 그대로** 출력하세요:
{{
"status": false,
"item": [],
"message": "\\n\\n고객센터로 연결해드리겠습니다."
}}

- ambiguous_count < 3인 경우 아래 응답을 **무조건 그대로** 출력하세요:
{{
"status": false,
"item": [],
"message": "\\n\\n질문을 잘 알아듣지 못했어요."
}}

4-1. 요금제 추천이 아니지만 요금제와 관련된 질문에는 **반드시 요금제를 추천하지 말고** 해당 질문에 대한 대답을 해주세요.

5. 추천이 가능할 때 출력은 반드시 아래 JSON 형식을 따릅니다:  
(링크는 반드시 {plans_json} 안의 planUrl 값으로)

{{
  "status": true,
  "item": [
    {{ "name": "요금제명1", "link": "https://..." }},
    {{ "name": "요금제명2", "link": "https://..." }},
    {{ "name": "요금제명3", "link": "https://..." }}
  ],
  "message": "<아래 형식으로 작성한 설명>"
}}

6. **반드시 아래 "message" 필드 출력 형식에 맞춰서 출력하세요.**  
※ \n\n 으로 시작하고 \n\n 으로 끝나야 합니다.  
※ 요금제 이름 옆에 "<", ">" 등 아무것도 붙이지 마세요.

\n\n  
정확한 요금제 이름만\n  
\n월 요금:  **정확한 숫자(천 단위 구분 쉼표 포함)**원\n데이터 제공량: \n음성통화: \nSMS: \n주요 혜택: \n\n
- (추천 사유는 간결하고 명확하게 한 줄)\n  
\n  
정확한 요금제 이름만\n  
\n월 요금:  **정확한 숫자(천 단위 구분 쉼표 포함)**원\n데이터 제공량: \n음성통화: \nSMS: \n주요 혜택: \n\n
- (추천 사유는 간결하고 명확하게 한 줄)\n  
\n  
정확한 요금제 이름만\n  
\n월 요금:  **정확한 숫자(천 단위 구분 쉼표 포함)**원\n데이터 제공량: \n음성통화: \nSMS: \n주요 혜택: \n\n
- (추천 사유는 간결하고 명확하게 한 줄)\n
\n추천 총 정리 한 줄\n\n

7. 사용자 표현 해석 기준:
| 표현               | 해석             |
|--------------------|------------------|
| 유튜브를 자주 봐요  | 데이터 사용량 많음 |
| 게임 자주 해요      | 데이터 사용량 많음 |
| 웹서핑만 해요       | 데이터 사용량 적음 |
| 영상을 조금만 봐요  | 데이터 사용량 적음 |

8. 사용자의 현재 요금제와 이름이 **같은 LG U+ 요금제는 절대 추천하지 마세요.**

────────────────────────

🧾 사용자 정보  
- 대상(eligibility): {new_eligibilityList}  
- 현재 통신사: {new_user_profile.get('telecomProvider')}  
- 현재 사용 요금제: {new_user_profile.get('planName')}  
- 가족 결합 여부: {new_user_profile.get('familyBundle')}  
- 통BTI 성향: {new_user_profile.get('tongResult')}  

📌 최종 발화:
"{query}"

→ 이 최종 메시지를 반드시 반영하여 추천 결과를 제시하세요.
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
                        except:
                            continue
        except Exception as e:
            print("GPT stream error:", e)

        if not first_line_sent:
            if ambiguous_count >= 3:
                yield json.dumps({"status": False, "item": []}) + "\n"
                yield "\n\n고객센터로 연결해드리겠습니다."
            else:
                yield json.dumps({"status": False, "item": []}) + "\n"
                yield "\n\n질문을 잘 알아듣지 못했어요."

    return StreamingResponse(get_response(), media_type="text/plain")
