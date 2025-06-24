import asyncio
import os
import json
from datetime import datetime
from typing import List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from openai import OpenAI
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

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    prefer_grpc=False,
    timeout=10.0,
)

collection_name = "plan_collection"
required_vector_dim = 1536

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

def generate_plan_text(plan: Plan) -> str:
    return (
        f"{plan.name} 요금제, 기본 데이터 {plan.baseDataGb}GB, "
        f"일일 {plan.dailyDataGb}GB, 공유 {plan.sharingDataGb}GB, "
        f"월 {plan.monthlyFee}원, 통화 {plan.voiceCallPrice}분, "
        f"SMS {plan.sms}건, 속도제한 {plan.throttleSpeedKbps}Kbps, "
        f"대상 {plan.eligibility}, 망 {plan.mobileType}, "
        f"데이터 {plan.isOnline}, 설명 {plan.description}"
    )

@app.post("/vectorize")
async def vectorize_plans(plans: List[Plan]):
    # ✅ 컬렉션이 없거나 벡터 차원이 다르면 재생성
    if qdrant.collection_exists(collection_name):
        info = qdrant.get_collection(collection_name)
        current_dim = info.vectors_config.size
        if current_dim != required_vector_dim:
            qdrant.delete_collection(collection_name)
            qdrant.create_collection(
                collection_name,
                vectors_config=VectorParams(size=required_vector_dim, distance=Distance.COSINE),
            )
    else:
        qdrant.create_collection(
            collection_name,
            vectors_config=VectorParams(size=required_vector_dim, distance=Distance.COSINE),
        )

    # ✅ 벡터 생성 및 업로드
    points = []
    for plan in plans:
        text = generate_plan_text(plan)
        embedding = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding
        points.append(
            PointStruct(id=plan.planId, vector=embedding, payload=plan.dict())
        )

    qdrant.upsert(collection_name, points)
    return {"status": "ok", "inserted": len(points)}

@app.post("/search")
async def search_and_recommend(request: Request):
    body = await request.json()
    query = body.get("query")
    user_profile_raw = body.get("userProfile")
    ambiguous_count = body.get("ambiguousCount", 0)
    history = body.get("history", "")

    if not query or not user_profile_raw:
        return JSONResponse(content={"status": False, "message": "query and userProfile are required"})

    # ✅ 연령별 분류
    birthday_str = user_profile_raw.get("birthdate")
    eligibilityList = ["ALL"]
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

    # ✅ 쿼리 임베딩 후 검색
    query_embedding = openai.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    hits = qdrant.search(collection_name, query_embedding, limit=3)
    similar = [h.payload for h in hits]
    plans_json = json.dumps(similar, ensure_ascii=False, indent=2)

    prompt = f"""[여기에 기존 prompt 문자열 삽입 — 너무 길어서 생략]"""

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

    return StreamingResponse(get_response(), media_type="text/plain")
