import os
import re
import asyncio

from openai import OpenAI
from dotenv import load_dotenv
from databaselib import get_db, post_message
from fastapi import HTTPException

from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics.answer_relevancy import AnswerRelevancyTemplate
from deepeval.metrics import FaithfulnessMetric
from deepeval.metrics.faithfulness import FaithfulnessTemplate
from google import genai
from graphrag import graph_rag_retrieve
from deepeval import evaluate

from rag.normal_rag_search import normal_rag_search
from fastapi.responses import StreamingResponse

from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    # TurnRelevancy, 
    # TurnFaithfulness
)



client = genai.Client(api_key=os.getenv("GOOGLE_AI_STUDIO_KEY"))

judge_model = GeminiModel(
    model="gemini-3-flash-preview",
    api_key=os.getenv("GOOGLE_AI_STUDIO_KEY"),
    temperature=0,
    # generation_kwargs={
    #     'max_output_tokens': 1000
    # }
)

class ShortRelevancyTemplate(AnswerRelevancyTemplate):
    @staticmethod
    def generate_reason(irrelevant_statements, relevant_statements, score):
        return f"""
        Based on the relevancy score of {score}, summarize WHY the answer is relevant or irrelevant.
        CRITICAL: 
        - Your reason must be ONE concise sentence.
        - Do not exceed 15 words.
        - Do not use bullet points.
        """

# 2. Custom Template for Faithfulness
class ShortFaithfulnessTemplate(FaithfulnessTemplate):
    @staticmethod
    def generate_reason(contradictions, score):
        return f"""
        Based on the faithfulness score of {score}, explain why the text is faithful or unfaithful.
        CRITICAL:
        - Your reason must be ONE concise sentence.
        - Do not exceed 15 words.
        - Do not use bullet points.
        """

def get_prompt(joined_text_normal, joined_text_graph, rag_mode, user_input):
    return f"""
                    You are a technical assistant.

                    You MUST answer using ONLY the information in the context.
                    If the answer is not present, say: "I don't know based on the provided context."

                    Context from VECTOR SEARCH (text similarity):
                    {joined_text_normal if rag_mode in ("normal", "both") else "N/A"}

                    Context from GRAPH SEARCH (entity relationships):
                    {joined_text_graph if rag_mode in ("graph", "both") else "N/A"}

                    Rules:
                    - Prefer GRAPH context when both provide relevant information
                    - Do NOT combine unrelated facts
                    - Do NOT use outside knowledge

                    Question:
                    {user_input}
                """

def call_LLM(prompt: str) -> str:

    metric = None
    # print(prompt)
    try:

        actual_response = client.models.generate_content(
            model="gemini-2.5-flash",
            # messages=[{"role": "user", "content": prompt}],
            contents=prompt
        )
        answer = actual_response.text
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

    return answer



async def event_generator(prompt, guardrail_enabled, user_input, joined_text_normal, joined_text_graph,rag_mode):
    full_response = ""
        
    # 1. Stream จาก Gemini
    # ใช้ stream=True สำหรับการตอบโต้อย่างรวดเร็ว
    response = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=prompt
    )

    for chunk in response:
        if chunk.text:
            full_response += chunk.text
            yield chunk.text
            await asyncio.sleep(0.01) # ป้องกัน buffer ค้าง
    try:
        if guardrail_enabled:
            # ส่งสัญญาณให้ UI รู้ว่ากำลังประเมินผล
            yield "\n\n--- [SYSTEM_EVALUATION_START] ---\n"
            full_response += "\n\n--- [SYSTEM_EVALUATION_START] ---\n"
            
            metric1 = AnswerRelevancyMetric(model=judge_model)
            metric2 = FaithfulnessMetric(model=judge_model)
            test_case = LLMTestCase(
                input=user_input,
                retrieval_context=[joined_text_normal[:3000], joined_text_graph[:3000]],
                actual_output=full_response
            )
            
            # await asyncio.gather(
            metric1.measure(test_case)
            full_response += f"\n[Answer Relevancy: {metric1.score:.2f}]"
            yield f"\n[Answer Relevancy: {metric1.score:.2f}]"
            if metric1.score < 0.6:
                full_response += f"\n[Relevancy Reason: {metric1.reason}]"
                yield f"\n[Relevancy Reason: {metric1.reason}]"
            metric2.measure(test_case)
            print("==============================================")
            full_response += f"\n[Faithfulness: {metric2.score:.2f}]"
            yield f"\n[Faithfulness: {metric2.score:.2f}]"
            if metric2.score < 0.6:
                full_response += f"\n[Relevancy Reason: {metric2.reason}]"
                yield f"\n[Faithfulness: {metric2.reason}]"

    except:
        yield f"\n[EVALUATION: ERROR]"
        full_response += f"\n[EVALUATION: ERROR]"

    # 3. บันทึกข้อมูลลง Database
    yield f"\n[RAG_MODE: {rag_mode}]"
    full_response += f"\n[RAG_MODE: {rag_mode}]"
    post_message("user", user_input)
    post_message("ai", full_response)

async def chat_stream(user_input: str, guardrail_enabled: bool, G, node_ids, node_embeddings, rag_mode):
    # เตรียม Context (เหมือนเดิม)
    joined_text_normal = ""
    joined_text_graph = ""
    
    if rag_mode != "none":
        if rag_mode in ("normal", "both"):
            result_normal = normal_rag_search(user_input)
            joined_text_normal = "\n\n".join(f"{r['text']}" for r in result_normal)
        if rag_mode in ("graph", "both"):
            results_graph = graph_rag_retrieve(user_input, G, node_ids, node_embeddings)
            joined_text_graph = results_graph["context"]
        prompt = get_prompt(joined_text_normal, joined_text_graph, rag_mode, user_input)
    else:
        prompt = user_input

    

    return StreamingResponse(
        event_generator(prompt, guardrail_enabled, user_input, joined_text_normal, joined_text_graph , rag_mode), 
        media_type="text/plain"
    )
