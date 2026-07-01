from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import os 
from dotenv import load_dotenv
load_dotenv()

# الـ prefix متناسق مع ترتيب الجداول والراوترات التانية في مشروعك
router = APIRouter(
    prefix="/api/rag",
    tags=["Medical RAG Chatbot"]
)


ENDPOINT = os.getenv("ENDPOINT")
API_KEY = os.getenv("API-Cloud")
COLLECTION_NAME = "mediascan_rag"

qdrant_client = QdrantClient(url=ENDPOINT, api_key=API_KEY, timeout=30.0)
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# 2. إعداد الـ LangChain وموديل OpenRouter
llm = ChatOpenAI(
    api_key=os.getenv("API_key"), 
    base_url=os.getenv("base_url"),
    model="openrouter/free",
    temperature=0.2 
)

memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", input_key="Question")

template = """أنت طبيب استشاري خبير ومتحدث لَبِق في نظام MediaScan-AI الطبي. وظيفتك هي الإجابة على أسئلة المرضى الطبية بناءً على السياق (Context) المرفق، وإذا لم يغطِّ السياق الحالة، أجب بناءً على معرفتك الطبية العامة كطبيب محترف.
متجبش اسماء خالص

شروط وإرشادات صارمة للصياغة (قواعد الإنتاج):
1. يجب أن تكون الإجابة باللغة العربية الفصحى الطبية السليمة 100%. يمنع منعاً باتاً دمج مصطلحات إنجليزية داخل الكلمات العربية (مثل كتابة benign أو simple أو ablation وسط الجمل). إذا اضطررت لكتابة مصطلح طبي، اكتبه باللغة العربية واكتب المصطلح الإنجليزي كاملاً بين قوسين، مثل: ورم حميد (Benign tumor).
2. لا تخبر المريض أبداً بأنك "بحثت في السياق ولم تجد" أو "أن معلوماتك في السياق لا تشمل المرض". ادمج معلوماتك العامة وسياقك بسلاسة كاملة لتظهر دائماً بمظهر الطبيب الواثق والمتمكن.
3. حافظ على أسلوب طبي محترف، منظم في نقاط أو جداول، دافئ، ومطمئن للمريض.
4. اذكر دائماً في نهاية الإجابة أن هذه المعلومات لأغراض استرشادية وتثقيفية فقط، ولا تغني عن الفحص السريري ومراجعة طبيب الكلى والمسالك البولية المختص.

تاريخ الحوار السابق (Chat History):
{chat_history}

المراجع الطبية (Context):
{context}

سؤال المريض الحالي (Question): {Question}
الإجابة الطبية الاحترافية:"""

prompt = PromptTemplate(input_variables=["chat_history", "context", "Question"], template=template)
chat_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

class QueryRequest(BaseModel):
    question: str

@router.post("/ask")
async def ask_doctor(request: QueryRequest):
    try:
        user_query = request.question
        print(f" Received question: {user_query}") 
        
        query_vector = embedding_model.encode(user_query).tolist()
        
        search_result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=2
            ).points
        print(f"Qdrant returned {len(search_result)} results.") 
        
        context_list = []
        for hit in search_result:
            if hit.payload and 'text' in hit.payload:
                context_list.append(str(hit.payload['text']))
            elif hit.payload:
                context_list.append(str(list(hit.payload.values())[0]))
        
        context_text = "\n\n".join(context_list)
        
        if not context_text:
            context_text = "لا يوجد سياق متاح"
            print("Warning: Context is empty!")

       
        response = chat_chain.run(
            Question=str(user_query), 
            context=str(context_text)
        )
        
        return {"answer": response}
        
    except Exception as e:
       
        print(f"CRITICAL ERROR IN RAG: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal RAG Error: {str(e)}")