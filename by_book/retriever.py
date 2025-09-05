# pip install dotenv
import os
from dotenv import load_dotenv

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 임베딩 모델 선언하기
from langchain_openai import OpenAIEmbeddings
EB_MODEL = "text-embedding-3-large" # text-embedding-3-large
embedding = OpenAIEmbeddings(model=EB_MODEL, api_key=OPENAI_API_KEY)

# 언어 모델 불러오기
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="openai/gpt-oss-120b",  # Hugging Face Router의 모델
    openai_api_key=HF_API_KEY,
    openai_api_base="https://router.huggingface.co/v1"  # base_url 대신 사용
)

# Load Chroma store
from langchain_chroma import Chroma
print("Loading existing Chroma store")
persist_directory = 'C:/ITStudy/Project/TechSeminar_Public/by_book/dbstore/chroma_store_recursive_text-embedding-3-large'

vectorstore = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embedding
)

# Create retriever
retriever = vectorstore.as_retriever(k=3)

# Create document chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser # 문자열 출력 파서를 불러옵니다.

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            # "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. you should user performed_context. You should consider all of special situation and general situation. If the 'perforemd_context' is null, you just say 'it is empyt'. Please write your answer in a markdown table format with the main points. Be sure to include all your source and page numbers like (3 ~ 10) in your answer. If you have over one source, you should include all of them. Answer in Korean. Also please write the keywords on user question that you think. \n#Example Format: \n(brief summary of the answer) \n (table) \n  (detailed answer to the question) \n**출처** \n- (file source) (page source and page number) (Please write the quoted text within 20 characters and follow it with ... )\n\n 키워드 : (keywords)\n #Context: {context}",
            "당신은 사용자를 위한 전문 대출 상담 챗봇입니다. "
            "당신의 역할은 일반 신용대출 및 부동산 관련 대출에 대해 정확하고 도움이 되는 상담을 제공하는 것입니다. "
            "사용자의 질문에 대해 반드시 벡터 데이터베이스에서 검색된 retrieved context(RAG) 기반으로만 답변해야 하며, "
            "retrieved context에 없는 정보를 추론하거나 임의로 만들어내지 마십시오. "
            "다음 규칙을 반드시 따르십시오:\n"
            "1. 답변은 retrieved context에 포함된 정보에 기반하여 작성합니다.\n"
            "2. retrieved context에 관련 정보가 없을 경우, \"해당 내용은 문서에 존재하지 않으므로 전문 상담사에게 연결해 드리겠습니다.\" 라고 답변합니다.\n"
            "3. 답변 작성 시 일반적인 상황뿐만 아니라 발생할 수 있는 특수 상황도 함께 고려하여 설명합니다.\n"
            "4. 답변은 다음 Markdown 표 형식으로 작성합니다:\n"
            "5. 답변할 수 있는 종류가 많다면 최대 5개까지 설명합니다.\n"
            "(간단 요약)\n"
            "(표)\n"
            "(질문에 대한 상세한 설명)\n"
            "**출처**\n"
            "- (문서명) (페이지 범위) “(20자 이하 인용문)…\"\n"
            "5. 마지막 줄에는 사용자 질문에서 핵심이 되는 키워드를 다음 형식으로 작성합니다: "
            "키워드 : (keyword1, keyword2, ...)\n"
            "#Context: {context}"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

document_chain = create_stuff_documents_chain(llm, question_answering_prompt) | StrOutputParser()

# query augmentation chain
query_augmentation_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="messages"), # 기존 대화 내용
        (
            "system",
            "기존의 대화 내용을 활용하여 사용자의 아래 질문의 의도를 파악하여 명료한 한 문장의 질문으로 변환하라. 대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라. :\n\n{query}",
        ),
    ]
)

query_augmentation_chain = query_augmentation_prompt | llm | StrOutputParser()