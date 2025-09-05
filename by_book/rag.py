import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
import retriever
import os
from dotenv import load_dotenv

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 모델 초기화
llm = ChatOpenAI(
    model="openai/gpt-oss-120b",  # Hugging Face Router의 모델
    openai_api_key=HF_API_KEY,
    openai_api_base="https://router.huggingface.co/v1"  # base_url 대신 사용
)

# 사용자의 메시지 처리하기 위한 함수
def get_ai_response(messages, docs):    
    response = retriever.document_chain.stream({
        "messages": messages,
        "context": docs
    })

    for chunk in response:
        yield chunk

# Streamlit 앱
st.title("💬 GPT-4o Langchain Chat")

# 스트림릿 session_state에 메시지 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage("당신은 사용자를 위한 전문 대출 상담 챗봇입니다. "
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
            "#Context: {context}"),  
        AIMessage("How can I help you?")
    ]

# 스트림릿 화면에 메시지 출력
for msg in st.session_state.messages:
    if msg.content:
        if isinstance(msg, SystemMessage):
            st.chat_message("system").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)

# 사용자 입력 처리
if prompt := st.chat_input():
    st.chat_message("user").write(prompt) # 사용자 메시지 출력
    st.session_state.messages.append(HumanMessage(prompt)) # 사용자 메시지 저장

    augmented_query = retriever.query_augmentation_chain.invoke({
        "messages": st.session_state["messages"],
        "query": prompt,
    })
    print("augmented_query\t", augmented_query)

    # 관련 문서 검색
    print("관련 문서 검색")
    docs = retriever.retriever.invoke(f"{prompt}\n{augmented_query}")

    for doc in docs:
        print('---------------')
        print(doc)   
    print("===============")

    with st.spinner(f"AI가 답변을 준비 중입니다... '{augmented_query}'"):
        response = get_ai_response(st.session_state["messages"], docs)
        result = st.chat_message("assistant").write_stream(response) # AI 메시지 출력
    st.session_state["messages"].append(AIMessage(result)) # AI 메시지 저장    