# RAG 전략 비교 실험
- 목적 : 금융 데이터에 맞는 적절한 RAG 전략을 찾기 위한 비교 실험
<br>
- 실험 이유 : 
    - RAG 기술을 LLM에 도입하기 위해서 다양한 부분을 고려해야 한다. 크게 LLM 모델, 청킹 방법, 벡터 DB, 임베딩 모델 선정에 따라 결과값에 차이가 있을 것이라 예상하였다. 또한 특정 도메인에 따라 적합한 RAG 전략이 다를 것이라 예상하여, 금융 도메인 중 대출에 집중해서 비교해보고자 하였다.
<br>
- 실험 방법 :
    - 청킹 기법, 임베딩 모델, 벡터 데이터베이스, LLM 모델의 요소를 달리하여 조합한 후 비교 분석을 수행한다.
    - 같은 질문에 따른 답변의 질로 각 RAG 모델의 성능을 판단한다.
    - Langchain을 활용하여 선정된 context가 무엇인지 확인 후 참고한다.
    - 팀원 개개인이 답변을 보고 좋은 답변, 나쁜 답변을 표시한다.
    - 각 조합별 성능을 평가한 후, 우수한 성능을 보인 모델의 구성 요소에는 ‘긍정적’ 표시를, 성능이 저조한 모델의 구성 요소에는 ‘부정적’ 표시를 부여하여 실험 결과를 수집한다

<br>

- 사용 데이터셋 (data 폴더에 저장) : 
    - 한국주택금융공사에서 제공하는 pdf (5개)
    - 우리 은행에서 제공하는 대출 안내 사이트 pdf로 저장 (8개)

<br>

- 스택 / 기술 

    <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white" height = 25> <img src="https://img.shields.io/badge/langchain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" height = 25> <img src="https://img.shields.io/badge/huggingface-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white" height = 25> <img src="https://img.shields.io/badge/openai-412991?style=for-the-badge&logo=openai&logoColor=white" height = 25> <img src="https://img.shields.io/badge/streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" height = 25>
    

## 각 구성 요소 선정
### 청킹 기법
-  Recursive
    - 사용 클래스 : RecursiveCharacterTextSplitter
    - 선정 이유 : Financial Report Chunking for Effective Retrieval Augmented Generation (2024) 에서 금융 데이터에 어울리는 청킹 방법으로 구조 기반 청킹을 제안하였다. 따라서 비슷한 개념의 재귀 청킹으로 선정하였다.
    <br>
- Semantic

    - 사용클래스 : SemanticChunker
    - 선정 이유 : 청킹 전략의 Level을 나누면 Sementic 방법은 Level 4로 높은 레벨로 평가된다. 따라서 재귀 청킹보다 높은 레벨의 청킹 기법으로 비교할 수 있도록 설정하였다.
    
### 임베딩 모델
- openai/text-embedding-3-large
    - 이유 : 고차원 임베딩으로 정밀한 의미 표현과 다양한 언어 환경에서 뛰어난 성능 제공
<br>
- stsb-xlm-r-multilinual
    - 이유 : local에 저장해서 사용하여 빠른 실행이 가능
<br>
- intfloat/multilingual-e5-large-instruct
    - 이유 : 다양한 언어 환경에서 뛰어난 성능 제공, 챗봇과 검색엔진 등에서 보편적으로 사용되는 모델
<br>
- kakaobank/kf-deberta-base
    - 이유 : 한국어 금융 도메인 특화로 언어와 도메인에서 특장점
<br>
- BAAI/bge-m3
    - 이유 : Dense, Sparse, Multi-vector 임베딩 방식으로 다양한 검색 가능 및 높은 한국어 성능

### 벡터 DB
- chroma
    - 이유 : Langchain, OpenAI 등과 원활하게 통합되어 사용하기 용이하여 보편적으로 사용하여 선정
<br>
- FAISS
    - 이유 : 고차원 벡터 유사도 검색이 가능한 라이브러리로 보편적으로 사용하기 때문에 벡터 DB 역할로 선정

### LLM 모델
- gpt-4o-mini
    - 이유 : 성능 대비 경량화된 모델로 보편적으로 사용
<br>
- openai/gpt-oss-120b
    - 고성능의 모델로 대용량 문서에 적합

## 결과
- 긍정적 평가와 부정적 평가의 비율을 Bar 그래프로 표현
- 구분 :
    - 청록색 : 긍정적 평가
    - 다홍색 : 부정적 평가

### 청킹 기법
<img src = "by_book\results\img\chunking.png" width = 400px>

왼쪽부터 semantic 기법과 recursive 기법이다. 두 기법의 성능 차이가 많이 나지 않는다는 것을 확인하였다.  

### 임베딩 모델
<img src = "by_book\results\img\embedding_model.png" width = 400px>

왼쪽부터 intfloat/multilingual-e5-large-instruct, BAAI/bge-m3, openai/text-embedding-3-large, stsb-xlm-r-multilinual, kakaobank/kf-deberta-base 모델이다.

가장 높은 긍정적 평가를 받은 모델은 **openai/text-embedding-3-large**이다. 반대로 가장 부정적 평가를 받은 모델은 **kakaobank/kf-deberta-base**이다.

text-embedding-3-large 같은 경우 고성능의 모델로 좋은 성능을 보일 것이라 예상을 했었다. 하지만 한국어 및 금융에 특화된 모델로 선정한 kakaobank/kf-deberta-base이 부정적 결과를 나타낸 것에 의문을 가졌다.

### 벡터 DB
<img src = "by_book\results\img\vector_db.png" width = 400px>

왼쪽부터 chroma, FASIS 이다. 청킹 기법과 마찬가지로 큰 차이를 보이지 않았다.

### LLM 모델
<img src = "by_book\results\img\llm_model.png" width = 400px>

왼쪽부터 openai/gpt-oss-120b, gpt-4o-mini 모델이다. 확실히 gpt-4o-mini는 경량화된 모델이고 gpt-oss-120b는 고성능 모델이라 확연한 차이가 난다고 판단했다. 

### 가장 최고의 답변
- 가장 답변을 잘 했다고 생각이 되는 것은 [다음]("https://github.com/YoonjinOh0909/techseminar_rag_public/blob/main/by_book/results/good_result.md")과 같다.

해당 답변의 조합은 
- chroma
- recursive 
- openai/text-embedding-3-large
- openai/gpt-oss-120b

이다.

나머지 답변들도 [Results 폴더]("ttps://github.com/YoonjinOh0909/techseminar_rag_public/blob/main/by_book/results")에서 확인 가능하다.

### 챗봇 프로토타입
``LLM을 활용한 AI 에이전트 개발 입문``의 코드를 활용하여 ``Streamlit``에 로컬로 RAG를 사용한 챗봇 프로토타입을 실행해보았다.

아래와 같이 작동이 가능하다.

<img src = "by_book\results\img\chatbot_prototype_1.png" width = 45%> <img src = "by_book\results\img\chatbot_prototype_2.png" width = 45%>
<img src = "by_book\results\img\chatbot_prototype_3.png" width = 45%> <img src = "by_book\results\img\chatbot_prototype_4.png" width = 45%>


## 느낀점 및 마무리
### 어려웠던 점
- 다양한 요소들을 조합하고자 했더니, 각 라이브러리가 요구하는 버전들이 서로 달라서 적정 버전을 찾는 것이 어려웠다. 파이썬 버전도 3.12 버전에서 3.10 버전으로 다운그레이드 하며 openai, langchain 등과 버전을 맞추는 작업을 진행하였다. 특히 ai 기술의 발전에 따라 라이브러리 업데이트도 빈번해서 빠르게 최신 버전이 지원이 되지 않는 상황도 발생하였다.
<br>
- 벡터 DB에 저장을 하기 위해 벡터화를 진행할 때 계속해서 제한된 토큰 수를 넘어서 중간에 멈춘 적이 많았다. 과금을 통해 해결하는 방법도 있었지만, split 된 문서들의 길이를 계산해서 제한 길이 이내에서만 임베딩이 진행되도록 하였다.


### 아쉬운 점 및 개선점
- 실험 결과의 신뢰도
    - 공통된 기준 없이 팀원들의 주관적인 평가로 각 모델들을 평가한 것이 아쉽다. 하나의 질문으로 비교 분석을 할 예정이었다면, 예상 답변을 미리 준비해서 그와 비슷한 정도로 답변의 질을 평가할 수 있을 것이라고 생각한다. 
    - 예를 들어 질문이 "우리 부부의 연소득은 총 합 6800만원이야. 한 명은 군에 종사하고 하나는 직장인이라는 것을 참고해줘. 받을 수 있는 대출은 뭐가 있을까?" 라면 '디딤돌 대출', '군인우대 대출', '직장인 대출', '주택담보대출', '보금자리론' 등 입력한 문서에서 찾을 수 있는 서비스를 모두 제시해야 한다는 것을 기대하고, 그것을 기준으로 점수를 매겼어야 한다고 생각한다.

- 청킹 기법 성능의 적은 차이
    - recursive와 semantic 기법에서 차이가 많이 날 것이라고 기대했지만 비등한 결과가 도출 되었다. 이 이유는 데이터 전처리가 되지 않아서 그렇다는 예상이다. 존재하는 pdf를 아무런 전처리 없이 바로 청킹을 하여 저장하였고, 그 결과 ``일부개정 2020. 05. 11``과 같은 필요한 내용과는 먼 정보들도 포함이 되었다. 
    좋은 데이터에서 좋은 결과가 나온다. 있는 그대로 사용하는 것이 아니라 어떻게 전처리할지 고민이 필요할 것 같다.

- 도메인 지식의 부족
    - LLM에 질문을 입력할 때 실제 사용자가 입력할 것 같은 질문을 하였다. 본인의 상황을 설명하고, 필요한 정보를 부탁하였다. 하지만 LLM 설계도 딱 사용자 입장에서만 한 것 같다. 프롬프트를 살펴보면, 먼저 입력된 context에서만 답변을 찾고 모르면 모른다고 답변하라고만 되어있다. 하지만 여기서 더 중요한 것은 고객에게 서비스를 제공하기 위해 필요한 정보가 있다면 그것을 요구하도록 하여야 한다.
    대출 서비스를 추천하고자 할 때 나이, 주택 소유 여부, 신혼부부 등 필요한 정보들이 있을 것이다. 더 현업에 있었다면 이런 부분까지 고민해서 프롬프트를 작성했을 것이라 생각한다.
    - 이와 같이 추가 질문을 하고, 재질문을 모두 기억하고 답변을 하기 위해 ``질의 확장`` 기능을 활용해야 한다.

- 자료 수집에 불필요한 단계
    - 우리은행 사이트에 있는 정보들을 복사해서 pdf에 저장하고, 다시 pdf를 활용해서 데이터를 저장하였다. pdf로 저장하는 단계를 없애고, 인터넷 링크를 활용해서 바로 정보를 수집할 수 있는 방법을 고민해야 한다.

### 참고 자료
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks(Lewis et al., NeurIPS 2020)]("https://arxiv.org/abs/2005.11401")
- [Retrieval-Augmented Generation for Large Language Models: A Survey(Yunfan Gao et al., arXiv, 2023)]("https://arxiv.org/abs/2312.10997")
- [Enhancing RAG with Multimodal Document Understanding]("https://arxiv.org/abs/2506.16035")
- [Is Semantic Chunking Worth the Computational Cost? (Renyi Qu et al., 2024)]("https://arxiv.org/abs/2410.13070")
- [Financial Report Chunking for Effective Retrieval Augmented Generation (2024)]("https://arxiv.org/abs/2402.05131")
- [LangChain으로 이미지가 있는 문서를 검색하는 RAG 시스템 구축하기 (Multimodal RAG Cookbook)]("https://www.ncloud-forums.com/topic/497/")
- [LLM을 활용한 AI 에이전트 개발 입문]("https://github.com/onlybooks/llm")

## 팀원


| <img src="https://avatars.githubusercontent.com/u/190356693?v=4" width="80"/> | <img src="https://avatars.githubusercontent.com/u/65223360?v=4" width="80"/> | <img src="https://avatars.githubusercontent.com/u/127960949?v=4" width="80"/> | <img src="https://avatars.githubusercontent.com/u/132493143?v=4" width="80"/> |
| :---: | :---:| :---: | :---: |
| [오윤진](https://github.com/YoonjinOh0909) | [유문희](https://github.com/muunioi) | [이계무](https://github.com/Gyemoo) | [최수빈](https://github.com/sbyy77dev) |

