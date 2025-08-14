# Mulitmodal RAG 실습

multimodal_RAG.ipynb [다운로드 링크]( https://www.ncloud-forums.com/topic/497/) : Naver Cloud Forum

## API key 발급
https://clovastudio.ncloud.com/info/api-key

CLOVA Studio의 API 키 발급 필요하다.

```
"프로필 > API 키 > 테스트 > 테스트 앱 발급" 
```
위에서 테스트 API 키를 발급 받으면 된다.

해당 api는 다음 링크에서 사용법을 확인할 수 있다.
https://api.ncloud-docs.com/docs/ai-naver-clovastudio-summary

```
"익스플로러 > 문단나누기, 임베딩 > 테스트 앱 생성"으로 이동
적절한 이름의 테스트 앱을 생성합니다. 
이 앱은 추후 chunking과 embedding 과정에서 활용됩니다.
```
하지만 현재 페이지에서는 테스트앱 생성하는 버튼이 있지 않다.

받은 api key는 multimodal_RAG.ipynb에서 다음 영역에서 입력할 수 있다. 해당 셀을 실행하면 ``CLOVA Studio API Key: ``입력창이 하나 뜨는데 여기에 api key를 입력하면 된다.

추후에는 .env 파일을 이용하면 좋을 것 같다.
<img width="600" src="https://github.com/user-attachments/assets/bab042ed-bdde-4fe5-96da-a3649a213ac0"/>

## PDF 문서에서 텍스트와 이미지 추출하기 (LOAD)
PyMuPDF를 통해서 pdf를 읽을 수 있다. ``extract_documents_from_pdf`` 여기에서 글자와 그림을 추출해서 글자는 documents 라는 변수에 LangChain Document로 변환하여 넣는다. 

merged_text 라는 변수에 전체 텍스트를 저장한다. 또한 Langchain Document를 반환해서 docs 변수에 저장해둔다.

## 이미지 to 텍스트 변환 및 요약하기
현재 코드는 HyperClova X 를 사용하고 있다. 해당 서비스를 이용하기 위해서는 이미지 크기 조정이 필요하다. 따라서 ``check_and_resize_image_to_outdir`` 함수를 이용해 사이즈가 괜찮으면 바로 filtered_images 폴더에 넘어가고 아니라면 리사이즈 이후 넘어간다.

**따라서 이미지가 없거나 필요없으면 진행하지 않아도 된다.**

따라서 이미지가 없고 text만 있다면 ``Ncloud Object Storage 에 이미지 저장 및 링크 생성`` 부분도 실행하지 않아도 된다.

## Convert

``Convert`` 부분에서는 LLM을 지정하고, 해당 이미지를 읽고 요약을 하라고 명시한다. 이미지 같은 경우에는 직전에 생성했던 AWS에 올라간 이미지 주소를 사용한다.

만약 이미지를 사용할 것이 없다면 LLM 설정만 하고 바꾸는 작업은 안해도 된다.

## Chunking
``chunking`` 부분에서 지금까지 모아둔 text, text화된 이미지를 가지고 청킹을 진행한다. 현재 코드에서는 규칙 기반 청킹을 진행한다.(문서에서 본 것은 아니고 문단 길이 균일화한다고 해서 규칙 기반이라 생각)

그 뒤는 이미지가 들어있든, text만 있든 상관없이 그대로 진행하면 됨.

# 코드 분할

1. 이미지 없이 글자로만 이루어진 pdf일 경우에 쓰기 위해 코드 진행을 간략하게 만든다.

2. 참조하는 문서가 2개 이상이라면 어떻게 해야될까?
    extract_documents_from_pdf 함수에 보면 return 하는 변수가 documents 인데 이는 LangChain Document이다.
    그래서 최종 documents 즉 docs에 모두 append하면 될 것 같은데 따라서 원래 있던 것에서 extend를 통해서 넣으면 될 것 같다.

    그러면 어떤 pdf에서 어떤 내용을 가져오는지 알 수 있다.

3. LLM 변경
현재는 naver꺼로 이루어져있다. LLM을 원하는대로 바꿀 수 있도록 한다. 

4. Chunking 하는 부분 변경
원하는 것은 청킹 방법에 따라 결과가 달라지는가에 대한 부분이다. 즉 청킹할 때 방법을 달리 할 수 있으면 좋을 것 같다.

이 코드 기반으로 했을 때는 고정된 청킹 api를 사용해야할 것 같다. 다른 부분과 연결된 것이 많아서.

하지만 다른 방법으로 하고 싶을 경우 LLM을 활용한 ai 에이전트 개발 입분 249 페이지를 참고해서 진행하면 될 것 같다.