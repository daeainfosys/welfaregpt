# 중복된 라이브러리 import를 제거하였습니다.
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.retrievers import MultiVectorRetriever
from transformers import AutoTokenizer, pipeline
import torch
import re
import unicodedata
from langchain.schema import Document
from typing import List
import uuid
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_vector import MultiVectorRetriever

# 이모지 및 특수문자 제거 함수
def remove_emojis_and_enclosed_chars(text):
    # 텍스트에서 이모지 및 특수문자를 제거합니다.
    return ''.join(
        ch for ch in text
        if not (
            unicodedata.category(ch).startswith('So')
            or (0x1F000 <= ord(ch) <= 0x1FFFF)
            or (0x2460 <= ord(ch) <= 0x24FF)
        )
    )

# 문서 전처리 함수
def preprocess_document(text):
    # 문서에서 불필요한 마크다운, 태그, 이모지 등을 정리합니다.
    text = remove_emojis_and_enclosed_chars(text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'(\*\*|\*|_)', '', text)
    text = re.sub(r'^[-\*\+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n+', '\n', text).strip()
    return text

# 문서 리스트 전처리 함수
def process_pages(pages: List[Document]) -> List[Document]:
    # 각 문서를 전처리하여 반환합니다.
    return [Document(page_content=preprocess_document(page.page_content), metadata=page.metadata) for page in pages]

# PDF 파일 경로
file_path = "test.pdf"

# PyMuPDFLoader로 문서 로드
loader = PyMuPDFLoader(file_path)
documents = loader.load()

# 문서 리스트 생성
doc_list = [doc for doc in documents]

# 문서 전처리
processed_data = process_pages(doc_list)
# 문서 ID를 생성합니다.

# 임베딩 모델 로드
embedding_model = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sroberta-nli',
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True},
)

# 벡터 저장소 생성
vectorstore = Chroma(
    collection_name="small_bigger_chunks",
    embedding_function=embedding_model,
)
# 부모 문서의 저장소 계층
store = InMemoryStore()

id_key = "doc_id"

# 검색기 (시작 시 비어 있음)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

# PDF 파일 경로
file_path = "test.pdf"

# PyMuPDFLoader로 문서 로드
loader = PyMuPDFLoader(file_path)
docs = loader.load()

# 문서 ID를 생성합니다.
doc_ids = [str(uuid.uuid4()) for _ in docs]

# RecursiveCharacterTextSplitter 객체를 생성합니다.
parent_text_splitter = RecursiveCharacterTextSplitter(chunk_size=600)

# 더 작은 청크를 생성하는 데 사용할 분할기
child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

# parent 문서 리스트를 생성합니다.
parent_docs = []

# 각 문서에 대해 반복합니다.
for i, doc in enumerate(docs):
    # 현재 문서의 ID를 가져옵니다.
    _id = doc_ids[i]
    # 현재 문서를 parent_text_splitter로 분할합니다.
    parent_doc_chunks = parent_text_splitter.split_documents([doc])

    # 각 분할된 청크의 metadata에 문서 ID를 저장합니다.
    for chunk in parent_doc_chunks:
        chunk.metadata[id_key] = _id
    # 분할된 청크들을 parent_docs에 추가합니다.
    parent_docs.extend(parent_doc_chunks)

# parent 문서 리스트를 생성합니다.
child_docs = []

# 각 문서에 대해 반복합니다.
for i, doc in enumerate(docs):
    # 현재 문서의 ID를 가져옵니다.
    _id = doc_ids[i]
    # 현재 문서를 child_text_splitter로 분할합니다.
    child_doc_chunks = child_text_splitter.split_documents([doc])

    # 각 분할된 청크의 metadata에 문서 ID를 저장합니다.
    for chunk in child_doc_chunks:
        chunk.metadata[id_key] = _id
    # 분할된 청크들을 child_docs에 추가합니다.
    child_docs.extend(child_doc_chunks)


# 배치 크기 제한으로 인한 에러 방지를 위해 작은 배치로 나누어 처리
def add_documents_in_batches(vectorstore, documents, batch_size=1000):
    """문서들을 배치 단위로 나누어 벡터 저장소에 추가"""
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        vectorstore.add_documents(batch)
        print(f"배치 {i//batch_size + 1}: {len(batch)}개 문서 추가 완료")

# 벡터 저장소에 parent + child 문서를 배치 단위로 추가
print(f"Parent 문서 수: {len(parent_docs)}")
print(f"Child 문서 수: {len(child_docs)}")
print(f"총 문서 수: {len(parent_docs) + len(child_docs)}")

print("\nParent 문서들을 배치 단위로 추가 중...")
add_documents_in_batches(retriever.vectorstore, parent_docs, batch_size=1000)

print("\nChild 문서들을 배치 단위로 추가 중...")
add_documents_in_batches(retriever.vectorstore, child_docs, batch_size=1000)

# docstore 에 원본 문서를 저장
retriever.docstore.mset(list(zip(doc_ids, docs)))
print("원본 문서 저장 완료")

# LLM 모델 로딩
model_name = "kakaocorp/kanana-1.5-8b-instruct-2505"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

llm_pipeline = pipeline(
    "text-generation",
    model=model_name,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    max_new_tokens=512,
    early_stopping=True,
    temperature=0.3,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.1,
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# 여러 정책들에 대한 답변을 얻는 함수
def ask(
    question,
    age=None,
    gender=None,
    location=None,
    income=None,
    family_size=None,
    marriage=None,
    children=None,
    basic_living=None,
    employment_status=None,
    pregnancy_status=None,
    nationality=None,
    disability=None,
    military_service=None,
    k=5
):
    """
    여러 정책들에 대한 답변을 생성하는 함수
    k: 검색할 문서 수 (기본값 5개)
    """
    try:
        # 텍스트 정제 함수
        def clean_text(text):
            text = re.sub(r'<[^>]+>', '', text)
            return re.sub(r'\s+', ' ', text.strip())

        docs = retriever.invoke(question)

        if not docs:
            return "관련 정보를 찾을 수 없습니다."

        # 검색된 문서 품질 확인
        quality_docs = []
        for doc in docs:
            cleaned_content = clean_text(doc.page_content)
            if len(cleaned_content.strip()) > 100:
                quality_docs.append(doc)

        if not quality_docs:
            return "검색된 문서의 품질이 낮아 답변을 생성할 수 없습니다."

        # 참고자료 및 출처 정리
        context_parts = []
        sources = []

        for i, doc in enumerate(quality_docs[:3]):
            clean_content = clean_text(doc.page_content)
            context_parts.append(f"[참고자료 {i+1}]\n{clean_content[:1024]}")
            page_num = doc.metadata.get('page', doc.metadata.get('source', '알 수 없음'))
            sources.append(f"출처: {page_num}")

        context = "\n\n".join(context_parts)

        # 프롬프트 생성 함수 (welfare_chatbot.py의 _process_query와 동일하게)
        def build_prompt(
            question,
            context,
            age=age,
            gender=gender,
            location=location,
            income=income,
            family_size=family_size,
            marriage=marriage,
            children=children,
            basic_living=basic_living,
            employment_status=employment_status,
            pregnancy_status=pregnancy_status,
            nationality=nationality,
            disability=disability,
            military_service=military_service
        ):
            # 사용자 정보 문자열 생성
            user_info = []
            if age is not None:
                user_info.append(f"나이: {age}")
            if gender is not None:
                user_info.append(f"성별: {gender}")
            if location is not None:
                user_info.append(f"거주지: {location}")
            if income is not None:
                user_info.append(f"소득: 중위 {income}%")
            if family_size is not None:
                user_info.append(f"가구 형태: {family_size}")
            if marriage is not None:
                user_info.append(f"결혼 유무: {marriage}")
            if children is not None:
                user_info.append(f"자녀 수: {children}명")
            if basic_living is not None:
                user_info.append(f"기초생활수급 여부: {basic_living}")
            if employment_status is not None:
                user_info.append(f"취업 여부: {employment_status}")
            if pregnancy_status is not None:
                user_info.append(f"임신/출산 상태: {pregnancy_status}")
            if nationality is not None:
                user_info.append(f"국적: {nationality}")
            if disability is not None:
                user_info.append(f"장애 유무: {disability}")
            if military_service is not None:
                user_info.append(f"군 복무 여부: {military_service}")
            user_info_str = "\n".join(user_info)

            prompt = f"""한국 복지정책 전문가로서 사용자 정보와 중요 지침과 주어진 검색 결과를 바탕으로 질문에 대해 복지 정책들을 정확하고 이해하기 쉽게 답변해주세요.

사용자 정보:
{user_info_str if user_info_str else "정보 없음"}

질문: 
{question}

검색 결과:
{context}

중요 지침:
1. 정확히 3개 정책만 답변
2. 각 정책마다 아래 6개 항목을 모두 작성
3. 마지막에 완전한 문장으로 마무리

필수 형식:
### 정책 [번호]: [정책명]
- 요약: [요약 내용]
- 대상: [대상 내용]
- 지원: [지원 내용]
- 방법: [방법 내용]
- 주의: [주의 내용]
- 문의: [문의 내용]

답변:"""
            return prompt

        prompt = build_prompt(
            question,
            context,
            age=age,
            gender=gender,
            location=location,
            income=income,
            family_size=family_size,
            marriage=marriage,
            children=children,
            basic_living=basic_living,
            employment_status=employment_status,
            pregnancy_status=pregnancy_status,
            nationality=nationality,
            disability=disability,
            military_service=military_service
        )

        response = llm.predict(prompt)

        # 출처 정보 추가
        source_info = f"\n\n참고자료: {', '.join(sources)}"

        print("생성 길이: ", len(response))
        return response + source_info

    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {str(e)}"