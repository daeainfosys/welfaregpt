import streamlit as st
from streamlit_chatbox import *
import time
import simplejson as json
import re
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch
from langchain.schema import Document
from typing import List
import unicodedata
import os
import streamlit as st
    
# 이모지 및 특수문자 제거 함수
def remove_emojis_and_enclosed_chars(text):
    """텍스트에서 이모지와 특수문자를 제거합니다."""
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
    """문서에서 불필요한 마크다운, 태그, 이모지 등을 정리합니다."""
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
    """각 문서를 전처리하여 반환합니다."""
    return [Document(page_content=preprocess_document(page.page_content), metadata=page.metadata) for page in pages]

# 문서 로드 및 전처리 함수
def load_and_process_documents(file_paths: List[str]):
    """여러 PDF 문서를 로드하고 전처리하여 반환합니다."""
    all_documents = []
    
    for file_path in file_paths:
        try:
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            all_documents.extend(documents)
        except Exception as e:
            st.error(f"파일 로드 중 오류 발생 ({file_path}): {str(e)}")
            continue
    
    if not all_documents:
        return []
    
    processed_data = process_pages(all_documents)
    
    # 문서를 텍스트 청크로 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", "。", ".", "!", "?", ";", ":", " "]
    )
    texts = text_splitter.split_documents(processed_data)
    return texts

# 임베딩 모델 로드 캐시 함수
@st.cache_resource
def load_embedding_model():
    """임베딩 모델을 로드하여 반환합니다."""
    embedding_model = HuggingFaceEmbeddings(
        model_name='jhgan/ko-sroberta-nli',
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True},
    )
    return embedding_model

# 벡터스토어 생성 함수
def create_vectorstore(texts, embedding_model):
    """텍스트 청크와 임베딩 모델을 사용하여 벡터스토어를 생성합니다."""
    if not texts:
        return None
    db = Chroma.from_documents(texts, embedding=embedding_model)
    return db

# LLM 모델 로드 캐시 함수
@st.cache_resource
def load_llm_model():
    """LLM 모델을 로드하여 반환합니다."""
    model_name = "kakaocorp/kanana-1.5-8b-instruct-2505"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm_pipeline = pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        max_new_tokens=512,
        early_stopping=True,
        temperature=0.5,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    return llm

# 공통 문서 처리 함수
def _process_query(question, age=None, gender=None, location=None, income=None, family_size=None, marriage=None, children=None, basic_living=None, employment_status=None, pregnancy_status=None, nationality=None, disability=None, military_service=None):    
    """질문을 처리하고 프롬프트와 출처를 생성합니다."""
    if st.session_state.get("db") is None:
        return None, "PDF 파일을 먼저 업로드해주세요.", []
    
    # clean_text 함수 정의
    def clean_text(text):
        text = re.sub(r'<[^>]+>', '', text)
        return re.sub(r'\s+', ' ', text.strip())

    # 벡터스토어에서 유사한 문서 검색
    retriever = st.session_state.db.as_retriever(search_type="similarity", k=3)
    docs = retriever.get_relevant_documents(question)

    if not docs:
        return None, "관련 정보를 찾을 수 없습니다.", []

    # 문서 품질 필터링
    quality_docs = []
    for doc in docs:
        cleaned_content = clean_text(doc.page_content)
        if len(cleaned_content.strip()) > 100:
            quality_docs.append(doc)

    if not quality_docs:
        return None, "검색된 문서의 품질이 낮아 답변을 생성할 수 없습니다.", []

    # 참고자료와 출처 정리
    context_parts = []
    sources = []
    search_results = []  # 검색 결과를 사용자에게 보여주기 위한 변수
    
    for i, doc in enumerate(quality_docs[:3]):
        clean_content = clean_text(doc.page_content)
        context_parts.append(f"[참고자료 {i+1}]\n{clean_content[:1024]}")
        
        # 페이지 정보 추출
        page_num = doc.metadata.get('page', '알 수 없음')
        source_file = doc.metadata.get('source', '알 수 없음')
        
        # 출처 정보 생성
        if page_num != '알 수 없음':
            source_info = f"📄 페이지 {page_num}"
            if source_file != '알 수 없음':
                source_info += f" ({source_file})"
        else:
            source_info = f"📄 {source_file}"
        
        sources.append(source_info)
        
        # 검색 결과 정보 (사용자에게 보여줄 용도)
        search_results.append({
            'content': clean_content[:500] + "..." if len(clean_content) > 500 else clean_content,
            'page': page_num,
            'source': source_file
        })

    context = "\n\n".join(context_parts)

    # 프롬프트 생성 (1줄씩만 띄우기)
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

답변"""

    return prompt, None, sources, search_results

# 답변에서 프롬프트 제거 함수
def _extract_answer_only(response):
    """LLM 응답에서 답변 부분만 추출합니다."""
    if not response:
        return response
    
    # "답변" 키워드 이후의 내용만 추출
    answer_markers = ["답변:", "답변", "Answer:", "Answer"]
    
    for marker in answer_markers:
        if marker in response:
            parts = response.split(marker, 1)
            if len(parts) > 1:
                return parts[1].strip()
    
    # 답변 마커가 없으면 ### 정책으로 시작하는 부분부터 추출
    lines = response.split('\n')
    answer_lines = []
    start_found = False
    
    for line in lines:
        if line.strip().startswith('### 정책'):
            start_found = True
        if start_found:
            answer_lines.append(line)
    
    if answer_lines:
        return '\n'.join(answer_lines)
    
    return response

# 일반 모드 답변 생성 함수
def generate_answer(question, age=None, gender=None, location=None, income=None, family_size=None, marriage=None, children=None, basic_living=None, employment_status=None, pregnancy_status=None, nationality=None, disability=None, military_service=None):
    """질문에 대한 답변을 생성합니다 (일반 모드)."""
    try:
        prompt, error_msg, sources, search_results = _process_query(question, age, gender, location, income, family_size, marriage, children, basic_living, employment_status, pregnancy_status, nationality, disability, military_service)
        
        if error_msg:
            return error_msg, [], []
        
        # LLM 응답 생성
        try:
            response = st.session_state.llm.predict(prompt)
            if response is None:
                return "모델에서 응답을 생성하지 못했습니다.", [], []
            
            # 답변에서 프롬프트 제거
            clean_response = _extract_answer_only(response)
            # clean_response = remove_emojis_and_enclosed_chars(clean_response)
            return clean_response, sources, search_results
            
        except Exception as e:
            return f"LLM 모델 응답 생성 중 오류가 발생했습니다: {str(e)}", [], []

    except Exception as e:
        return f"답변 생성 중 예상치 못한 오류가 발생했습니다: {str(e)}", [], []

# 스트리밍 모드 답변 생성 함수
def generate_answer_streaming(question, age=None, gender=None, location=None, income=None, family_size=None, marriage=None, children=None, basic_living=None, employment_status=None, pregnancy_status=None, nationality=None, disability=None, military_service=None):
    """질문에 대한 답변을 생성합니다 (스트리밍 모드)."""
    try:
        prompt, error_msg, sources, search_results = _process_query(question, age, gender, location, income, family_size, marriage, children, basic_living, employment_status, pregnancy_status, nationality, disability, military_service)
        
        if error_msg:
            yield error_msg, [], []
            return
        
        # LLM 응답 생성
        try:
            response = st.session_state.llm.predict(prompt)
            if response is None:
                yield "모델에서 응답을 생성하지 못했습니다.", [], []
                return
            
            # 답변에서 프롬프트 제거
            clean_response = _extract_answer_only(response)
            
            clean_response = remove_emojis_and_enclosed_chars(clean_response)
            # 스트리밍 시뮬레이션
            words = clean_response.split()
            for i in range(0, len(words), 5):  # 5단어씩 출력
                yield " ".join(words[:i+5]), sources, search_results
                time.sleep(0.1)
                
        except Exception as e:
            yield f"LLM 모델 응답 생성 중 오류가 발생했습니다: {str(e)}", [], []

    except Exception as e:
        yield f"답변 생성 중 예상치 못한 오류가 발생했습니다: {str(e)}", [], []

# 피드백 처리 함수
def on_feedback(feedback, chat_history_id: str = "", history_index: int = -1):
    """피드백을 처리합니다."""
    reason = feedback.get("text", "")
    score = feedback.get("score", 0)
    
    # 피드백 저장 로직 (실제 구현에서는 데이터베이스에 저장)
    if "feedback_history" not in st.session_state:
        st.session_state.feedback_history = []
    
    st.session_state.feedback_history.append({
        "chat_history_id": chat_history_id,
        "history_index": history_index,
        "score": score,
        "reason": reason,
        "timestamp": time.time()
    })
    
    st.session_state["need_rerun"] = True

# 채팅 세션 변경 함수
def on_chat_change():
    """채팅 세션이 변경될 때 호출됩니다."""
    st.session_state.chat_box.use_chat_name(st.session_state["chat_name"])
    st.session_state.chat_box.context_to_session()

# 기본 문서 자동 로드 함수
def load_default_documents():
    """페이지 시작 시 기본 복지 문서를 자동으로 로드합니다."""
    if st.session_state.get("default_documents_loaded", False):
        return
    
    pdf_dir = "./pdf/welfare"
    
    if os.path.exists(pdf_dir):
        pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
        
        if pdf_files:
            try:
                with st.spinner("기본 복지 문서를 로드하고 있습니다..."):
                    # 문서 로드 및 처리
                    texts = load_and_process_documents(pdf_files)
                    
                    if texts:
                        # 임베딩 모델 로드
                        if "embedding_model" not in st.session_state:
                            st.session_state.embedding_model = load_embedding_model()
                        
                        # 벡터스토어 생성
                        db = create_vectorstore(texts, st.session_state.embedding_model)
                        st.session_state.db = db
                        
                        # LLM 모델 로드
                        if "llm" not in st.session_state:
                            st.session_state.llm = load_llm_model()
                        
                        st.session_state.default_documents_loaded = True
                        st.session_state.documents_loaded = True
                        st.success(f"✅ 기본 복지 문서 {len(texts)}개 청크가 로드되었습니다!")
                        
            except Exception as e:
                st.error(f"기본 문서 로드 중 오류 발생: {str(e)}")

# 메인 앱
def main():
    st.set_page_config(
        page_title="복지PT",
        page_icon="🏛️",
        layout="wide"
    )
    
    # ChatBox 초기화
    if "chat_box" not in st.session_state:
        st.session_state.chat_box = ChatBox(
            use_rich_markdown=False,
            user_theme="green",
            assistant_theme="blue",
        )
        st.session_state.chat_box.use_chat_name("welfare_chat")
    
    chat_box = st.session_state.chat_box
    
    # 기본 문서 자동 로드
    load_default_documents()
    
    # 사이드바 구성
    with st.sidebar:        
        # 채팅 세션 선택
        chat_name = st.selectbox(
            "채팅 세션:", 
            ["welfare_chat", "general_chat"], 
            key="chat_name", 
            on_change=on_chat_change
        )
        chat_box.use_chat_name(chat_name)
        
        # 설정 옵션
        streaming = st.checkbox('스트리밍 모드', key="streaming")
        show_history = st.checkbox('세션 상태 보기', key="show_history")
        
        chat_box.context_from_session(exclude=["chat_name"])
        
        st.divider()
        
        # 사용자 정보 입력
        st.subheader("사용자 정보")
        age = st.text_input("나이", value="", placeholder="나이를 입력하세요")
        gender = st.radio("성별", options=["남", "여"], index=0)
        family_size = ["1인 가구", "한부모가족", "다자녀가정"]
        family_size = st.selectbox("가구 형태", options=family_size, index=0)
        
        # 결혼 유무 입력
        marriage = ["미혼", "기혼", "이혼"]
        marriage = st.selectbox("결혼 유무", options=marriage, index=0)
        
        # 국적 입력
        nationality = ["내국인", "외국인", "재외국민", "난민"]
        nationality = st.selectbox("국적", options=nationality, index=0)
        
        # 장애 유무 입력
        disability = st.radio("장애 유무", options=["없음", "있음"], index=0)
        
        # 병역 유무 입력
        military_service = ["군필", "복무 중", "미필"]
        military_service = st.selectbox("병역 유무", options=military_service, index=0)
        
        # 취업 여부 (실직자/구직자/재직자)
        employment_status = ["재직자", "구직자", "실직자"]
        employment_status = st.selectbox("취업 여부", options=employment_status, index=0)
        
        # 임신/출산 상태 (임산부, 출산 후 6개월 이내, 해당 없음)
        pregnancy_status = ["해당 없음", "임산부", "출산 후 6개월 이내"]
        pregnancy_status = st.selectbox("임신/출산", options=pregnancy_status, index=0)
        
        # 자녀 수 선택
        children_options = ["0명", "1명", "2명", "3명", "4명", "5명", "6명", "7명", "8명", "9명", "10명"]
        children = st.selectbox("자녀 수", options=children_options, index=0)

        # 거주지
        locations = ["서울", "수원", "부산", "대구", "인천", "광주", "대전", "울산", "세종", 
                    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]
        location = st.selectbox("거주지", options=locations, index=0)
        
        # 소득 (중위 %)
        income = st.slider("소득 (중위 %)", min_value=10, max_value=90, value=50, step=1)
        
        # 기초생활수급 여부 (수급자/비수급자)
        basic_living = st.radio("기초생활수급 여부", options=["비수급자", "수급자"], index=0)
        
        st.divider()
        
        # 문서 상태 표시
        st.subheader("문서 상태")
        
        pdf_dir = "./pdf/welfare"
        
        # 기본 문서 상태 표시
        if st.session_state.get("default_documents_loaded", False):
            if os.path.exists(pdf_dir):
                default_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
                st.success(f"✅ 기본 복지 정책 문서 {len(default_files)}개 파일이 로드되었습니다.")
                with st.expander("기본 문서 목록 보기"):
                    for file in default_files:
                        st.write(f"📄 {file}")
            else:
                st.warning("⚠️ 기본 복지 정책 문서를 로드할 수 없습니다.")
        else:
            if os.path.exists(pdf_dir) and len([f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]) > 0:
                st.info("⏳ 기본 복지 정책 문서를 로드 중입니다...")
            else:
                st.warning("⚠️ pdf/welfare 폴더에 기본 복지 정책 문서가 없습니다.")
        
        # 추가 문서 업로드 섹션
        st.subheader("추가 문서 업로드")
        
        uploaded_files = st.file_uploader(
            "추가 복지 정책 PDF 파일들을 업로드하세요 (선택사항)",
            type=["pdf"],
            accept_multiple_files=True,
            help="기본 문서에 추가로 더 많은 정책 문서를 업로드할 수 있습니다."
        )
        
        # 추가 문서 로드 버튼
        if st.button("추가 문서 로드", type="secondary"):
            with st.spinner("추가 문서를 처리하고 있습니다..."):
                try:
                    # 업로드된 파일 저장
                    additional_files = []
                    for uploaded_file in uploaded_files:
                        with open(uploaded_file.name, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        additional_files.append(uploaded_file.name)
                    
                    # 기본 문서와 추가 문서 합치기
                    all_files = []
                    
                    # 기본 문서 추가
                    if os.path.exists(pdf_dir):
                        pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
                        all_files.extend(pdf_files)
                    
                    # 추가 문서 추가
                    all_files.extend(additional_files)
                    
                    st.info(f"기본 문서 포함 총 {len(all_files)}개 파일을 처리합니다...")

                    # 문서 로드 및 처리
                    texts = load_and_process_documents(all_files)
                    
                    if texts:
                        # 벡터스토어 재생성
                        try:
                            db = create_vectorstore(texts, st.session_state.embedding_model)
                            st.session_state.db = db
                        except Exception as e:
                            st.error(f"벡터스토어 생성 실패: {str(e)}")
                            return
                        
                        st.success(f"✅ 추가 문서 포함 총 {len(texts)}개 문서 청크가 성공적으로 로드되었습니다!")
                        st.session_state.documents_loaded = True
                    else:
                        st.error("추가 문서 로드에 실패했습니다.")
                        
                except Exception as e:
                    st.error(f"추가 문서 처리 중 오류 발생: {str(e)}")
        
        st.divider()
        
        # 내보내기 버튼
        btns = st.container()
        
        if btns.button("🗑️ 대화 내역 삭제"):
            chat_box.init_session(clear=True)
            st.rerun()
    
    # 메인 채팅 영역
    # 멋진 첫 화면 문구와 폰트 크기 조정
    st.markdown(
        """
        <h1 style='text-align: center; font-size: 3.2em;'>복지PT에 오신 것을 환영합니다!</h1>
        <p style='text-align: center; font-size: 1.5em; color: #555;'>
            당신의 상황에 꼭 맞는 복지 정책을 <b>AI</b>가 쉽고 빠르게 찾아드립니다.<br>
            궁금한 점을 자유롭게 입력해보세요!
        </p>
        """,
        unsafe_allow_html=True
    )
    
    # 사용 팁 표시
    if not st.session_state.get("chat_started", False):
        st.info("""
        💡 **사용 팁**
        - 왼쪽 사이드바에서 개인정보를 입력하면 더 정확한 답변을 받을 수 있습니다
        - 예시: "30대 신혼부부를 위한 주거 지원 정책을 알려주세요"
        - 채팅 후 아래 사용설명서를 확인해보세요!
        - 추가적인 문서(PDF)를 업로드하면 더 정확한 답변을 받을 수 있습니다 (선택사항)
        """)
    
    # 채팅이 시작되었는지 확인
    if len(st.session_state.chat_box.history) > 0:
        st.session_state.chat_started = True
    
    # 채팅 박스 초기화 및 출력
    chat_box.init_session()
    chat_box.output_messages()
    
    # 피드백 설정
    feedback_kwargs = {
        "feedback_type": "thumbs",
        "optional_text_label": "피드백을 남겨주세요",
    }
    
    # 채팅 입력 처리
    if query := st.chat_input('나에게 알맞는 복지 혜택 알려주세요.'):
        # 기본 문서가 로드되었는지 확인
        if not st.session_state.get("default_documents_loaded", False):
            st.error("⏳ 기본 복지 정책 문서를 로드 중입니다. 잠시만 기다려주세요!")
            return
        

        chat_box.user_say(query)
        
        age_val = age if age.strip() else None
        
        if streaming:
            # 스트리밍 모드
            try:
                # 답변 및 참고자료 영역을 텍스트로 초기화합니다.
                elements = chat_box.ai_say([
                    "답변을 생성하고 있습니다...",
                    "",
                ])
                
                generator = generate_answer_streaming(
                    question=query,
                    age=age_val,
                    gender=gender,
                    location=location,
                    income=income,
                    family_size=family_size,
                    marriage=marriage,
                    children=children,
                    employment_status=employment_status,
                    pregnancy_status=pregnancy_status,
                    nationality=nationality,
                    disability=disability,
                    military_service=military_service,
                    basic_living=basic_living
                )
                
                text = ""
                sources = []
                search_results = []
                try:
                    for response, doc_sources, doc_search_results in generator:
                        text = response
                        sources = doc_sources
                        search_results = doc_search_results
                        chat_box.update_msg(text, element_index=0, streaming=True)
                except Exception as e:
                    text = f"스트리밍 중 오류 발생: {str(e)}"
                    sources = []
                    search_results = []
                
                chat_box.update_msg(text, element_index=0, streaming=False, state="complete")
                
                # 검색 결과와 참고자료 표시
                reference_text = ""
                if search_results:
                    reference_text += "검색된 관련 정보:\n\n"
                    for i, result in enumerate(search_results, 1):
                        reference_text += f"{i}. {os.path.basename(result['source'])} (페이지 {result['page']})\n"
                        reference_text += f"{result['content']}\n\n"
                    reference_text += "---\n\n"
                
                if sources:
                    reference_text += "참고자료:\n" + "\n".join(sources)
                else:
                    reference_text += "참고자료: 없음"
                
                chat_box.update_msg(reference_text, element_index=1, streaming=False, state="complete")
                
                # 피드백 표시
                chat_history_id = f"chat_{len(chat_box.history)}"
                chat_box.show_feedback(
                    **feedback_kwargs,
                    key=chat_history_id,
                    on_submit=on_feedback,
                    kwargs={"chat_history_id": chat_history_id, "history_index": len(chat_box.history) - 1}
                )
            except Exception as e:
                chat_box.ai_say([
                    f"스트리밍 모드 오류: {str(e)}",
                    "📄 참고자료: 없음",
                ])
        else:
            # 일반 모드
            with st.spinner("답변을 생성하고 있습니다..."):
                try:
                    result = generate_answer(
                        question=query,
                        age=age_val,
                        gender=gender,
                        location=location,
                        income=income,
                        family_size=family_size,
                        basic_living=basic_living,
                        marriage=marriage,
                        children=children,
                        employment_status=employment_status,
                        pregnancy_status=pregnancy_status,
                        nationality=nationality,
                        disability=disability,
                        military_service=military_service
                    )
                    # 반환값이 tuple인지 확인
                    if isinstance(result, tuple) and len(result) == 3:
                        text, sources, search_results = result
                    else:
                        text = str(result)
                        sources = []
                        search_results = []

                    
                except Exception as e:
                    text = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
                    sources = []
                    search_results = []
            
            reference_text = ""
            reference_text += "---\n\n"            
            if search_results:
                reference_text += "검색된 관련 정보:\n\n"
                for i, result in enumerate(search_results, 1):
                    reference_text += f"{i}. {os.path.basename(result['source'])} (페이지 {result['page']})\n"
                    reference_text += f"{result['content']}\n\n"
            
            # 답변과 참고자료를 텍스트로 제공합니다. (Markdown이 아닌 plain text)
            chat_box.ai_say([
                text,
                reference_text,
            ])
    
    # 세션 상태 보기
    if show_history:
        st.subheader("세션 상태")
        st.write(st.session_state)

if __name__ == "__main__":
    main() 