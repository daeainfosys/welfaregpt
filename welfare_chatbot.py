import time
import re
import os

import streamlit as st
from streamlit_chatbox import *

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch
from langchain.schema import Document
from typing import List
import unicodedata
from kiwipiepy import Kiwi
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

_kiwi = None

# 채팅 목록 관리 함수
def get_chat_list():
    """채팅 목록을 반환합니다."""
    if "chat_list" not in st.session_state:
        st.session_state.chat_list = ["welfare_chat"]
    return st.session_state.chat_list

def add_new_chat():
    """새 채팅을 목록에 추가합니다."""
    if "chat_list" not in st.session_state:
        st.session_state.chat_list = ["welfare_chat"]
    
    # 최대 채팅 번호 관리
    if "max_chat_number" not in st.session_state:
        st.session_state.max_chat_number = 0
    
    # 새 채팅 번호는 현재 최대 번호 + 1
    st.session_state.max_chat_number += 1
    new_chat_name = f"새 채팅 {st.session_state.max_chat_number}"
    
    st.session_state.chat_list.append(new_chat_name)
    return new_chat_name

def delete_chat(chat_name):
    """채팅을 목록에서 삭제합니다."""
    if "chat_list" in st.session_state and chat_name in st.session_state.chat_list:
        st.session_state.chat_list.remove(chat_name)
        # 삭제된 채팅이 현재 채팅이면 첫 번째 채팅으로 변경
        if st.session_state.get("current_chat") == chat_name:
            if st.session_state.chat_list:
                st.session_state.current_chat = st.session_state.chat_list[0]
                st.session_state.chat_box.use_chat_name(st.session_state.current_chat)
            else:
                # 모든 채팅이 삭제되면 기본 채팅 생성
                st.session_state.chat_list = ["welfare_chat"]
                st.session_state.current_chat = "welfare_chat"
                st.session_state.chat_box.use_chat_name("welfare_chat")

def start_new_chat():
    """새 채팅을 시작합니다."""
    new_chat_name = add_new_chat()
    st.session_state.current_chat = new_chat_name
    st.session_state.chat_box.use_chat_name(new_chat_name)
    st.session_state.chat_box.init_session(clear=True)
    st.session_state.chat_started = False
    st.rerun()

# 캐시 메모리 관리 함수
def get_conversation_cache_key(chat_name):
    """채팅별 대화 캐시 키를 생성합니다."""
    return f"conv_cache_{chat_name}"

def save_conversation_to_cache(chat_name, question, answer):
    """대화를 캐시에 저장합니다."""
    cache_key = get_conversation_cache_key(chat_name)
    if cache_key not in st.session_state:
        st.session_state[cache_key] = []
    
    st.session_state[cache_key].append({
        "question": question,
        "answer": answer,
        "timestamp": time.time()
    })
    
    # 최근 10개 대화만 유지
    if len(st.session_state[cache_key]) > 10:
        st.session_state[cache_key] = st.session_state[cache_key][-10:]

def get_conversation_history(chat_name):
    """채팅별 대화 기록을 반환합니다."""
    cache_key = get_conversation_cache_key(chat_name)
    return st.session_state.get(cache_key, [])

def format_conversation_history(chat_name):
    """대화 기록을 프롬프트 형식으로 포맷팅합니다."""
    history = get_conversation_history(chat_name)
    if not history:
        return ""
    
    formatted_history = "\n[이전 대화 기록]:\n"
    for i, conv in enumerate(history[-3:], 1):  # 최근 3개만 참조 (성능 향상)
        formatted_history += f"Q{i}: {conv['question']}\n"
        formatted_history += f"A{i}: {conv['answer'][:150]}...\n"  # 답변은 150자로 제한
    
    return formatted_history

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

def get_kiwi_instance():
    """Kiwi 인스턴스를 싱글톤으로 관리"""
    global _kiwi
    if _kiwi is None:
        _kiwi = Kiwi()
    return _kiwi

def kiwi_tokenize(text):
    """최적화된 Kiwi 토큰화 함수"""
    if not text or not text.strip():
        return []
    
    kiwi = get_kiwi_instance()
    try:
        # 긴 텍스트는 잘라서 처리 (메모리 절약)
        if len(text) > 1000:
            text = text[:1000]
        
        tokens = kiwi.tokenize(text)
        return [token.form for token in tokens if len(token.form) > 1]  # 한 글자 토큰 제거
    except Exception as e:
        # 토큰화 실패 시 간단한 공백 분할
        return [word for word in text.split() if len(word) > 1]

# 배치 처리 함수
def add_documents_in_batches(vectorstore, documents, batch_size=1000):
    """문서들을 배치 단위로 나누어 벡터 저장소에 추가"""
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        vectorstore.add_documents(batch)

# 문서 로드 및 전처리 함수
def load_and_process_documents(file_paths: List[str], embedding_model):
    """여러 PDF 문서를 로드하고 EnsembleRetriever를 생성"""
    try:
        all_documents = []
        successful_files = []
        for file_path in file_paths:
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            all_documents.extend(documents)
            successful_files.append(file_path)
        
        if not all_documents:
            return None, None, None
        
        print(f"총 {len(all_documents)}개 페이지 로딩 완료")
        
        processed_data = process_pages(all_documents)
        
        processed_data = [
            doc for doc in processed_data 
        ]
        
        get_kiwi_instance()
        
        kiwi_bm25 = BM25Retriever.from_documents(
            processed_data, 
            preprocess_func=kiwi_tokenize
        )
        
        kiwi_bm25.k = 3
        
        vectorstore = FAISS.from_documents(
            processed_data,
            embedding_model,
        )
        
        faiss_retriever = vectorstore.as_retriever(
        )

        retriever = EnsembleRetriever(
            retrievers=[kiwi_bm25, faiss_retriever],
            weights=[0.7, 0.3],  # BM25에 더 높은 가중치
            search_type="mmr"
        )

        return retriever, len(processed_data), processed_data
    except Exception as e:
        st.error(f"문서 처리 중 치명적 오류 발생: {str(e)}")
        return None, None, None

# 임베딩 모델 로드 캐시 함수
@st.cache_resource
def load_embedding_model():
    """임베딩 모델을 로드하여 반환합니다."""
    embedding_model = HuggingFaceEmbeddings(
        model_name='jhgan/ko-sroberta-multitask',
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True},
    )
    return embedding_model


# LLM 모델 로드 캐시 함수 
@st.cache_resource
def load_llm_model():
    """LLM 모델을 로드하여 반환합니다."""
    model_name = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm_pipeline = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        max_new_tokens=512,
        temperature=1.0,  # 의미 없음, 제거 가능
        do_sample=False,  # 확률적 출력 제거
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
        num_beams=4,  # 빔 서치로 정확도 향상
    )

    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    return llm

# 공통 문서 처리 함수
def _process_query(question, chat_name):    
    """질문을 처리하고 프롬프트와 출처를 생성합니다."""
    if st.session_state.get("retriever") is None:
        return None, "PDF 파일을 먼저 업로드해주세요.", [], []
    
    # clean_text 함수 정의
    def clean_text(text):
        text = re.sub(r'<[^>]+>', '', text)
        return re.sub(r'\s+', ' ', text.strip())

    # EnsembleRetriever 사용 (설정된 k 값에 따라 검색)
    try:
        docs = st.session_state.retriever.invoke(question, k=3)
    except Exception as e:
        return None, f"문서 검색 중 오류 발생: {str(e)}", [], []

    if not docs:
        return None, "관련 정보를 찾을 수 없습니다. 다른 키워드나 표현으로 다시 질문해주세요.", [], []

    # 문서 품질 필터링
    quality_docs = []
    for doc in docs:
        cleaned_content = clean_text(doc.page_content)
        if len(cleaned_content.strip()) > 100:
            quality_docs.append(doc)

    if not quality_docs:
        return None, "검색된 문서의 품질이 낮아 답변을 생성할 수 없습니다. 더 구체적인 질문을 해주세요.", [], []

    # 참고자료와 출처 정리
    context_parts = []
    sources = []
    search_results = []  # 검색 결과를 사용자에게 보여주기 위한 변수
    
    for i, doc in enumerate(quality_docs[:3]):
        clean_content = clean_text(doc.page_content)
        context_parts.append(f"[참고자료 {i+1}]\n{clean_content[:500]}")
        
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

    # 대화 기록 가져오기
    conversation_history = format_conversation_history(chat_name)
    history = get_conversation_history(chat_name)
    
    # 첫 번째 질문인지 확인
    is_first_question = len(history) == 0
    
    # 프롬프트 생성 (첫 번째 질문 여부에 따라 다르게 생성)
    if is_first_question:
        # 첫 번째 질문 - 구조화된 정책 번호별 추천
        prompt = f"""한국 복지정책 전문가로서, 아래 참고자료를 바탕으로 질문에 대해 알기 쉽고 정확하게 복지 정책을 추천해 주세요.

만약 문서에 답이 없거나 불완전하다면, 다음과 같이 답변해 주세요:
1. 현재 제공 가능한 정보를 먼저 알려주세요.
2. 부족한 정보에 대해 "추가로 다음 정보가 필요합니다:" 형태로 명시해 주세요.
3. 사용자가 어떤 정보를 더 제공하면 도움이 될지 구체적으로 안내해 주세요.

[사용자 질문]:
{question}

[참고자료]:
{context}

[답변 중요 지침]:
1. 최대 3개 정책을 추천해 주세요.  
2. 각 정책마다 아래 6개 항목을 모두 작성해 주세요.  
3. 답변 마지막은 완전한 문장으로 마무리해 주세요.
4. 정보가 부족한 경우, 어떤 개인정보나 상황 정보가 추가로 필요한지 구체적으로 안내해 주세요.

[필수 형식]:
### 정책 [번호]: [정책명]
- 요약: [요약 내용]
- 대상: [대상 내용]
- 지원: [지원 내용]
- 방법: [방법 내용]
- 주의: [주의 내용]
- 문의: [문의 내용]

[정보 부족 시 추가 안내]:
더 정확한 정책 추천을 위해 다음 정보가 필요합니다:
- 나이, 성별, 거주지역
- 소득 수준, 가구 형태
- 결혼 여부, 자녀 수
- 취업 상태, 특별한 상황(임신, 장애 등)

답변:"""
    else:
        # 후속 질문 - 유동적인 답변
        prompt = f"""한국 복지정책 전문가로서, 아래 참고자료와 이전 대화 기록을 바탕으로 질문에 대해 자연스럽고 유용한 답변을 제공해 주세요.

{conversation_history}

[현재 사용자 질문]:
{question}

[참고자료]:
{context}

[답변 중요 지침]:
1. 이전 대화 기록을 참고하여 연속성 있는 답변을 제공해 주세요.
2. 사용자의 질문과 상황에 맞게 유동적으로 답변해 주세요.
3. 추가 정보가 필요한 경우, 구체적으로 안내해 주세요.
4. 정책 추천 시에는 사용자의 상황에 가장 적합한 정책을 우선적으로 소개해 주세요.
5. 답변은 자연스럽고 이해하기 쉽게 작성해 주세요.

답변:"""

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
        if line.strip().startswith('### 정책: '):
            start_found = True
        if start_found:
            answer_lines.append(line)
    
    if answer_lines:
        return '\n'.join(answer_lines)
    
    return response

# 스트리밍 모드 답변 생성 함수
def generate_answer_streaming(question, chat_name):
    """질문에 대한 답변을 생성합니다 (스트리밍 모드)."""
    try:
        result = _process_query(question, chat_name)
        
        if len(result) == 4:
            prompt, error_msg, sources, search_results = result
        else:
            # 이전 형식 지원 (3개 반환값)
            prompt, error_msg, sources = result
            search_results = []
        
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

            save_conversation_to_cache(chat_name, question, clean_response)

            # 스트리밍 시뮬레이션
            words = clean_response.split(' ')

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

# 추가 문서 로드 함수
def load_additional_documents(uploaded_files):
    """추가 문서를 로드합니다."""
    pdf_dir = "./pdf/welfare"
    
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

            # EnsembleRetriever 방식으로 문서 로드 및 처리
            result = load_and_process_documents(all_files, st.session_state.embedding_model)
            
            if result[0] is not None:  # retriever가 성공적으로 생성되었는지 확인
                retriever, total_chunks, processed_docs = result
                st.session_state.retriever = retriever
                st.session_state.processed_docs = processed_docs
                
                st.success(f"✅ 추가 문서 포함 총 {len(all_files)}개 문서가 성공적으로 로드되었습니다!")
                st.session_state.documents_loaded = True
                return True
            else:
                st.error("추가 문서 로드에 실패했습니다.")
                return False
                
        except Exception as e:
            st.error(f"추가 문서 처리 중 오류 발생: {str(e)}")
            return False

# 기본 문서 자동 로드 함수 (EnsembleRetriever 방식으로 수정)
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
                    # 임베딩 모델 로드
                    if "embedding_model" not in st.session_state:
                        st.session_state.embedding_model = load_embedding_model()
                    
                    # EnsembleRetriever 방식으로 문서 로드 및 처리
                    result = load_and_process_documents(pdf_files, st.session_state.embedding_model)
                    if result[0] is not None:  # retriever가 성공적으로 생성되었는지 확인
                        retriever, total_chunks, processed_docs = result
                        st.session_state.retriever = retriever
                        st.session_state.processed_docs = processed_docs
                        
                        # LLM 모델 로드
                        if "llm" not in st.session_state:
                            st.session_state.llm = load_llm_model()
                        
                        st.session_state.default_documents_loaded = True
                        st.session_state.documents_loaded = True
                        st.success(f"✅ 기본 복지 문서 {total_chunks}개 청크가 로드되었습니다!")
                        st.success(f"✅ 기본 복지 문서 로드되었습니다!")
                        
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
    st.session_state.chat_box = ChatBox(
        use_rich_markdown=True,  # True로 변경
        user_theme="green",
        assistant_theme="blue",
    )
    st.session_state.chat_box.use_chat_name("welfare_chat")
    
    # 현재 채팅 초기화
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "welfare_chat"
    
    # 마이크 상태 초기화
    if "mic_active" not in st.session_state:
        st.session_state.mic_active = False
    
    chat_box = st.session_state.chat_box
    
    # 기본 문서 자동 로드
    load_default_documents()
    
    # 사이드바 구성
    with st.sidebar:        
        # 새 채팅 추가 버튼
        # "새 채팅" 버튼을 회색(secondary) 스타일로 변경
        if st.button("📝 새 채팅", type="secondary", use_container_width=True):
            start_new_chat()
        
        st.divider()
        
        # 채팅 표시
        st.subheader("채팅")
        chat_list = get_chat_list()
        
        for i, chat in enumerate(chat_list):
            display_text = f"{chat[:20]}..." if len(chat) > 20 else chat
            
            # 현재 채팅 표시
            if chat == st.session_state.current_chat:
                # 현재 채팅은 녹색 배경, 삭제 버튼 포함
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    # 진한 회색 배경, 사이즈 동일하게 현재 채팅 표시
                    st.markdown(
                        f'<div style="background-color: #444444; padding: 8px; border-radius: 8px; font-weight: bold; color: #FFFFFF; width: 100%;">'
                        f'{display_text}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                with col2:
                    if len(chat_list) > 1:  # 최소 하나의 채팅은 유지
                        if st.button("🗑️", key=f"delete_current_{i}", help="현재 채팅 삭제", 
                                    use_container_width=False):
                            delete_chat(chat)
                            st.rerun()
            else:
                # 다른 채팅은 회색 배경
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    # 회색 배경 채팅 영역
                    if st.button(
                        f"{display_text}",
                        key=f"chat_{i}",
                        use_container_width=True,
                        help="채팅 선택"
                    ):
                        st.session_state.current_chat = chat
                        st.session_state.chat_box.use_chat_name(chat)
                        st.rerun()

                
                with col2:
                    if len(chat_list) > 1:  # 최소 하나의 채팅은 유지
                        if st.button("🗑️", key=f"delete_other_{i}", help="채팅 삭제", 
                                    use_container_width=False):
                            delete_chat(chat)
                            st.rerun()
        
        st.divider()
        
        # 설정 옵션
        show_history = st.checkbox('세션 상태 보기', key="show_history")
        
        chat_box.context_from_session(exclude=["current_chat"])
        
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
            if uploaded_files:
                success = load_additional_documents(uploaded_files)
                if success:
                    st.rerun()
            else:
                st.warning("업로드할 파일을 선택해주세요.")
        
        st.divider()
    
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
        - 구체적인 상황을 설명하면 더 정확한 답변을 받을 수 있습니다
        - 예시: "30대 신혼부부를 위한 주거 지원 정책을 알려주세요"
        - 나이, 소득, 가구 형태 등의 정보를 함께 제공해주세요
        - 추가적인 문서(PDF)를 업로드하면 더 정확한 답변을 받을 수 있습니다 (선택사항)
        - 이전 대화 내용을 참고하여 연속성 있는 답변을 제공합니다
        - 왼쪽 사이드바에서 새 채팅을 생성하거나 기존 채팅을 선택할 수 있습니다
        """)
    
    # 채팅이 시작되었는지 확인
    if len(st.session_state.chat_box.history) > 0:
        st.session_state.chat_started = True
    
    # 현재 채팅으로 설정
    chat_box.use_chat_name(st.session_state.current_chat)
    
    # 채팅 박스 초기화 및 출력
    chat_box.init_session()
    chat_box.output_messages()
    
    # 피드백 설정
    feedback_kwargs = {
        "feedback_type": "thumbs",
        "optional_text_label": "피드백을 남겨주세요",
    }
    
    # 채팅 입력창 하단 고정을 위한 CSS
    st.markdown(
        """
        <style>
        /* 페이지 전체 스크롤 시 채팅 입력창 고정 */
        .main .block-container {
            padding-bottom: 120px !important;
        }
        
        /* 채팅 입력창 고정 스타일 */
        .fixed-bottom {
            position: fixed !important;
            bottom: 0 !important;
            left: 0 !important;
            right: 0 !important;
            background: rgba(255, 255, 255, 0.98) !important;
            backdrop-filter: blur(15px) !important;
            padding: 20px !important;
            border-top: 2px solid #e6e6e6 !important;
            box-shadow: 0 -4px 20px rgba(0,0,0,0.15) !important;
            z-index: 9999 !important;
        }
        
        /* 사이드바가 있는 경우 왼쪽 여백 조정 */
        .fixed-bottom {
            left: 21rem !important;
        }
        
        /* 데스크톱에서 사이드바 너비 조정 */
        @media (max-width: 768px) {
            .fixed-bottom {
                left: 0 !important;
                right: 0 !important;
            }
        }
        
        /* 마이크 버튼 스타일 */
        .mic-button {
            display: flex !important;
            align-items: center !important;
            height: 100% !important;
        }
        
        /* 채팅 입력창 스타일 개선 */
        .stChatInput > div {
            margin-bottom: 0 !important;
        }
        
        /* 전체 채팅 입력창 컨테이너 */
        .fixed-bottom [data-testid="column"] {
            gap: 10px !important;
        }
        
        /* 채팅 입력창 자체 스타일 */
        .fixed-bottom .stChatInput {
            margin-bottom: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # 하단 고정 입력 영역
    st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
    
    # 채팅 입력 UI - 마이크 버튼 추가 (하단 고정)
    col1, col2 = st.columns([10, 1])
    
    with col1:
        # 첫 번째 질문인지 확인하여 플레이스홀더 메시지 변경
        current_chat = st.session_state.current_chat
        history = get_conversation_history(current_chat)
        is_first_question = len(history) == 0 and len(st.session_state.chat_box.history) == 0
        
        if is_first_question:
            placeholder_text = "복지 정책에 대해 질문해주세요. (예: 30대 신혼부부 주거 지원 정책)"
        else:
            placeholder_text = "추가 질문이나 더 자세한 정보가 필요하시면 입력해주세요."
        
        query = st.chat_input(placeholder_text)
    
    with col2:
        st.markdown('<div class="mic-button">', unsafe_allow_html=True)
        # 마이크 버튼 (기능 없음, 시각적 효과만)
        if st.button("🎤", key="mic_button", type="secondary" if not st.session_state.mic_active else "primary"):
            st.session_state.mic_active = not st.session_state.mic_active
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
        
    # 채팅 입력 처리
    if query:
        # 기본 문서가 로드되었는지 확인
        if not st.session_state.get("default_documents_loaded", False):
            st.error("⏳ 기본 복지 정책 문서를 로드 중입니다. 잠시만 기다려주세요!")
            return
        
        # 현재 채팅 이름 설정
        current_chat = st.session_state.current_chat
        
        # 현재 채팅이 비어있고 기본 이름인 경우 자동 이름 생성
        if (current_chat == "welfare_chat" and 
            len(get_conversation_history(current_chat)) == 0 and
            len(st.session_state.chat_box.history) == 0):
            
            # 질문 기반으로 채팅 이름 생성
            chat_name_preview = query[:20] + "..." if len(query) > 20 else query
            chat_name_preview = re.sub(r'[^\w\s가-힣]', '', chat_name_preview)
            
            # 채팅 리스트에서 기본 이름 교체
            if "welfare_chat" in st.session_state.chat_list:
                index = st.session_state.chat_list.index("welfare_chat")
                st.session_state.chat_list[index] = chat_name_preview
                st.session_state.current_chat = chat_name_preview
                current_chat = chat_name_preview
        
        chat_box.use_chat_name(current_chat)
        chat_box.user_say(query)
        
        try:
            elements = chat_box.ai_say([
                "답변을 생성하고 있습니다...",
                "",
            ])
            
            generator = generate_answer_streaming(question=query, chat_name=current_chat)
            
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
    
    # 세션 상태 보기
    if show_history:
        st.subheader("세션 상태")
        st.write(f"현재 채팅: {st.session_state.current_chat}")
        st.write(f"마이크 상태: {'활성화' if st.session_state.mic_active else '비활성화'}")
        st.write(f"채팅 목록: {get_chat_list()}")
        
        # 대화 기록 표시
        st.write("대화 기록:")
        history = get_conversation_history(st.session_state.current_chat)
        for i, conv in enumerate(history, 1):
            st.write(f"{i}. Q: {conv['question'][:50]}...")
            st.write(f"   A: {conv['answer'][:100]}...")
        
        with st.expander("전체 세션 상태 보기"):
            st.write(st.session_state)

if __name__ == "__main__":
    main() 