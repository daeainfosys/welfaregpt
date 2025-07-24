import time
import re
import os
from typing import List

import streamlit as st
from streamlit_chatbox import *

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch
from langchain.schema import Document
import unicodedata
from kiwipiepy import Kiwi
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# =====================================================
# 전역 변수 및 상수
# =====================================================

_kiwi = None

SIDO_CITY_MAP = {
    "서울": ["종로", "중구", "용산", "성동", "광진", "동대문", "중랑", "성북", "강북", "도봉", "노원", "은평", "서대문", "마포", "양천", "강서", "구로", "금천", "영등포", "동작", "관악", "서초", "강남", "송파", "강동"],
    "부산": ["중구", "서구", "동구", "영도구", "부산진구", "동래구", "남구", "북구", "해운대구", "사하구", "금정구", "강서구", "연제구", "수영구", "사상구", "기장군"],
    "대구": ["중구", "동구", "서구", "남구", "북구", "수성구", "달서구", "달성군"],
    "인천": ["중구", "동구", "미추홀구", "연수구", "남동구", "부평구", "계양구", "서구", "강화군", "옹진군"],
    "광주": ["동구", "서구", "남구", "북구", "광산구"],
    "대전": ["동구", "중구", "서구", "유성구", "대덕구"],
    "울산": ["중구", "남구", "동구", "북구", "울주군"],
    "세종": ["세종"],
    "경기도": ["수원", "성남", "고양", "용인", "부천", "안산", "안양", "남양주", "화성", "평택", "의정부", "시흥", "파주", "김포", "광명", "광주", "군포", "오산", "이천", "안성", "양주", "구리", "포천", "의왕", "하남", "여주", "동두천", "과천", "가평", "연천"],
    "강원도": ["춘천", "원주", "강릉", "동해", "태백", "속초", "삼척", "홍천", "횡성", "영월", "평창", "정선", "철원", "화천", "양구", "인제", "고성", "양양"],
    "충청북도": ["청주", "충주", "제천", "보은", "옥천", "영동", "진천", "괴산", "음성", "단양", "증평"],
    "충청남도": ["천안", "공주", "보령", "아산", "서산", "논산", "계룡", "당진", "금산", "부여", "서천", "청양", "홍성", "예산", "태안"],
    "전라북도": ["전주", "군산", "익산", "정읍", "남원", "김제", "완주", "진안", "무주", "장수", "임실", "순창", "고창", "부안"],
    "전라남도": ["목포", "여수", "순천", "나주", "광양", "담양", "곡성", "구례", "고흥", "보성", "화순", "장흥", "강진", "해남", "영암", "무안", "함평", "영광", "장성", "완도", "진도", "신안"],
    "경상북도": ["포항", "경주", "김천", "안동", "구미", "영주", "영천", "상주", "문경", "경산", "군위", "의성", "청송", "영양", "예천", "봉화", "울진", "울릉"],
    "경상남도": ["창원", "진주", "통영", "사천", "김해", "밀양", "거제", "양산", "의령", "함안", "창녕", "고성", "남해", "하동", "산청", "함양", "거창", "합천"],
    "제주특별자치도": ["제주시", "서귀포시"]
}

FOLLOWUP_KEYWORDS = [
    "어떻게", "방법", "절차", "신청", "접수", "자격", "조건", "요건", 
    "서류", "문서", "준비", "필요", "언제", "어디서", "누구", "얼마",
    "자세히", "더", "구체적", "상세", "추가", "정확", "정확히",
    "그 정책", "해당 정책", "위 정책", "이 정책", "그것", "그거", "이것", "이거",
    "첫 번째", "두 번째", "세 번째", "1번", "2번", "3번"
]

# =====================================================
# 채팅 관리 클래스
# =====================================================

class ChatManager:
    @staticmethod
    def get_chat_list():
        """채팅 목록을 반환합니다."""
        if "chat_list" not in st.session_state:
            st.session_state.chat_list = ["welfare_chat"]
        return st.session_state.chat_list

    @staticmethod
    def add_new_chat():
        """새 채팅을 목록에 추가합니다."""
        if "chat_list" not in st.session_state:
            st.session_state.chat_list = ["welfare_chat"]
        
        if "max_chat_number" not in st.session_state:
            st.session_state.max_chat_number = 0
        
        st.session_state.max_chat_number += 1
        new_chat_name = f"새 채팅 {st.session_state.max_chat_number}"
        st.session_state.chat_list.append(new_chat_name)
        return new_chat_name

    @staticmethod
    def delete_chat(chat_name):
        """채팅을 목록에서 삭제합니다."""
        if "chat_list" not in st.session_state or chat_name not in st.session_state.chat_list:
            return
            
        st.session_state.chat_list.remove(chat_name)
        
        # 관련 메모리 정리
        MemoryManager.cleanup_chat_data(chat_name)
        
        # 현재 채팅이 삭제된 경우 처리
        if st.session_state.get("current_chat") == chat_name:
            if st.session_state.chat_list:
                st.session_state.current_chat = st.session_state.chat_list[0]
                st.session_state.chat_box.use_chat_name(st.session_state.current_chat)
            else:
                st.session_state.chat_list = ["welfare_chat"]
                st.session_state.current_chat = "welfare_chat"
                st.session_state.chat_box.use_chat_name("welfare_chat")

    @staticmethod
    def start_new_chat():
        """새 채팅을 시작합니다."""
        new_chat_name = ChatManager.add_new_chat()
        st.session_state.current_chat = new_chat_name
        st.session_state.chat_box.use_chat_name(new_chat_name)
        st.session_state.chat_box.init_session(clear=True)
        
        # 메모리 초기화
        MemoryManager.clear_conversation_memory(new_chat_name)
        MemoryManager.save_policy_context(new_chat_name, [])
        
        st.session_state.chat_started = False
        st.rerun()

# =====================================================
# 메모리 관리 클래스
# =====================================================

class MemoryManager:
    @staticmethod
    def get_conversation_memory(chat_name):
        """채팅별 ConversationBufferMemory 인스턴스를 반환합니다."""
        memory_key = f"memory_{chat_name}"
        if memory_key not in st.session_state:
            st.session_state[memory_key] = ConversationBufferMemory(
                memory_key="chat_history",
                input_key="question",
                output_key="text",
                return_messages=False,
                max_token_limit=2000,
            )
        return st.session_state[memory_key]

    @staticmethod
    def get_conversation_history(chat_name):
        """채팅별 대화 기록을 반환합니다."""
        memory = MemoryManager.get_conversation_memory(chat_name)
        chat_memory = memory.chat_memory
        
        history = []
        messages = chat_memory.messages
        
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                human_msg = messages[i]
                ai_msg = messages[i + 1]
                history.append({
                    "question": human_msg.content,
                    "answer": ai_msg.content,
                    "timestamp": time.time()
                })
        
        return history

    @staticmethod
    def clear_conversation_memory(chat_name):
        """특정 채팅의 대화 기록을 지웁니다."""
        memory = MemoryManager.get_conversation_memory(chat_name)
        memory.clear()

    @staticmethod
    def save_policy_context(chat_name, policies_info):
        """채팅별로 제시된 정책 정보를 저장합니다."""
        context_key = f"policy_context_{chat_name}"
        st.session_state[context_key] = policies_info

    @staticmethod
    def get_policy_context(chat_name):
        """채팅별 정책 정보를 반환합니다."""
        context_key = f"policy_context_{chat_name}"
        return st.session_state.get(context_key, [])

    @staticmethod
    def cleanup_chat_data(chat_name):
        """채팅 관련 모든 데이터를 정리합니다."""
        keys_to_remove = [
            f"memory_{chat_name}",
            f"policy_context_{chat_name}"
        ]
        for key in keys_to_remove:
            if key in st.session_state:
                del st.session_state[key]

# =====================================================
# 텍스트 처리 유틸리티
# =====================================================

class TextProcessor:
    @staticmethod
    def remove_emojis_and_special_chars(text):
        """텍스트에서 이모지와 특수문자를 제거합니다."""
        return ''.join(
            ch for ch in text
            if not (
                unicodedata.category(ch).startswith('So')
                or (0x1F000 <= ord(ch) <= 0x1FFFF)
                or (0x2460 <= ord(ch) <= 0x24FF)
            )
        )

    @staticmethod
    def find_region_in_text(text: str) -> str:
        """텍스트에서 지역을 추출합니다."""
        city_to_sido = {city: sido for sido, cities in SIDO_CITY_MAP.items() for city in cities}
        
        cleaned_text = re.sub(r'[_\.]', ' ', text.lower())
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        # 금지 문구 제거
        banned_phrases = ["무주택자", "무주택 세대", "무주택세대구성원", "무주택", "주택도시기금", "주택공급", "주택건설", "주택관리"]
        for phrase in banned_phrases:
            cleaned_text = cleaned_text.replace(phrase, " ")
        
        # 시군구 우선 매칭
        for city in sorted(city_to_sido.keys(), key=len, reverse=True):
            pattern = rf'{re.escape(city)}(시|군|구)?'
            if re.search(pattern, cleaned_text):
                return city_to_sido[city]
        
        # 시도 매칭
        for sido in sorted(SIDO_CITY_MAP.keys(), key=len, reverse=True):
            pattern = rf'{re.escape(sido)}(광역시|특별시|도)?'
            if re.search(pattern, cleaned_text):
                return sido
        
        return "전국"

    @staticmethod
    def preprocess_document(text):
        """문서 전처리를 수행합니다."""
        text = TextProcessor.remove_emojis_and_special_chars(text)
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'(\*\*|\*|_)', '', text)
        text = re.sub(r'^[-\*\+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\n+', '\n', text).strip()
        return text

    @staticmethod
    def clean_text(text):
        """HTML 태그와 공백을 정리합니다."""
        text = re.sub(r'<[^>]+>', '', text)
        return re.sub(r'\s+', ' ', text.strip())

    @staticmethod
    def extract_answer_only(response):
        """LLM 응답에서 답변 부분만 추출합니다."""
        if not response:
            return response

        answer_markers = ["답변:", "답변", "Answer:", "Answer"]
        
        for marker in answer_markers:
            if marker in response:
                parts = response.split(marker, 1)
                if len(parts) > 1:
                    answer = parts[1].strip()
                    break
        else:
            lines = response.split('\n')
            answer_lines = []
            start_found = False

            for line in lines:
                if line.strip().startswith('### 정책: '):
                    start_found = True
                if start_found:
                    answer_lines.append(line)

            answer = '\n'.join(answer_lines) if answer_lines else response

        # 감사 인사 제거
        thank_you_markers = ["감사합니다.", "감사합니다", "Thank you.", "Thank you"]
        for marker in thank_you_markers:
            if marker in answer:
                answer = answer.split(marker, 1)[0].strip()
                break

        return answer

# =====================================================
# 후속 질문 처리 클래스
# =====================================================

class FollowupHandler:
    @staticmethod
    def is_followup_question(question, chat_history):
        """후속 질문인지 판단합니다."""
        question_lower = question.lower()
        has_previous_policy = "정책" in chat_history
        has_followup_keyword = any(keyword in question_lower for keyword in FOLLOWUP_KEYWORDS)
        return has_previous_policy and has_followup_keyword

    @staticmethod
    def extract_referenced_policy(question, chat_history):
        """질문에서 참조하는 정책을 추출합니다."""
        question_lower = question.lower()
        
        # 숫자로 정책을 참조하는 경우
        policy_patterns = {
            ("1번", "첫 번째"): r'### 정책 1: ([^\n]+)',
            ("2번", "두 번째"): r'### 정책 2: ([^\n]+)',
            ("3번", "세 번째"): r'### 정책 3: ([^\n]+)'
        }
        
        for keywords, pattern in policy_patterns.items():
            if any(keyword in question_lower for keyword in keywords):
                policy_match = re.search(pattern, chat_history)
                if policy_match:
                    return f"### 정책 {keywords[0][0]}: {policy_match.group(1)}"
        
        # 일반적인 참조의 경우 가장 최근 정책 반환
        policies = re.findall(r'### 정책 \d+: [^\n]+(?:\n- [^\n]+)*', chat_history)
        return policies[-1] if policies else ""

# =====================================================
# Kiwi 토크나이저 클래스
# =====================================================

class KiwiTokenizer:
    @staticmethod
    def get_instance():
        """Kiwi 인스턴스를 싱글톤으로 관리"""
        global _kiwi
        if _kiwi is None:
            _kiwi = Kiwi()
        return _kiwi

    @staticmethod
    def tokenize(text):
        """최적화된 Kiwi 토큰화 함수"""
        if not text or not text.strip():
            return []
        
        kiwi = KiwiTokenizer.get_instance()
        try:
            if len(text) > 1000:
                text = text[:1000]
            
            tokens = kiwi.tokenize(text)
            return [token.form for token in tokens if len(token.form) > 1]
        except Exception:
            return [word for word in text.split() if len(word) > 1]

# =====================================================
# 문서 처리 클래스
# =====================================================

class DocumentProcessor:
    @staticmethod
    def process_pages_with_region(pages: List[Document]) -> List[Document]:
        """페이지에 지역 정보를 추가합니다."""
        new_docs = []
        for page in pages:
            content = TextProcessor.preprocess_document(page.page_content)
            
            file_region = TextProcessor.find_region_in_text(page.metadata.get('source', ''))
            content_region = TextProcessor.find_region_in_text(content)
            region = file_region if file_region != "전국" else content_region
            
            metadata = dict(page.metadata) if page.metadata else {}
            metadata["region"] = region
            new_docs.append(Document(page_content=content, metadata=metadata))
        return new_docs

    @staticmethod
    def load_and_process_documents(file_paths: List[str], embedding_model):
        """여러 PDF 문서를 로드하고 EnsembleRetriever를 생성합니다."""
        try:
            all_documents = []
            for file_path in file_paths:
                loader = PyMuPDFLoader(file_path)
                documents = loader.load()
                all_documents.extend(documents)
            
            if not all_documents:
                return None, None, None
            processed_data = DocumentProcessor.process_pages_with_region(all_documents)
            
            KiwiTokenizer.get_instance()
            
            kiwi_bm25 = BM25Retriever.from_documents(
                processed_data, 
                preprocess_func=KiwiTokenizer.tokenize
            )
            kiwi_bm25.k = 5
            
            vectorstore = FAISS.from_documents(processed_data, embedding_model)
            faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

            retriever = EnsembleRetriever(
                retrievers=[kiwi_bm25, faiss_retriever],
                weights=[0.5, 0.5]
            )

            return retriever, len(processed_data), processed_data
        except Exception as e:
            st.error(f"문서 처리 중 치명적 오류 발생: {str(e)}")
            return None, None, None

    @staticmethod
    def load_additional_documents(uploaded_files):
        """추가 문서를 로드합니다."""
        pdf_dir = "./pdf/welfare"
        
        with st.spinner("추가 문서를 처리하고 있습니다..."):
            try:
                additional_files = []
                for uploaded_file in uploaded_files:
                    with open(uploaded_file.name, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    additional_files.append(uploaded_file.name)
                
                all_files = []
                if os.path.exists(pdf_dir):
                    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
                    all_files.extend(pdf_files)
                all_files.extend(additional_files)
                
                st.info(f"기본 문서 포함 총 {len(all_files)}개 파일을 처리합니다...")

                result = DocumentProcessor.load_and_process_documents(all_files, st.session_state.embedding_model)
                
                if result[0] is not None:
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

    @staticmethod
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
                        if "embedding_model" not in st.session_state:
                            st.session_state.embedding_model = ModelLoader.load_embedding_model()
                        
                        result = DocumentProcessor.load_and_process_documents(pdf_files, st.session_state.embedding_model)
                        if result[0] is not None:
                            retriever, total_chunks, processed_docs = result
                            st.session_state.retriever = retriever
                            st.session_state.processed_docs = processed_docs
                            
                            if "chains" not in st.session_state:
                                st.session_state.chains = ModelLoader.load_llm_model()
                            
                            st.session_state.default_documents_loaded = True
                            st.session_state.documents_loaded = True
                            st.success(f"✅ 기본 복지 문서 {total_chunks}개 청크가 로드되었습니다!")
                            
                except Exception as e:
                    st.error(f"기본 문서 로드 중 오류 발생: {str(e)}")

# =====================================================
# 모델 로더 클래스
# =====================================================

class ModelLoader:
    @staticmethod
    @st.cache_resource
    def load_embedding_model():
        """임베딩 모델을 로드하여 반환합니다."""
        return HuggingFaceEmbeddings(
            model_name='jhgan/ko-sroberta-multitask',
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True},
        )

    @staticmethod
    @st.cache_resource
    def load_llm_model():
        """LLM 모델을 로드하여 반환합니다."""
        model_name = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        llm_pipeline = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            max_new_tokens=1024,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
        llm = HuggingFacePipeline(pipeline=llm_pipeline)

        # 프롬프트 템플릿들
        policy_prompt = PromptTemplate(
            input_variables=["question", "context", "chat_history"],
            template="""당신은 한국 복지정책 전문가입니다. 이전 대화 내용을 참고하여 아래 참고자료를 바탕으로 질문에 대해 구체적이고 실용적인 복지 정책을 추천해 주세요.

{chat_history}

[사용자 질문]:
{question}

[참고자료]:
{context}

[답변 지침]:
1. 이전 대화 내용이 있다면 그 맥락을 고려하여 답변해 주세요.
2. 최대 3개의 관련 정책을 추천해 주세요.
3. 각 정책별로 아래 형식을 따라 작성해 주세요.
4. 정보가 부족한 경우, 가능한 범위에서 답변하고 추가 정보가 필요함을 명시해 주세요.

### 정책 [번호]: [정책명]
- 요약: [요약 내용]
- 대상: [대상 내용]
- 지원: [지원 내용]
- 방법: [방법 내용]
- 주의: [주의 내용]
- 문의: [문의 내용]

답변:"""
        )

        followup_prompt = PromptTemplate(
            input_variables=["selected_policy", "followup_question", "chat_history", "context"],
            template="""당신은 한국 복지정책 전문가입니다. 아래 정책에 대해 사용자가 후속 질문을 했습니다. 이전 대화 내용을 참고하여 구체적이고 실용적인 정보를 제공해 주세요.

[이전 대화 기록]
{chat_history}

[참고자료]
{context}

[선택된 정책]
{selected_policy}

[사용자의 후속 질문]
{followup_question}

[답변 지침]
1. 선택된 정책과 관련된 정보에 집중하여 상세하게 설명해 주세요.
2. 필요한 경우 신청 절차, 대상 요건, 유의사항 등을 구체적으로 안내해 주세요.
3. 정보가 부족한 경우, 가능한 범위 내에서 답변하고 추가 정보가 필요함을 알려 주세요.

답변:
"""
        )

        return {
            "policy": LLMChain(prompt=policy_prompt, llm=llm),
            "followup": LLMChain(prompt=followup_prompt, llm=llm)
        }

# =====================================================
# 쿼리 처리 클래스
# =====================================================

class QueryProcessor:
    @staticmethod
    def process_query(question, chat_name):
        """질문을 처리하고 답변을 생성합니다."""
        if st.session_state.get("retriever") is None or st.session_state.get("chains") is None:
            return "복지 정책 문서와 AI 모델을 먼저 로드해주세요.", []

        # 사용자 질문에서 지역 추출
        user_region = TextProcessor.find_region_in_text(question)

        try:
            # 문서 검색
            initial_docs = st.session_state.retriever.invoke(question)
        except Exception as e:
            return f"문서 검색 중 오류 발생: {str(e)}", []

        # 지역 필터링
        docs = []
        for doc in initial_docs:
            doc_region = doc.metadata.get("region", "전국")
            if (doc_region == user_region or doc_region == "전국" or user_region == "전국" or
                (user_region in SIDO_CITY_MAP.get(doc_region, []))):
                docs.append(doc)

        # 문서 품질 검증
        quality_docs = []
        for doc in docs[:3]:
            cleaned_content = TextProcessor.clean_text(doc.page_content)
            if len(cleaned_content.strip()) > 30:
                quality_docs.append(doc)
                
        if not quality_docs:
            return """죄송합니다. 해당 질문에 대한 적절한 정보를 찾을 수 없습니다. 
                
더 정확한 답변을 위해 다음 정보를 추가로 제공해 주시면 도움이 됩니다:
- 구체적인 거주지역 (시/군/구)
- 나이 및 가구 형태
- 소득 수준 및 특별한 상황
- 찾고 계신 복지 분야 (주거, 육아, 취업 등)""", []

        # 참고자료 구성
        context_parts = []
        search_results = []
        for i, doc in enumerate(quality_docs[:3]):
            clean_content = TextProcessor.clean_text(doc.page_content)
            context_parts.append(f"[참고자료 {i+1}]\n{clean_content[:1000]}")
            
            source = doc.metadata.get('source', '알 수 없음')
            page = doc.metadata.get('page', '')
            
            search_results.append({
                'content': clean_content[:500] + "..." if len(clean_content) > 500 else clean_content,
                'page': page,
                'source': source
            })

        context = "\n\n".join(context_parts)
        try:
            memory = MemoryManager.get_conversation_memory(chat_name)
            chat_history = memory.buffer
            
            # 후속 질문 판단
            is_followup = FollowupHandler.is_followup_question(question, chat_history)
            
            if is_followup:
                selected_policy = FollowupHandler.extract_referenced_policy(question, chat_history)
                response = st.session_state.chains["followup"].run({
                    "followup_question": question,
                    "selected_policy": selected_policy,
                    "context": context,
                    "chat_history": chat_history
                })
            else:
                response = st.session_state.chains["policy"].run({
                    "question": question, 
                    "context": context,
                    "chat_history": chat_history
                })
            
            processed_response = TextProcessor.extract_answer_only(response)
            processed_response = TextProcessor.remove_emojis_and_special_chars(processed_response)

            # 메모리에 저장
            memory.save_context({"question": question}, {"text": processed_response})

            return processed_response, search_results
        except Exception as e:
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}", []

    @staticmethod
    def generate_answer_streaming(question, chat_name):
        """질문에 대한 답변을 생성합니다 (스트리밍 모드)."""
        try:
            response, search_results = QueryProcessor.process_query(question, chat_name)
            
            if not response or "오류" in response:
                yield response, [], []
                return
            
            words = response.split(' ')
            sources = []
            if search_results:
                for result in search_results:
                    source_info = f"{result['source']} (페이지 {result['page']})"
                    sources.append(source_info)

            for i in range(0, len(words), 5):
                yield " ".join(words[:i+5]), sources, search_results
                time.sleep(0.1)
                    
        except Exception as e:
            yield f"답변 생성 중 예상치 못한 오류가 발생했습니다: {str(e)}", [], []

# =====================================================
# UI 관리 클래스
# =====================================================

class UIManager:
    @staticmethod
    def render_css():
        """CSS 스타일을 렌더링합니다."""
        st.markdown(
            """
            <style>
            /* 페이지 전체 스크롤 시 채팅 입력창을 위한 여백 */
            .main .block-container {
                padding-bottom: 120px !important;
            }
            
            /* 마이크 버튼 정렬 */
            .mic-button {
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                height: 48px !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    @staticmethod
    def render_header():
        """헤더를 렌더링합니다."""
        st.markdown(
            """
            <h1 style='text-align: center; font-size: 3.2em;'>생애주기별 개인 맞춤형 복지 추천 AI 에이전트</h1>
            <p style='text-align: center; font-size: 1.5em; color: #555;'>
                당신의 상황에 꼭 맞는 복지 정책을 <b>AI</b>가 쉽고 빠르게 찾아드립니다.<br>
                궁금한 점을 자유롭게 입력해보세요!
            </p>
            """,
            unsafe_allow_html=True
        )

    @staticmethod
    def render_usage_tips():
        """사용 팁을 렌더링합니다."""
        if not st.session_state.get("chat_started", False):
            st.info("""
            💡 **사용 팁**
            - 구체적인 상황을 설명하면 더 정확한 답변을 받을 수 있습니다
            - 예시: "30대 신혼부부를 위한 주거 지원 정책을 알려주세요"
            - 나이, 소득, 가구 형태 등의 정보를 함께 제공해주세요
            - 추가적인 문서(PDF)를 업로드하면 더 정확한 답변을 받을 수 있습니다 (선택사항)
            - 이전 대화 내용을 참고하여 연속성 있는 답변을 제공합니다
            - **후속 질문**: 정책이 제시된 후 "1번 정책의 신청 방법을 알려주세요", "그 정책의 자격 요건은 무엇인가요?" 같은 구체적인 질문을 할 수 있습니다
            - 왼쪽 사이드바에서 새 채팅을 생성하거나 기존 채팅을 선택할 수 있습니다
            """)

    @staticmethod
    def render_sidebar():
        """사이드바를 렌더링합니다."""
        with st.sidebar:
            # 새 채팅 버튼
            if st.button("📝 새 채팅", type="secondary", use_container_width=True):
                ChatManager.start_new_chat()
            
            st.divider()
            
            # 채팅 목록
            st.subheader("채팅")
            chat_list = ChatManager.get_chat_list()
            
            for i, chat in enumerate(chat_list):
                display_text = f"{chat[:20]}..." if len(chat) > 20 else chat
                
                if chat == st.session_state.current_chat:
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(
                            f'<div style="background-color: #444444; padding: 8px; border-radius: 8px; font-weight: bold; color: #FFFFFF; width: 100%;">'
                            f'{display_text}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    with col2:
                        if len(chat_list) > 1:
                            if st.button("🗑️", key=f"delete_current_{i}", help="현재 채팅 삭제", use_container_width=False):
                                ChatManager.delete_chat(chat)
                                st.rerun()
                else:
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        if st.button(f"{display_text}", key=f"chat_{i}", use_container_width=True, help="채팅 선택"):
                            st.session_state.current_chat = chat
                            st.session_state.chat_box.use_chat_name(chat)
                            st.rerun()
                    with col2:
                        if len(chat_list) > 1:
                            if st.button("🗑️", key=f"delete_other_{i}", help="채팅 삭제", use_container_width=False):
                                ChatManager.delete_chat(chat)
                                st.rerun()
            
            st.divider()
            
            # 문서 상태
            UIManager.render_document_status()
            
            # 추가 문서 업로드
            UIManager.render_document_upload()

    @staticmethod
    def render_document_status():
        """문서 상태를 렌더링합니다."""
        st.subheader("문서 상태")
        pdf_dir = "./pdf/welfare"
        
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

    @staticmethod
    def render_document_upload():
        """문서 업로드 섹션을 렌더링합니다."""
        st.subheader("추가 문서 업로드")
        
        uploaded_files = st.file_uploader(
            "추가 복지 정책 PDF 파일들을 업로드하세요 (선택사항)",
            type=["pdf"],
            accept_multiple_files=True,
            help="기본 문서에 추가로 더 많은 정책 문서를 업로드할 수 있습니다."
        )
        
        if st.button("추가 문서 로드", type="secondary"):
            if uploaded_files:
                success = DocumentProcessor.load_additional_documents(uploaded_files)
                if success:
                    st.rerun()
            else:
                st.warning("업로드할 파일을 선택해주세요.")

# =====================================================
# 피드백 처리
# =====================================================

def on_feedback(feedback, chat_history_id: str = "", history_index: int = -1):
    """피드백을 처리합니다."""
    reason = feedback.get("text", "")
    score = feedback.get("score", 0)
    
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

# =====================================================
# 메인 애플리케이션
# =====================================================

def main():
    """메인 애플리케이션 함수"""
    st.set_page_config(
        page_title="생애주기별 개인 맞춤형 복지 추천 AI 에이전트",
        page_icon="🏛️",
        layout="wide"
    )
    
    # 초기화
    if "chat_box" not in st.session_state:
        st.session_state.chat_box = ChatBox(
            use_rich_markdown=True,
            user_theme="green",
            assistant_theme="blue",
        )
        st.session_state.chat_box.use_chat_name("welfare_chat")
    
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "welfare_chat"
    
    if "mic_active" not in st.session_state:
        st.session_state.mic_active = False
    
    chat_box = st.session_state.chat_box
    
    # 기본 문서 로드
    DocumentProcessor.load_default_documents()
    
    # UI 렌더링
    UIManager.render_sidebar()
    UIManager.render_header()
    UIManager.render_usage_tips()
    
    # 채팅 상태 확인
    if len(st.session_state.chat_box.history) > 0:
        st.session_state.chat_started = True
    
    # 채팅 박스 설정
    chat_box.use_chat_name(st.session_state.current_chat)
    st.markdown('<div class="chat-history-container">', unsafe_allow_html=True)
    chat_box.init_session()
    chat_box.output_messages()
    
    feedback_kwargs = {
        "feedback_type": "thumbs",
        "optional_text_label": "피드백을 남겨주세요",
    }
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # CSS 적용
    UIManager.render_css()
    
    # 채팅 입력창
    col1, col2 = st.columns([8, 1])

    with col1:
        current_chat = st.session_state.current_chat
        history = MemoryManager.get_conversation_history(current_chat)
        is_first_question = len(history) == 0 and len(st.session_state.chat_box.history) == 0

        placeholder_text = ("복지 정책에 대해 질문해주세요. (예: 30대 신혼부부 주거 지원 정책)" 
                          if is_first_question 
                          else "추가 질문이나 더 자세한 정보가 필요하시면 입력해주세요.")

        st.markdown('<div style="height: 48px; display: flex; align-items: center;">', unsafe_allow_html=True)
        query = st.chat_input(placeholder_text)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="mic-button" style="height: 48px; display: flex; align-items: center; justify-content: center;">', unsafe_allow_html=True)
        if st.button("🎤", key="mic_button", type="secondary" if not st.session_state.mic_active else "primary"):
            st.session_state.mic_active = not st.session_state.mic_active
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
        
    # 채팅 입력 처리
    if query:
        if not st.session_state.get("default_documents_loaded", False) or "chains" not in st.session_state:
            st.error("⏳ 기본 복지 정책 문서와 AI 모델을 로드 중입니다. 잠시만 기다려주세요!")
            return
        
        current_chat = st.session_state.current_chat
        
        # 채팅 이름 자동 생성
        if (current_chat == "welfare_chat" and 
            len(MemoryManager.get_conversation_history(current_chat)) == 0 and
            len(st.session_state.chat_box.history) == 0):
            
            chat_name_preview = query[:20] + "..." if len(query) > 20 else query
            chat_name_preview = re.sub(r'[^\w\s가-힣]', '', chat_name_preview)
            
            if "welfare_chat" in st.session_state.chat_list:
                index = st.session_state.chat_list.index("welfare_chat")
                st.session_state.chat_list[index] = chat_name_preview
                st.session_state.current_chat = chat_name_preview
                current_chat = chat_name_preview
        
        chat_box.use_chat_name(current_chat)
        chat_box.user_say(query)
        
        try:
            elements = chat_box.ai_say(["답변을 생성하고 있습니다...", ""])
            generator = QueryProcessor.generate_answer_streaming(question=query, chat_name=current_chat)
            
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
            
            # 참고자료 표시
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
            
            # 피드백
            chat_history_id = f"chat_{len(chat_box.history)}"
            chat_box.show_feedback(
                **feedback_kwargs,
                key=chat_history_id,
                on_submit=on_feedback,
                kwargs={"chat_history_id": chat_history_id, "history_index": len(chat_box.history) - 1}
            )
            
        except Exception as e:
            chat_box.ai_say([f"스트리밍 모드 오류: {str(e)}", "📄 참고자료: 없음"])

if __name__ == "__main__":
    main() 