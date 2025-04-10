from attr import dataclass
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_teddynote import logging

# from langchain_teddynote.messages import random_uuid
import uuid
from modules.agent import create_agent_executor
from modules.handler import stream_handler, format_search_result
from modules.tools import WebSearchTool

import deepl


@dataclass
class ChatMessageWithType:
    chat_message: ChatMessage
    msg_type: str
    tool_name: str


st.title("DeepLens!")
st.title("넷스루 행동데이터 AI분석 💬")
st.markdown(
    "**고객행동데이터**를 사용자의 자연어 요청에 따라 LLM으로 자동분석하는 솔루션"
)

# 대화기록을 저장하기 위한 용도로 생성
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ReAct Agent 초기화
if "react_agent" not in st.session_state:
    st.session_state["react_agent"] = None

# include_domains 초기화
if "include_domains" not in st.session_state:
    st.session_state["include_domains"] = []


# 사이드바 생성
with st.sidebar:

    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    st.markdown("made by inwoo")

    # 모델 선택 메뉴
    selected_model = st.selectbox("LLM 선택", ["gpt-4o", "gpt-4o-mini"], index=0)

    # 검색 결과 개수 설정
    search_result_count = st.slider("검색 결과", min_value=1, max_value=10, value=3)

    # include_domains 설정
    st.subheader("검색 도메인 설정")
    search_topic = st.selectbox("검색 주제", ["general", "news"], index=0)
    new_domain = st.text_input("추가할 도메인 입력")
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("도메인 추가", key="add_domain"):
            if new_domain and new_domain not in st.session_state["include_domains"]:
                st.session_state["include_domains"].append(new_domain)

    # 현재 등록된 도메인 목록 표시
    st.write("등록된 도메인 목록:")
    for idx, domain in enumerate(st.session_state["include_domains"]):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text(domain)
        with col2:
            if st.button("삭제", key=f"del_{idx}"):
                st.session_state["include_domains"].pop(idx)
                st.rerun()

    # 설정 버튼
    apply_btn = st.button("설정 완료", type="primary")


# 이전 대화를 출력
def print_messages():
    for message in st.session_state["messages"]:
        if message.msg_type == "text":
            st.chat_message(message.chat_message.role).write(
                message.chat_message.content
            )
        elif message.msg_type == "tool_result":
            with st.expander(f"✅ {message.tool_name}"):
                st.markdown(message.chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message, msg_type="text", tool_name=""):
    if msg_type == "text":
        st.session_state["messages"].append(
            ChatMessageWithType(
                chat_message=ChatMessage(role=role, content=message),
                msg_type="text",
                tool_name=tool_name,
            )
        )
    elif msg_type == "tool_result":
        st.session_state["messages"].append(
            ChatMessageWithType(
                chat_message=ChatMessage(
                    role="assistant", content=format_search_result(message)
                ),
                msg_type="tool_result",
                tool_name=tool_name,
            )
        )


def request_to_llm(user_input):
    """ClickHouse 연결, 정보 추출, FAISS 인덱스 생성, 쿼리 생성 및 실행을 수행합니다."""
    db_driver = deepl.connect_to_clickhouse()
    if not db_driver:
        return

    df_tablemeta = deepl.get_tablemeta(db_driver, deepl.CLICKHOUSE_DATABASE)

    # df_templatequery = deepl.get_templatequery()

    # df_query_exec_list = deepl.execute_query()

    # df_prompt = deepl.set_prompt(df_query_exec_list)

    # retriever = deepl.get_retriever(df_tablemeta, df_templatequery)

    # 자연어 쿼리 예시
    # user_request = "1,2,3번 페이지를 방문한 고객의 명단을 최다방문순 TOP 100개 추출해줘"  # 예시 쿼리. 실제 쿼리는 데이터에 맞게 수정해야 합니다.
    # user_request = "1번 페이지를 방문한 고객의 추천상품을 선정해줘"  # 예시 쿼리. 실제 쿼리는 데이터에 맞게 수정해야 합니다.
    # user_request = "moniClckStream테이블에서 memberid가 10000001인 고객을 분석해줘"  # 예시 쿼리. 실제 쿼리는 데이터에 맞게 수정해야 합니다.

    # ans, request_target, request_subject = split_question(user_request)

    # request_target, request_subject = deepl.split_sentence_AI(user_input)

    # # print(ans)
    # print(request_target)
    # print(request_subject)

    # if True:
    #     clickhouse_query = deepl.generate_query_target(user_input, retriever)
    # # print(f"생성된 ClickHouse 쿼리: {clickhouse_query}")

    # target_list = db_driver.execute(clickhouse_query)
    # result = deepl.analysis_target(target_list)
    # # df = pd.DataFrame(result)
    # print(result.content)

    # try:
    #     target_list = db_driver.execute(clickhouse_query)
    #     result = deepl.analysis_target(target_list)
    #     # df = pd.DataFrame(result)
    #     print(result)
    # except Exception as e:
    #     print(f"쿼리 실행 오류: {e}")

    # else:
    #     return
    return df_tablemeta  # , df_templatequery, df_query_exec_list, df_prompt


# 환경변수 초기화 호출
deepl.init_env()

# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []
    st.session_state["thread_id"] = uuid.uuid4()

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

apply_btn = True

# 설정 버튼이 눌리면...
if apply_btn:
    tool = WebSearchTool().create()
    tool.max_results = search_result_count
    tool.include_domains = st.session_state["include_domains"]
    tool.topic = search_topic
    st.session_state["react_agent"] = create_agent_executor(
        model_name=selected_model,
        tools=[tool],
    )
    st.session_state["thread_id"] = uuid.uuid4()

# 만약에 사용자 입력이 들어오면...
if user_input:
    agent = st.session_state["react_agent"]
    # Config 설정

    if agent is not None:
        config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
        # 사용자의 입력
        st.chat_message("user").write(user_input)

        prompt = deepl.set_request_prompt(user_input)

        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            container_messages, tool_args, agent_answer = stream_handler(
                container,
                agent,
                {
                    "messages": [
                        (
                            "human",
                            prompt,
                        ),
                    ]
                },
                config,
            )

            # 대화기록을 저장한다.
            add_message("user", user_input)
            for tool_arg in tool_args:
                add_message(
                    "assistant",
                    tool_arg["tool_result"],
                    "tool_result",
                    tool_arg["tool_name"],
                )
            add_message("assistant", agent_answer)
    else:
        warning_msg.warning("사이드바에서 설정을 완료해주세요.")
