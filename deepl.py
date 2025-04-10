import os
import re
import pandas as pd
from dotenv import load_dotenv

import clickhouse_driver
import openai

# API 키를 환경변수로 관리하기 위한 설정 파일
from langchain_openai import ChatOpenAI

# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.chains import create_sql_query_chain
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from langchain_teddynote import logging
from langchain_community.utilities import SQLDatabase
from langchain.document_loaders import DataFrameLoader
from langchain_core.prompts import load_prompt
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1)

# 환경 변수에서 ClickHouse 접속 정보 가져오기
CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST")
CLICKHOUSE_PORTNATIVE = os.getenv("CLICKHOUSE_PORTNATIVE")  # 기본 포트 9000 (NATIVE)
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD")
CLICKHOUSE_DATABASE = os.getenv("CLICKHOUSE_DATABASE")


# # 환경변수 초기화
def init_env():

    load_dotenv()

    # llm API 설정
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

    # LangSmith 추적을 설정합니다. https://smith.langchain.com
    # 프로젝트 이름을 입력합니다.


#    logging.langsmith("DeepLens-01")
# openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)


def connect_to_clickhouse():
    """ClickHouse에 연결하고 연결 객체를 반환합니다."""
    try:
        driver = clickhouse_driver.Client(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORTNATIVE,
            database=CLICKHOUSE_DATABASE,
            user=CLICKHOUSE_USER,
            password=CLICKHOUSE_PASSWORD,
        )
        return driver
    except Exception as e:
        print(f"ClickHouse 연결 오류: {e}")
        return None


def get_tablemeta():
    """ClickHouse 연결, 정보 추출, FAISS 인덱스 생성, 쿼리 생성 및 실행을 수행합니다."""
    db_driver = connect_to_clickhouse()
    if not db_driver:
        return

    """ClickHouse 데이터베이스의 테이블 및 컬럼 정보를 가져옵니다."""
    try:
        # 테이블 목록 가져오기
        table_query = f"""
            SELECT a.name AS table_name,
                CAST(b.table_comment AS String) table_comment 
            FROM system.tables a 
            INNER JOIN information_schema.tables b 
                on a.database = '{CLICKHOUSE_DATABASE}'
            AND a.database = b.table_schema 
            AND a.name IN ('moniClckStream', 'userProfile', 'page-metadata')
            AND a.name = b.table_name
        """
        tables = db_driver.execute(table_query)

        table_meta = []
        for table in tables:
            table_name, table_comment = table
            table_comment = (
                table_comment.decode("utf-8", errors="replace")
                if isinstance(table_comment, bytes)
                else table_comment
            )

            columns_query = f"""
            SELECT column_name as column_name, data_type as column_type, CAST(column_comment AS String) as column_comment
            FROM information_schema.columns
            WHERE table_schema = '{CLICKHOUSE_DATABASE}' AND table_name = '{table_name}'
            """
            columns = db_driver.execute(columns_query)

            for col in columns:
                column_name, column_type, column_comment = col
                column_comment = (
                    column_comment.decode("utf-8", errors="replace")
                    if isinstance(column_comment, bytes)
                    else column_comment
                )
                table_meta.append(
                    {
                        "table_name": table_name,
                        "table_comment": table_comment if table_comment else "",
                        "column_name": column_name,
                        "column_comment": column_comment if column_comment else "",
                        "column_type": column_type,
                        # "description": f"Table: {table_name}, Column: {column_name}, Type: {column_type}"
                        "description": f"Table: {table_name} ({table_comment}), Column: {column_name} ({column_comment}), Type: {column_type}",
                    }
                )

        return pd.DataFrame(table_meta)

    except Exception as e:
        print(f"테이블 정보 가져오기 오류: {e}")
        return None


def get_templatequery():

    try:
        # 사전 정의된 Sample Queries 및 한글 설명 추가
        sample_template_queries = [
            {
                "query": """
                SELECT T1.accesstime,
                    toDateTime(T1.accesstime) action_time,
                    T1.pcid,
                    T1.nthsessionid,
                    T1.url,
                    T1.pageid,
                    T1.pagetitle,
                    T1.tagtype,
                    T1.clicktext
                FROM moniClckStream T1
                LEFT OUTER JOIN page_metadata T2
                    ON T1.pageid = T2.page_id
                AND T1.siteid = T2.site_id
               -- LEFT OUTER JOIN userProfile T3
               --     ON T1.memberid = T3.memberId
                WHERE T1.pagetitle LIKE '%%'    
                -- AND T1.memberid = '10000001'
                AND toDate(T1.accesstime) between '2025-04-01' AND today()
                AND T1.siteid = 'KNBANK'
                LIMIT 30
                ;    
            """,
                "description": "특정페이지를 방문한 고객의 행동데이터 추출 쿼리",
            }
        ]

        return pd.DataFrame(sample_template_queries)

    except Exception as e:
        print(f"Template Query 등록 에러: {e}")
        return None


def execute_query(input_query):
    """ClickHouse 연결, 정보 추출, FAISS 인덱스 생성, 쿼리 생성 및 실행을 수행합니다."""
    db_driver = connect_to_clickhouse()
    if not db_driver:
        return

    try:
        query_exec_list = db_driver.execute(input_query)
        # result = analysis_target(target_list)
        # df = pd.DataFrame(result)
        # print(query_exec_list)
        return query_exec_list
    except Exception as e:
        print(f"쿼리 실행 오류: {e}")

    else:
        return None


def set_request_prompt(user_input):

    prompt_template = load_prompt("prompts/analysis1.yaml", encoding="utf-8")

    df_query_script = get_templatequery()

    table_meta = get_tablemeta()

    query_result = execute_query(df_query_script["query"][0])

    prompt = prompt_template.format(
        table_meta=table_meta,
        query_script=df_query_script,
        query_result=query_result,
        user_input=user_input,
    )

    return prompt


def analysis_target(input_query_result):
    # LLM을 사용하여 쿼리 결과 분석 요청

    analysis_prompt = """
    당신은 다양한 분야에서 고객의 행동수집하고 의미있는 행동을 찾아내서 마케팅성과를 끌어내는 전세계에서 가장 유능한 고객행동데이터 AI분석가입니다.
    다음은 특정 고객의 행동데이터로 이 데이터를 요약하고, [FORMAT]을 참고하여 사용자 행동 패턴을 분석해 주세요.

    #조회결과:
    {query_result}

    #FORMAT:
    - 전체건수:
    - 주요방문페이지:
    - 시간별 체류시간 및 총체류시간(시간은 datetime_action 정보이용):
    - 이동경로 순서도요약(순서,페이지,방문시간):
    - 고객추천상품(최빈도순위3개):
    """

    prompt = PromptTemplate.from_template(analysis_prompt)

    # LLM을 사용하여 결과 요약 및 분석 요청
    chain = prompt | llm

    input = {"query_result": input_query_result}

    analysis_result = chain.invoke(input)

    return analysis_result
    # # print("\n🔹 LLM 분석 결과:\n", analysis_result)


def get_retriever(df_tablemeta, df_templatequery):
    """테이블 및 컬럼 정보를 FAISS에 저장합니다."""
    combined_df = pd.concat([df_tablemeta, df_templatequery], ignore_index=True)

    loader = DataFrameLoader(combined_df, page_content_column="description")

    docs = loader.load()
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    # retriever = vectorstore.as_retriever()

    return vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )  # 유사도 검색을 위한 k값 설정


def split_question(question):
    """
    주어진 질문을 '분석할 대상의 조건'과 '분석 주제'로 나눕니다.

    지원하는 패턴:
    1) "A의 B" -> "VIP 고객의 평균 구매 금액을 알려줘"
    2) "A한 고객의 B" -> "회원 가입한 고객의 재방문율을 분석해줘"
    3) "A에 대한 B" -> "회원 등급에 대한 구매 패턴을 분석해줘"
    4) "A인 고객 B" -> "고객번호가 11111인 고객의 구매 패턴을 분석해줘"

    :param question: 사용자가 입력한 질문
    :return: (대상 조건, 분석 주제)
    """

    patterns = [
        (re.compile(r"(.+?)의 (.+)"), 1, 2),  # "A의 B"
        (re.compile(r"(.+?한 고객)의 (.+)"), 1, 2),  # "A한 고객의 B"
        (re.compile(r"(.+?)에 대한 (.+)"), 1, 2),  # "A에 대한 B"
        (re.compile(r"(.+?)인 고객 (.+)"), 1, 2),  # "A인 고객 B"
    ]

    for pattern, target_idx, subject_idx in patterns:
        match = pattern.match(question)
        if match:
            analysis_target = match.group(target_idx).strip()
            analysis_subject = match.group(subject_idx).strip()
            ans = True
            return ans, analysis_target, analysis_subject

    # 분리가 안 될 경우 전체 질문을 그대로 반환

    return False, question, ""


# def split_sentence_AI(user_input):
#     """
#     사용자의 질문을 GPT-4o를 통해 분석하여
#     1. 고객 추출 조건
#     2. 결과 요청 사항
#     두 개의 요약 문장으로 분리합니다.
#     """

#     prompt = f"""
#     사용자가 입력한 문장을 두 개의 문장으로 요약하세요:
#     1. 고객 추출 조건 (예: 고객을 필터링하는 기준)
#     2. 결과 요청 사항 (예: 고객 데이터를 분석하는 작업)

#     예제 입력: "고객번호가 111인 고객의 추천상품을 선정해줘"
#     예제 출력:
#     고객 추출 조건: 고객번호가 111인 고객
#     결과 요청 사항: 추천상품을 선정해줘

#     입력: "{user_input}"
#     출력:
#     """

#     # ✅ 최신 OpenAI SDK 방식
#     response = openai_client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "당신은 사용자의 질문을 분석하여 데이터를 정리하는 AI입니다.",
#             },
#             {"role": "user", "content": prompt},
#         ],
#     )

#     # 🔹 GPT 응답에서 결과 추출
#     result_text = response.choices[0].message.content
#     lines = result_text.split("\n")

#     # 🔹 결과값 초기화
#     condition, request = "", ""

#     for line in lines:
#         if "고객 추출 조건:" in line:
#             condition = line.replace("고객 추출 조건:", "").strip()
#         elif "결과 요청 사항:" in line:
#             request = line.replace("결과 요청 사항:", "").strip()

#     return condition, request
