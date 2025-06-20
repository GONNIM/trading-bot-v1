from urllib.parse import urlencode

# import os
import streamlit as st
from style import style_main
from config import MIN_CASH

# Initialize
if "selected_site" not in st.session_state:
    st.session_state.selected_site = None

# Page Config
st.set_page_config(
    page_title="Upbit Live Trading Bot v1", page_icon="🤖", layout="wide"
)

# Header Section
st.markdown(style_main, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* 헤더와 본문 사이 간격 제거 */
    div.block-container {
        padding-top: 1rem;  /* 기본값은 3rem */
    }

    /* 제목 상단 마진 제거 */
    h1 {
        margin-top: 0 !important;
    }

    [data-testid="stSidebarHeader"],
    [data-testid="stSidebarNavItems"],
    [data-testid="stSidebarNavSeparator"] { display: none !important; }
    div.stButton > button, div.stForm > form > button {
        height: 60px !important;
        font-size: 30px !important;
        font-weight: 900 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize
if "virtual_krw" not in st.session_state:
    st.session_state.virtual_krw = 0

if "virtual_over" not in st.session_state:
    st.session_state.virtual_over = False

st.title("🤖 Upbit Live Trading Bot v1")

# st.write("현재 작업 디렉토리:", os.getcwd())

st.subheader("🔧 가상 보유자산 설정")
with st.form("input_form"):
    cash = st.number_input(
        "가상 보유자산(KRW)", 10_000, 100_000_000_000, 1_000_000, 10_000
    )
    str_submitted = "🧪 테스트 모드 가상 보유자산 설정하기"
    submitted = st.form_submit_button(str_submitted, use_container_width=True)

if submitted:
    if MIN_CASH > cash:
        st.error(
            f"설정한 가상 보유자산이 최소주문가능금액({MIN_CASH} KRW)보다 작습니다."
        )
        st.stop()

    st.session_state.virtual_krw = cash
    st.session_state.virtual_over = True

if st.session_state.virtual_over:
    st.subheader("가상 보유자산")
    st.info(f"{st.session_state.virtual_krw:.0f} KRW")

    st.subheader("테스트 모드 실행")
    if st.button(
        "Upbit Live Trading Bot v1 테스트 모드 실행하기", use_container_width=True
    ):
        params = urlencode({"virtual_krw": st.session_state.virtual_krw})
        st.markdown(
            f"""
            <meta http-equiv="refresh" content="0; url=./app_live?{params}">
        """,
            unsafe_allow_html=True,
        )
        st.switch_page("pages/app_live.py")
