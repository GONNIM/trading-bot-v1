import streamlit as st

from style import style_main


# Initialize
if "selected_site" not in st.session_state:
    st.session_state.selected_site = None

# Page Config
st.set_page_config(
    page_title="Upbit Live Trading Bot v1", page_icon="🤖", layout="wide"
)

st.title("🤖 Upbit Live Trading Bot v1")

# Header Section
st.markdown(style_main, unsafe_allow_html=True)

# Chatbot List Section
sites = {
    "BACKTEST-v1": {
        "description": "📈 업비트 자동매매 백테스트 - MACD, EMA",
        "link": "https://llm-trading.streamlit.app/",
    },
    "BACKTEST-v2": {
        "description": "📈 업비트 자동매매 백테스트 - MACD, EMA v2",
        "link": "https://llm-trading-v2.streamlit.app/",
    },
}

selected_site = None
columns = st.columns(len(sites))

for i, (site, info) in enumerate(sites.items()):
    if columns[i].button(site, key=site):
        st.session_state.selected_site = site
        st.rerun()

if st.session_state.selected_site:
    selected_site = st.session_state.selected_site
    site_info = sites[selected_site]

    site_card = f"""
    <div class="site-card">
        <div class="site-name">{selected_site}</div>
        <div class="site-description">{site_info['description']}</div>
    </div>
    """
    if not site_info["link"].startswith("https://"):
        st.markdown(site_card, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.write("")
            st.write("")
            if st.button(f"Open {selected_site}", key=f"open_{selected_site}"):
                st.switch_page(site_info["link"])
    else:
        site_card = f"""
        <div class="site-card">
            <div class="site-name">{selected_site}</div>
            <div class="site-description">{site_info['description']}</div>
            <a href="{site_info['link']}" target="_blank" class="site-link">Open {selected_site}</a>
        </div>
        """
        st.markdown(site_card, unsafe_allow_html=True)
else:
    site_card = f"""
    <div class="site-card">
        <div class="site-name">상단의 사이트를 선택해 주세요!</div>
        <div class="site-description">대박이 납니다!!!</div>
        <!-- <a href="#" target="_self" class="site-link">🤖 Upbit Live Trading Bot</a> //-->
    </div>
    """
    st.markdown(site_card, unsafe_allow_html=True)

# Footer
st.write("")
st.markdown(
    '<div class="footer"><p>© Upbit Live Trading Bot</p></div>', unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    [data-testid="stSidebarContent"] {display:none;}<br>
    </style>
    """,
    unsafe_allow_html=True,
)
