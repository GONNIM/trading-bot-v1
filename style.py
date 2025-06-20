style_main = """
<style>
/* ────────────────────────────────────────────────
   🔒 Streamlit 사이드바 완전 숨김
   (data-testid 선택자를 쓰면 버전 변동에 가장 안전함)
──────────────────────────────────────────────── */
[data-testid="stSidebar"],          /* 사이드바 컨테이너 */
[data-testid="stSidebarNav"] {      /* 내비 항목 */
    display: none !important;
}

/* 사이드바가 사라져도 남을 수 있는 left-margin 제거 */
.css-1d391kg { margin-left: 0rem !important; }

[data-testid="stMainBlockContainer"] { max-width: 80rem; }

/* ────────────────────────────────────────────────
   아래는 기존 스타일
──────────────────────────────────────────────── */
.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 0;
    background: #f8f9fa;
    border-bottom: 1px solid #e9ecef;
    margin-bottom: 2rem;
}
.header input {
    flex-grow: 1;
    margin: 0 1.25rem;
    padding: 0.9rem; /* 조금 더 크게 */
    border: 1px solid #ced4da;
    border-radius: 0.5rem;
    background: #fff;
    font-size: 1.2rem; /* 20% 증가 */
    font-weight: 500; /* 굵기 증가 */
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    transition: border-color 0.2s, box-shadow 0.2s;
}
.header input:focus {
    border-color: #4e9af1;
    box-shadow: 0 0 0 0.2rem rgba(78,154,241,0.25);
    outline: none;
}
.header a {
    margin-left: 1rem;
    color: #4e9af1;
    text-decoration: none; /* 밑줄 제거 */
    font-weight: 500;
    font-size: 1.2rem; /* 20% 증가 */
}
.footer {
    text-align: center;
    padding: 1.8rem 0; /* 살짝 크게 */
    margin-top: 2.4rem; /* 살짝 크게 */
    background: #f8f9fa;
    border-top: 1px solid #e9ecef;
    color: #6c757d;
    font-size: 1.08rem; /* 20% 증가 */
    font-weight: 500; /* 굵기 증가 */
}
.site-card {
    border: 1px solid #e9ecef;
    border-radius: 0.8rem;
    padding: 1.8rem; /* 살짝 크게 */
    text-align: center;
    background: #fff;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: box-shadow 0.3s, transform 0.2s;
    margin-top: 1.8rem; /* 살짝 크게 */
}
.site-card:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    transform: translateY(-2px);
}
.site-icon {
    width: 4.8rem; /* 살짝 크게 */
    height: 4.8rem;
    margin: 0 auto;
    border-radius: 50%;
    background: #f1f8ff;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #4e9af1;
    font-size: 2.16rem; /* 20% 증가 */
    font-weight: 600; /* 굵기 증가 */
}
.site-name {
    font-size: 1.8rem; /* 20% 증가 */
    margin: 1.2rem 0; /* 살짝 크게 */
    color: #212529;
    font-weight: 700; /* 굵기 증가 */
}
.site-description {
    font-size: 1.2rem; /* 20% 증가 */
    color: #6c757d;
    margin-bottom: 1.8rem; /* 살짝 크게 */
    font-weight: 500; /* 굵기 증가 */
}
.site-link,
.site-link:link,
.site-link:visited,
.site-link:hover,
.site-link:active {
    display: inline-block;
    margin-top: 1.2rem;
    padding: 0.9rem 1.8rem;
    background-color: #4e9af1;
    color: #fff;
    text-decoration: none !important;
    border-radius: 0.6rem;
    font-weight: 600;
    font-size: 1.2rem;
    transition: background-color 0.2s, transform 0.2s;
}
.site-link:hover {
    background-color: #3a7ac9;
    transform: translateY(-1px);
}

.stButton>button {
    width: 100% !important;
    height: 10rem !important;
    border-radius: 1.25rem !important;
    border: none !important;
    background: #4e9af1 !important;
    color: white !important;
    font-weight: 900 !important;
    font-size: 50rem !important; /* 실제로 눈에 띄게 크게 */
    transition: background-color 0.2s, transform 0.2s !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: visible !important;
    white-space: normal !important;
    line-height: 1.2 !important;
}
.stButton>button:hover {
    background: #3a7ac9 !important;
    transform: translateY(-2px) !important;
    font-weight: 900;
}

/* --------- 반응형 --------- */
@media (max-width: 48rem) {
    .header { flex-direction: column; padding: 1.2rem 0; }
    .header input { margin: 0.6rem 0; width: 90%; }
    .site-card { padding: 1.2rem; margin-top: 1.2rem; }
    .site-name { font-size: 1.44rem; }
    .site-description { font-size: 1.08rem; }
    .site-link { padding: 0.72rem 1.44rem; }
}
@media (max-width: 30rem) {
    .header { padding: 0.96rem 0; }
    .site-card { padding: 0.96rem; margin-top: 0.96rem; }
    .site-name { font-size: 1.2rem; }
    .site-description { font-size: 0.96rem; }
    .site-link { padding: 0.6rem 1.2rem; }
}
</style>
"""
