from pathlib import Path
from datetime import date
import base64
import html
import mimetypes
import uuid

import pandas as pd
import streamlit as st

from src.matching_engine import TfidfMatcher, TFIDF_THRESHOLD
from src.evaluation import evaluate
from src.database import (
    init_db,
    add_report,
    get_reports,
    get_found_reports_for_matcher,
    UPLOAD_DIR,
)
from src.auth import initialize_auth_db, register_user, verify_user

st.set_page_config(
    page_title="ReuniteAI — Lost Items, New Beginnings",
    page_icon="🔑",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PROJECT_ROOT = Path(__file__).resolve().parent
ASSET_DIR = PROJECT_ROOT / "assets"

@st.cache_data(show_spinner=False)
def image_data_uri(path_string: str) -> str:
    """Return an embedded image URI for safe inline HTML artwork."""
    path = Path(path_string)
    if not path.is_file():
        return ""
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"

PROMO_ART_URI = image_data_uri(str(ASSET_DIR / "reuniteai_scene_floating_objects.jpg"))

# ---- Premium chocolate / gold / amber theme ----
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700;800&display=swap');
    :root { --gold:#f6c15e; --amber:#ff9e2c; --cream:#fff0d5; --muted:#d7c0a2; }
    .stApp {
      background: radial-gradient(ellipse at 78% 7%,rgba(175,91,28,.25),transparent 34%),
                  radial-gradient(ellipse at 8% 48%,rgba(113,59,25,.20),transparent 35%),
                  linear-gradient(135deg,#140b06 0%,#2c170c 48%,#170b05 100%);
      color:var(--cream);font-family:'DM Sans',sans-serif;
    }
    [data-testid="stHeader"]{background:rgba(20,11,6,.72)}
    .block-container{max-width:1480px;padding-top:1rem;padding-bottom:2.4rem}
    h1,h2,h3{font-family:'Playfair Display',Georgia,serif!important;color:var(--cream)!important}
    p,label,.stMarkdown{color:var(--cream)}
    .topbar{display:flex;align-items:center;justify-content:space-between;padding:.35rem 0 1rem;margin-bottom:1rem;border-bottom:1px solid #f6c15e38;animation:fadeDown .65s ease both}
    .brand{font-family:'Playfair Display',Georgia,serif;font-size:1.9rem;font-weight:800;letter-spacing:-.04em;color:#fff0d5}
    .brand span{color:var(--gold)}
    .brand-mark{display:inline-flex;width:39px;height:39px;align-items:center;justify-content:center;border:1px solid #f6c15e99;color:var(--gold);border-radius:50%;margin-right:.55rem;font-size:1.2rem}
    .topnote{color:#d7c0a2;font-size:.76rem;letter-spacing:.22em;text-transform:uppercase}
    .eyebrow,.section-label{font-size:.72rem;letter-spacing:.28em;text-transform:uppercase;color:var(--gold);margin-bottom:.8rem}
    .hero-title{font-family:'Playfair Display',Georgia,serif;font-size:clamp(2.8rem,5.2vw,5.4rem);line-height:.92;letter-spacing:-.055em;font-weight:800;color:#fff0d5;margin:.2rem 0 1.25rem}
    .hero-title em{font-style:normal;color:#f6bd54}
    .hero-sub{max-width:570px;color:#e0cbb0;line-height:1.7;font-size:1.04rem}
    .gold-rule{height:2px;width:74px;background:linear-gradient(90deg,#f6c15e,transparent);margin:1.2rem 0}
    .hero-image{border-radius:22px;overflow:hidden;border:1px solid #f6c15e42;box-shadow:0 22px 60px #0007}
    .hero-image img{border-radius:22px}
    .microline{color:#d7c0a2;font-size:.82rem;margin-top:.9rem}
    .stat-row{display:flex;gap:1.5rem;flex-wrap:wrap;margin-top:1.5rem}
    .stat{border-left:1px solid #f6c15e52;padding-left:.85rem}
    .stat strong{display:block;font-size:1.05rem;color:#ffcf71}.stat span{font-size:.77rem;color:#d7c0a2}
    .feature-card{height:100%;min-height:130px;padding:1.1rem;border:1px solid #f6c15e30;border-radius:17px;background:linear-gradient(145deg,#5a321b80,#1c1009c9);transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease;animation:rise .7s ease both}
    .feature-card:hover{transform:translateY(-5px);border-color:#f6c15eaa;box-shadow:0 12px 28px #0004}
    .feature-icon{font-size:1.65rem;margin-bottom:.5rem}.feature-name{font-weight:700;color:#fff0d5;margin-bottom:.35rem}.feature-desc{font-size:.86rem;line-height:1.5;color:#d7c0a2}
    .step{border-top:1px solid #f6c15e38;padding:1rem .2rem}.step-num{display:inline-flex;align-items:center;justify-content:center;width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#ffe29a,#e99a2b);color:#321b0e;font-weight:800;margin-bottom:.7rem}
    .stButton>button{background:linear-gradient(110deg,#ffdc80,#f3a12e)!important;color:#291608!important;border:1px solid #ffe19b!important;border-radius:999px!important;font-weight:800!important;min-height:3rem;box-shadow:0 7px 22px #e9952b26;transition:transform .2s ease,box-shadow .2s ease,filter .2s ease}
    .stButton>button:hover{transform:translateY(-3px);box-shadow:0 12px 30px #e9952b55;filter:brightness(1.04)}
    [data-testid="stTextArea"] textarea,[data-testid="stTextInput"] input,[data-testid="stSelectbox"] div[data-baseweb="select"]>div,[data-testid="stDateInput"] input{background:#21130b!important;color:#fff0d5!important;border:1px solid #956536!important;border-radius:12px!important}
    [data-testid="stTabs"] button{color:#d7c0a2!important;font-weight:700!important}
    [data-testid="stTabs"] button[aria-selected="true"]{color:#ffcc6b!important}
    [data-testid="stTabs"] [data-baseweb="tab-highlight"]{background:#f6c15e!important}
    [data-testid="stAlert"]{border-radius:13px}
    .footer{border-top:1px solid #f6c15e30;padding-top:1rem;margin-top:2rem;color:#d7c0a2;font-size:.82rem}

    /* Motion and depth: deliberately subtle so the app remains comfortable to use. */
    .stApp:before{content:"";position:fixed;inset:-18%;z-index:0;pointer-events:none;
      background:radial-gradient(circle at 18% 28%,#f6c15e0a 0,transparent 26%),
                 radial-gradient(circle at 78% 62%,#ff9e2c0c 0,transparent 30%);
      animation:ambientGlow 16s ease-in-out infinite alternate}
    [data-testid="stMain"] .block-container{position:relative;animation:pageEnter .55s cubic-bezier(.2,.75,.25,1) both}
    [data-testid="stTabs"] [data-testid="stTabContent"]{animation:fadeUp .38s ease both}
    [data-testid="stMetric"]{padding:1rem 1.1rem;border:1px solid #f6c15e30;border-radius:16px;
      background:linear-gradient(145deg,#5a321b55,#1c1009a8);transition:transform .22s ease,border-color .22s ease}
    [data-testid="stMetric"]:hover{transform:translateY(-3px);border-color:#f6c15e88}
    [data-testid="stMetricLabel"]{color:#d7c0a2!important}
    [data-testid="stMetricValue"]{color:#ffcf71!important;font-weight:800}
    .feature-card:nth-child(2){animation-delay:80ms}.feature-card:nth-child(3){animation-delay:150ms}.feature-card:nth-child(4){animation-delay:220ms}

    .match-section-head{display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap;
      padding:.85rem 1rem;margin:.8rem 0 1rem;border:1px solid #f6c15e38;border-radius:14px;
      background:linear-gradient(110deg,#5a321b77,#1c1009a8);color:#fff0d5}
    .match-section-head small{color:#d7c0a2}
    .match-card{height:100%;min-height:220px;position:relative;overflow:hidden;padding:1.1rem 1.1rem 1rem;
      margin:0 0 1rem;border:1px solid #f6c15e40;border-radius:18px;
      background:linear-gradient(145deg,#392012eF 0%,#201109ef 68%,#180c07f2 100%);
      box-shadow:0 12px 30px #0002;animation:cardReveal .55s cubic-bezier(.2,.75,.25,1) both;
      animation-delay:var(--card-delay,0ms);transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease}
    .match-card:before{content:"";position:absolute;top:0;left:0;right:0;height:2px;
      background:linear-gradient(90deg,transparent,#f6c15e, #ff9e2c,transparent);opacity:.8}
    .match-card:hover{transform:translateY(-4px);border-color:#f6c15eaa;box-shadow:0 18px 38px #0005}
    .match-card.featured-match{border-color:#f6c15e99;box-shadow:0 12px 32px #f6c15e0b,0 12px 30px #0003}
    .match-top{display:flex;align-items:flex-start;gap:.75rem;margin-bottom:.9rem}
    .match-icon{display:flex;align-items:center;justify-content:center;flex:0 0 48px;height:48px;
      border:1px solid #f6c15e55;border-radius:14px;background:linear-gradient(145deg,#f6c15e22,#ff9e2c0b);font-size:1.55rem}
    .match-heading{flex:1;min-width:0}
    .match-rank{font-size:.67rem;color:#f6c15e;letter-spacing:.16em;text-transform:uppercase;margin-bottom:.25rem}
    .match-title{font-size:1.07rem;line-height:1.35;font-weight:800;color:#fff0d5;overflow-wrap:anywhere}
    .match-id{font-size:.75rem;color:#bfa587;margin-top:.3rem}
    .match-badge{white-space:nowrap;font-size:.68rem;font-weight:800;letter-spacing:.02em;padding:.33rem .55rem;border-radius:999px;
      background:#f6c15e19;border:1px solid #f6c15e55;color:#ffcf71}
    .match-badge.low{background:#d7c0a20c;border-color:#d7c0a233;color:#d7c0a2}
    .match-details{font-size:.87rem;line-height:1.58;color:#e0cbb0;overflow-wrap:anywhere}
    .match-meta{display:flex;flex-wrap:wrap;gap:.45rem .9rem;font-size:.76rem;color:#d7c0a2;margin:.8rem 0}
    .match-meta span{display:inline-block}
    .match-score-row{display:flex;justify-content:space-between;align-items:center;gap:.7rem;margin:.9rem 0 .42rem;font-size:.75rem;color:#d7c0a2}
    .match-score-row strong{color:#ffcf71;font-size:.9rem}
    .match-progress{height:6px;border-radius:99px;overflow:hidden;background:#fff0d512;border:1px solid #f6c15e15}
    .match-progress-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,#c47b27,#ffdf8a,#ff9e2c);
      box-shadow:0 0 12px #f6c15e35;animation:meterFill .9s cubic-bezier(.2,.75,.25,1) both;transform-origin:left}
    .match-note{margin:.8rem 0 0;padding:.8rem 1rem;border-left:2px solid #f6c15e88;border-radius:0 10px 10px 0;background:#f6c15e0a;
      font-size:.78rem;line-height:1.55;color:#d7c0a2}
    .search-tip{border:1px dashed #f6c15e55;border-radius:14px;padding:.9rem 1rem;color:#d7c0a2;background:#f6c15e08}
    .footer strong{letter-spacing:.02em}

    @keyframes rise{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
    @keyframes fadeDown{from{opacity:0;transform:translateY(-9px)}to{opacity:1;transform:translateY(0)}}
    @keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
    @keyframes pageEnter{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:translateY(0)}}
    @keyframes cardReveal{from{opacity:0;transform:translateY(12px) scale(.99)}to{opacity:1;transform:translateY(0) scale(1)}}
    @keyframes meterFill{from{transform:scaleX(.03)}to{transform:scaleX(1)}}
    @keyframes ambientGlow{from{transform:translate3d(-1%,0,0) scale(1)}to{transform:translate3d(1%,1%,0) scale(1.035)}}

    /* Phase 4: clearly visible cinematic motion, plus a reduced-motion fallback. */
    .brand-mark{animation:brandPulse 3.2s ease-in-out infinite;box-shadow:0 0 0 0 #f6c15e00}
    .hero-title{animation:heroReveal .9s cubic-bezier(.18,.72,.22,1) both}
    .hero-sub{animation:fadeUp .9s .12s ease both}
    .hero-image{animation:heroFloat 7s ease-in-out infinite;transform-origin:center;}
    .auth-stage{position:relative;overflow:hidden;isolation:isolate;min-height:220px;margin:.5rem 0 1rem;padding:1.5rem 1.4rem 1.4rem;
      border:1px solid #f6c15e48;border-radius:24px;background:linear-gradient(135deg,#5b321c99,#231108eF 58%,#160a05f5);
      box-shadow:0 24px 70px #0005;animation:authPanelReveal .9s cubic-bezier(.18,.72,.22,1) both}
    .auth-stage:before{content:"";position:absolute;inset:-40% -10% auto auto;width:280px;height:280px;border:1px solid #f6c15e30;border-radius:50%;
      box-shadow:0 0 0 22px #f6c15e08,0 0 0 48px #f6c15e05;animation:orbitDrift 18s linear infinite;z-index:-1}
    .auth-stage:after{content:"";position:absolute;inset:auto auto -95px -45px;width:235px;height:235px;border-radius:50%;
      background:radial-gradient(circle,#ff9e2c2b,transparent 68%);filter:blur(4px);animation:orbGlow 5s ease-in-out infinite alternate;z-index:-1}
    .auth-eyebrow{font-size:.7rem;letter-spacing:.25em;text-transform:uppercase;color:#ffcf71;margin-bottom:.7rem}
    .auth-title{font-family:'Playfair Display',Georgia,serif;font-size:clamp(2.1rem,3.4vw,3.5rem);line-height:.98;letter-spacing:-.05em;color:#fff0d5;font-weight:800;margin:0 0 .8rem;animation:fadeUp .9s .15s both}
    .auth-title em{font-style:normal;color:#f6bd54}
    .auth-copy{color:#e0cbb0;line-height:1.7;max-width:520px;font-size:.95rem;animation:fadeUp .9s .26s both}
    .auth-chip-row{display:flex;gap:.55rem;flex-wrap:wrap;margin-top:1.1rem}
    .auth-chip{padding:.45rem .7rem;border:1px solid #f6c15e42;border-radius:99px;background:#140b06a8;color:#ffdf9b;font-size:.72rem;animation:chipRise .75s both}
    .auth-chip:nth-child(2){animation-delay:.12s}.auth-chip:nth-child(3){animation-delay:.24s}
    [data-testid="stForm"]{padding:1.05rem 1.1rem;border:1px solid #f6c15e35;border-radius:18px;background:linear-gradient(145deg,#43261565,#180d07aa);animation:authPanelReveal .75s .12s both}
    [data-testid="stTextInput"] input{transition:border-color .25s ease,box-shadow .25s ease,transform .25s ease}
    [data-testid="stTextInput"] input:focus{border-color:#ffcf71!important;box-shadow:0 0 0 3px #f6c15e20!important;transform:translateY(-1px)}
    .signed-in-strip{display:flex;align-items:center;gap:.5rem;padding:.65rem .9rem;margin:.1rem 0 .7rem;border:1px solid #f6c15e30;border-radius:12px;
      background:linear-gradient(90deg,#4d2b1866,#1b0e07a8);color:#e0cbb0;font-size:.82rem;animation:fadeUp .45s ease both}
    .signed-in-strip strong{color:#ffcf71}
    @keyframes heroReveal{from{opacity:0;transform:translateY(24px) scale(.985);filter:blur(4px)}to{opacity:1;transform:translateY(0) scale(1);filter:blur(0)}}
    @keyframes authPanelReveal{from{opacity:0;transform:translateY(22px) scale(.985)}to{opacity:1;transform:translateY(0) scale(1)}}
    @keyframes chipRise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
    @keyframes heroFloat{0%,100%{transform:translateY(0) rotate(0deg)}50%{transform:translateY(-7px) rotate(.25deg)}}
    @keyframes brandPulse{0%,100%{box-shadow:0 0 0 0 #f6c15e00}50%{box-shadow:0 0 0 6px #f6c15e0e}}
    @keyframes orbitDrift{from{transform:rotate(0deg) translateX(8px)}to{transform:rotate(360deg) translateX(8px)}}
    @keyframes orbGlow{from{transform:scale(.92);opacity:.6}to{transform:scale(1.12);opacity:1}}
    @media(max-width:850px){.auth-title{font-size:2.4rem}.auth-stage{padding:1.1rem;min-height:0}.auth-copy{font-size:.9rem}}

    @media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}}
    @media(max-width:750px){.topnote{display:none}.hero-title{font-size:3rem}.stat-row{gap:1rem}.match-card{min-height:0}.match-badge{font-size:.63rem}}


    /* Phase 5: diagonal split-panel sign-in/sign-up inspired by the reference clip. */
    .stApp [data-testid="stHorizontalBlock"]:has(.auth-promo) {
      gap:0 !important; align-items:stretch !important; overflow:hidden !important;
      border-radius:28px; border:1px solid #f6c15e66;
      box-shadow:0 28px 85px #0007, 0 0 0 1px #fff0d50a;
      background:#fff0d5; margin:.35rem 0 1.2rem;
      animation:authSceneArrive .78s cubic-bezier(.2,.78,.2,1) both;
    }
    .stApp [data-testid="stHorizontalBlock"]:has(.auth-promo) > [data-testid="stColumn"] {
      min-width:0; padding:0 !important; flex:1 1 50% !important;
    }
    .stApp [data-testid="stHorizontalBlock"]:has(.auth-promo) [data-testid="stElementContainer"] {
      width:100%; margin:0 !important;
    }
    .stApp [data-testid="stColumn"]:has(.auth-promo) {background:#211108; padding:0 !important;}
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) {
      background:linear-gradient(145deg,#fff8e9 0%,#fff0d5 72%,#f7e0b6 100%) !important;
      padding:2.1rem 2rem 1.8rem !important; display:flex; flex-direction:column; justify-content:center;
      min-height:530px; box-sizing:border-box; color:#321b0e !important;
      animation:authFormEnter .8s cubic-bezier(.18,.72,.22,1) both;
    }
    .auth-promo {
      position:relative; display:flex; flex-direction:column; justify-content:center;
      box-sizing:border-box; isolation:isolate; overflow:hidden; min-height:530px;
      height:100%; padding:2.6rem 2.2rem 2.2rem;
      color:#fff0d5; background:
        radial-gradient(ellipse at 80% 14%,#ff9e2c25,transparent 35%),
        radial-gradient(ellipse at 7% 90%,#f6c15e17,transparent 42%),
        linear-gradient(145deg,#38200f 0%,#211108 57%,#140b06 100%);
    }
    .auth-promo:before {
      content:""; position:absolute; z-index:-1; width:340px;height:340px;right:-130px;top:-110px;
      border:1px solid #f6c15e42;border-radius:50%;
      box-shadow:0 0 0 24px #f6c15e0d,0 0 0 56px #f6c15e08,0 0 0 92px #f6c15e05;
      animation:authOrbit 22s linear infinite;
    }
    .auth-promo:after {
      content:""; position:absolute; z-index:0; pointer-events:none; top:-12%; bottom:-12%; width:34%;
      right:-22%; transform:skewX(-9deg);
      background:linear-gradient(90deg,#f6c15e00,#f6c15e24 44%,#fff0d51c 72%,#fff0d500);
      animation:diagonalWipe 1.12s cubic-bezier(.2,.75,.2,1) both;
    }
    .auth-promo.promo-right:before{right:auto;left:-135px;top:-95px;}
    .auth-promo.promo-right:after{right:auto;left:-22%;animation-name:diagonalWipeReverse;}
    .auth-promo.promo-left{border-radius:27px 0 0 27px;}
    .auth-promo.promo-right{border-radius:0 27px 27px 0;}
    .auth-promo-content{position:relative;z-index:1;max-width:420px;animation:authWordsIn .82s .12s cubic-bezier(.18,.72,.22,1) both;}
    .auth-promo-kicker{color:#f6c15e;font-size:.7rem;font-weight:800;letter-spacing:.24em;text-transform:uppercase;margin-bottom:1rem;}
    .auth-promo-title{font-family:'Playfair Display',Georgia,serif!important;font-size:clamp(2.5rem,4vw,4.15rem);font-weight:800;line-height:.95;letter-spacing:-.055em;color:#fff0d5;margin:0 0 1.1rem;}
    .auth-promo-title em{font-style:normal;color:#f6c15e;}
    .auth-promo-copy{font-size:.92rem;line-height:1.75;color:#e0cbb0;max-width:360px;margin:0;}
    .auth-promo-rule{width:72px;height:2px;margin:1.3rem 0;background:linear-gradient(90deg,#f6c15e,transparent);}
    .auth-promo-pills{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:1.25rem;}
    .auth-promo-pill{border:1px solid #f6c15e4d;border-radius:99px;padding:.42rem .65rem;background:#140b0670;color:#ffe1a1;font-size:.7rem;white-space:nowrap;}
    .auth-floating-items{display:flex;gap:.7rem;margin-top:1.8rem;align-items:center;}
    .auth-floating-item{display:flex;align-items:center;justify-content:center;width:48px;height:48px;border:1px solid #f6c15e55;border-radius:16px;background:linear-gradient(145deg,#f6c15e20,#ff9e2c09);font-size:1.35rem;box-shadow:0 10px 30px #0002;animation:authFloat 4s ease-in-out infinite;}
    .auth-floating-item:nth-child(2){animation-delay:.55s}.auth-floating-item:nth-child(3){animation-delay:1.1s}
    .auth-form-marker{height:0;overflow:hidden;}
    .auth-form-kicker{font-size:.68rem;letter-spacing:.22em;text-transform:uppercase;color:#976522;font-weight:800;margin:.1rem 0 .6rem;}
    .auth-form-title{font-family:'Playfair Display',Georgia,serif;font-size:2.45rem;line-height:1.05;letter-spacing:-.04em;color:#2c170c;margin:0 0 .5rem;font-weight:800;}
    .auth-form-subtitle{font-size:.88rem;line-height:1.55;color:#75563b;margin:0 0 1.35rem;}
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) .stMarkdown,
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) .stMarkdown p,
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) label,
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) [data-testid="stCaptionContainer"] {
      color:#49301d !important;
    }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) [data-testid="stForm"] {
      border:0 !important; background:transparent !important; box-shadow:none !important;
      padding:0 !important; animation:authFormElementsIn .7s .12s ease both;
    }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) [data-testid="stTextInput"] input {
      color:#321b0e !important; background:#fffaf0 !important; border:1px solid #c8a36b !important;
      border-radius:11px !important; min-height:2.8rem; transition:border-color .25s ease,box-shadow .25s ease,transform .25s ease;
    }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) [data-testid="stTextInput"] input:focus {
      border-color:#99621f !important; box-shadow:0 0 0 3px #f6c15e38 !important; transform:translateY(-1px);
    }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) [data-testid="stTextInput"] input::placeholder {color:#a28b70 !important;}
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) .stButton>button,
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) [data-testid="stFormSubmitButton"] button {
      background:linear-gradient(110deg,#4a2815,#211108) !important; color:#fff0d5 !important;
      border:1px solid #956536 !important; border-radius:999px !important; min-height:2.9rem;
      box-shadow:0 8px 20px #321b0e20 !important; letter-spacing:.01em;
    }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) .stButton>button:hover,
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) [data-testid="stFormSubmitButton"] button:hover {
      background:linear-gradient(110deg,#5c341b,#2c170c) !important; border-color:#f6c15e !important;
      transform:translateY(-2px); box-shadow:0 12px 28px #321b0e32 !important;
    }
    .auth-toggle-note{color:#896847!important;font-size:.78rem;line-height:1.55;margin:.7rem 0 .4rem;}
    .auth-footnote{color:#bfa587;font-size:.72rem;text-align:center;line-height:1.5;margin:.9rem 0 .3rem;}
    @keyframes authSceneArrive{from{opacity:0;transform:translateY(18px) scale(.99);filter:blur(3px)}to{opacity:1;transform:translateY(0) scale(1);filter:blur(0)}}
    @keyframes authFormEnter{from{opacity:.45;transform:translateX(22px)}to{opacity:1;transform:translateX(0)}}
    @keyframes authWordsIn{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
    @keyframes authFormElementsIn{from{opacity:0;transform:translateY(13px)}to{opacity:1;transform:translateY(0)}}
    @keyframes diagonalWipe{0%{right:-38%;opacity:.05}45%{opacity:1}100%{right:-22%;opacity:.9}}
    @keyframes diagonalWipeReverse{0%{left:-38%;opacity:.05}45%{opacity:1}100%{left:-22%;opacity:.9}}
    @keyframes authOrbit{from{transform:rotate(0deg) translateX(5px)}to{transform:rotate(360deg) translateX(5px)}}
    @keyframes authFloat{0%,100%{transform:translateY(0) rotate(0)}50%{transform:translateY(-7px) rotate(2deg)}}
    @media(max-width:850px){
      .stApp [data-testid="stHorizontalBlock"]:has(.auth-promo){flex-direction:column!important;border-radius:22px;}
      .stApp [data-testid="stColumn"]:has(.auth-form-marker){min-height:0;padding:1.6rem 1.3rem!important;}
      .auth-promo{min-height:360px;padding:1.8rem 1.4rem;}
      .auth-promo-title{font-size:2.8rem}.auth-promo.promo-left,.auth-promo.promo-right{border-radius:0;}
      .auth-form-title{font-size:2rem;}
    }



    /* Phase 6: deeper, silkier, more cinematic motion in the existing chocolate / gold palette. */
    .stApp [data-testid="stHorizontalBlock"]:has(.auth-promo) {
      perspective: 1800px !important;
      transform-style: preserve-3d !important;
      will-change: transform, opacity, filter;
      animation: phase6SceneBloom 1.35s cubic-bezier(.16,1,.3,1) both !important;
      box-shadow: 0 35px 100px #0008, 0 0 0 1px #f6c15e22, 0 0 90px #ff9e2c10 !important;
    }
    .stApp [data-testid="stHorizontalBlock"]:has(.auth-promo) > [data-testid="stColumn"] {
      min-width: 0 !important;
      transform-style: preserve-3d !important;
      backface-visibility: hidden;
      will-change: transform, opacity, filter, clip-path;
    }
    .stApp [data-testid="stColumn"]:has(.auth-promo.mode-signin) {
      animation: phase6PromoFromRight 1.45s cubic-bezier(.16,1,.3,1) both !important;
    }
    .stApp [data-testid="stColumn"]:has(.auth-promo.mode-signup) {
      animation: phase6PromoFromLeft 1.45s cubic-bezier(.16,1,.3,1) both !important;
    }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker.mode-signin) {
      animation: phase6FormFromLeft 1.35s .05s cubic-bezier(.16,1,.3,1) both !important;
    }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker.mode-signup) {
      animation: phase6FormFromRight 1.35s .05s cubic-bezier(.16,1,.3,1) both !important;
    }
    .auth-promo {
      isolation: isolate !important;
      overflow: hidden !important;
      background-size: 190% 190%, 180% 180%, 160% 160%, 100% 100% !important;
      animation: phase6AuroraFlow 13s ease-in-out infinite alternate !important;
    }
    .auth-promo::before {
      width: 420px !important; height: 420px !important;
      border: 1px solid #f6c15e55 !important;
      box-shadow: 0 0 0 19px #f6c15e0c, 0 0 0 48px #f6c15e08,
                  0 0 0 82px #f6c15e05, 0 0 70px #ff9e2c12 !important;
      animation: phase6Orbit 28s linear infinite, phase6RingPulse 6s ease-in-out infinite alternate !important;
    }
    .auth-promo::after {
      content: "" !important;
      top: -25% !important; bottom: -25% !important;
      width: 68% !important; right: auto !important; left: -95% !important;
      opacity: 0 !important;
      transform: skewX(-19deg) translateX(-30%) !important;
      background: linear-gradient(90deg, transparent 0%, #f6c15e08 20%, #f6c15e30 48%, #fff0d548 58%, #ff9e2c25 70%, transparent 100%) !important;
      filter: blur(2px) !important;
      animation: phase6LightSail 8.5s cubic-bezier(.65,0,.35,1) 1s infinite !important;
    }
    .auth-promo.promo-right::after { left: -95% !important; right: auto !important; }
    .auth-promo.promo-left::after { left: -95% !important; right: auto !important; animation-name: phase6LightSailReverse !important; }
    .auth-promo-content {
      transform-style: preserve-3d !important;
      will-change: transform, opacity, filter;
      animation: phase6ContentFloatIn 1.2s .18s cubic-bezier(.16,1,.3,1) both !important;
    }
    .auth-promo-kicker { animation: phase6LetterReveal .8s .22s both; }
    .auth-promo-title {
      text-wrap: balance;
      text-shadow: 0 8px 42px #0004;
      animation: phase6TitleIn 1.15s .16s cubic-bezier(.16,1,.3,1) both;
    }
    .auth-promo-copy { animation: phase6CopyIn 1s .34s cubic-bezier(.16,1,.3,1) both; }
    .auth-promo-rule {
      transform-origin: left center;
      animation: phase6RuleDraw 1.1s .48s cubic-bezier(.16,1,.3,1) both;
      box-shadow: 0 0 16px #f6c15e20;
    }
    .auth-promo-pills .auth-promo-pill {
      transition: transform .45s cubic-bezier(.16,1,.3,1), background .45s ease, border-color .45s ease, box-shadow .45s ease;
      animation: phase6PillDrift .85s .38s cubic-bezier(.16,1,.3,1) both;
    }
    .auth-promo-pill:nth-child(2) { animation-delay: .5s; }
    .auth-promo-pill:hover {
      transform: translateY(-4px) scale(1.035);
      background: #f6c15e18; border-color: #f6c15e99; box-shadow: 0 8px 22px #0002;
    }
    .auth-floating-item {
      position: relative !important;
      transform-style: preserve-3d !important;
      transition: transform .55s cubic-bezier(.16,1,.3,1), border-color .4s ease, box-shadow .4s ease !important;
      animation: phase6ObjectFloat 5.6s ease-in-out infinite !important;
    }
    .auth-floating-item:nth-child(1) { animation-delay: -.4s !important; }
    .auth-floating-item:nth-child(2) { animation-delay: -2.2s !important; }
    .auth-floating-item:nth-child(3) { animation-delay: -3.8s !important; }
    .auth-floating-item:hover {
      transform: translateY(-10px) translateZ(34px) rotateY(14deg) rotateX(8deg) scale(1.12) !important;
      border-color: #ffe29a !important; box-shadow: 0 22px 35px #0005, 0 0 26px #f6c15e20 !important;
    }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) {
      position: relative !important;
      overflow: hidden !important;
      isolation: isolate !important;
      box-shadow: inset 1px 0 #ffffff99, -12px 0 50px #00008 !important;
    }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker)::before {
      content: ""; position: absolute; inset: 0; pointer-events: none; z-index: 0;
      background: radial-gradient(ellipse at 10% 0%, #f6c15e27, transparent 42%),
                  radial-gradient(ellipse at 100% 100%, #ff9e2c19, transparent 38%);
      animation: phase6PaperGlow 9s ease-in-out infinite alternate;
    }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) > div { position: relative; z-index: 1; }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) .auth-form-title {
      animation: phase6FormTitle 1s .22s cubic-bezier(.16,1,.3,1) both !important;
    }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) .auth-form-subtitle {
      animation: phase6FormSubtitle 1s .32s cubic-bezier(.16,1,.3,1) both !important;
    }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) [data-testid="stTextInput"] {
      animation: phase6FieldAppear .75s .34s cubic-bezier(.16,1,.3,1) both;
    }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) [data-testid="stTextInput"]:nth-of-type(2) { animation-delay: .43s; }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) [data-testid="stTextInput"]:nth-of-type(3) { animation-delay: .52s; }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) [data-testid="stTextInput"]:nth-of-type(4) { animation-delay: .61s; }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) [data-testid="stFormSubmitButton"] button,
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) .stButton>button {
      position: relative !important; overflow: hidden !important; isolation: isolate !important;
      transition: transform .5s cubic-bezier(.16,1,.3,1), box-shadow .5s ease, filter .5s ease !important;
    }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) [data-testid="stFormSubmitButton"] button::before,
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) .stButton>button::before {
      content: ""; position: absolute; z-index: -1; top: -60%; bottom: -60%; left: -80%; width: 42%;
      background: linear-gradient(90deg, transparent, #fff0d566, transparent); transform: skewX(-22deg);
      transition: left .75s cubic-bezier(.16,1,.3,1);
    }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) [data-testid="stFormSubmitButton"] button:hover::before,
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) .stButton>button:hover::before { left: 145%; }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) [data-testid="stFormSubmitButton"] button:hover,
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) .stButton>button:hover {
      transform: translateY(-3px) scale(1.015) !important; filter: brightness(1.06);
      box-shadow: 0 16px 32px #321b0e28, 0 0 0 1px #f6c15e55 !important;
    }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) [data-testid="stTextInput"] input {
      transition: border-color .4s ease, box-shadow .4s ease, transform .4s cubic-bezier(.16,1,.3,1), background .4s ease !important;
    }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) [data-testid="stTextInput"] input:focus {
      transform: translateY(-2px) scale(1.008) !important; background: #fffdf8 !important;
      box-shadow: 0 0 0 4px #f6c15e26, 0 10px 25px #321b0e10 !important;
    }
    .auth-toggle-note { animation: phase6SoftUp .75s .55s both; }
    @keyframes phase6SceneBloom {
      0% { opacity:0; transform:translateY(22px) scale(.965) rotateX(2deg); filter:blur(12px); clip-path:inset(8% 0 8% 0 round 30px); }
      55% { opacity:1; filter:blur(0); clip-path:inset(0 0 0 0 round 28px); }
      100% { opacity:1; transform:translateY(0) scale(1) rotateX(0); filter:blur(0); clip-path:inset(0 0 0 0 round 28px); }
    }
    @keyframes phase6PromoFromRight {
      0% { opacity:0; transform:translate3d(95px,0,-80px) rotateY(-13deg) scale(.93); filter:blur(12px); clip-path:polygon(18% 0,100% 0,100% 100%,0 100%); }
      65% { opacity:1; filter:blur(0); clip-path:polygon(0 0,100% 0,100% 100%,0 100%); }
      100% { opacity:1; transform:translate3d(0,0,0) rotateY(0) scale(1); filter:blur(0); clip-path:polygon(0 0,100% 0,100% 100%,0 100%); }
    }
    @keyframes phase6PromoFromLeft {
      0% { opacity:0; transform:translate3d(-95px,0,-80px) rotateY(13deg) scale(.93); filter:blur(12px); clip-path:polygon(0 0,82% 0,100% 100%,0 100%); }
      65% { opacity:1; filter:blur(0); clip-path:polygon(0 0,100% 0,100% 100%,0 100%); }
      100% { opacity:1; transform:translate3d(0,0,0) rotateY(0) scale(1); filter:blur(0); clip-path:polygon(0 0,100% 0,100% 100%,0 100%); }
    }
    @keyframes phase6FormFromLeft {
      0% { opacity:0; transform:translate3d(-70px,14px,65px) rotateY(9deg) scale(.94); filter:blur(10px); clip-path:polygon(0 0,100% 0,100% 100%,17% 100%); }
      68% { opacity:1; filter:blur(0); clip-path:polygon(0 0,100% 0,100% 100%,0 100%); }
      100% { opacity:1; transform:translate3d(0,0,0) rotateY(0) scale(1); filter:blur(0); clip-path:polygon(0 0,100% 0,100% 100%,0 100%); }
    }
    @keyframes phase6FormFromRight {
      0% { opacity:0; transform:translate3d(70px,14px,65px) rotateY(-9deg) scale(.94); filter:blur(10px); clip-path:polygon(0 0,83% 0,100% 100%,0 100%); }
      68% { opacity:1; filter:blur(0); clip-path:polygon(0 0,100% 0,100% 100%,0 100%); }
      100% { opacity:1; transform:translate3d(0,0,0) rotateY(0) scale(1); filter:blur(0); clip-path:polygon(0 0,100% 0,100% 100%,0 100%); }
    }
    @keyframes phase6AuroraFlow { 0%{background-position:0% 50%,80% 10%,10% 90%,0 0} 50%{background-position:100% 20%,10% 70%,90% 10%,0 0} 100%{background-position:20% 100%,70% 20%,0% 60%,0 0} }
    @keyframes phase6Orbit { from{transform:rotate(0deg) translateX(8px)} to{transform:rotate(360deg) translateX(8px)} }
    @keyframes phase6RingPulse { from{opacity:.65;filter:blur(0)} to{opacity:1;filter:blur(1px)} }
    @keyframes phase6LightSail { 0%,12%{left:-95%;opacity:0} 17%{opacity:.95} 42%{left:145%;opacity:0} 100%{left:145%;opacity:0} }
    @keyframes phase6LightSailReverse { 0%,12%{left:145%;opacity:0} 17%{opacity:.95} 42%{left:-95%;opacity:0} 100%{left:-95%;opacity:0} }
    @keyframes phase6ContentFloatIn { 0%{opacity:0;transform:translate3d(0,22px,55px) scale(.96);filter:blur(7px)} 70%{opacity:1;filter:blur(0)} 100%{opacity:1;transform:translate3d(0,0,0) scale(1);filter:blur(0)} }
    @keyframes phase6TitleIn { 0%{opacity:0;transform:translateY(28px) rotateX(-12deg);filter:blur(9px);letter-spacing:-.08em} 100%{opacity:1;transform:translateY(0) rotateX(0);filter:blur(0);letter-spacing:-.055em} }
    @keyframes phase6CopyIn { 0%{opacity:0;transform:translateY(15px)} 100%{opacity:1;transform:translateY(0)} }
    @keyframes phase6RuleDraw { 0%{transform:scaleX(0);opacity:0} 100%{transform:scaleX(1);opacity:1} }
    @keyframes phase6PillDrift { 0%{opacity:0;transform:translateY(12px) rotateX(25deg)} 100%{opacity:1;transform:translateY(0) rotateX(0)} }
    @keyframes phase6ObjectFloat { 0%,100%{transform:translate3d(0,0,0) rotateY(0) rotateZ(0)} 33%{transform:translate3d(0,-9px,14px) rotateY(5deg) rotateZ(-2deg)} 66%{transform:translate3d(0,3px,4px) rotateY(-4deg) rotateZ(2deg)} }
    @keyframes phase6PaperGlow { from{opacity:.45;transform:scale(1)} to{opacity:1;transform:scale(1.08)} }
    @keyframes phase6FormTitle { 0%{opacity:0;transform:translateX(15px) rotateY(-7deg);filter:blur(5px)} 100%{opacity:1;transform:translateX(0) rotateY(0);filter:blur(0)} }
    @keyframes phase6FormSubtitle { 0%{opacity:0;transform:translateY(10px)} 100%{opacity:1;transform:translateY(0)} }
    @keyframes phase6FieldAppear { 0%{opacity:0;transform:translateY(13px) scale(.985);filter:blur(3px)} 100%{opacity:1;transform:translateY(0) scale(1);filter:blur(0)} }
    @keyframes phase6LetterReveal { 0%{opacity:0;letter-spacing:.42em;transform:translateX(-8px)} 100%{opacity:1;letter-spacing:.24em;transform:translateX(0)} }
    @keyframes phase6SoftUp { 0%{opacity:0;transform:translateY(7px)} 100%{opacity:1;transform:translateY(0)} }
    @media (max-width:850px) {
      .stApp [data-testid="stHorizontalBlock"]:has(.auth-promo) { perspective:none !important; }
      .stApp [data-testid="stHorizontalBlock"]:has(.auth-promo) > [data-testid="stColumn"] { animation:none !important; }
      .auth-promo::after { animation-duration:11s !important; }
    }
    @media (prefers-reduced-motion:reduce) {
      .stApp [data-testid="stHorizontalBlock"]:has(.auth-promo),
      .stApp [data-testid="stHorizontalBlock"]:has(.auth-promo) > [data-testid="stColumn"],
      .auth-promo, .auth-promo::before, .auth-promo::after, .auth-promo-content,
      .auth-promo-title, .auth-promo-copy, .auth-floating-item,
      .stApp [data-testid="stColumn"]:has(.auth-form-marker)::before { animation:none !important; }
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Phase 8: sunlit sandstone / caramel / bronze visual direction ----
st.markdown(
    """
    <style>
    :root {
      --sun-cream:#FFF3DF; --ivory:#F9EBD5; --sand:#E6C9A0;
      --honey:#DCA465; --bronze:#956F4D; --caramel:#B8753D;
      --chestnut:#593218; --ink:#493629; --muted:#806B57;
      --line:#D9B98A; --paper:#FFF9EF;
    }
    .stApp {
      background:
        radial-gradient(ellipse at 84% 3%, rgba(220,164,101,.38), transparent 28%),
        radial-gradient(ellipse at 5% 37%, rgba(247,209,157,.42), transparent 34%),
        linear-gradient(135deg, #F6E7CE 0%, #FFF5E5 45%, #EBD0A8 100%) !important;
      color:var(--ink) !important; font-family:'DM Sans',sans-serif !important;
    }
    [data-testid="stAppViewContainer"], [data-testid="stMain"] { background:transparent !important; }
    [data-testid="stHeader"], [data-testid="stToolbar"] { background:rgba(249,235,213,.6) !important; }
    .block-container { max-width:1480px; padding-top:1rem; padding-bottom:2.6rem; }
    h1,h2,h3,h4 { color:var(--chestnut) !important; }
    .stApp .stMarkdown { color:var(--ink); }
    .stApp p, .stApp label, .stApp [data-testid="stCaptionContainer"] { color:var(--ink); }
    .topbar { border-bottom:1px solid #956F4D55 !important; }
    .brand { color:#593218 !important; }
    .brand span { color:#B8753D !important; }
    .brand-mark { color:#956F4D !important; border-color:#B8753D99 !important; background:#FFF4E1AA; }
    .topnote, .microline { color:#806B57 !important; }
    .eyebrow, .section-label { color:#956F4D !important; }
    .hero-title { color:#593218 !important; text-shadow:0 2px 18px #FFF9EF66; }
    .hero-title em { color:#B8753D !important; }
    .hero-sub { color:#71553F !important; }
    .gold-rule { background:linear-gradient(90deg,#DCA465,transparent) !important; }
    .hero-image { border:1px solid #B88B5B77 !important; box-shadow:0 22px 55px #704a271e,0 3px 10px #704a2711 !important; }
    .hero-image img { border-radius:22px !important; }
    .stat { border-left:1px solid #956F4D80 !important; }
    .stat strong { color:#87502D !important; }
    .stat span { color:#806B57 !important; }

    .feature-card {
      border:1px solid #D9B98A !important; border-radius:20px !important;
      background:linear-gradient(145deg,#FFFAF1 0%,#F5E3C8 100%) !important;
      box-shadow:0 9px 26px #75502e12 !important;
    }
    .feature-card:hover { border-color:#B8753D !important; box-shadow:0 17px 32px #75502e24 !important; }
    .feature-icon { filter:drop-shadow(0 4px 5px #8e5c2d25); }
    .feature-name { color:#593218 !important; }
    .feature-desc { color:#806B57 !important; }
    .step { border-top-color:#D9B98A !important; }
    .step-num { background:linear-gradient(135deg,#F7D19D,#DCA465) !important; color:#593218 !important; }

    .stButton>button, [data-testid="stFormSubmitButton"] button {
      background:linear-gradient(115deg,#A66B3D 0%,#704326 100%) !important;
      color:#FFF4E4 !important; border:1px solid #B17A4B !important; border-radius:999px !important;
      font-weight:800 !important; min-height:2.9rem; box-shadow:0 8px 22px #70432624 !important;
      transition:transform .28s cubic-bezier(.2,.75,.25,1),box-shadow .28s ease,filter .28s ease !important;
    }
    .stButton>button:hover, [data-testid="stFormSubmitButton"] button:hover {
      background:linear-gradient(115deg,#BA7D4B 0%,#80502F 100%) !important;
      transform:translateY(-3px) !important; box-shadow:0 15px 30px #70432632 !important; filter:brightness(1.03);
    }
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] div[data-baseweb="select"]>div,
    [data-testid="stDateInput"] input, [data-testid="stNumberInput"] input {
      background:#FFFAF2 !important; color:#493629 !important; border:1px solid #C7A477 !important;
      border-radius:12px !important;
    }
    [data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus,
    [data-testid="stDateInput"] input:focus {
      border-color:#A66B3D !important; box-shadow:0 0 0 3px #DCA46535 !important;
    }
    [data-testid="stTabs"] button { color:#806B57 !important; font-weight:700 !important; }
    [data-testid="stTabs"] button[aria-selected="true"] { color:#704326 !important; }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:#B8753D !important; }
    [data-testid="stAlert"] { border-radius:15px !important; background:#FFF9EF !important; }
    [data-testid="stMetric"] {
      padding:1rem 1.1rem; border:1px solid #D9B98A !important; border-radius:18px !important;
      background:linear-gradient(145deg,#FFFAF1,#F3DFC0) !important; box-shadow:0 8px 23px #70432612;
    }
    [data-testid="stMetricLabel"] { color:#806B57 !important; }
    [data-testid="stMetricValue"] { color:#87502D !important; font-weight:800 !important; }
    [data-testid="stDataFrame"] { border:1px solid #D9B98A !important; border-radius:14px !important; overflow:hidden; }
    [data-testid="stExpander"] { border-color:#D9B98A !important; border-radius:14px !important; background:#FFF9EF88 !important; }
    [data-testid="stImage"] img { border-radius:16px !important; }
    .footer { border-top:1px solid #D9B98A !important; color:#806B57 !important; }
    .footer strong { color:#956F4D !important; }

    /* Authentication: keep a cinematic bronze feature panel, but let the page breathe. */
    .stApp [data-testid="stHorizontalBlock"]:has(.auth-promo) {
      border:1px solid #CFAE7E !important; background:#FFF3DF !important;
      box-shadow:0 26px 72px #77522a21,0 3px 12px #77522a0d !important;
    }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) {
      background:linear-gradient(145deg,#FFFAF0 0%,#F7E6CA 76%,#F0D5AA 100%) !important;
      color:#493629 !important;
    }
    .auth-promo {
      background:
        radial-gradient(ellipse at 84% 14%,#F7D19D55,transparent 36%),
        linear-gradient(145deg,#8B6242 0%,#654930 56%,#593218 100%) !important;
      color:#FFF3DF !important;
    }
    .auth-promo-art {
      position:absolute !important; inset:0 !important; width:100% !important; height:100% !important;
      object-fit:cover !important; object-position:65% center !important; z-index:0 !important;
      opacity:.42 !important; pointer-events:none !important; mix-blend-mode:screen !important;
      filter:saturate(.8) contrast(.94) !important; animation:phase8ArtFloat 16s ease-in-out infinite alternate !important;
      mask-image:linear-gradient(to bottom,rgba(0,0,0,.35),rgba(0,0,0,.92)) !important;
    }
    .auth-promo-content { z-index:2 !important; }
    .auth-promo-kicker { color:#F7D19D !important; }
    .auth-promo-title { color:#FFF3DF !important; text-shadow:0 3px 28px #2D1E1B40; }
    .auth-promo-title em { color:#F7D19D !important; }
    .auth-promo-copy { color:#F7E8D3 !important; }
    .auth-promo-rule { background:linear-gradient(90deg,#F7D19D,transparent) !important; }
    .auth-promo-pill { border-color:#F7D19D66 !important; background:#59321863 !important; color:#FFF0D4 !important; }
    .auth-floating-item { border-color:#F7D19D88 !important; background:linear-gradient(145deg,#FFF3DF35,#DCA46535) !important; box-shadow:0 12px 28px #2D1E1B26 !important; }
    .auth-form-kicker { color:#956F4D !important; }
    .auth-form-title { color:#593218 !important; }
    .auth-form-subtitle, .auth-toggle-note { color:#806B57 !important; }
    .auth-footnote { color:#806B57 !important; }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) .stMarkdown,
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) .stMarkdown p,
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) label,
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) [data-testid="stCaptionContainer"] { color:#493629 !important; }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) [data-testid="stTextInput"] input {
      color:#493629 !important; background:#FFFCF7 !important; border-color:#C7A477 !important;
    }
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) .stButton>button,
    .stApp [data-testid="stColumn"]:has(.auth-form-marker) [data-testid="stFormSubmitButton"] button {
      background:linear-gradient(115deg,#A66B3D,#704326) !important; color:#FFF3DF !important; border-color:#B17A4B !important;
    }

    .match-section-head { background:linear-gradient(110deg,#F7E4C7,#FFF9EF) !important; border-color:#D9B98A !important; color:#593218 !important; }
    .match-section-head small { color:#806B57 !important; }
    .match-card { background:linear-gradient(145deg,#FFFCF6 0%,#F6E6CE 100%) !important; border-color:#D9B98A !important; box-shadow:0 12px 30px #75502e14 !important; }
    .match-card:before { background:linear-gradient(90deg,transparent,#DCA465,#956F4D,transparent) !important; }
    .match-card:hover { border-color:#B8753D !important; box-shadow:0 19px 38px #75502e23 !important; }
    .match-icon { border-color:#D9B98A !important; background:linear-gradient(145deg,#F7D19D77,#DCA46522) !important; }
    .match-rank { color:#956F4D !important; }
    .match-title { color:#593218 !important; }
    .match-id, .match-details, .match-meta, .match-score-row, .match-note { color:#806B57 !important; }
    .match-badge { background:#E6C9A066 !important; border-color:#C7A477 !important; color:#704326 !important; }
    .match-badge.low { background:#F3E7D4 !important; border-color:#D9B98A !important; color:#806B57 !important; }
    .match-score-row strong { color:#87502D !important; }
    .match-progress { background:#E6C9A0 !important; border-color:#D9B98A !important; }
    .match-progress-fill { background:linear-gradient(90deg,#956F4D,#DCA465,#F7D19D) !important; box-shadow:0 0 12px #B8753D32 !important; }
    .match-note { border-left-color:#B8753D !important; background:#DCA46515 !important; }
    .search-tip { border-color:#C7A477 !important; color:#806B57 !important; background:#FFF9EF !important; }

    .visual-gallery { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1rem; margin:.8rem 0 1.6rem; }
    .visual-card { overflow:hidden; border-radius:20px; border:1px solid #D9B98A; background:linear-gradient(150deg,#FFFCF6,#F3DFC0); box-shadow:0 10px 28px #75502e12; transition:transform .45s cubic-bezier(.2,.75,.25,1),box-shadow .45s ease,border-color .35s ease; animation:phase8CardArrive .8s cubic-bezier(.2,.75,.25,1) both; }
    .visual-card:nth-child(2) { animation-delay:.12s; }
    .visual-card:nth-child(3) { animation-delay:.24s; }
    .visual-card:hover { transform:translateY(-7px) perspective(800px) rotateX(1deg) rotateY(-1deg); border-color:#B8753D; box-shadow:0 20px 36px #75502e25; }
    .visual-card img { display:block; width:100%; height:205px; object-fit:cover; transition:transform 1.1s cubic-bezier(.16,1,.3,1),filter .8s ease; }
    .visual-card:hover img { transform:scale(1.045); filter:saturate(1.07); }
    .visual-card-copy { padding:.9rem 1rem 1rem; }
    .visual-card-kicker { color:#956F4D; text-transform:uppercase; letter-spacing:.18em; font-size:.62rem; font-weight:800; margin-bottom:.35rem; }
    .visual-card-title { color:#593218; font-family:'Playfair Display',Georgia,serif; font-size:1.12rem; font-weight:700; margin-bottom:.3rem; }
    .visual-card-description { color:#806B57; line-height:1.5; font-size:.8rem; }
    @keyframes phase8ArtFloat { from { transform:scale(1.02) translate3d(-.5%,0,0); } to { transform:scale(1.09) translate3d(.8%,-.6%,0); } }
    @keyframes phase8CardArrive { from { opacity:0; transform:translateY(17px) scale(.985); filter:blur(3px); } to { opacity:1; transform:translateY(0) scale(1); filter:blur(0); } }
    @media(max-width:850px) {
      .visual-gallery { grid-template-columns:1fr; }
      .visual-card img { height:230px; }
      .auth-promo-art { opacity:.28 !important; }
    }
    @media(prefers-reduced-motion:reduce) { .visual-card,.visual-card img,.auth-promo-art { animation:none !important; transition:none !important; } }
    </style>
    """,
    unsafe_allow_html=True,
)

def render_match_card(row, rank: int) -> None:
    """Render a polished, safely escaped candidate card; score is text similarity only."""
    raw_score = row.get("score", 0.0)
    try:
        score = float(raw_score) if not pd.isna(raw_score) else 0.0
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))
    percent = round(score * 100)

    category = str(row.get("category", "other") or "other").strip()
    category_key = category.lower()
    icons = {
        "accessories": "🧢", "bag": "🎒", "books": "📚", "bottle": "🧴",
        "clothing": "🧥", "container": "🥡", "electronics": "🎧",
        "id card": "🪪", "keys": "🔑", "stationery": "✏️",
        "umbrella": "☂️", "other": "📦", "wallet": "👛", "phone": "📱",
    }
    icon = icons.get(category_key, "📦")
    if score >= max(TFIDF_THRESHOLD * 2, 0.35):
        label, badge_class = "Higher text similarity", ""
    elif score >= TFIDF_THRESHOLD:
        label, badge_class = "Possible text similarity", ""
    else:
        label, badge_class = "Low text similarity", "low"

    def safe(value, fallback="Not provided"):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            value = ""
        value = str(value).strip()
        return html.escape(value if value else fallback)

    item_name = safe(row.get("item_name", ""), "Unnamed item")
    report_id = safe(row.get("report_id", ""), "No report ID")
    description_raw = str(row.get("description", "") or "").strip()
    if len(description_raw) > 260:
        description_raw = description_raw[:257].rstrip() + "..."
    description = safe(description_raw, "No description provided.")
    location = safe(row.get("location", ""), "Location not specified")
    report_date = row.get("date_found", "")
    if not str(report_date or "").strip():
        report_date = row.get("report_date", "")
    report_date = safe(report_date, "Date not provided")
    category_safe = safe(category, "Other")
    rank_label = f"POSSIBLE MATCH {rank:02d}"
    featured = " featured-match" if rank == 1 else ""
    card_delay = min((rank - 1) * 65, 390)

    st.markdown(
        f'''<div class="match-card{featured}" style="--card-delay:{card_delay}ms">
          <div class="match-top">
            <div class="match-icon">{icon}</div>
            <div class="match-heading">
              <div class="match-rank">{rank_label}</div>
              <div class="match-title">{item_name}</div>
              <div class="match-id">Report ID · {report_id}</div>
            </div>
            <div class="match-badge {badge_class}">{label}</div>
          </div>
          <div class="match-details">{description}</div>
          <div class="match-meta"><span>◈ {category_safe}</span><span>⌖ {location}</span><span>◷ {report_date}</span></div>
          <div class="match-score-row"><span>Text similarity</span><strong>{percent}%</strong></div>
          <div class="match-progress"><div class="match-progress-fill" style="width:{percent}%"></div></div>
        </div>''',
        unsafe_allow_html=True,
    )


# Create tables if needed. Existing report records are retained during migration.
init_db()
initialize_auth_db()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

brand_markup = '<div class="topbar"><div><span class="brand-mark">♡</span><span class="brand">Reunite<span>AI</span></span></div><div class="topnote">Lost items · New beginnings</div></div>'

# Show a cinematic split-panel sign-in/sign-up experience before the dashboard.
current_user = st.session_state.get("auth_user")
if not current_user:
    st.markdown(brand_markup, unsafe_allow_html=True)
    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "signin"

    auth_mode = st.session_state.get("auth_mode", "signin")
    login_pressed = False
    signup_pressed = False

    with st.container():
        auth_left, auth_right = st.columns([1, 1], gap="small", vertical_alignment="center")

        def render_promo(side: str, mode: str) -> None:
            if mode == "signin":
                kicker = "REUNITEAI MEMBERS"
                title = "Welcome<br><em>back.</em>"
                copy = "Your reports, your search, your reunions — all in one place. Sign in to pick up where you left off."
                chips = '<span class="auth-promo-pill">✦ Your reports</span><span class="auth-promo-pill">◈ Smart matching</span>'
            else:
                kicker = "A SMALL SEARCH. A BIG REUNION."
                title = "Start the<br><em>first page.</em>"
                copy = "Create your account to report a lost item, share a found belonging, and help a story find its way home."
                chips = '<span class="auth-promo-pill">✦ Free demo account</span><span class="auth-promo-pill">◈ Community powered</span>'
            st.markdown(f'''<div class="auth-promo promo-{side} mode-{mode}">
              <img class="auth-promo-art" src="{PROMO_ART_URI}" alt="A warm still life of everyday belongings" aria-hidden="true" />
              <div class="auth-promo-content">
                <div class="auth-promo-kicker">{kicker}</div>
                <div class="auth-promo-title">{title}</div>
                <p class="auth-promo-copy">{copy}</p>
                <div class="auth-promo-rule"></div>
                <div class="auth-promo-pills">{chips}</div>
                <div class="auth-floating-items"><span class="auth-floating-item">🔑</span><span class="auth-floating-item">🎒</span><span class="auth-floating-item">🎧</span></div>
              </div>
            </div>''', unsafe_allow_html=True)

        def render_form_heading(mode: str) -> None:
            if mode == "signin":
                kicker, title, copy = "GOOD TO SEE YOU", "Sign in", "Enter your details to open your ReuniteAI dashboard."
            else:
                kicker, title, copy = "JOIN THE COMMUNITY", "Create account", "Start your account in a few simple steps."
            st.markdown(f'''<div class="auth-form-marker mode-{mode}"></div>
              <div class="auth-form-kicker">{kicker}</div>
              <div class="auth-form-title">{title}</div>
              <div class="auth-form-subtitle">{copy}</div>''', unsafe_allow_html=True)

        if auth_mode == "signin":
            with auth_left:
                render_form_heading("signin")
                with st.form("login_form", clear_on_submit=False):
                    login_email = st.text_input("Email address", placeholder="you@example.com", autocomplete="username", key="login_email")
                    login_password = st.text_input("Password", type="password", placeholder="Enter your password", autocomplete="current-password", key="login_password")
                    login_pressed = st.form_submit_button("Sign in  →", use_container_width=True)
                st.markdown('<div class="auth-toggle-note">New to ReuniteAI? Create a free demo account.</div>', unsafe_allow_html=True)
                if st.button("Create an account  ↗", key="show_signup", use_container_width=True):
                    st.session_state["auth_mode"] = "signup"
                    st.rerun()
            with auth_right:
                render_promo("right", "signin")
        else:
            with auth_left:
                render_promo("left", "signup")
            with auth_right:
                render_form_heading("signup")
                with st.form("signup_form", clear_on_submit=False):
                    signup_name = st.text_input("Full name", placeholder="Your name", key="signup_name")
                    signup_email = st.text_input("Email address", placeholder="you@example.com", autocomplete="email", key="signup_email")
                    signup_password = st.text_input("Create password", type="password", placeholder="At least 8 characters", autocomplete="new-password", key="signup_password")
                    signup_confirm = st.text_input("Confirm password", type="password", placeholder="Enter it again", autocomplete="new-password", key="signup_confirm")
                    signup_pressed = st.form_submit_button("Create my account  ✦", use_container_width=True)
                st.markdown('<div class="auth-toggle-note">Already have a ReuniteAI account? Sign in.</div>', unsafe_allow_html=True)
                if st.button("Back to sign in  ↗", key="show_signin", use_container_width=True):
                    st.session_state["auth_mode"] = "signin"
                    st.rerun()

    st.markdown('<div class="auth-footnote">Local capstone demo · Your login details are stored locally in SQLite · Never use a reused or sensitive password</div>', unsafe_allow_html=True)

    if login_pressed:
        account = verify_user(login_email, login_password)
        if account:
            st.session_state["auth_user"] = account
            st.rerun()
        else:
            st.error("We couldn't sign you in with those details. Check your email and password, or create an account.")
    if signup_pressed:
        if signup_password != signup_confirm:
            st.error("Your passwords do not match.")
        else:
            ok, message, account = register_user(signup_name, signup_email, signup_password)
            if ok and account:
                st.session_state["auth_user"] = account
                st.rerun()
            else:
                st.error(message)

    st.markdown('<div class="footer"><strong style="color:#956F4D;">♡ ReuniteAI</strong> &nbsp; Lost items. New beginnings. <span style="float:right;">Local capstone demo</span></div>', unsafe_allow_html=True)
    st.stop()

# Signed-in dashboard: all existing features remain below this gate.
st.markdown(brand_markup, unsafe_allow_html=True)
account_left, account_right = st.columns([6, 1], vertical_alignment="center")
with account_left:
    safe_name = html.escape(str(current_user.get("full_name", "Member")))
    safe_email = html.escape(str(current_user.get("email", "")))
    st.markdown(f'<div class="signed-in-strip">✦ Signed in as <strong>{safe_name}</strong> <span>· {safe_email}</span></div>', unsafe_allow_html=True)
with account_right:
    if st.button("Sign out", key="sign_out", use_container_width=True):
        st.session_state.pop("auth_user", None)
        st.rerun()

hero_left, hero_right = st.columns([0.91, 1.09], gap="large", vertical_alignment="center")
with hero_left:
    st.markdown(
        """
        <div class="eyebrow">People · Belongings · Stories</div>
        <div class="hero-title">LOST<br>SOMETHING?<br><em>LET'S BRING IT HOME.</em></div>
        <div class="hero-sub">A little search can make a big reunion. Report a missing belonging, browse found items, and let AI help surface possible matches.</div>
        <div class="gold-rule"></div>
        <div class="microline">🔑 Smart matching &nbsp; · &nbsp; 🛡️ Verify safely &nbsp; · &nbsp; 🤝 Community powered</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="stat-row"><div class="stat"><strong>TF-IDF + cosine</strong><span>Text retrieval</span></div><div class="stat"><strong>SQLite</strong><span>Local report storage</span></div><div class="stat"><strong>Human verified</strong><span>Ownership checks</span></div></div>',
        unsafe_allow_html=True,
    )
with hero_right:
    hero_image = ASSET_DIR / "reuniteai_hero.jpg"
    if hero_image.exists():
        st.markdown('<div class="hero-image">', unsafe_allow_html=True)
        st.image(str(hero_image), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Add assets/reuniteai_hero.jpg to show the hero artwork.")

# A visual collection of purpose-made 3D scenes, all using the reference palette.
st.markdown('<div class="section-label">A sunlit collection</div><h2>Small things. Big reunions.</h2><p style="color:#806B57;margin-top:-.45rem;">Beautiful details help people describe what they have lost.</p>', unsafe_allow_html=True)
gallery_items = [
    ("reuniteai_scene_keys_wallet.jpg", "THE DETAILS MATTER", "Keys & keepsakes", "Keys, a cardholder and a wallet—small clues can make a match clearer."),
    ("reuniteai_scene_journal_umbrella.jpg", "EVERYDAY ESSENTIALS", "Useful little things", "Headphones, a notebook and an umbrella, rendered in warm sandstone light."),
    ("reuniteai_scene_floating_objects.jpg", "READY TO REUNITE", "Find the familiar", "Belongings brought into focus with a cinematic, golden-hour feel."),
]
gallery_markup = ['<div class="visual-gallery">']
for image_name, kicker, title, description in gallery_items:
    src_uri = image_data_uri(str(ASSET_DIR / image_name))
    gallery_markup.append(
        '<article class="visual-card">'
        f'<img src="{src_uri}" alt="{html.escape(title)}" loading="lazy">'
        '<div class="visual-card-copy">'
        f'<div class="visual-card-kicker">{html.escape(kicker)}</div>'
        f'<div class="visual-card-title">{html.escape(title)}</div>'
        f'<div class="visual-card-description">{html.escape(description)}</div>'
        '</div></article>'
    )
gallery_markup.append('</div>')
st.markdown("".join(gallery_markup), unsafe_allow_html=True)

st.write("")
feature_cols = st.columns(4, gap="medium")
for col, (icon, title, desc) in zip(feature_cols, [
    ("🔍", "Smart Matching", "Rank found reports by text similarity."),
    ("📸", "Photo Reports", "Attach an optional image to a report."),
    ("🗂️", "Saved Records", "Keep local reports in the SQLite database."),
    ("🛡️", "Safe & Simple", "Verify potential matches with authorised staff."),
]):
    with col:
        st.markdown(f'<div class="feature-card"><div class="feature-icon">{icon}</div><div class="feature-name">{title}</div><div class="feature-desc">{desc}</div></div>', unsafe_allow_html=True)

st.write("")
st.markdown('<div class="section-label">Your lost-and-found workspace</div>', unsafe_allow_html=True)
st.markdown("## Reports, matches and insights")

tab_home, tab_report, tab_reports, tab_search, tab_eval, tab_about = st.tabs([
    "✦ Overview", "＋ Submit a Report", "🔒 My Reports",
    "⌕ AI Match Search", "▥ Evaluation", "ⓘ About & Privacy"
])

with tab_home:
    all_reports = get_reports(
        created_by=current_user["email"],
        owner_user_id=int(current_user["user_id"]),
    )
    lost_count = int((all_reports["report_type"] == "lost").sum()) if not all_reports.empty else 0
    found_count = int((all_reports["report_type"] == "found").sum()) if not all_reports.empty else 0
    m1, m2, m3 = st.columns(3)
    m1.metric("Lost reports submitted", lost_count)
    m2.metric("Found reports submitted", found_count)
    m3.metric("Total saved reports", len(all_reports))
    st.caption("These counts include only reports linked to your signed-in account. The original synthetic CSV and community found reports remain available to the matching engine.")
    st.markdown("### Recently submitted")
    if all_reports.empty:
        st.info("No user-submitted reports yet. Open **Submit a Report** to add the first one.")
    else:
        display_cols = ["report_id", "report_type", "item_name", "category", "location", "report_date", "status"]
        st.dataframe(all_reports[display_cols].head(8), use_container_width=True, hide_index=True)

with tab_report:
    st.markdown('<div class="section-label">Add to the community board</div>', unsafe_allow_html=True)
    st.header("Submit a lost or found item")
    st.write("Use the form to record an item. Contact information is not shown in the public report table.")
    with st.form("report_form", clear_on_submit=True):
        report_type = st.radio("What are you reporting?", ["lost", "found"], horizontal=True, format_func=lambda x: "I lost an item" if x == "lost" else "I found an item")
        left, right = st.columns(2, gap="large")
        with left:
            item_name = st.text_input("Item name *", placeholder="e.g., Blue metal water bottle")
            category_options = ["accessories", "bag", "books", "bottle", "clothing", "container", "electronics", "id card", "keys", "stationery", "umbrella", "other"]
            category = st.selectbox("Category *", category_options)
            item_date = st.date_input("Date lost / found", value=date.today())
        with right:
            location = st.text_input("Location", placeholder="e.g., Library reading room")
            contact_name = st.text_input("Contact name (optional)", max_chars=100)
            contact_info = st.text_input("Contact method (optional; private)", max_chars=250, placeholder="Email or phone — avoid sensitive details")
        description = st.text_area("Description *", placeholder="Describe colour, material, brand, stickers, marks or other identifying details.", height=120, max_chars=2000)
        uploaded = st.file_uploader("Item photograph (optional)", type=["png", "jpg", "jpeg", "webp"], help="Images only, maximum 5 MB. Avoid photos showing IDs, addresses or other private information.")
        submitted = st.form_submit_button("Save Report  →", use_container_width=True)

    if submitted:
        if not item_name.strip() or not description.strip():
            st.error("Please enter both the item name and description.")
        elif uploaded is not None and uploaded.size > 5 * 1024 * 1024:
            st.error("The image is larger than 5 MB. Please choose a smaller image.")
        else:
            saved_image = ""
            try:
                if uploaded is not None:
                    ext = Path(uploaded.name).suffix.lower()
                    if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
                        raise ValueError("Unsupported image type.")
                    # Verify actual image content instead of trusting the extension alone.
                    from PIL import Image
                    import io
                    image = Image.open(io.BytesIO(uploaded.getvalue()))
                    image.verify()
                    saved_name = f"{uuid.uuid4().hex}{ext}"
                    destination = UPLOAD_DIR / saved_name
                    destination.write_bytes(uploaded.getvalue())
                    saved_image = str(destination.relative_to(Path(__file__).resolve().parent))

                report_id = add_report(
                    report_type=report_type,
                    item_name=item_name,
                    category=category,
                    description=description,
                    location=location,
                    report_date=item_date.isoformat(),
                    image_path=saved_image,
                    contact_name=contact_name,
                    contact_info=contact_info,
                    created_by=current_user["email"],
                    owner_user_id=int(current_user["user_id"]),
                )
                st.success(f"Report saved successfully! Your report ID is **{report_id}**.")
                st.info("This is a local demo. Keep your report ID. Contact details are stored locally and are not shown in the report table.")
                if uploaded is not None:
                    st.image(uploaded, caption="Uploaded item photograph", width=220)
            except Exception as exc:
                st.error(f"Could not save the report: {exc}")

with tab_reports:
    st.markdown('<div class="section-label">Your local records</div>', unsafe_allow_html=True)
    st.header("My Reports")
    st.caption("🔒 Private account view: only reports linked to your signed-in account are listed here. Older unassigned demo records remain in SQLite but are hidden from individual accounts.")
    filter_type = st.selectbox("Filter reports", ["all", "lost", "found"], format_func=lambda x: {"all":"All reports", "lost":"Lost items", "found":"Found items"}[x])
    stored = get_reports(
        None if filter_type == "all" else filter_type,
        created_by=current_user["email"],
        owner_user_id=int(current_user["user_id"]),
    )
    if stored.empty:
        st.info("No reports in this view yet. Submit a report to see it here.")
    else:
        st.dataframe(
            stored[["report_id", "report_type", "item_name", "category", "description", "location", "report_date", "status"]],
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("View an uploaded photo"):
            image_rows = stored[stored["image_path"].fillna("").astype(str).str.strip().ne("")]
            if image_rows.empty:
                st.caption("No photos have been uploaded for the selected reports.")
            else:
                chosen_id = st.selectbox("Choose report ID", image_rows["report_id"].tolist())
                chosen = image_rows[image_rows["report_id"] == chosen_id].iloc[0]
                image_path = Path(__file__).resolve().parent / str(chosen["image_path"])
                if image_path.exists():
                    st.image(str(image_path), caption=f"{chosen_id} — {chosen['item_name']}", width=360)
                else:
                    st.warning("The image file is not present at its saved path.")

with tab_search:
    st.markdown('<div class="section-label">Find the closest description</div>', unsafe_allow_html=True)
    st.header("AI Match Search")
    st.caption("Searches the original sample reports plus found items submitted through the dashboard.")
    st.markdown('<div class="search-tip">💡 <strong>Search tip:</strong> Include the item type, colour, material, brand, stickers or other distinctive details. The system compares words in your description with found-item reports.</div>', unsafe_allow_html=True)
    q = st.text_area("Describe the item you lost", placeholder="e.g., Dark blue metal bottle with a white mountain sticker", height=120, key="match_query")
    categories = ["Any"] + sorted(set(reports_category for reports_category in pd.read_csv("data/found_items.csv")["category"].dropna().astype(str).tolist()))
    match_category = st.selectbox("Category filter", categories, key="match_category")
    top_k = st.slider("Candidates to show", 1, 10, 3, key="match_topk")
    if st.button("✨ Find Possible Matches", key="do_match", use_container_width=True):
        if not q.strip():
            st.error("Please enter an item description.")
        else:
            try:
                with st.spinner("ReuniteAI is comparing descriptions…"):
                    candidate_reports = get_found_reports_for_matcher("data/found_items.csv")
                    matcher = TfidfMatcher(candidate_reports)
                    results = matcher.search(q, top_k=top_k, category=match_category)
                if results.empty:
                    st.warning("No candidates were found.")
                else:
                    if float(results["score"].iloc[0]) < TFIDF_THRESHOLD:
                        st.warning("No strong text match found. Try adding identifying details; you can still review these candidates.")
                    else:
                        st.success("Possible text matches, ranked from highest similarity.")
                    st.markdown(
                        f'''<div class="match-section-head"><div><strong>Candidate results</strong><br><small>Ranked by description similarity</small></div><div><strong>{len(results)}</strong> <small>result{'s' if len(results) != 1 else ''}</small></div></div>''',
                        unsafe_allow_html=True,
                    )
                    result_columns = st.columns(2, gap="medium")
                    for result_index, (_, result_row) in enumerate(results.iterrows(), start=1):
                        with result_columns[(result_index - 1) % 2]:
                            render_match_card(result_row, result_index)
                    st.markdown(
                        '<div class="match-note"><strong>Important:</strong> The percentage represents text similarity, not the probability of ownership. A high score is only a lead. Compare distinctive details and verify ownership with authorised staff before returning an item.</div>',
                        unsafe_allow_html=True,
                    )
            except Exception as exc:
                st.error(f"Search failed: {exc}")

with tab_eval:
    st.markdown('<div class="section-label">Check the method</div>', unsafe_allow_html=True)
    st.header("Evaluation (synthetic demo)")
    st.write("The evaluation continues to use the original labelled CSV cases, so existing results remain comparable.")
    if st.button("▶ Run Evaluation", key="eval_button"):
        try:
            with st.spinner("Evaluating the baseline and TF-IDF matcher…"):
                summary, details = evaluate()
            st.subheader("Summary")
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.subheader("Query-level details")
            st.dataframe(details, use_container_width=True, hide_index=True)
            st.caption("The evaluation uses synthetic demo data; do not present it as real-world performance.")
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

with tab_about:
    st.markdown('<div class="section-label">Transparent by design</div>', unsafe_allow_html=True)
    st.header("About & Privacy")
    st.markdown("""
    **AI method**
    - TF-IDF converts text into numerical vectors.
    - Cosine similarity ranks descriptions by textual similarity.
    - This is text retrieval, not a supervised model trained on ownership decisions.

    **Data and safety**
    - The sample CSV is synthetic; user reports are stored in a local SQLite database.
    - Optional photos are saved in the local `uploads` folder.
    - Sign-up accounts are stored in SQLite; passwords are salted and hashed, not stored as plain text.
    - My Reports is filtered by the signed-in account's user ID, with an email-based fallback for reports created before the ownership ID migration. Unassigned legacy records are kept but hidden from individual accounts.
    - Found-item descriptions, categories, and locations can be used as community matching candidates; contact fields are not shown in search results.
    - Contact fields are stored locally and omitted from the report table.
    - This is basic demo authentication only: there is no email verification, password reset, multi-factor authentication, or production-grade access-control service. Do not expose it publicly or enter sensitive personal information.
    - Similarity scores do not prove ownership; verify with authorised staff.
    """)

st.markdown('<div class="footer"><strong style="color:#956F4D;">♡ ReuniteAI</strong> &nbsp; Lost items. New beginnings. <span style="float:right;">AI mini-capstone · Local demo</span></div>', unsafe_allow_html=True)
