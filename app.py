import os,json,time,streamlit as st
from dotenv import load_dotenv
from lexisync.pdf_reader import read_bundle
from lexisync.search import SearchEngine
from lexisync.qa import GroundedQA
from lexisync.benchmark import starter_set,run_benchmark
load_dotenv()
st.set_page_config(page_title="LexiSyncAI | Casework Console",page_icon="⚖️",layout="wide")
st.markdown("""
<style>
:root { --ink:#202833; --muted:#687483; --accent:#c47735; --paper:#f7f5f0; }
.stApp { background:var(--paper); color:var(--ink); }
.block-container { max-width:1420px; padding-top:1.2rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { background:#202833; }
[data-testid="stSidebar"] * { color:#f4f1eb !important; }
[data-testid="stSidebar"] input { color:#202833 !important; }
[data-testid="stTabs"] button { border-radius:4px 4px 0 0; font-weight:650; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#a85f25; border-bottom:3px solid #c47735; }
div.stButton > button, div.stDownloadButton > button { border-radius:5px; border:0; background:#263746; color:white; font-weight:650; }
div.stButton > button:hover, div.stDownloadButton > button:hover { background:#c47735; color:white; }
[data-testid="stMetric"] { background:#fffdfa; border:1px solid #e5dfd5; padding:14px 16px; border-radius:7px; }
div[data-testid="stExpander"] { background:#fffdfa; border:1px solid #e7e0d5; border-radius:7px; }
section[data-testid="stFileUploaderDropzone"] { background:#fffdfa; border:1px dashed #b9a58c; }
h1,h2,h3 { letter-spacing:-.025em; }
</style>
""",unsafe_allow_html=True)
hero_left, hero_right = st.columns([4,1])
with hero_left:
    st.markdown("<div style='font-size:12px;letter-spacing:.18em;font-weight:800;color:#a85f25'>LEGAL RESEARCH / DOCUMENT INTELLIGENCE</div><div style='font-size:38px;font-weight:800;letter-spacing:-.04em;color:#202833;margin-top:5px'>LexiSyncAI <span style='font-size:16px;color:#a85f25;font-weight:600'>CASEWORK CONSOLE</span></div><div style='font-size:15px;color:#687483;margin-top:4px'>A structured workspace for searching, checking and citing legal evidence.</div>",unsafe_allow_html=True)
with hero_right:
    st.markdown("<div style='text-align:right;padding-top:22px;color:#687483;font-size:12px'>IMPLEMENTATION<br><b style='color:#202833;font-size:15px'>EDITION B</b></div>",unsafe_allow_html=True)
st.markdown("<hr style='border:0;border-top:1px solid #ded6ca;margin:18px 0 22px 0'>",unsafe_allow_html=True)
with st.sidebar:
 st.markdown("### CONTROL PANEL")
 st.caption("MODEL & ACCESS")
 key=st.text_input("Groq API key",os.getenv("GROQ_API_KEY",""),type="password")
 model=st.selectbox("Groq model",["openai/gpt-oss-120b","openai/gpt-oss-20b"])
 st.caption("Groq uses API-hosted inference, not a local model.")
if "docs" not in st.session_state: st.session_state.docs=[]
if "passages" not in st.session_state: st.session_state.passages=[]
if "engine" not in st.session_state: st.session_state.engine=None
if "answer" not in st.session_state: st.session_state.answer=None
w,q,e,m=st.tabs(["01  DOCUMENT INTAKE","02  ASK THE FILES","03  QUALITY CHECK","04  SYSTEM MAP"])
with w:
 st.subheader("Document intake")
 st.caption("Register a public-document bundle, inspect extraction quality, then build its search index.")
 st.text_input("Case label","Public legal document review")
 uploads=st.file_uploader("Select 2–5 public PDF files",type=["pdf"],accept_multiple_files=True)
 if uploads and not 2<=len(uploads)<=5: st.warning("The assignment specifies a bundle of 2–5 PDFs.")
 if uploads and st.button("Extract documents",type="primary",disabled=not 2<=len(uploads)<=5):
  reports=[]; passages=[]; bar=st.progress(0)
  for i,f in enumerate(uploads):
   r=read_bundle(f.getvalue(),f.name); reports.append(r["report"]); passages+=r["passages"]; bar.progress((i+1)/len(uploads))
  st.session_state.docs=reports; st.session_state.passages=passages; st.session_state.engine=None
  st.success(f"Extracted {len(reports)} documents into {len(passages)} evidence passages.")
 for d in st.session_state.docs:
  with st.expander(f"{d['file']} · {d['pages']} pages"):
   st.write(f"Digital pages: {d['digital']} · Scanned pages: {d['scanned']} · Tables detected: {d['tables']}")
   st.write("Detected headings:",", ".join(d["headings"][:30]) or "None")
   for note in d["notes"]: st.warning(note)
 if st.session_state.passages and st.button("Build hybrid index",type="primary"):
  with st.spinner("Loading lexical, embedding and reranking models..."):
   st.session_state.engine=SearchEngine(st.session_state.passages)
  st.success("Hybrid index is ready.")
with q:
 st.subheader("Evidence-led question desk")
 st.caption("Ask a focused question and inspect the source passages behind the response.")
 if not st.session_state.engine: st.info("Extract PDFs and build the index in Workspace.")
 else:
  question=st.text_area("Legal question",height=100,placeholder="Does the liability limitation cover the indemnity obligations?")
  a,b=st.columns(2); k=a.slider("Evidence passages",3,12,7); gate=b.slider("Evidence overlap threshold",.05,.8,.18,.01)
  if st.button("Search and generate answer",type="primary",disabled=not question.strip()):
   t=time.perf_counter(); st.session_state.answer=GroundedQA(key,model).answer(question,st.session_state.engine,k,gate)
   st.session_state.answer["elapsed"]=round(time.perf_counter()-t,2)
  r=st.session_state.answer
  if r:
   if r["confidence"]=="INSUFFICIENT EVIDENCE": st.error("INSUFFICIENT EVIDENCE · Abstained")
   elif r["confidence"]=="HIGH": st.success("HIGH confidence")
   elif r["confidence"]=="MEDIUM": st.info("MEDIUM confidence")
   else: st.warning("LOW confidence")
   st.markdown("### Answer"); st.write(r["answer"])
   st.caption(f"Elapsed: {r.get('elapsed','—')} s · Evidence overlap: {r['overlap']:.2f}")
   if r["explanation"]: st.markdown("**Evidence explanation:** "+r["explanation"])
   st.markdown("### Citations and evidence")
   for x in r["sources"]:
    with st.expander(f"{x['file']} · Page {x['page']} · {x['section']}"):
     st.write(x["text"]); st.caption("Passage ID: "+x["id"])
   if r["conflicts"]:
    st.warning("Potential source conflict")
    for x in r["conflicts"]: st.write("• "+x)
   with st.expander("Retrieval diagnostics"): st.json(r["diagnostics"])
with e:
 st.subheader("Retrieval quality check")
 st.warning("Starter questions are templates. Add document-specific expected answers and gold passage IDs before reporting academic metrics.")
 if st.session_state.engine:
  qs=starter_set(); st.write(f"{len(qs)} questions, including factual, clause, multi-hop and unanswerable prompts.")
  if st.button("Run retrieval comparison"):
   st.session_state.bench=run_benchmark(qs,st.session_state.engine)
  if "bench" in st.session_state:
   r=st.session_state.bench; a,b,c=st.columns(3); a.metric("Questions",r["count"])
   b.metric("Recall@5",f"{r['recall5']:.1%}" if r["recall5"] is not None else "N/A")
   c.metric("Mean hybrid latency",f"{r['latency']:.0f} ms")
   st.dataframe(r["rows"],use_container_width=True)
   st.download_button("Download JSON",json.dumps(r,indent=2),"benchmark.json")
 else: st.info("Build the search index first.")
with m:
 st.subheader("How the evidence pipeline works")
 st.code("""2–5 public PDFs
   ↓ PyMuPDF block extraction / reading order
   ↓ scanned-page flag → optional Tesseract OCR
   ↓ clause and section-aware segmentation
   ├─ BM25 lexical retrieval (exact clause IDs)
   └─ SentenceTransformer dense embeddings
           ↓ reciprocal-rank fusion
           ↓ cross-encoder reranking
           ↓ evidence sufficiency / abstention
           ↓ Groq API (retrieved context only)
           ↓ cited answer + page/section evidence
           ↓ held-out evaluation and ablation""")
 st.write("Modules: pdf_reader.py, segmenter.py, search.py, qa.py, benchmark.py.")
