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
  st.session_state.pop("bench",None)
  st.session_state.pop("bench_candidates",None)
  st.success(f"Extracted {len(reports)} documents into {len(passages)} evidence passages.")
 for d in st.session_state.docs:
  with st.expander(f"{d['file']} · {d['pages']} pages"):
   st.write(f"Digital pages: {d['digital']} · Scanned pages: {d['scanned']} · Tables detected: {d['tables']}")
   st.write("Detected headings:",", ".join(d["headings"][:30]) or "None")
   for note in d["notes"]: st.warning(note)
 if st.session_state.passages and st.button("Build hybrid index",type="primary"):
  with st.spinner("Loading lexical, embedding and reranking models..."):
   st.session_state.engine=SearchEngine(st.session_state.passages)
  st.session_state.pop("bench",None)
  st.session_state.pop("bench_candidates",None)
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
 if st.session_state.engine:
  if "bench_questions" not in st.session_state:
   st.session_state.bench_questions=starter_set()
  st.caption("Compare retrieval configurations and calibrate the evidence-overlap abstention gate against the labelled questions.")
  with st.expander("Question set and gold passage labels",expanded=True):
   st.caption("Recall@k needs judged relevant passages. Edit the question set and add their exact passage IDs (comma-separated); leave unanswerable questions without gold IDs.")
   editor_rows=[{**question,"gold_ids":", ".join(question["gold_ids"])} for question in st.session_state.bench_questions]
   edited_questions=st.data_editor(
    editor_rows,
    key="bench_questions_editor",
    hide_index=True,
    use_container_width=True,
    num_rows="fixed",
    column_config={
     "id":st.column_config.TextColumn("ID",disabled=True),
     "question":st.column_config.TextColumn("Question",width="large"),
     "answerable":st.column_config.CheckboxColumn("Answerable"),
     "type":st.column_config.TextColumn("Question type",disabled=True),
     "gold_ids":st.column_config.TextColumn("Gold passage IDs",width="large"),
    }
   )
   updated_questions=[
    {**question,"question":str(question.get("question") or ""),"gold_ids":[value.strip() for value in (question.get("gold_ids") or "").split(",") if value.strip()]}
    for question in edited_questions
   ]
   if updated_questions!=st.session_state.bench_questions:
    st.session_state.pop("bench",None)
    st.session_state.pop("bench_candidates",None)
   st.session_state.bench_questions=updated_questions
  qs=st.session_state.bench_questions
  labelled_count=sum(bool(question["gold_ids"]) for question in qs)
  st.info(f"Gold passage labels: {labelled_count}/{len(qs)} questions. Without gold labels, Recall@k is shown as a token-overlap estimate.")
  if st.button("Find passage IDs to label"):
   candidates={}
   with st.spinner("Finding BM25 candidate passages for answerable questions..."):
    for question in qs:
     if question["answerable"]:
      hits,_,_=st.session_state.engine.find(question["question"],3,"bm25")
      candidates[question["id"]]=[{
       "Passage ID":hit["id"],
       "Source":f"{hit.get('file','Unknown')} · page {hit.get('page','?')} · {hit.get('section','Unknown')}",
       "Text preview":hit.get("text","")[:350],
      } for hit in hits]
   st.session_state.bench_candidates=candidates
  if st.session_state.get("bench_candidates"):
   for question in qs:
    if question["answerable"]:
     with st.expander(f"{question['id']} · {question['question']}"):
      rows=st.session_state.bench_candidates.get(question["id"],[])
      if rows:
       st.dataframe(rows,use_container_width=True,hide_index=True)
      else:
       st.write("No BM25 candidates found.")
      st.caption("Judge which candidate passages actually answer this question, then copy their IDs into Gold passage IDs above.")
  st.write(f"{len(qs)} questions, including factual, clause, multi-hop and unanswerable prompts.")
  if st.button("Run retrieval comparison"):
   st.session_state.bench=run_benchmark(st.session_state.bench_questions,st.session_state.engine)
  if "bench" in st.session_state:
   r=st.session_state.bench
   a,b,c,d=st.columns(4)
   a.metric("Questions used",r["count"])
   b.metric("Recall@5 (full pipeline)",f"{r['recall5']:.1%}" if r["recall5"] is not None else "N/A",help=r["note"])
   c.metric("Mean latency (full)",f"{r['latency']:.1f} ms" if r["latency"] is not None else "N/A")
   d.metric("Abstention accuracy",f"{r['abstention_accuracy']:.1%}" if r["abstention_accuracy"] is not None else "N/A",help="Evidence-overlap gate accuracy at threshold 0.20.")
   st.subheader("Recall@k per configuration")
   config_rows=[{
    "Configuration":item["configuration"],
    "Models loaded":item["models_loaded"],
    "Recall@1":f"{item['recall1']:.3f}" if item["recall1"] is not None else "N/A",
    "Recall@3":f"{item['recall3']:.3f}" if item["recall3"] is not None else "N/A",
    "Recall@5":f"{item['recall5']:.3f}" if item["recall5"] is not None else "N/A",
    "Latency (ms)":f"{item['latency_ms']:.1f}" if item["latency_ms"] is not None else "N/A",
   } for item in r["configurations"]]
   st.dataframe(config_rows,use_container_width=True,hide_index=True)
   if r["recall_method"]=="estimated_token_overlap":
    st.warning(f"Estimated Recall@k based on token-overlap pseudo-labels for {r['recall_questions']} questions. Add judged gold passage IDs for validated scores.")
   else:
    st.caption(f"Recall labels: {r['labelled_questions']} questions. {r['note']}")
   if r["invalid_gold_ids"]:
    st.warning(f"Skipped {len(r['invalid_gold_ids'])} gold passage ID(s) not found in the current index.")
   st.subheader("Abstention-gate calibration (threshold sweep)")
   st.dataframe([{
    "threshold":f"{row['threshold']:.2f}",
    "abstention_accuracy":f"{row['abstention_accuracy']:.1%}" if row["abstention_accuracy"] is not None else "N/A",
    "false_abstain_rate":f"{row['false_abstain_rate']:.1%}" if row["false_abstain_rate"] is not None else "N/A",
   } for row in r["calibration"]],use_container_width=True,hide_index=True)
   with st.expander("Per-question retrieval results"):
    st.dataframe(r["rows"],use_container_width=True,hide_index=True)
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
