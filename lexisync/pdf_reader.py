import fitz,io
from .segmenter import make_passages
def read_bundle(raw,name):
 doc=fitz.open(stream=raw,filetype="pdf"); pages=[]; digital=0; scanned=0; tables=0; heads=[]; notes=[]
 for n,p in enumerate(doc):
  blocks=p.get_text("blocks",sort=True); text="\\n".join(str(b[4]).strip() for b in blocks if len(b)>4 and str(b[4]).strip())
  if len(text.strip())<30:
   scanned+=1
   try:
    import pytesseract
    from PIL import Image
    pix=p.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False)
    text=pytesseract.image_to_string(Image.open(io.BytesIO(pix.tobytes("png"))))
    notes.append(f"Page {n+1}: OCR fallback attempted; inspect OCR accuracy.")
   except Exception: notes.append(f"Page {n+1}: image-only page flagged; install Tesseract for OCR.")
  else: digital+=1
  try: tables+=len(p.find_tables().tables)
  except Exception: pass
  if text.strip():
   pages.append({"file":name,"page":n+1,"text":text})
   for line in text.splitlines():
    t=line.strip()
    if len(t)<100 and (t.isupper() or (t[:1].isdigit() and "." in t[:8])) and t not in heads: heads.append(t)
 passages=make_passages(pages)
 return {"passages":passages,"report":{"file":name,"pages":len(doc),"digital":digital,"scanned":scanned,"tables":tables,"headings":heads,"notes":notes}}
