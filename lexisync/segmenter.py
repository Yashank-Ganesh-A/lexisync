import re,hashlib
PAT=re.compile(r"(?im)^\\s*((?:(?:\\d+\\.){1,5}\\d+[A-Za-z]?)|(?:article|section|clause)\\s+[A-Za-z0-9().-]+)")
def make_passages(pages,limit=1400,overlap=150):
 out=[]
 for p in pages:
  bits=re.split(r"(?im)(?=^\\s*(?:(?:\\d+\\.){1,5}\\d+[A-Za-z]?)|(?:article|section|clause)\\s+[A-Za-z0-9().-]+)",p["text"])
  section="Unlabelled"
  for bit in bits:
   bit=bit.strip()
   if not bit: continue
   m=PAT.match(bit)
   if m: section=m.group(1).strip()
   for start in range(0,len(bit),limit-overlap):
    chunk=bit[start:start+limit]
    if not chunk: continue
    uid=hashlib.sha1(f"{p['file']}|{p['page']}|{section}|{chunk}".encode()).hexdigest()[:12]
    out.append({"id":uid,"file":p["file"],"page":p["page"],"section":section,"text":chunk})
    if len(bit)<=limit: break
 return out
