import time
def starter_set():
 data=[
 ("Q01","What is the purpose of the document?",True,"factual"),("Q02","Who are the parties?",True,"factual"),
 ("Q03","What is the commencement date?",True,"factual"),("Q04","What payment terms apply?",True,"factual"),
 ("Q05","What termination notice is required?",True,"factual"),("Q06","What does clause 2.1 state?",True,"clause"),
 ("Q07","What is the liability cap?",True,"clause"),("Q08","Which warranties are included?",True,"clause"),
 ("Q09","Which duties continue after termination?",True,"clause"),("Q10","How do indemnity and liability provisions interact?",True,"multi-hop"),
 ("Q11","Compare notice and termination provisions.",True,"multi-hop"),("Q12","Which sections govern dispute resolution?",True,"multi-hop"),
 ("Q13","What is the signatory's passport number?",False,"unanswerable"),("Q14","What is the supplier's email password?",False,"unanswerable"),
 ("Q15","What was the weather on signing day?",False,"unanswerable"),("Q16","What will a court decide next year?",False,"unanswerable")]
 return [{"id":i,"question":q,"answerable":a,"type":t,"gold_ids":[]} for i,q,a,t in data]
def run_benchmark(qs,engine):
 rows=[]; labelled=[]; times=[]
 for q in qs:
  t=time.perf_counter(); h,ms,_=engine.find(q["question"],5,"hybrid"); d,_,_=engine.find(q["question"],5,"dense")
  times.append((time.perf_counter()-t)*1000)
  if q["gold_ids"]: labelled.append(bool(set(x["id"] for x in h)&set(q["gold_ids"])))
  rows.append({"ID":q["id"],"type":q["type"],"answerable":q["answerable"],"hybrid passages":len(h),"dense passages":len(d),"gold labels":bool(q["gold_ids"]),"latency ms":round(ms,1)})
 return {"count":len(rows),"recall5":sum(labelled)/len(labelled) if labelled else None,"latency":sum(times)/len(times) if times else 0,"rows":rows,"note":"Recall requires manually labelled gold_ids; manual correctness, citation precision and calibrated abstention still required."}
