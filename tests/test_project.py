from lexisync.segmenter import make_passages
from lexisync.benchmark import starter_set
def test_page_metadata_and_passage_id():
 p=make_passages([{"file":"x.pdf","page":2,"text":"7.2 Liability. Total liability is capped."}])
 assert p and p[0]["page"]==2 and p[0]["id"]
def test_benchmark_size():
 q=starter_set()
 assert len(q)>=15 and sum(not x["answerable"] for x in q)>=3
