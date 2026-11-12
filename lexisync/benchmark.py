from math import ceil

from lexisync.qa import evidence_overlap, is_broad_question, meaningful_tokens


CONFIGURATIONS = (
    ("BM25 only", "bm25"),
    ("Dense only", "dense"),
    ("Hybrid (BM25 + dense, RRF)", "hybrid_rrf"),
    ("Hybrid + cross-encoder rerank", "hybrid_rerank"),
)

THRESHOLDS = tuple(round(step / 20, 2) for step in range(1, 11))


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


def _gold_ids(question):
    gold_ids = question.get("gold_ids", [])
    if isinstance(gold_ids, str):
        gold_ids = gold_ids.split(",")
    return {str(value).strip() for value in gold_ids if str(value).strip()}


def _recall(retrieved, relevant):
    if not relevant:
        return None
    return sum(
        len(set(item["id"] for item in hits) & gold_ids)
        for hits, gold_ids in zip(retrieved, relevant)
    ) / sum(len(gold_ids) for gold_ids in relevant)


def _estimated_gold_ids(question, passages):
    if not question["answerable"]:
        return set()
    tokens = meaningful_tokens(question["question"])
    if not tokens:
        return set()
    minimum_matches = max(1, ceil(len(tokens) * 0.25))
    return {
        str(passage["id"])
        for passage in passages
        if len(tokens & meaningful_tokens(passage.get("text", "")))
        >= minimum_matches
    }


def run_benchmark(qs, engine):
    available = {
        "bm25": True,
        "dense": engine.encoder is not None,
        "hybrid_rrf": engine.encoder is not None,
        "hybrid_rerank": (
            engine.encoder is not None and engine.cross is not None
        ),
    }
    results = {mode: [] for _, mode in CONFIGURATIONS}
    latencies = {mode: [] for _, mode in CONFIGURATIONS}
    valid_gold = []
    invalid_gold_ids = set()
    passage_ids = {str(item["id"]) for item in engine.passages}

    for question in qs:
        gold_ids = _gold_ids(question)
        valid = gold_ids & passage_ids
        invalid_gold_ids.update(gold_ids - passage_ids)
        valid_gold.append(valid)

        for _, mode in CONFIGURATIONS:
            if not available[mode]:
                results[mode].append([])
                continue
            hits, elapsed_ms, _ = engine.find(
                question["question"], 5, mode
            )
            results[mode].append(hits)
            latencies[mode].append(elapsed_ms)

    has_manual_labels = any(valid_gold)
    if has_manual_labels:
        relevant_by_question = valid_gold
        recall_method = "manual"
        recall_questions = sum(bool(ids) for ids in valid_gold)
    else:
        relevant_by_question = [
            _estimated_gold_ids(question, engine.passages)
            for question in qs
        ]
        recall_method = "estimated_token_overlap"
        recall_questions = sum(bool(ids) for ids in relevant_by_question)

    configurations = []
    for name, mode in CONFIGURATIONS:
        if available[mode]:
            retrieved = results[mode]
            recall = {
                k: _recall(
                    [hits[:k] for hits in retrieved],
                    relevant_by_question
                )
                for k in (1, 3, 5)
            }
            if recall_method == "estimated_token_overlap":
                recall = {
                    k: value if value is not None else 0.0
                    for k, value in recall.items()
                }
            latency = (
                sum(latencies[mode]) / len(latencies[mode])
                if latencies[mode] else 0
            )
        else:
            recall = {k: None for k in (1, 3, 5)}
            latency = None
        configurations.append({
            "configuration": name,
            "models_loaded": available[mode],
            "recall1": recall[1],
            "recall3": recall[3],
            "recall5": recall[5],
            "latency_ms": latency,
        })

    full_pipeline = results["hybrid_rerank"]
    threshold_rows = []
    for threshold in THRESHOLDS:
        correct = 0
        false_abstains = 0
        answerable_count = 0
        for question, hits in zip(qs, full_pipeline):
            answerable = bool(question["answerable"])
            if answerable:
                answerable_count += 1
            overlap = evidence_overlap(question["question"], hits)
            abstains = not hits or (
                not is_broad_question(question["question"])
                and overlap < threshold
            )
            correct += abstains != answerable
            if answerable and abstains:
                false_abstains += 1
        can_calibrate = available["hybrid_rerank"] and bool(qs)
        threshold_rows.append({
            "threshold": threshold,
            "abstention_accuracy": (
                correct / len(qs) if can_calibrate else None
            ),
            "false_abstain_rate": (
                false_abstains / answerable_count
                if can_calibrate and answerable_count else None
            ),
        })

    baseline = next(
        (item for item in configurations
         if item["configuration"] == "Hybrid + cross-encoder rerank"),
        None
    )
    default_calibration = next(
        (row for row in threshold_rows if row["threshold"] == 0.2),
        None
    )
    question_rows = [{
        "ID": question["id"],
        "type": question["type"],
        "answerable": bool(question["answerable"]),
        "gold labels": bool(gold_ids),
        "hybrid passages": len(hits),
        "dense passages": len(dense_hits),
    } for question, gold_ids, hits, dense_hits in zip(
        qs, valid_gold, full_pipeline, results["dense"]
    )]

    return {
        "count": len(qs),
        "labelled_questions": sum(bool(ids) for ids in valid_gold),
        "recall_questions": recall_questions,
        "recall_method": recall_method,
        "invalid_gold_ids": sorted(invalid_gold_ids),
        "recall5": baseline["recall5"] if baseline else None,
        "latency": baseline["latency_ms"] if baseline else None,
        "abstention_accuracy": (
            default_calibration["abstention_accuracy"]
            if default_calibration else None
        ),
        "configurations": configurations,
        "calibration": threshold_rows,
        "rows": question_rows,
        "note": (
            "Recall uses manually labelled gold passage IDs."
            if recall_method == "manual" else
            "No gold passage IDs were provided. Recall values are estimates "
            "using answerable questions and passage/query token overlap "
            "(at least 25% of meaningful query terms); manually judged gold "
            "IDs are required for validated Recall@k."
        ) + (
            " Abstention metrics measure the evidence-overlap gate against "
            "each question's answerable label; they do not measure "
            "generated-answer correctness."
        ),
    }
