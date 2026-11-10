
import re
import json
from groq import Groq

ABSTAIN = "The documents do not contain enough evidence to answer this."

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or",
    "that", "the", "this", "to", "was", "what", "when", "where",
    "which", "who", "why", "with", "document", "documents",
    "file", "files", "uploaded"
}


def meaningful_tokens(text):
    """Extract meaningful words from text."""
    return {
        token
        for token in re.findall(r"\b[a-z0-9]+\b", text.lower())
        if token not in STOP_WORDS and len(token) > 1
    }


def normalize_confidence(value):
    """
    Convert model confidence into HIGH, MEDIUM or LOW.
    Supports both text labels and numeric scores.
    """
    if isinstance(value, (int, float)):
        score = float(value)

        if score >= 0.80:
            return "HIGH"
        elif score >= 0.50:
            return "MEDIUM"
        else:
            return "LOW"

    text = str(value).strip().upper()

    if text in {"HIGH", "MEDIUM", "LOW"}:
        return text

    try:
        score = float(text)

        if score >= 0.80:
            return "HIGH"
        elif score >= 0.50:
            return "MEDIUM"
        else:
            return "LOW"

    except (ValueError, TypeError):
        return "LOW"


class GroundedQA:

    def __init__(self, key, model):
        self.key = key
        self.model = model

    def answer(self, q, engine, k, threshold):

        # STEP 1: Retrieve relevant passages.
        hits, ms, diag = engine.find(q, k, "hybrid")

        # STEP 2: Calculate meaningful question terms.
        query_tokens = meaningful_tokens(q)

        evidence_tokens = set()

        for hit in hits:
            evidence_tokens.update(
                meaningful_tokens(hit.get("text", ""))
            )

        # STEP 3: Calculate evidence overlap.
        overlap = (
            len(query_tokens & evidence_tokens) / len(query_tokens)
            if query_tokens else 1.0
        )

        # STEP 4: Identify broad document-summary questions.
        q_lower = q.lower()

        broad_question = any(
            phrase in q_lower
            for phrase in (
                "main subject",
                "main topic",
                "what is this document about",
                "what are these documents about",
                "summarize",
                "summary of the document",
                "overview of the document"
            )
        )

        # STEP 5: Evidence sufficiency check.
        if not hits or (
            not broad_question and overlap < threshold
        ):

            return {
                "answer": ABSTAIN,
                "confidence": "INSUFFICIENT EVIDENCE",
                "explanation": (
                    "No passages were retrieved."
                    if not hits else
                    "Retrieved passages did not contain enough "
                    "meaningful question terms to pass the "
                    "evidence-overlap gate."
                ),
                "sources": hits[:3],
                "overlap": overlap,
                "conflicts": [],
                "diagnostics": diag
            }

        # STEP 6: Check Groq API key.
        if not self.key:

            return {
                "answer": (
                    "Enter a Groq API key to generate a grounded "
                    "response. Retrieved passages are shown below."
                ),
                "confidence": "LOW",
                "explanation": (
                    "Generation skipped because no API key was supplied."
                ),
                "sources": hits[:3],
                "overlap": overlap,
                "conflicts": [],
                "diagnostics": diag
            }

        # STEP 7: Prepare retrieved evidence.
        ctx = "\n\n".join(
            f"[E{i}] FILE={x.get('file', 'Unknown')} "
            f"PAGE={x.get('page', 'Unknown')} "
            f"SECTION={x.get('section', 'Unknown')}\n"
            f"{x.get('text', '')}"
            for i, x in enumerate(hits, 1)
        )

        # STEP 8: Define the legal research instructions.
        system = (
            "You are a cautious legal document analyst. "
            "Use ONLY the supplied evidence. Do not use outside "
            "law, general knowledge or unsupported assumptions. "
            "Answer the user's question directly when the evidence "
            "supports it. Cite every factual claim using [E1], "
            "[E2] style identifiers. If evidence is insufficient, "
            "clearly state that. Never invent page numbers, clauses "
            "or legal provisions. "
            "Return valid JSON with these keys: "
            "answer, confidence, explanation, cited_ids, conflicts. "
            "Confidence should preferably be HIGH, MEDIUM or LOW."
        )

        # STEP 9: Call Groq API.
        try:

            client = Groq(api_key=self.key)

            response = client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": system
                    },
                    {
                        "role": "user",
                        "content": (
                            f"QUESTION: {q}\n\n"
                            f"EVIDENCE:\n{ctx}"
                        )
                    }
                ]
            )

            # STEP 10: Parse model response.
            raw_response = response.choices[0].message.content

            data = json.loads(raw_response)

            answer = str(data.get("answer", ABSTAIN))

            model_confidence = data.get("confidence", "LOW")

            confidence = normalize_confidence(model_confidence)

            # STEP 11: Preserve model abstention.
            if (
                ABSTAIN.lower() in answer.lower()
                or str(model_confidence).strip().upper()
                == "INSUFFICIENT EVIDENCE"
            ):

                return {
                    "answer": ABSTAIN,
                    "confidence": "INSUFFICIENT EVIDENCE",
                    "explanation": data.get("explanation", ""),
                    "sources": hits[:3],
                    "overlap": overlap,
                    "conflicts": data.get("conflicts", []),
                    "diagnostics": diag
                }

            # STEP 12: Resolve citation IDs.
            cited_ids = data.get("cited_ids", [])

            if isinstance(cited_ids, str):
                cited_ids = [cited_ids]

            sources = []

            for value in cited_ids:

                try:
                    index = int(
                        str(value).strip()
                        .replace("E", "")
                        .replace("e", "")
                    ) - 1

                    if 0 <= index < len(hits):
                        sources.append(hits[index])

                except (TypeError, ValueError):
                    continue

            # Keep retrieved evidence visible if citation IDs
            # are missing or incorrectly formatted.
            if not sources:
                sources = hits[:3]

            # STEP 13: Return grounded answer.
            return {
                "answer": answer,
                "confidence": confidence,
                "explanation": data.get("explanation", ""),
                "sources": sources,
                "overlap": overlap,
                "conflicts": data.get("conflicts", []),
                "diagnostics": diag
            }

        # STEP 14: Handle API and response errors.
        except Exception as exc:

            return {
                "answer": f"Groq request failed: {exc}",
                "confidence": "LOW",
                "explanation": (
                    "No generated legal conclusion is asserted. "
                    "Inspect the retrieved evidence and API response."
                ),
                "sources": hits[:3],
                "overlap": overlap,
                "conflicts": [],
                "diagnostics": diag
            }
