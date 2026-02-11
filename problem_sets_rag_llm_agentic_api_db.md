# Problem Sets: RAG/LLM, Agentic Protocols, APIs, Databases

> **Purpose:** LeetCode-style problems for FirstThen interview prep.
> Each problem is directly checkable — code output, multiple choice, or fill-in.
> Complexity calibrated to role relevance.

---

## Section A: RAG & LLM (12 Problems)

**Priority: HIGH** — Core to the role. The STAR framework relies on LLM for predicate extraction and response generation, RAG for grounding in vetted clinical sources.

---

### Problem A1: Cosine Similarity (Easy)

**Type:** Code Writing

Given two embedding vectors, compute their cosine similarity. Do NOT use any ML library — implement from scratch.

```python
def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    Return a float between -1 and 1.
    Raise ValueError if vectors have different lengths or are zero vectors.
    """
    pass

# Test Cases:
assert abs(cosine_similarity([1, 0, 0], [1, 0, 0]) - 1.0) < 1e-6
assert abs(cosine_similarity([1, 0, 0], [0, 1, 0]) - 0.0) < 1e-6
assert abs(cosine_similarity([1, 1], [1, -1]) - 0.0) < 1e-6
assert abs(cosine_similarity([3, 4], [6, 8]) - 1.0) < 1e-6
assert abs(cosine_similarity([1, 2, 3], [-1, -2, -3]) - (-1.0)) < 1e-6
```

**What it tests:** Foundation of vector search — every RAG system uses this.

---

### Problem A2: Text Chunking with Overlap (Easy)

**Type:** Code Writing

Implement a text chunker that splits a document into overlapping chunks for embedding. This is the first step in any RAG ingestion pipeline.

```python
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into chunks of approximately chunk_size characters
    with overlap characters of overlap between consecutive chunks.

    Rules:
    - Split on sentence boundaries ('. ') when possible
    - If no sentence boundary exists within chunk_size, split at chunk_size
    - overlap defines how many characters from the end of one chunk
      appear at the start of the next
    - Return list of chunk strings

    Example:
    text = "The child has ADHD. The parent needs strategies. Visual schedules work well. Timers help transitions."
    chunk_size = 50, overlap = 20

    Expected output (approximately):
    ["The child has ADHD. The parent needs strategies.",
     "The parent needs strategies. Visual schedules work well.",
     "Visual schedules work well. Timers help transitions."]
    """
    pass

# Test Cases:
text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
chunks = chunk_text(text, 40, 15)
assert len(chunks) >= 2
assert all(len(c) <= 55 for c in chunks)  # chunk_size + some tolerance for sentence boundary
# Verify overlap: end of chunk[i] should overlap with start of chunk[i+1]
for i in range(len(chunks) - 1):
    assert chunks[i][-10:] in chunks[i + 1] or chunks[i + 1][:15] in chunks[i]
```

**What it tests:** RAG ingestion pipeline — chunking strategy directly affects retrieval quality.

---

### Problem A3: BM25 Scoring (Medium)

**Type:** Code Writing

Implement BM25 scoring for keyword-based retrieval. FirstThen uses hybrid retrieval (BM25 + FAISS vector search).

```python
import math

def bm25_score(query_terms: list[str], document: list[str],
               corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> float:
    """
    Compute BM25 score for a single document given a query.

    Args:
        query_terms: tokenized query (e.g., ["adhd", "strategies"])
        document: tokenized document (e.g., ["visual", "schedules", "help", "adhd"])
        corpus: list of all tokenized documents (for IDF computation)
        k1: term frequency saturation parameter
        b: length normalization parameter

    Returns:
        BM25 score (float)

    Formula:
        score = sum over query terms of:
            IDF(term) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))

        where IDF(term) = log((N - n(term) + 0.5) / (n(term) + 0.5) + 1)
              N = total docs in corpus
              n(term) = number of docs containing term
              tf = frequency of term in document
              doc_len = length of document
              avg_doc_len = average document length in corpus
    """
    pass

# Test Cases:
corpus = [
    ["adhd", "child", "strategies", "visual", "schedules"],
    ["medication", "dosage", "adhd", "treatment"],
    ["parent", "coaching", "behavioral", "strategies"],
    ["adhd", "adhd", "diagnosis", "criteria", "dsm"],
]
query = ["adhd", "strategies"]
doc = corpus[0]  # contains both query terms

score = bm25_score(query, doc, corpus)
assert score > 0  # document contains query terms
assert bm25_score(["xyz"], corpus[0], corpus) == 0  # term not in doc
# doc with more "adhd" occurrences should score differently
score_doc3 = bm25_score(["adhd"], corpus[3], corpus)
score_doc0 = bm25_score(["adhd"], corpus[0], corpus)
assert score_doc3 > score_doc0  # corpus[3] has "adhd" twice
```

**What it tests:** Keyword retrieval — the BM25 half of FirstThen's hybrid retrieval.

---

### Problem A4: Reciprocal Rank Fusion (Medium)

**Type:** Code Writing

Given results from two different retrieval systems (BM25 and vector search), combine them using Reciprocal Rank Fusion (RRF). This is how hybrid retrieval merges results.

```python
def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> list[str]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: list of ranked result lists (each is a list of doc IDs)
                      e.g., [["doc_3", "doc_1", "doc_7"], ["doc_1", "doc_3", "doc_5"]]
        k: RRF constant (default 60)

    Returns:
        Combined ranked list of doc IDs, sorted by RRF score descending.

    Formula:
        RRF_score(doc) = sum over all lists of: 1 / (k + rank(doc))
        where rank is 1-indexed position in the list.
        If doc not in a list, it contributes 0 from that list.
    """
    pass

# Test Cases:
bm25_results = ["doc_A", "doc_B", "doc_C", "doc_D"]
vector_results = ["doc_C", "doc_A", "doc_E", "doc_B"]

fused = reciprocal_rank_fusion([bm25_results, vector_results])
assert fused[0] == "doc_A"  # rank 1 + rank 2 = best combined
assert "doc_C" in fused[:3]
assert "doc_E" in fused  # only in vector results but still included
assert len(fused) == 5  # union of all docs

# Single list should return same order
assert reciprocal_rank_fusion([["x", "y", "z"]]) == ["x", "y", "z"]
```

**What it tests:** Hybrid retrieval fusion — directly used in FirstThen's BM25 + FAISS pipeline.

---

### Problem A5: Predicate Extraction Parser (Medium)

**Type:** Code Writing

Given an LLM's text output containing extracted predicates, parse them into structured Python objects. This is the bridge between the LLM layer and the ASP reasoning engine in the STAR framework.

```python
def parse_predicates(llm_output: str) -> list[dict]:
    """
    Parse LLM-generated predicate text into structured form.

    The LLM outputs predicates in this format (one per line):
        predicate_name(arg1, arg2, ...)

    Args can be:
        - atoms: lowercase words (e.g., adhd, anxiety)
        - strings: quoted text (e.g., "visual schedules")
        - numbers: integers (e.g., 7)

    Return a list of dicts, each with:
        {"name": str, "args": list[str | int]}

    Ignore blank lines and lines starting with '#' (comments).
    Strip whitespace from args. Preserve string quotes content but remove the quotes.

    Example:
        Input:
        '''
        child_age(7)
        has_diagnosis(adhd)
        challenge("morning routines")
        strategy_preference(parent, "visual schedules")
        # this is a comment
        '''
        Output:
        [
            {"name": "child_age", "args": [7]},
            {"name": "has_diagnosis", "args": ["adhd"]},
            {"name": "challenge", "args": ["morning routines"]},
            {"name": "strategy_preference", "args": ["parent", "visual schedules"]}
        ]
    """
    pass

# Test Cases:
output = """child_age(8)
has_diagnosis(adhd)
challenge("homework focus")
parent_concern(mother, "screen time")
# ignore this
severity(moderate)
"""
result = parse_predicates(output)
assert len(result) == 5
assert result[0] == {"name": "child_age", "args": [8]}
assert result[2] == {"name": "challenge", "args": ["homework focus"]}
assert result[3] == {"name": "parent_concern", "args": ["mother", "screen time"]}

# Edge cases
assert parse_predicates("") == []
assert parse_predicates("# only comments\n# here") == []
assert parse_predicates("fact()") == [{"name": "fact", "args": []}]  # zero-arity
```

**What it tests:** The literal core task — predicate extraction is the #1 technical need for the role.

---

### Problem A6: Context Window Manager (Medium)

**Type:** Code Writing

Implement a context window manager that fits as many relevant retrieved chunks as possible into an LLM's context window, prioritized by relevance score.

```python
def build_context(
    chunks: list[dict],  # [{"text": str, "score": float, "source": str}]
    max_tokens: int,
    system_prompt_tokens: int,
    reserve_output_tokens: int
) -> list[dict]:
    """
    Select chunks to fit within the LLM context window.

    Available tokens = max_tokens - system_prompt_tokens - reserve_output_tokens

    Rules:
    1. Sort chunks by score descending (highest relevance first)
    2. Estimate tokens per chunk as: len(chunk["text"]) // 4  (rough char-to-token)
    3. Greedily add chunks until budget is exhausted
    4. Return selected chunks in their original relevance order

    Return list of selected chunk dicts.
    """
    pass

# Test Cases:
chunks = [
    {"text": "A" * 400, "score": 0.9, "source": "guide.pdf"},   # ~100 tokens
    {"text": "B" * 800, "score": 0.7, "source": "faq.pdf"},      # ~200 tokens
    {"text": "C" * 200, "score": 0.95, "source": "manual.pdf"},  # ~50 tokens
    {"text": "D" * 1200, "score": 0.6, "source": "ref.pdf"},     # ~300 tokens
]

selected = build_context(chunks, max_tokens=1000, system_prompt_tokens=200, reserve_output_tokens=300)
# Available: 1000 - 200 - 300 = 500 tokens
# Sorted by score: C(50), A(100), B(200), D(300)
# C(50) + A(100) + B(200) = 350 <= 500 ✓, + D(300) = 650 > 500 ✗
assert len(selected) == 3
assert selected[0]["source"] == "manual.pdf"  # highest score
assert selected[1]["source"] == "guide.pdf"
assert selected[2]["source"] == "faq.pdf"

# Edge: no room for anything
assert build_context(chunks, 100, 50, 50) == []
```

**What it tests:** Practical RAG engineering — managing context windows for reliable retrieval-augmented generation.

---

### Problem A7: Retrieval-Augmented Prompt Builder (Medium)

**Type:** Code Writing

Given retrieved context chunks and a user query, construct a structured prompt that constrains the LLM to only use provided context (critical for healthcare — prevents hallucination).

```python
def build_rag_prompt(
    user_query: str,
    context_chunks: list[dict],  # [{"text": str, "source": str}]
    extracted_predicates: list[str],  # from ASP reasoning
    system_rules: list[str]  # clinician-defined constraints
) -> str:
    """
    Build a constrained RAG prompt for the response generation layer.

    The prompt must follow this structure:
    1. System instruction (hardcoded - see below)
    2. Clinician rules (from system_rules)
    3. Retrieved context (from context_chunks, with source attribution)
    4. Current predicates (from ASP reasoning output)
    5. User query

    System instruction:
        "You are a parent coaching assistant for childhood ADHD.
        ONLY use information from the provided context.
        If the answer is not in the context, say 'I don't have information about that.'
        NEVER provide medical advice, medication recommendations, or diagnoses."

    Format rules section as:
        "## Rules\n- rule1\n- rule2\n..."

    Format context section as:
        "## Context\n[Source: source1]\ntext1\n\n[Source: source2]\ntext2\n..."

    Format predicates section as:
        "## Current Session State\n- pred1\n- pred2\n..."

    Format query section as:
        "## Parent Question\nquery"

    Return the complete prompt string.
    """
    pass

# Test Cases:
prompt = build_rag_prompt(
    user_query="How can I help my child with morning routines?",
    context_chunks=[
        {"text": "Visual schedules help children with ADHD...", "source": "guide.pdf"},
        {"text": "Timers can reduce transition anxiety...", "source": "strategies.pdf"},
    ],
    extracted_predicates=["child_age(7)", "has_diagnosis(adhd)", "challenge(morning_routines)"],
    system_rules=["Only suggest strategies from approved list", "Do not discuss medication"]
)

assert "ONLY use information from the provided context" in prompt
assert "NEVER provide medical advice" in prompt
assert "[Source: guide.pdf]" in prompt
assert "Visual schedules" in prompt
assert "child_age(7)" in prompt
assert "morning routines" in prompt
assert "Do not discuss medication" in prompt

# Verify section ordering (system before rules before context before predicates before query)
sys_idx = prompt.index("NEVER provide medical advice")
rules_idx = prompt.index("## Rules")
ctx_idx = prompt.index("## Context")
pred_idx = prompt.index("## Current Session State")
query_idx = prompt.index("## Parent Question")
assert sys_idx < rules_idx < ctx_idx < pred_idx < query_idx
```

**What it tests:** The response generation layer of STAR — constrained prompt construction for clinical safety.

---

### Problem A8: Hallucination Detector (Hard)

**Type:** Code Writing

Given a set of context chunks and an LLM response, detect claims in the response that are NOT supported by the provided context. This is a critical safety mechanism for healthcare chatbots.

```python
def detect_unsupported_claims(
    response_sentences: list[str],
    context_chunks: list[str],
    threshold: float = 0.3
) -> list[dict]:
    """
    Detect sentences in the LLM response that aren't supported by context.

    Simple approach (no ML needed):
    1. For each response sentence, compute word overlap with each context chunk
    2. Word overlap = |words_in_sentence ∩ words_in_chunk| / |words_in_sentence|
       (use lowercased, alpha-only words; ignore stopwords)
    3. A sentence is "supported" if its max overlap with any chunk >= threshold
    4. Return list of unsupported sentences with their best overlap score

    Stopwords to ignore: {"the", "a", "an", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall", "to",
        "of", "in", "for", "on", "with", "at", "by", "from", "it", "this",
        "that", "and", "or", "but", "not", "no", "if", "then", "so", "as",
        "your", "you", "i", "my", "we", "our", "they", "their", "he", "she"}

    Return: [{"sentence": str, "best_overlap": float}] for unsupported sentences only
    """
    pass

# Test Cases:
context = [
    "Visual schedules help children with ADHD maintain morning routines",
    "Timers reduce transition anxiety for children aged 5 to 10",
    "Positive reinforcement improves behavioral outcomes"
]

response = [
    "Visual schedules can help your child with morning routines.",     # supported
    "Timers are effective for reducing transition anxiety.",            # supported
    "Medication such as Adderall should be considered.",               # NOT supported
    "Studies show 85% improvement rate with these techniques.",        # NOT supported
]

unsupported = detect_unsupported_claims(response, context)
assert len(unsupported) == 2
assert any("Medication" in u["sentence"] or "Adderall" in u["sentence"] for u in unsupported)
assert any("85%" in u["sentence"] for u in unsupported)
assert all(u["best_overlap"] < 0.3 for u in unsupported)
```

**What it tests:** Healthcare safety — detecting when the LLM goes beyond provided context.

---

### Problem A9: LLM Concepts (Multiple Choice)

**Type:** Multiple Choice (6 questions)

**Q1.** In a RAG system, what is the primary purpose of the retrieval step?

- A) To fine-tune the model on domain-specific data
- B) To ground the LLM's response in relevant source documents, reducing hallucination
- C) To reduce the size of the model for faster inference
- D) To translate the user's query into a different language

**Answer:** B

**Explanation:** RAG retrieves relevant documents and injects them into the prompt context, so the LLM generates responses grounded in actual source material rather than relying solely on parametric knowledge. This is especially critical in healthcare where accuracy and traceability matter.

---

**Q2.** In FirstThen's STAR architecture, why is predicate extraction done by an LLM rather than rule-based NLP?

- A) LLMs are faster than rule-based systems
- B) Natural language is ambiguous and varied — LLMs handle paraphrasing, typos, and implicit meaning that rigid patterns miss
- C) Rule-based NLP cannot process English text
- D) LLMs guarantee 100% accuracy on predicate extraction

**Answer:** B

**Explanation:** Parents describe situations in countless ways ("he can't sit still", "focusing is hard for him", "attention problems"). An LLM can map all of these to `challenge(attention)` because it understands semantic meaning, while rules would need explicit patterns for each phrasing.

---

**Q3.** What is the key difference between "retrieval" and "reranking" in a RAG pipeline?

- A) Retrieval uses neural models, reranking uses keyword matching
- B) Retrieval casts a wide net finding candidate documents efficiently; reranking uses a more expensive cross-encoder to precisely score relevance of those candidates
- C) Retrieval and reranking are the same operation
- D) Reranking happens before retrieval

**Answer:** B

**Explanation:** Retrieval (e.g., BM25, bi-encoder) is fast but approximate — it finds top-K candidates. Reranking (e.g., cross-encoder) is slower but more accurate — it jointly encodes query + document to score relevance. This two-stage approach balances speed and quality. Henry built this exact pattern at eXRealityAI with BM25 + FAISS retrieval and BAAI cross-encoder reranking.

---

**Q4.** Why does FirstThen use ASP for conversation logic instead of just prompting the LLM with rules?

- A) ASP is cheaper to run
- B) LLMs cannot follow instructions
- C) ASP provides formal guarantees — rules are deterministic, verifiable, and explainable via justification trees, which is required for clinical safety and regulatory approval
- D) ASP handles natural language better than LLMs

**Answer:** C

**Explanation:** Prompting an LLM with "never recommend medication" is a suggestion it might violate. An ASP integrity constraint `:- recommended(X), medication(X).` is a mathematical guarantee — the solver literally cannot produce an answer set containing a medication recommendation. This is why Forsante Oy achieved EU MDR Class IIb certification using ASP.

---

**Q5.** What is "temperature" in LLM inference, and what setting would you use for predicate extraction in a healthcare chatbot?

- A) A measure of computational load; set high for speed
- B) A sampling parameter controlling output randomness; set LOW (near 0) for deterministic, consistent predicate extraction
- C) A measure of model accuracy; set high for better results
- D) A parameter controlling context window size; irrelevant to extraction

**Answer:** B

**Explanation:** Temperature controls the softmax distribution over next tokens. Temperature ~0 gives near-deterministic output (always picks highest probability token), which is what you want for structured predicate extraction — the same input should always produce the same predicates. For the response generation layer, you might use slightly higher temperature (0.3–0.7) for more natural language.

---

**Q6.** In hybrid retrieval (BM25 + dense vector search), when would BM25 outperform vector search?

- A) When queries contain domain-specific jargon, acronyms, or exact technical terms that embeddings may not capture well
- B) When the corpus is very small
- C) When queries are long paragraphs
- D) BM25 never outperforms vector search

**Answer:** A

**Explanation:** BM25 excels at exact lexical matching. A query like "DSM-5 ADHD criteria" benefits from BM25 because the exact tokens "DSM-5" match directly. Dense embeddings might map this to semantically similar but lexically different passages. This is why hybrid retrieval works — BM25 catches exact matches while vectors catch semantic matches. FirstThen's clinical terminology makes hybrid retrieval essential.

---

### Problem A10: Embedding Index Lookup (Easy)

**Type:** Code Writing

Implement a simple in-memory vector index that stores document embeddings and retrieves the top-K most similar documents to a query.

```python
def create_index(documents: list[dict]) -> dict:
    """
    Create a simple vector index from documents.
    Each document has: {"id": str, "text": str, "embedding": list[float]}
    Return the index structure (you choose the format).
    """
    pass

def query_index(index: dict, query_embedding: list[float], top_k: int) -> list[str]:
    """
    Find the top_k most similar document IDs to the query embedding.
    Use cosine similarity.
    Return list of document IDs sorted by similarity descending.
    """
    pass

# Test Cases:
docs = [
    {"id": "doc1", "text": "ADHD morning routines", "embedding": [1.0, 0.0, 0.0]},
    {"id": "doc2", "text": "medication guidelines", "embedding": [0.0, 1.0, 0.0]},
    {"id": "doc3", "text": "behavioral strategies", "embedding": [0.7, 0.3, 0.0]},
    {"id": "doc4", "text": "sleep hygiene tips", "embedding": [0.0, 0.0, 1.0]},
]

index = create_index(docs)
results = query_index(index, [0.9, 0.1, 0.0], top_k=2)
assert results[0] == "doc1"  # most similar to [0.9, 0.1, 0.0]
assert results[1] == "doc3"  # second most similar
assert len(results) == 2
```

**What it tests:** Core vector search mechanic — what FAISS does under the hood.

---

### Problem A11: Predicate Validation Pipeline (Hard)

**Type:** Code Writing

Implement a validation pipeline that checks LLM-extracted predicates against a schema of allowed predicates and value ranges. This prevents malformed or dangerous predicates from reaching the ASP engine.

```python
PREDICATE_SCHEMA = {
    "child_age": {"arity": 1, "arg_types": ["int"], "constraints": {"0": {"min": 2, "max": 17}}},
    "has_diagnosis": {"arity": 1, "arg_types": ["enum"], "constraints": {"0": {"values": ["adhd", "anxiety", "asd", "odd"]}}},
    "challenge": {"arity": 1, "arg_types": ["enum"], "constraints": {"0": {"values": [
        "morning_routines", "homework", "transitions", "emotional_regulation",
        "social_skills", "sleep", "focus", "impulsivity"
    ]}}},
    "severity": {"arity": 1, "arg_types": ["enum"], "constraints": {"0": {"values": ["mild", "moderate", "severe"]}}},
    "parent_concern": {"arity": 2, "arg_types": ["enum", "str"], "constraints": {"0": {"values": ["mother", "father", "guardian"]}}},
    "session_phase": {"arity": 1, "arg_types": ["enum"], "constraints": {"0": {"values": ["intake", "strategy", "practice", "followup"]}}},
}

def validate_predicates(
    predicates: list[dict],  # [{"name": str, "args": list}]
    schema: dict
) -> dict:
    """
    Validate extracted predicates against schema.

    Return:
    {
        "valid": [list of valid predicate dicts],
        "invalid": [{"predicate": dict, "error": str} for each invalid one]
    }

    Validation rules:
    1. Predicate name must exist in schema
    2. Number of args must match arity
    3. Each arg must match its declared type:
       - "int": must be an integer
       - "enum": must be in the allowed values list
       - "str": any string is ok
    4. For "int" type, value must satisfy min/max constraints if present
    """
    pass

# Test Cases:
predicates = [
    {"name": "child_age", "args": [7]},           # valid
    {"name": "has_diagnosis", "args": ["adhd"]},   # valid
    {"name": "child_age", "args": [25]},           # invalid: age > 17
    {"name": "challenge", "args": ["flying"]},     # invalid: not in enum
    {"name": "unknown_pred", "args": ["x"]},       # invalid: not in schema
    {"name": "child_age", "args": [7, 8]},         # invalid: wrong arity
    {"name": "severity", "args": ["moderate"]},    # valid
    {"name": "parent_concern", "args": ["mother", "screen time"]},  # valid
]

result = validate_predicates(predicates, PREDICATE_SCHEMA)
assert len(result["valid"]) == 4
assert len(result["invalid"]) == 4
assert any("not in schema" in e["error"].lower() or "unknown" in e["error"].lower()
           for e in result["invalid"])
assert any("arity" in e["error"].lower() or "argument" in e["error"].lower()
           for e in result["invalid"])
```

**What it tests:** The validation layer between LLM extraction and ASP reasoning — critical for system reliability.

---

### Problem A12: Conversation History Compression (Medium)

**Type:** Code Writing

When conversations get long, you need to compress older turns to fit within the context window while preserving key information. Implement a conversation compressor.

```python
def compress_conversation(
    turns: list[dict],  # [{"role": "parent"|"coach", "text": str, "turn_num": int}]
    max_recent_turns: int,
    max_summary_tokens: int
) -> dict:
    """
    Compress a conversation history for context window management.

    Strategy:
    1. Keep the last max_recent_turns turns verbatim (most recent context)
    2. For older turns, create a summary:
       - Extract key facts mentioned (sentences containing predicates-like
         patterns: ages, diagnoses, challenges)
       - A "key fact" is any sentence containing a number, or any of these
         keywords: ["adhd", "diagnosis", "challenge", "age", "medication",
                    "strategy", "concern", "behavior", "school", "routine"]
       - Deduplicate key facts (case-insensitive exact match)
    3. Estimate tokens as len(text) // 4

    Return:
    {
        "summary": str,  # joined key facts from older turns, separated by ". "
        "recent_turns": list[dict],  # last N turns unchanged
        "total_estimated_tokens": int
    }
    """
    pass

# Test Cases:
turns = [
    {"role": "parent", "text": "My child is 7 years old and has ADHD.", "turn_num": 1},
    {"role": "coach", "text": "Thank you for sharing. What challenges are you facing?", "turn_num": 2},
    {"role": "parent", "text": "Morning routines are really hard. The school says he can't focus.", "turn_num": 3},
    {"role": "coach", "text": "That's common with ADHD. Let me suggest some strategies.", "turn_num": 4},
    {"role": "parent", "text": "We tried visual schedules but they didn't work.", "turn_num": 5},
    {"role": "coach", "text": "Let's adjust the approach. How about timers?", "turn_num": 6},
]

result = compress_conversation(turns, max_recent_turns=2, max_summary_tokens=200)
assert len(result["recent_turns"]) == 2
assert result["recent_turns"][0]["turn_num"] == 5
assert result["recent_turns"][1]["turn_num"] == 6
assert "7 years old" in result["summary"] or "ADHD" in result["summary"]
assert "school" in result["summary"].lower() or "routine" in result["summary"].lower()
assert result["total_estimated_tokens"] > 0
```

**What it tests:** Practical conversation management — maintaining context across long coaching sessions.

---

## Section B: Agentic Protocols (8 Problems)

**Priority: MEDIUM-HIGH** — The role involves agentic protocols for orchestrating the chatbot system. Henry has LangGraph experience.

---

### Problem B1: Conversation State Machine (Medium)

**Type:** Code Writing

Implement a finite state machine that manages conversation phases for the ADHD coaching chatbot. This is the Python-side orchestrator that works alongside the ASP reasoning engine.

```python
from enum import Enum

class Phase(Enum):
    INTAKE = "intake"
    STRATEGY = "strategy"
    PRACTICE = "practice"
    FOLLOWUP = "followup"
    ENDED = "ended"

TRANSITIONS = {
    Phase.INTAKE: {
        "intake_complete": Phase.STRATEGY,
        "user_exit": Phase.ENDED,
    },
    Phase.STRATEGY: {
        "strategy_selected": Phase.PRACTICE,
        "need_more_info": Phase.INTAKE,
        "user_exit": Phase.ENDED,
    },
    Phase.PRACTICE: {
        "practice_complete": Phase.FOLLOWUP,
        "change_strategy": Phase.STRATEGY,
        "user_exit": Phase.ENDED,
    },
    Phase.FOLLOWUP: {
        "new_challenge": Phase.INTAKE,
        "session_end": Phase.ENDED,
        "user_exit": Phase.ENDED,
    },
    Phase.ENDED: {},
}

class ConversationFSM:
    def __init__(self):
        self.state = Phase.INTAKE
        self.history: list[tuple[Phase, str, Phase]] = []  # (from, event, to)

    def trigger(self, event: str) -> bool:
        """
        Attempt a state transition. Return True if successful, False if invalid.
        Log successful transitions to history.
        """
        pass

    def get_available_events(self) -> list[str]:
        """Return list of valid events for the current state."""
        pass

    def can_reach(self, target: Phase) -> bool:
        """
        Return True if target phase is reachable from current state
        via any sequence of transitions. Use BFS/DFS.
        """
        pass

# Test Cases:
fsm = ConversationFSM()
assert fsm.state == Phase.INTAKE
assert fsm.get_available_events() == ["intake_complete", "user_exit"]

assert fsm.trigger("intake_complete") == True
assert fsm.state == Phase.STRATEGY

assert fsm.trigger("intake_complete") == False  # invalid from STRATEGY
assert fsm.state == Phase.STRATEGY  # unchanged

assert fsm.trigger("strategy_selected") == True
assert fsm.state == Phase.PRACTICE

assert fsm.trigger("change_strategy") == True
assert fsm.state == Phase.STRATEGY

assert len(fsm.history) == 3

# Reachability
fsm2 = ConversationFSM()
assert fsm2.can_reach(Phase.FOLLOWUP) == True
assert fsm2.can_reach(Phase.ENDED) == True

fsm3 = ConversationFSM()
fsm3.trigger("user_exit")
assert fsm3.can_reach(Phase.STRATEGY) == False  # ENDED has no transitions
```

**What it tests:** Conversation orchestration — the agentic layer that manages coaching session flow.

---

### Problem B2: Tool Call Parser & Executor (Medium)

**Type:** Code Writing

Implement a system that parses LLM-generated tool calls and executes them safely. This is the core of agent-tool interaction.

```python
import json

AVAILABLE_TOOLS = {
    "lookup_strategy": {
        "description": "Look up an ADHD strategy by name",
        "params": {"strategy_name": "str"},
        "handler": lambda params: f"Strategy '{params['strategy_name']}': Use visual cues and timers."
    },
    "get_child_profile": {
        "description": "Get the child's profile",
        "params": {"child_id": "str"},
        "handler": lambda params: f"Child {params['child_id']}: age 7, diagnosis ADHD, challenges: morning routines"
    },
    "log_session_note": {
        "description": "Log a note for the current session",
        "params": {"note": "str", "category": "str"},
        "handler": lambda params: f"Logged '{params['note']}' under {params['category']}"
    },
}

def parse_and_execute_tool_calls(llm_output: str, tools: dict) -> list[dict]:
    """
    Parse tool calls from LLM output and execute them.

    LLM output format contains tool calls as JSON blocks:
        <tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>

    There may be regular text between tool calls.

    For each tool call:
    1. Parse the JSON
    2. Validate tool name exists in available tools
    3. Validate all required params are present
    4. Execute the handler
    5. Collect results

    Return: [
        {"tool": str, "arguments": dict, "result": str, "status": "success"|"error", "error": str|None}
    ]
    """
    pass

# Test Cases:
llm_output = """Let me look up that strategy for you.
<tool_call>{"name": "lookup_strategy", "arguments": {"strategy_name": "visual schedules"}}</tool_call>
And also check the child's profile.
<tool_call>{"name": "get_child_profile", "arguments": {"child_id": "child_001"}}</tool_call>
"""

results = parse_and_execute_tool_calls(llm_output, AVAILABLE_TOOLS)
assert len(results) == 2
assert results[0]["status"] == "success"
assert "visual schedules" in results[0]["result"]
assert results[1]["status"] == "success"
assert "child_001" in results[1]["result"]

# Error case: unknown tool
error_output = '<tool_call>{"name": "delete_user", "arguments": {"id": "1"}}</tool_call>'
error_results = parse_and_execute_tool_calls(error_output, AVAILABLE_TOOLS)
assert error_results[0]["status"] == "error"

# Error case: missing params
bad_params = '<tool_call>{"name": "lookup_strategy", "arguments": {}}</tool_call>'
bad_results = parse_and_execute_tool_calls(bad_params, AVAILABLE_TOOLS)
assert bad_results[0]["status"] == "error"
```

**What it tests:** Agentic tool use — how agents interact with external capabilities safely.

---

### Problem B3: Supervisor Agent Router (Medium)

**Type:** Code Writing

Implement a supervisor agent that routes incoming messages to specialized sub-agents based on intent classification. This mirrors FirstThen's multi-agent architecture.

```python
AGENT_REGISTRY = {
    "intake_agent": {
        "description": "Handles initial child/family information gathering",
        "keywords": ["age", "child", "name", "diagnosis", "family", "tell me about", "introduce"],
        "priority": 1,
    },
    "strategy_agent": {
        "description": "Recommends ADHD management strategies",
        "keywords": ["strategy", "help", "technique", "approach", "how to", "what can i do", "tip", "suggest"],
        "priority": 2,
    },
    "safety_agent": {
        "description": "Handles safety-critical topics (medication, crisis, self-harm)",
        "keywords": ["medication", "medicine", "drug", "crisis", "emergency", "harm", "suicide", "hurt", "dangerous"],
        "priority": 0,  # highest priority — always checked first
    },
    "progress_agent": {
        "description": "Tracks and discusses progress over time",
        "keywords": ["progress", "better", "worse", "improvement", "working", "update", "last time", "before"],
        "priority": 3,
    },
}

def route_message(
    message: str,
    agent_registry: dict,
    conversation_phase: str  # current FSM phase
) -> dict:
    """
    Route a message to the appropriate agent.

    Routing logic:
    1. Check all agents by priority order (lower number = higher priority)
    2. For each agent, count keyword matches (case-insensitive, substring match)
    3. Safety agent ALWAYS wins if it has ANY match (priority 0 override)
    4. Otherwise, agent with most keyword matches wins
    5. Ties broken by priority (lower = wins)
    6. If no keywords match any agent, route based on conversation_phase:
       - "intake" -> intake_agent
       - "strategy" -> strategy_agent
       - "practice" -> strategy_agent
       - "followup" -> progress_agent

    Return: {"agent": str, "confidence": "high"|"medium"|"low", "matches": int}
    - high: 3+ keyword matches
    - medium: 1-2 keyword matches
    - low: 0 matches (routed by phase)
    """
    pass

# Test Cases:
result = route_message("My child is 7 years old and was diagnosed last year",
                       AGENT_REGISTRY, "intake")
assert result["agent"] == "intake_agent"

result = route_message("What strategies can help with morning routines?",
                       AGENT_REGISTRY, "strategy")
assert result["agent"] == "strategy_agent"

# Safety override
result = route_message("Should we try medication for focus?",
                       AGENT_REGISTRY, "strategy")
assert result["agent"] == "safety_agent"

# Phase-based fallback
result = route_message("Okay, sounds good",
                       AGENT_REGISTRY, "intake")
assert result["agent"] == "intake_agent"
assert result["confidence"] == "low"

# Safety always wins even with other matches
result = route_message("My child's medication is making their behavior worse, any strategy tips?",
                       AGENT_REGISTRY, "strategy")
assert result["agent"] == "safety_agent"  # safety override
```

**What it tests:** Multi-agent orchestration — routing with safety overrides, directly relevant to FirstThen's architecture.

---

### Problem B4: Agent Memory Store (Easy)

**Type:** Code Writing

Implement a simple agent memory that persists facts across conversation turns, supporting both short-term (session) and long-term (cross-session) storage.

```python
class AgentMemory:
    def __init__(self):
        self.short_term: dict[str, any] = {}  # session-scoped
        self.long_term: dict[str, any] = {}   # persists across sessions

    def store(self, key: str, value, long_term: bool = False) -> None:
        """Store a fact. If long_term=True, store in long-term memory too."""
        pass

    def recall(self, key: str) -> any:
        """
        Recall a fact. Check short-term first, then long-term.
        Return None if not found.
        """
        pass

    def get_session_summary(self) -> dict:
        """Return all short-term memory as a dict."""
        pass

    def clear_session(self) -> None:
        """Clear short-term memory (simulate new session). Long-term persists."""
        pass

    def search(self, query: str) -> list[tuple[str, any]]:
        """
        Search for keys containing the query substring (case-insensitive).
        Search both short-term and long-term.
        Return list of (key, value) tuples. No duplicates — short-term wins.
        """
        pass

# Test Cases:
mem = AgentMemory()
mem.store("child_age", 7, long_term=True)
mem.store("current_topic", "morning routines")
mem.store("child_name", "Alex", long_term=True)

assert mem.recall("child_age") == 7
assert mem.recall("current_topic") == "morning routines"
assert mem.recall("nonexistent") is None

# Session clear
mem.clear_session()
assert mem.recall("current_topic") is None  # short-term cleared
assert mem.recall("child_age") == 7         # long-term persists

# Search
mem.store("child_challenge_1", "focus")
mem.store("child_challenge_2", "transitions")
results = mem.search("child")
assert len(results) >= 3  # child_age, child_name, child_challenge_1, child_challenge_2
```

**What it tests:** Agent state management — maintaining context across interactions.

---

### Problem B5: Retry with Exponential Backoff (Easy)

**Type:** Code Writing

Implement a retry decorator for unreliable API calls (LLM APIs, database connections). Essential for production agent systems.

```python
import time
import random

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (Exception,)
):
    """
    Decorator that retries a function with exponential backoff.

    Backoff formula: delay = min(base_delay * 2^attempt, max_delay)
    If jitter=True: delay = delay * random.uniform(0.5, 1.5)

    Should re-raise the last exception if all retries fail.
    Should NOT retry exceptions not in retryable_exceptions.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            pass  # implement
        return wrapper
    return decorator

# Test Cases:
call_count = 0

@retry_with_backoff(max_retries=3, base_delay=0.01, jitter=False,
                     retryable_exceptions=(ConnectionError,))
def flaky_api_call():
    global call_count
    call_count += 1
    if call_count < 3:
        raise ConnectionError("API timeout")
    return "success"

call_count = 0
result = flaky_api_call()
assert result == "success"
assert call_count == 3  # failed twice, succeeded on third

# Non-retryable exception should propagate immediately
@retry_with_backoff(max_retries=3, base_delay=0.01,
                     retryable_exceptions=(ConnectionError,))
def bad_input():
    raise ValueError("Invalid input")

try:
    bad_input()
    assert False, "Should have raised"
except ValueError:
    pass  # correct — ValueError is not retryable

# All retries exhausted
fail_count = 0
@retry_with_backoff(max_retries=2, base_delay=0.01, jitter=False,
                     retryable_exceptions=(ConnectionError,))
def always_fails():
    global fail_count
    fail_count += 1
    raise ConnectionError("down")

fail_count = 0
try:
    always_fails()
    assert False
except ConnectionError:
    assert fail_count == 3  # initial + 2 retries
```

**What it tests:** Production reliability — every agentic system needs robust error handling for API calls.

---

### Problem B6: Agent Pipeline Orchestrator (Hard)

**Type:** Code Writing

Implement the 3-stage STAR pipeline orchestrator that sequences: (1) LLM predicate extraction → (2) ASP reasoning → (3) LLM response generation. Handle errors at each stage with fallbacks.

```python
class STARPipeline:
    """
    Orchestrates the 3-stage STAR pipeline:
    Stage 1: extract_predicates(user_input) -> list[dict]
    Stage 2: asp_reason(predicates, context) -> dict with "actions" and "response_hints"
    Stage 3: generate_response(asp_output, user_input) -> str
    """

    def __init__(self, extractor, reasoner, generator):
        """
        extractor: callable(str) -> list[dict]  # can raise ExtractionError
        reasoner: callable(list[dict], dict) -> dict  # can raise ReasoningError
        generator: callable(dict, str) -> str  # can raise GenerationError
        """
        self.extractor = extractor
        self.reasoner = reasoner
        self.generator = generator
        self.logs: list[dict] = []

    def run(self, user_input: str, context: dict) -> dict:
        """
        Execute the full pipeline with error handling.

        Fallback behavior:
        - If extraction fails: use empty predicates [] and log warning
        - If reasoning fails: use default_actions = {"actions": ["acknowledge"],
          "response_hints": ["I understand. Could you tell me more?"]}
        - If generation fails: return the response_hints[0] as plain text

        Return:
        {
            "response": str,
            "predicates": list[dict],
            "asp_output": dict,
            "stages_completed": list[str],  # e.g., ["extraction", "reasoning", "generation"]
            "errors": list[dict]  # [{"stage": str, "error": str}]
        }
        """
        pass

# Test Cases:
class ExtractionError(Exception): pass
class ReasoningError(Exception): pass
class GenerationError(Exception): pass

# Happy path
pipeline = STARPipeline(
    extractor=lambda inp: [{"name": "child_age", "args": [7]}],
    reasoner=lambda preds, ctx: {"actions": ["suggest_strategy"], "response_hints": ["Try visual schedules"]},
    generator=lambda asp, inp: "Based on your child's age, I'd recommend visual schedules!"
)

result = pipeline.run("My child is 7", {})
assert result["response"] == "Based on your child's age, I'd recommend visual schedules!"
assert len(result["stages_completed"]) == 3
assert len(result["errors"]) == 0

# Extraction failure — should still produce a response
pipeline_bad_extract = STARPipeline(
    extractor=lambda inp: (_ for _ in ()).throw(ExtractionError("parse failed")),
    reasoner=lambda preds, ctx: {"actions": ["acknowledge"], "response_hints": ["Tell me more"]},
    generator=lambda asp, inp: "Could you tell me more about your child?"
)

result2 = pipeline_bad_extract.run("asdfgh", {})
assert result2["predicates"] == []
assert "extraction" not in result2["stages_completed"]
assert len(result2["errors"]) == 1
assert result2["response"] == "Could you tell me more about your child?"

# All stages fail — should still return something
pipeline_all_fail = STARPipeline(
    extractor=lambda inp: (_ for _ in ()).throw(ExtractionError("fail")),
    reasoner=lambda preds, ctx: (_ for _ in ()).throw(ReasoningError("fail")),
    generator=lambda asp, inp: (_ for _ in ()).throw(GenerationError("fail")),
)

result3 = pipeline_all_fail.run("hello", {})
assert result3["response"] == "I understand. Could you tell me more?"
assert len(result3["errors"]) == 3
```

**What it tests:** The actual STAR pipeline orchestration — the core architecture of FirstThen's system.

---

### Problem B7: Rate-Limited Agent Queue (Medium)

**Type:** Code Writing

Implement a task queue that respects API rate limits when making LLM calls. Agents often need to batch and throttle requests.

```python
from collections import deque
import time

class RateLimitedQueue:
    def __init__(self, max_calls_per_minute: int):
        self.max_calls = max_calls_per_minute
        self.call_timestamps: deque = deque()
        self.queue: deque = deque()

    def can_call_now(self, current_time: float) -> bool:
        """
        Check if a call can be made right now without exceeding rate limit.
        Clean up timestamps older than 60 seconds.
        """
        pass

    def next_available_time(self, current_time: float) -> float:
        """
        Return the earliest time a call can be made.
        If can_call_now, return current_time.
        Otherwise, return when the oldest call in the window expires + epsilon.
        """
        pass

    def record_call(self, current_time: float) -> None:
        """Record that a call was made at current_time."""
        pass

    def enqueue(self, task: dict) -> None:
        """Add a task to the queue. Task has: {"id": str, "payload": any}"""
        pass

    def process_next(self, current_time: float) -> dict | None:
        """
        If queue is non-empty and we can call now:
        - Record the call
        - Dequeue and return the task
        If queue is empty, return None.
        If rate limited, return {"wait_until": float}
        """
        pass

# Test Cases:
q = RateLimitedQueue(max_calls_per_minute=3)
t = 1000.0

q.enqueue({"id": "1", "payload": "extract predicates"})
q.enqueue({"id": "2", "payload": "generate response"})
q.enqueue({"id": "3", "payload": "validate output"})
q.enqueue({"id": "4", "payload": "log session"})

# First 3 should process immediately
r1 = q.process_next(t)
assert r1["id"] == "1"
r2 = q.process_next(t + 0.1)
assert r2["id"] == "2"
r3 = q.process_next(t + 0.2)
assert r3["id"] == "3"

# 4th should be rate limited
r4 = q.process_next(t + 0.3)
assert "wait_until" in r4

# After 60 seconds, should work again
r4_retry = q.process_next(t + 61)
assert r4_retry["id"] == "4"
```

**What it tests:** Production agent infrastructure — managing API rate limits for LLM calls.

---

### Problem B8: Agentic Concepts (Multiple Choice)

**Type:** Multiple Choice (4 questions)

**Q1.** What is the "supervisor" pattern in multi-agent systems?

- A) A single agent that handles all tasks sequentially
- B) A top-level agent that routes tasks to specialized sub-agents, aggregates their results, and decides next steps
- C) A monitoring system that logs agent performance
- D) A pattern where agents supervise human users

**Answer:** B

**Explanation:** The supervisor pattern (used in LangGraph, CrewAI, etc.) has a coordinator agent that understands each sub-agent's capabilities and delegates work accordingly. In FirstThen's context, a supervisor could route to intake, strategy, safety, and progress agents. Henry built this pattern in his Suspect Detection project with LangGraph.

---

**Q2.** Why would FirstThen's chatbot benefit from an agentic architecture over a single monolithic LLM?

- A) Agents are faster than a single LLM
- B) Separation of concerns: each agent specializes (safety, intake, strategy), making the system more maintainable, testable, and allowing the safety agent to have hard override authority
- C) Agents use less memory
- D) Agents don't need LLMs

**Answer:** B

**Explanation:** A monolithic prompt trying to handle safety checks, intake gathering, strategy recommendation, AND response generation would be fragile and hard to test. With agents, you can unit-test the safety agent independently, update strategy recommendations without touching intake logic, and ensure the safety agent always has veto power regardless of other agents' outputs.

---

**Q3.** What is "tool use" (function calling) in the context of LLM agents?

- A) The LLM directly executes code on the server
- B) The LLM generates structured requests to invoke pre-defined functions; an orchestrator executes them and returns results to the LLM
- C) The user manually calls functions and pastes results
- D) Tool use means the LLM can access the internet

**Answer:** B

**Explanation:** The LLM outputs a structured tool call (e.g., JSON with function name and arguments). The host application validates and executes the function, then feeds the result back to the LLM for the next step. This is how agents interact with databases, APIs, ASP solvers, etc. The LLM never directly executes code — the orchestrator mediates.

---

**Q4.** In a healthcare chatbot, why should the safety agent have "hard override" authority rather than just advisory input?

- A) It's more efficient
- B) Advisory safety can be overridden by other agents or the LLM — hard override ensures dangerous responses (medication advice, crisis mishandling) are BLOCKED regardless of what other components suggest, providing a deterministic safety guarantee
- C) Hard override makes the chatbot friendlier
- D) It's required by HIPAA

**Answer:** B

**Explanation:** If the safety agent merely advises "this might be about medication," another agent could still generate a response mentioning dosages. With hard override, the safety agent can intercept and replace any response that violates clinical boundaries — similar to how ASP integrity constraints provide mathematical guarantees. This is defense-in-depth for clinical safety.

---

## Section C: APIs (5 Problems)

**Priority: LOW-MEDIUM** — The role mentions "production software development (APIs, databases, deployment pipelines)" but the focus is on AI/ML. These are conceptual/easy to confirm baseline competency.

---

### Problem C1: REST API Design (Multiple Choice)

**Type:** Multiple Choice (4 questions)

**Q1.** A FastAPI endpoint receives a parent's message and returns the chatbot's response. Which HTTP method and status code are most appropriate?

- A) GET /chat?message=hello → 200 OK
- B) POST /chat with JSON body {"message": "hello"} → 200 OK
- C) PUT /chat with JSON body {"message": "hello"} → 201 Created
- D) POST /chat with JSON body {"message": "hello"} → 201 Created

**Answer:** B

**Explanation:** POST is correct because sending a message is an action with side effects (creates a response, updates conversation state). GET should be idempotent and not have side effects. 200 OK (not 201) because we're not creating a persisted resource — we're processing a request and returning a result. PUT implies replacing a resource at that URL, which doesn't fit a chat interaction.

---

**Q2.** You need to design an API for the STAR pipeline that returns intermediate results (predicates extracted, ASP reasoning output, and final response). Which pattern is most appropriate?

- A) Three separate endpoints called sequentially by the client
- B) One endpoint that returns all stages in a nested JSON response
- C) WebSocket connection streaming each stage as it completes
- D) B for synchronous use, C for real-time UI updates — both are valid depending on the client's needs

**Answer:** D

**Explanation:** For a simple integration, a single POST endpoint returning `{"predicates": [...], "asp_output": {...}, "response": "..."}` is clean and easy. For a real-time UI that shows "Extracting predicates... Reasoning... Generating response...", a WebSocket or SSE stream is better. Good API design considers multiple access patterns.

---

**Q3.** What is the purpose of request validation (e.g., Pydantic models in FastAPI)?

- A) To make the API slower for security
- B) To ensure incoming data matches expected types and constraints BEFORE processing, providing clear error messages and preventing invalid data from reaching business logic
- C) To encrypt the request
- D) To compress the request payload

**Answer:** B

**Explanation:** In a healthcare chatbot, you want to validate that `child_age` is an integer between 2-17, that `message` isn't empty, and that session IDs are valid UUIDs — all before hitting the LLM or ASP engine. Pydantic in FastAPI does this automatically and returns 422 errors with specific field-level feedback.

---

**Q4.** Your chatbot API needs to handle a scenario where the LLM API is down. What's the best practice?

- A) Return 200 with an empty response
- B) Return 503 Service Unavailable with a retry-after header and a fallback message from the ASP layer's response hints
- C) Crash the server
- D) Wait indefinitely for the LLM to respond

**Answer:** B

**Explanation:** 503 tells the client the issue is temporary. The retry-after header helps clients implement backoff. Returning a fallback from the ASP layer (which doesn't depend on the LLM API) means the system degrades gracefully rather than failing completely. This is the benefit of the hybrid architecture — the ASP layer is always available.

---

### Problem C2: FastAPI Endpoint (Easy)

**Type:** Code Writing

Write a FastAPI endpoint for the chatbot that validates input, calls the pipeline, and returns structured output.

```python
from pydantic import BaseModel, Field, field_validator

# Define the request/response models and endpoint

class ChatRequest(BaseModel):
    """
    Validate:
    - message: non-empty string, max 2000 chars
    - session_id: non-empty string
    - child_age: optional int, if provided must be 2-17
    """
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., min_length=1)
    child_age: int | None = Field(None, ge=2, le=17)

class ChatResponse(BaseModel):
    response: str
    session_id: str
    predicates_extracted: list[dict]
    confidence: float = Field(..., ge=0.0, le=1.0)

# Write the endpoint function signature and logic:
# POST /chat
# - Accept ChatRequest
# - Return ChatResponse
# - If message contains the word "medication" or "drug", return a safety response
#   with confidence 1.0 and predicates [{"name": "safety_flag", "args": ["medication"]}]
# - Otherwise return a normal response with confidence 0.8

# Your code here:
# from fastapi import FastAPI
# app = FastAPI()
#
# @app.post("/chat", response_model=ChatResponse)
# def chat(request: ChatRequest) -> ChatResponse:
#     pass
```

**What it tests:** API development basics — structuring endpoints for the chatbot service.

---

### Problem C3: API Authentication Middleware (Easy)

**Type:** Code Writing

Implement a simple API key authentication check. Production chatbot APIs need authentication.

```python
def authenticate_request(
    headers: dict[str, str],
    valid_api_keys: set[str]
) -> dict:
    """
    Check if the request has a valid API key.

    Rules:
    - API key should be in header "X-API-Key"
    - Return {"authenticated": True, "key": str} if valid
    - Return {"authenticated": False, "error": str} if:
      - Header missing: "Missing X-API-Key header"
      - Key invalid: "Invalid API key"
    """
    pass

# Test Cases:
valid_keys = {"key_abc123", "key_xyz789"}

assert authenticate_request({"X-API-Key": "key_abc123"}, valid_keys)["authenticated"] == True
assert authenticate_request({"X-API-Key": "wrong_key"}, valid_keys)["authenticated"] == False
assert authenticate_request({}, valid_keys)["authenticated"] == False
assert "Missing" in authenticate_request({}, valid_keys)["error"]
```

**What it tests:** Basic API security — authentication for the chatbot service.

---

## Section D: Databases (5 Problems)

**Priority: LOW** — Mentioned in JD but clearly not the focus. These confirm baseline knowledge.

---

### Problem D1: Database Concepts (Multiple Choice)

**Type:** Multiple Choice (4 questions)

**Q1.** FirstThen needs to store conversation history. Which database type is most appropriate?

- A) Relational (PostgreSQL) with a fixed schema for messages
- B) Document store (MongoDB) with flexible session documents containing nested message arrays
- C) Graph database (Neo4j)
- D) Both A and B are reasonable — relational for structured data (user profiles, session metadata) and document store for flexible conversation data. Many production systems use both.

**Answer:** D

**Explanation:** Conversation data is semi-structured (variable-length turns, nested metadata, different types of extracted predicates). A document store handles this naturally. But user profiles, billing, and analytics benefit from relational structure. The right answer in practice is often a polyglot persistence approach.

---

**Q2.** You're designing the schema to store extracted predicates from conversations. Which approach is best?

- A) One column per possible predicate (child_age, has_diagnosis, etc.)
- B) A JSON/JSONB column storing predicates as a flexible structure
- C) A normalized predicate table: (session_id, predicate_name, arg_position, arg_value)
- D) B for fast reads and simple queries; C if you need to query across sessions by specific predicate values (e.g., "find all sessions where child_age > 10")

**Answer:** D

**Explanation:** JSONB is great for storing and retrieving predicates for a single session. But if you need analytics queries like "what's the most common challenge?" or "average child age?", a normalized table lets you use SQL aggregations. Many teams start with JSONB and add a normalized projection when analytics needs arise.

---

**Q3.** What is the purpose of database indexing, and which column would you index first for the conversation history table?

- A) Indexing speeds up writes; index the message text column
- B) Indexing speeds up reads by creating a lookup structure; index session_id since most queries fetch all messages for a given session
- C) Indexing prevents data corruption; index the primary key only
- D) Indexing compresses data; index all columns

**Answer:** B

**Explanation:** The most common query pattern is "get conversation history for session X" → index session_id. You might also index (session_id, created_at) for ordered retrieval. Indexing message text would require a full-text index (like GIN in PostgreSQL) and is only needed if you're searching message content.

---

**Q4.** What does ACID stand for, and which property is most important for the chatbot's conversation state?

- A) Atomicity, Consistency, Isolation, Durability — Durability, so conversation history isn't lost
- B) Atomicity, Consistency, Isolation, Durability — Consistency, so the conversation state is always valid
- C) Authentication, Compression, Indexing, Duplication — all equally important
- D) Both A and B are important: Durability ensures history persists; Consistency ensures state invariants (e.g., session must have a user) are maintained

**Answer:** D

**Explanation:** For a healthcare chatbot, losing conversation history (durability failure) would be a serious issue — parents shouldn't have to repeat intake information. But consistency is also critical — you never want a session in an invalid state (e.g., predicates without a session, messages without a user). In practice, both matter.

---

### Problem D2: SQL for Conversation Analytics (Easy)

**Type:** Code Writing

Write SQL queries for analyzing chatbot conversation data.

```sql
-- Given these tables:
--
-- sessions(session_id, user_id, created_at, phase, ended_at)
-- messages(message_id, session_id, role, content, created_at)
-- predicates(pred_id, session_id, name, args_json, extracted_at)

-- Q1: Find the average number of messages per session
-- Your query:


-- Q2: Find the top 5 most frequently extracted predicate names
-- Your query:


-- Q3: Find all sessions where the child_age predicate was extracted with a value
--      (args_json) containing a number greater than 10
--      Assume args_json is stored like '[7]' or '[12]'
-- Your query:


-- Q4: Find users who had sessions stuck in "intake" phase for more than 10 messages
--     (potential UX issue — intake should be brief)
-- Your query:
```

**Expected Answers:**

```sql
-- Q1
SELECT AVG(msg_count) AS avg_messages_per_session
FROM (
    SELECT session_id, COUNT(*) AS msg_count
    FROM messages
    GROUP BY session_id
) sub;

-- Q2
SELECT name, COUNT(*) AS frequency
FROM predicates
GROUP BY name
ORDER BY frequency DESC
LIMIT 5;

-- Q3
SELECT DISTINCT s.*
FROM sessions s
JOIN predicates p ON s.session_id = p.session_id
WHERE p.name = 'child_age'
  AND CAST(REPLACE(REPLACE(p.args_json, '[', ''), ']', '') AS INTEGER) > 10;

-- Q4
SELECT s.user_id, s.session_id, COUNT(m.message_id) AS msg_count
FROM sessions s
JOIN messages m ON s.session_id = m.session_id
WHERE s.phase = 'intake'
GROUP BY s.user_id, s.session_id
HAVING COUNT(m.message_id) > 10;
```

**What it tests:** Basic SQL competency for analytics on chatbot data.

---

## Summary: Problem Distribution

| Section | Topic | # Problems | Complexity | Rationale |
|---------|-------|-----------|------------|-----------|
| A | RAG & LLM | 12 (9 code + 6 MC) | Easy-Hard | Core to role — predicate extraction, retrieval, safety |
| B | Agentic Protocols | 8 (7 code + 4 MC) | Easy-Hard | Important — pipeline orchestration, multi-agent routing |
| C | APIs | 5 (2 code + 4 MC) | Easy | Secondary — confirm baseline production skills |
| D | Databases | 5 (1 code + 4 MC) | Easy | Secondary — confirm baseline data skills |
| **Total** | | **30 problems** | | |

### Recommended Study Order:
1. **A5, A7, A11** (Predicate extraction & validation — the #1 job requirement)
2. **B1, B3, B6** (Pipeline orchestration — the STAR architecture)
3. **A3, A4, A10** (Retrieval mechanics — hybrid search)
4. **A8, A9** (Safety & LLM concepts — healthcare constraints)
5. **B2, B4, B5** (Agent infrastructure — tool calls, memory, reliability)
6. **A1, A2, A6, A12** (RAG foundations — vectors, chunking, context management)
7. **B7, B8** (Agent concepts & rate limiting)
8. **C1-C3, D1-D2** (API & DB basics — quick review)
