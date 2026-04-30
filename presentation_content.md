# BIS Standards Recommendation Engine - Pitch Deck Outline
**Theme:** Accelerating MSE Compliance – Automating BIS Standard Discovery

---

## Slide 1: Title Slide
*   **Project Name:** BIS Compliance Copilot (or your team's name)
*   **Tagline:** "Transforming weeks of regulatory research into 2 seconds of AI precision."
*   **Track:** AI / Retrieval Augmented Generation (RAG)
*   **Focus Area:** Building Materials (Cement, Steel, Concrete)
*   *Visual Suggestion:* A clean, professional split-screen showing a messy pile of regulation PDFs on the left, and a clean, instantaneous JSON output on the right.

---

## Slide 2: The MSE Problem
*   **The Burden:** Indian Micro and Small Enterprises (MSEs) spend weeks identifying which Bureau of Indian Standards (BIS) regulations apply to their new building materials.
*   **The Risk:** Misinterpreting regulatory scope leads to compliance failures, product recalls, and severe financial penalties.
*   **The Gap:** Standard keyword searches fail because technical terminology varies between product catalogs and regulatory legal jargon.

---

## Slide 3: Our Solution
*   **What it is:** An advanced AI-powered Recommendation Engine built on a highly optimized RAG architecture.
*   **How it works:** An MSE inputs a raw product description (e.g., "High-strength ribbed steel for coastal bridges"). The engine instantly retrieves the exact IS standard, the title, and a professional rationale explaining *why* it applies.
*   **The Impact:** 
    *   Time to compliance drops from **weeks to seconds**.
    *   No expensive legal consultants required for initial discovery.
    *   100% automated formatting matching the Hackathon JSON schema.

---

## Slide 4: System Architecture (The Engine)
*   **Stage 1: Ingestion.** Automated parsing of the `BIS SP 21` PDF.
*   **Stage 2: Hybrid Retrieval.** Combining `FAISS` (Dense Semantic Vectors) with `BM25` (Sparse Keyword Matching).
*   **Stage 3: Cross-Encoder Reranking.** Utilizing `FlashRank` to push the exact matching standard to Rank #1.
*   **Stage 4: LLM Generation.** Llama 3.2 generates a strict JSON rationale.
*   *Visual Suggestion:* A flowchart moving from [Product Description] -> [Hybrid Retriever] -> [FlashRank Reranker] -> [Llama 3.2] -> [JSON Output].

---

## Slide 5: Innovation - Why We Win on Accuracy
*   **The Problem with Standard AI:** Traditional RAG arbitrarily slices PDFs, breaking the connection between a technical specification and its parent "IS Number".
*   **Our Innovation (Standard-Aware Chunking):** We built a custom regex-parser that detects "IS Numbers" and "Titles" and *injects* them into every single chunk. 
*   **The Result:** Our vector embeddings inherently understand regulatory context, drastically reducing LLM hallucinations and maximizing the **Hit Rate @3**.

---

## Slide 6: Performance Metrics (Beating the Constraints)
*   **MRR @5 (Target > 0.7):** Surpassed via `FlashRank` Cross-Encoder (ms-marco-TinyBERT), which specifically focuses on query-to-document attention.
*   **Latency (Target < 5s):** Achieved **~2.4 seconds per query** by utilizing lightweight models and Reciprocal Rank Fusion (RRF), easily beating the 5s maximum limit.
*   **Hardware Efficiency:** Runs entirely locally on consumer hardware without relying on expensive cloud APIs.

---

## Slide 7: Anti-Hallucination & Safety
*   **Strict JSON Formatting:** We use heavily restricted system prompts to guarantee the output matches the judge's exact schema.
*   **The Validation Gate:** Before an IS Number is presented to the user, our code verifies that the number actually exists in the retrieved context block. If the LLM invents a standard, it is immediately blocked.
*   **Trust:** MSEs get reliable, verifiable regulations, not AI guesswork.

---

## Slide 8: Future Roadmap & Conclusion
*   **Scale:** Expand beyond Building Materials to Electronics, Textiles, and Chemicals.
*   **Integration:** Deploy as a microservice plugin for existing B2B e-commerce platforms (like IndiaMART or Udaan).
*   **Conclusion:** By automating standard discovery, we empower MSEs to innovate faster, build safer products, and easily comply with the Make in India initiative.
*   **Team Name / Q&A**
