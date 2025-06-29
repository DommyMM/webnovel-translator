# RAG-Enhanced Neural Translation System

A self-improving Chinese-to-English translation pipeline designed for web novels, combining rule-based consistency with RAG terminology precision.

## Current Status

**Working Components:**
- **Parallel Extraction**: Simultaneous rule learning and terminology extraction via AI comparison
- **Rule Application**: Applies learned style/structure patterns for consistent translation quality
- **RAG Terminology**: ChromaDB vector database with BGE-M3 embeddings for cultivation term mappings via semantic search
- **Semantic Chunking**: Two-level text splitting (lines → sentences) for focused term retrieval
- **Naive Evaluation**: Direct comparison between minimal prompt and enhanced pipeline results
- **Async Processing**: Concurrent translation and evaluation across multiple chapters

**Performance**: ~90s translation time per chapter with rules + RAG terminology application

---

## System Architecture

### **Streamlined Translation Pipeline (6 Steps)**
```
Chinese Input
    ↓ (Async DeepSeek)
1. Baseline Translation 
    ↓ (Parallel AI Comparison with Ground Truth)
2a. Style Rules Extraction + 2b. Terminology Extraction (PARALLEL)
    ↓ (Cleaning and Structuring)
3. Clean Style Rules 
    ↓ (ChromaDB Vector Database Build)
4. ChromaDB RAG Database (BGE-M3 Embeddings)
    ↓ (Semantic Chunking + Rules + RAG Application)
5. Final Enhanced Translation 
    ↓ (Naive Evaluation)
6. Compare Minimal Prompt vs Enhanced Pipeline
```

### **Example Translation Flow**

**Input Chinese:** `龙尘突破到聚气境，成为了丹帝传人`

**Step 1 - Baseline Translation:**
```
"Long Chen broke through to the Wind Gathering stage and became the Alchemy Emperor's successor"
```

**Step 2 - Parallel Extraction (2a + 2b simultaneously):**
```python
# 2a: Style rules extracted
style_rules = ["Prioritize active voice in action scenes", "Use dynamic verbs for cultivation breakthroughs"]

# 2b: Terminology differences identified
terminology_diffs = {
    "龙尘": "Long Chen",
    "聚气境": "Qi Condensation Realm",   # ← Corrects "wind gathering stage"
    "丹帝": "Pill God"                   # ← Corrects "Alchemy Emperor"
}
```

**Step 5 - Final Translation (Rules + RAG):**
```
"Long Chen broke through to the Qi Condensation realm and became the Pill God's successor"
```

**Step 6 - Naive Evaluation:**
```
Naive (minimal prompt): "Long Chen broke through to the gathering realm and became the Dan Emperor's heir"
Enhanced (rules + RAG): "Long Chen broke through to the Qi Condensation realm and became the Pill God's successor"

Improvement: Consistent terminology + natural flow
```

---

## Project Structure

```
/backend
├── data/
│   ├── chapters/
│   │   ├── clean/                   # Processed Chinese text
│   │   └── ground_truth/            # Reference translations
│   ├── rules/
│   │   ├── extracted_raw.json       # Initial rule database
│   │   └── cleaned.json             # Filtered actionable rules
│   └── terminology/
│       ├── extracted_terminology.json    # Raw terminology differences
│       ├── chroma_db_bge/                # BGE-M3 vector database (default)
│       ├── chroma_db_rag/                # Qwen3-8B vector database
│       └── chroma_db/                    # MPNet vector database
├── scripts/
│   ├── 1_baseline_translate.py      # Baseline translation pipeline
│   ├── 2_parallel_extraction.py     # Coordinates 2a + 2b in parallel
│   ├── 2a_extract_rules.py          # Style rule learning via AI comparison
│   ├── 2b_extract_terminology.py    # Terminology extraction via AI comparison
│   ├── 3_clean_rules.py             # AI-powered rule refinement
│   ├── 4_build_chromadb.py          # Build ChromaDB vector database
│   ├── 5_final_translate.py         # Final Rules + RAG translation
│   ├── 6_evaluate.py                # Naive evaluation coordinator
│   ├── 6a_naive_translate.py        # Generate naive translations
│   ├── 6b_compare_naive_enhanced.py # Compare naive vs enhanced
│   └── scrape/                      # Data collection tools
├── debug/
│   └── prompts/                     # Debug prompts (with --debug flag)
└── results/
    ├── baseline/                    # Initial translation results
    ├── final/                       # Rules + RAG translation results
    ├── naive/                       # Minimal prompt translations
    └── comparison/                  # Naive vs enhanced comparisons
```

## Streamlined Workflow (6 Steps)

### 0. Data Collection
```bash
# Scrape and process Chinese chapters
python scripts/scrape/scrape.py
python scripts/scrape/clean.py
```
Or just use your own txt files in `data/chapters/clean/` and `data/chapters/ground_truth/`.

### 1. Baseline Translation
```bash
# Translate with DeepSeek API (concurrent processing)
python scripts/1_baseline_translate.py --start 1 --end 3 --concurrent 10
```

### 2. Parallel Extraction (Rules + Terminology)
```bash
# Extract both style rules AND terminology differences simultaneously
python scripts/2_parallel_extraction.py --start 1 --end 10 --concurrent 3

# Or run individually if needed:
# python scripts/2a_extract_rules.py --start 1 --end 10 --concurrent 3
# python scripts/2b_extract_terminology.py --start 1 --end 10 --concurrent 2
```

### 3. Rule Cleaning
```bash
# Clean and filter rules with Cerebras AI
python scripts/3_clean_rules.py
```

### 4. Build Vector Database
```bash
# Build ChromaDB vector database (BGE-M3 by default)
python scripts/4_build_chromadb.py

# Alternative embedding models:
# python scripts/4_build_chromadb.py --qwen    # Qwen3-8B embeddings
# python scripts/4_build_chromadb.py --lite    # MPNet embeddings
```

### 5. Final Translation (Rules + RAG)
```bash
# Final translation with rules + RAG terminology (BGE-M3 by default)
python scripts/5_final_translate.py --start 1 --end 10 --concurrent 3

# Test RAG system
python scripts/5_final_translate.py --test

# Debug prompts without API calls
python scripts/5_final_translate.py --start 1 --end 1 --dry-run --debug
```

### 6. Naive Evaluation
```bash
# Compare minimal prompt vs enhanced pipeline
python scripts/6_evaluate.py --start 1 --end 10 --concurrent 3
```

## Key Innovations

### **1. Parallel Extraction Architecture**
```python
# Steps 2a and 2b run simultaneously (same inputs: baseline + ground truth)
async def run_parallel_extraction():
    tasks = [
        run_script_async("2a_extract_rules.py", args),      # Style/structure rules
        run_script_async("2b_extract_terminology.py", args) # Terminology differences
    ]
    results = await asyncio.gather(*tasks)  # ~50% faster than sequential
```

### **2. Hybrid Rules + RAG Architecture**
```python
# Style rules ensure consistent flow and tone
style_rules = extract_from_professional_translations()
# Semantic chunking for focused embeddings
semantic_units = chunk_by_lines_and_sentences(chinese_text)
# RAG ensures perfect terminology with 0.15 threshold
rag_lookup = semantic_search_chromadb(semantic_units, threshold=0.15)
# Combined application
final_translation = translate_with_rules_and_rag(chinese_text, style_rules, rag_lookup)
```

### **3. ChromaDB Vector RAG with Semantic Chunking**
```python
# Two-level text splitting for focused embeddings
semantic_units = chunk_chinese_text_by_semantic_units(text)
# BGE-M3 embeddings with lowered threshold for context-diluted terms
results = collection.query(
    query_texts=semantic_units,
    n_results=10,
    include=['documents', 'metadatas', 'distances']
)
# Professional examples approach (not rigid mappings)
terminology = {doc: metadata['english_term'] for doc, metadata, distance in results 
               if (1.0 - distance) >= 0.15}
```

### **4. Honest Naive Evaluation**
```python
# Compare against truly minimal prompt, not professional human translations
naive_prompt = "Translate this Chinese text to English prose:"
enhanced_result = rules_plus_rag_translation(chinese_text)
# Shows what your pipeline actually adds over raw ChatGPT
comparison = side_by_side_analysis(naive_result, enhanced_result)
```

## Tech Stack

### **APIs & Models**
- **DeepSeek V3**: Translation engine + rule extraction + naive evaluation
- **Cerebras Qwen-3-32B**: Rule cleaning + terminology extraction
- **ChromaDB**: Vector database for terminology RAG
- **BGE-M3 Embeddings**: Multilingual embeddings for semantic search (default)
- **Qwen3-8B Embeddings**: Chinese-specialized embeddings (optional)
- **MPNet Embeddings**: Basic multilingual embeddings (lightweight option)

### **Processing Architecture**
- **Async Processing**: Concurrent chapter translation and evaluation
- **Parallel Extraction**: Simultaneous rule and terminology extraction
- **Semantic Chunking**: Two-level splitting (lines → sentences) for focused retrieval
- **Vector RAG**: Semantic similarity search with 0.15 threshold for terminology consistency
- **Professional Examples**: Guidance-based terminology application (not rigid replacement)

## Results & Performance

### Translation Quality
```
Chinese: 龙尘看着那个丹帝传承，心中涌起一股聚气三重天的力量

Naive Translation (minimal prompt):
"Long Chen looked at the Dan Emperor's inheritance, feeling Qi Gathering third heaven power surge in his heart"

Enhanced (Rules + RAG):
"Long Chen gazed at the Pill God's inheritance, feeling third Heavenstage of Qi Condensation power surge through him"

Improvements:
- "Dan Emperor" → "Pill God" (RAG terminology via semantic search)
- "looked at" → "gazed at" (style rules)  
- "Qi Gathering third heaven" → "third Heavenstage of Qi Condensation" (RAG precision)
- "in his heart" → "through him" (natural flow from rules)
```

### System Performance
- **Speed**: ~90s per chapter (Rules + RAG), ~30s naive
- **Real-time Progress**: Live token streaming with multi-chapter progress bars
- **Terminology**: BGE-M3 semantic search with 0.15 threshold captures context-diluted terms
- **Scalability**: Async processing handles 10+ chapters concurrently
- **Cost**: ~$0.004 per chapter with DeepSeek API

### RAG Breakthrough Example
```
Problematic Chinese: "我是傲视天下，睥睨九霄的绝世丹帝——龙尘？"

Previous (Line-level): Only found 龙尘 → Long Chen (context dilution)
Current (Semantic chunking): Found both 龙尘 → Long Chen AND 丹帝 → Pill God

Result: "Am I the peerless Pill God who looks down on the world—Long Chen?"
```

## Why This Approach Works

### vs. Simple LLM Translation
- **Consistency**: RAG ensures "丹帝" always becomes "Pill God" via semantic search
- **Quality**: Style rules maintain professional translation standards
- **Learning**: System improves by learning from professional translations

### vs. Traditional CAT Tools  
- **Intelligence**: BGE-M3 embeddings understand semantic context, not just exact matches
- **Adaptability**: Self-improves from ground truth examples
- **Context**: Semantic chunking prevents context dilution in long sentences

### vs. Human Translation
- **Speed**: ~90s per chapter vs hours for human translation
- **Availability**: 24/7 operation with consistent quality
- **Consistency**: Never forgets established terminology or style rules
- **Cost**: $0.004/chapter vs significantly higher human costs

### Honest Evaluation
- **Realistic Baseline**: Compares against minimal prompt, not professional translations
- **Clear Value**: Shows exactly what pipeline adds over "just ask ChatGPT"
- **Measurable**: Direct metrics (terminology consistency, flow improvements, length ratios)

## Quick Start

```bash
# 1. Set up environment
echo "DEEPSEEK_API_KEY=your_key_here" > .env
echo "CEREBRAS_API_KEY=your_key_here" >> .env

# 2. Run streamlined pipeline (chapters 1-3)
python scripts/1_baseline_translate.py --start 1 --end 3 --concurrent 10
python scripts/2_parallel_extraction.py --start 1 --end 3 --concurrent 3
python scripts/3_clean_rules.py
python scripts/4_build_chromadb.py  # Uses BGE-M3 by default
python scripts/5_final_translate.py --start 1 --end 3 --concurrent 3        # Uses BGE-M3
python scripts/6_evaluate.py --start 1 --end 3 --concurrent 3               # Naive evaluation

# 3. Test RAG system
python scripts/5_final_translate.py --test

# 4. Debug specific chapter
python scripts/5_final_translate.py --start 1 --end 1 --dry-run --debug

# 5. Check results
cat results/comparison/comparison_summary.txt
```

## Advanced Usage

```bash
# Use different embedding models
python scripts/4_build_chromadb.py --qwen    # Qwen3-8B embeddings
python scripts/4_build_chromadb.py --lite    # MPNet embeddings

python scripts/5_final_translate.py --qwen --start 1 --end 3    # Use Qwen3 RAG
python scripts/5_final_translate.py --lite --start 1 --end 3    # Use MPNet RAG

# Run individual extraction steps if needed
python scripts/2a_extract_rules.py --start 1 --end 3 --concurrent 3
python scripts/2b_extract_terminology.py --start 1 --end 3 --concurrent 2

# Run evaluation steps individually
python scripts/6a_naive_translate.py --start 1 --end 3 --concurrent 3
python scripts/6b_compare_naive_enhanced.py --start 1 --end 3
```

---

*A hybrid translation system combining learned style rules with BGE-M3 vector RAG and semantic chunking for domain-specific terminology, with honest evaluation comparing enhanced pipeline against minimal baseline prompts.*