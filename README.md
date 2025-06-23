# RAG-Enhanced Neural Translation System

A self-improving Chinese-to-English translation pipeline designed for web novels, combining rule-based consistency with RAG terminology precision.

## Current Status

**Working Components:**
- **Rule Learning**: Extracts style/structure patterns from professional translations via AI comparison
- **Rule Application**: Applies learned rules for consistent translation style  
- **RAG Terminology**: ChromaDB vector database with BGE-M3 embeddings for cultivation term mappings via semantic search
- **Semantic Chunking**: Two-level text splitting (lines → sentences) for focused term retrieval
- **Evaluation Pipeline**: AI-powered quality assessment comparing baseline vs final translations
- **Async Processing**: Concurrent translation and evaluation across multiple chapters

**Performance**: ~90s translation time per chapter with rules + RAG terminology application

---

## System Architecture

### **Hybrid Translation Pipeline**
```
Chinese Input
    ↓ (Async DeepSeek)
Baseline Translation 
    ↓ (AI Comparison with Ground Truth)
Style Rules Extraction 
    ↓ (Cerebras AI Cleaning)
Clean Style Rules 
    ↓ (Terminology Extraction & Cleaning)
ChromaDB RAG Database (BGE-M3 Embeddings)
    ↓ (Semantic Chunking + Rules + RAG Application)
Final Enhanced Translation 
    ↓ (AI Quality Evaluation)
Quality Improvement Metrics
```

### **Example Translation Flow**

**Input Chinese:** `龙尘突破到聚气境，成为了丹帝传人`

**Step 1 - Baseline Translation:**
```
"Long Chen broke through to the Wind Gathering stage and became the Alchemy Emperor's successor"
```

**Step 2 - Semantic Chunking + RAG Term Extraction:**
```python
# Semantic units: ["龙尘突破到聚气境", "成为了丹帝传人"]
extracted_terms = ["龙尘", "聚气境", "丹帝"]
rag_mappings = {
    "龙尘": "Long Chen",
    "聚气境": "Qi Condensation Realm",   # ← Corrects "wind gathering stage"
    "丹帝": "Pill God"  # ← Corrects "Alchemy Emperor"
}
```

**Step 3 - Enhanced Translation (Rules + RAG):**
```
"Long Chen broke through to the Qi Condensation realm and became the Pill God's successor"
```

**Result:** Perfect terminology consistency with natural English flow

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
│   ├── 2_extract_rules.py           # Style rule learning via AI comparison
│   ├── 3_clean_rules.py             # AI-powered rule refinement
│   ├── 4_enhanced_translate.py      # Rules-only translation
│   ├── 5_extract_terminology.py     # Extract terminology differences
│   ├── 6_clean_terms.py             # Build ChromaDB vector database
│   ├── 7_final_translate.py         # Final Rules + RAG translation
│   ├── 8_evaluate.py                # Quality assessment pipeline
│   ├── test_rag.py                  # RAG system testing and debugging
│   └── scrape/                      # Data collection tools
├── debug/
│   └── prompts/                     # Debug prompts (with --debug flag)
└── results/
    ├── baseline/                    # Initial translation results
    ├── enhanced/                    # Rules-only translation results
    ├── final/                       # Rules + RAG translation results
    └── evaluation/                  # Comparative analysis
```

## Complete Workflow

### 1. Data Collection
```bash
# Scrape and process Chinese chapters
python scripts/scrape/scrape.py
python scripts/scrape/clean.py
```

### 2. Baseline Translation
```bash
# Translate with DeepSeek API (concurrent processing)
python scripts/1_baseline_translate.py
```

### 3. Rule Learning  
```bash
# Extract style rules by comparing with ground truth
python scripts/2_extract_rules.py --start 1 --end 10 --concurrent 3

# Clean and filter rules with Cerebras AI
python scripts/3_clean_rules.py
```

### 4. Enhanced Translation (Rules Only)
```bash
# Re-translate with learned rules
python scripts/4_enhanced_translate.py --start 1 --end 10 --concurrent 3
```

### 5. Terminology Extraction
```bash
# Extract terminology differences via AI comparison
python scripts/5_extract_terminology.py --start 1 --end 10 --concurrent 2

# Build ChromaDB vector database (BGE-M3 by default)
python scripts/6_clean_terms.py
```

### 6. Final Translation (Rules + RAG)
```bash
# Final translation with rules + RAG terminology (BGE-M3 by default)
python scripts/7_final_translate.py --start 1 --end 10 --concurrent 3

# Test RAG system
python scripts/7_final_translate.py --test

# Debug prompts without API calls
python scripts/7_final_translate.py --start 1 --end 1 --dry-run --debug
```

### 7. Quality Evaluation
```bash
# Comprehensive quality assessment
python scripts/8_evaluate.py --start 1 --end 10 --concurrent 3
```

## Key Innovations

### **1. Hybrid Rules + RAG Architecture**
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

### **2. AI-Powered Learning Pipeline**
```python
# Extract patterns by comparing translations
rules = ai_compare(my_translation, professional_translation)
# Clean with specialized model  
clean_rules = cerebras_clean(raw_rules)
# Apply in production
final_translation = translate_with_rules_and_rag(chinese_text, clean_rules, rag_terminology)
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

## Tech Stack

### **APIs & Models**
- **DeepSeek V3**: Primary translation engine (baseline, enhanced, final)
- **Cerebras Qwen-3-32B**: Rule extraction and cleaning
- **ChromaDB**: Vector database for terminology RAG
- **BGE-M3 Embeddings**: Multilingual embeddings for semantic search (default)
- **Qwen3-8B Embeddings**: Chinese-specialized embeddings (optional)
- **MPNet Embeddings**: Basic multilingual embeddings (lightweight option)

### **Processing Architecture**
- **Async Processing**: Concurrent chapter translation and evaluation
- **Semantic Chunking**: Two-level splitting (lines → sentences) for focused retrieval
- **Vector RAG**: Semantic similarity search with 0.15 threshold for terminology consistency
- **Professional Examples**: Guidance-based terminology application (not rigid replacement)

## Results & Performance

### Translation Quality
```
Chinese: 龙尘看着那个丹帝传承，心中涌起一股聚气三重天的力量

Baseline Translation:
"Long Chen looked at the Alchemy Emperor's inheritance, feeling Qi Gathering third level power surge in his heart"

Final (Rules + RAG):
"Long Chen gazed at the Pill God's inheritance, feeling third Heavenstage of Qi Condensation power surge through him"

Improvements:
- "Alchemy Emperor" → "Pill God" (RAG terminology via semantic search)
- "looked at" → "gazed at" (style rules)  
- "Qi Gathering third level" → "third Heavenstage of Qi Condensation" (RAG precision)
- "in his heart" → "through him" (natural flow from rules)
```

### System Performance
- **Speed**: ~90s per chapter (Rules + RAG), ~30s baseline
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

## Quick Start

```bash
# 1. Set up environment
echo "DEEPSEEK_API_KEY=your_key_here" > .env
echo "CEREBRAS_API_KEY=your_key_here" >> .env

# 2. Run complete pipeline (chapters 1-3)
python scripts/1_baseline_translate.py
python scripts/2_extract_rules.py --start 1 --end 3 --concurrent 3
python scripts/3_clean_rules.py
python scripts/4_enhanced_translate.py --start 1 --end 3 --concurrent 3
python scripts/5_extract_terminology.py --start 1 --end 3 --concurrent 2
python scripts/6_clean_terms.py  # Uses BGE-M3 by default
python scripts/7_final_translate.py --start 1 --end 3 --concurrent 3  # Uses BGE-M3
python scripts/8_evaluate.py --start 1 --end 3 --concurrent 3

# 3. Test RAG system
python scripts/7_final_translate.py --test

# 4. Debug specific chapter
python scripts/7_final_translate.py --start 1 --end 1 --dry-run --debug

# 5. Check results
cat results/evaluation/reports/evaluation_report.txt
```

## Advanced Usage

```bash
# Use different embedding models
python scripts/6_clean_terms.py --qwen    # Qwen3-8B embeddings
python scripts/6_clean_terms.py --lite    # MPNet embeddings

python scripts/7_final_translate.py --qwen --start 1 --end 3    # Use Qwen3 RAG
python scripts/7_final_translate.py --lite --start 1 --end 3    # Use MPNet RAG

# Test RAG retrieval with different models
python scripts/test_rag.py --bge --test specific    # Test specific queries
python scripts/test_rag.py --bge --test chapter     # Test chapter processing
```

---

*A hybrid translation system combining learned style rules with BGE-M3 vector RAG and semantic chunking for domain-specific terminology, creating translations that maintain both technical accuracy and natural flow.*