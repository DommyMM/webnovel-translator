# RAG-Enhanced Neural Translation System

A self-improving Chinese-to-English translation pipeline designed for web novels, combining rule-based consistency with RAG terminology precision.

## Current Status

**Working Components:**
- **Rule Learning**: Extracts style/structure patterns from professional translations via AI comparison
- **Rule Application**: Applies learned rules for consistent translation style  
- **RAG Terminology**: ChromaDB vector database for cultivation term mappings with semantic search
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
ChromaDB RAG Database
    ↓ (Rules + RAG Application)
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

**Step 2 - RAG Term Extraction:**
```python
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
│       └── chroma_db_rag/                # ChromaDB vector database
├── scripts/
│   ├── 1_baseline_translate.py      # Baseline translation pipeline
│   ├── 2_extract_rules.py           # Style rule learning via AI comparison
│   ├── 3_clean_rules.py             # AI-powered rule refinement
│   ├── 4_enhanced_translate.py      # Rules-only translation
│   ├── 5_extract_terminology.py     # Extract terminology differences
│   ├── 6_clean_terms.py             # Build ChromaDB vector database
│   ├── 7_final_translate.py         # Final Rules + RAG translation
│   ├── 8_evaluate.py                # Quality assessment pipeline
│   └── scrape/                      # Data collection tools
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

# Build ChromaDB vector database
python scripts/6_clean_terms.py
```

### 6. Final Translation (Rules + RAG)
```bash
# Final translation with rules + RAG terminology
python scripts/7_final_translate.py --start 1 --end 10 --concurrent 3
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
# RAG ensures perfect terminology
rag_lookup = semantic_search_chromadb(chinese_text)
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

### **3. ChromaDB Vector RAG**
```python
# Semantic similarity search with Chinese-specialized embeddings
results = collection.query(
    query_texts=[chinese_line],
    n_results=10,
    include=['documents', 'metadatas', 'distances']
)
# Context-aware terminology application
terminology = {doc: metadata['english_term'] for doc, metadata, distance in results}
```

## Tech Stack

### **APIs & Models**
- **DeepSeek V3**: Primary translation engine (baseline, enhanced, final)
- **Cerebras Qwen-3-32B**: Rule extraction and cleaning
- **ChromaDB**: Vector database for terminology RAG
- **Qwen3-8B Embeddings**: Chinese-specialized embeddings for semantic search

### **Processing Architecture**
- **Async Processing**: Concurrent chapter translation and evaluation
- **Vector RAG**: Semantic similarity search for terminology consistency
- **Rule Application**: Learned style patterns from professional translations

## Results & Performance

### Translation Quality
```
Chinese: 龙尘看着那个丹帝传承，心中涌起一股聚气三重天的力量

Baseline Translation:
"Long Chen looked at the Alchemy Emperor's inheritance, feeling Qi Gathering third level power surge in his heart"

Final (Rules + RAG):
"Long Chen gazed at the Pill God's inheritance, feeling third Heavenstage of Qi Condensation power surge through him"

Improvements:
- "Alchemy Emperor" → "Pill God" (RAG terminology)
- "looked at" → "gazed at" (style rules)  
- "Qi Gathering third level" → "third Heavenstage of Qi Condensation" (RAG precision)
- "in his heart" → "through him" (natural flow from rules)
```

### System Performance
- **Speed**: ~90s per chapter (Rules + RAG), ~30s baseline
- **Terminology**: Semantic search with high accuracy for cultivation terms
- **Scalability**: Async processing handles 10+ chapters concurrently
- **Cost**: ~$0.004 per chapter with DeepSeek API

## Why This Approach Works

### vs. Simple LLM Translation
- **Consistency**: RAG ensures "丹帝" always becomes "Pill God"
- **Quality**: Style rules maintain professional translation standards
- **Learning**: System improves by learning from professional translations

### vs. Traditional CAT Tools  
- **Intelligence**: Learns patterns via AI comparison, not just exact matches
- **Adaptability**: Self-improves from ground truth examples
- **Context**: Semantic search understands cultivation terminology in context

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
python scripts/2_extract_rules.py  
python scripts/3_clean_rules.py
python scripts/4_enhanced_translate.py
python scripts/5_extract_terminology.py
python scripts/6_clean_terms.py
python scripts/7_final_translate.py
python scripts/8_evaluate.py

# 3. Check results
cat results/evaluation/reports/evaluation_report.txt
```

---

*A hybrid translation system combining learned style rules with vector RAG for domain-specific terminology, creating translations that maintain both technical accuracy and natural flow.*