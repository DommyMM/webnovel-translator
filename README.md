# RAG-Enhanced Neural Translation System

A self-improving Chinese-to-English translation pipeline specifically designed for web novels, combining rule-based consistency with RAG terminology precision for optimal translation quality.

## Current Status: Functional but WiP

**Working Components:**
- **Rule Learning**: Extracts style/structure patterns from professional translations via AI comparison
- **Rule Application**: Applies 10 high-quality rules for consistent translation style  
- **RAG Terminology**: Database for exact cultivation term mappings with pattern matching
- **Pattern Extrapolation**: Learns cultivation patterns (realms, techniques, titles) from ground truth
- **Evaluation Pipeline**: AI-powered quality assessment comparing baseline vs final translations
- **Async Processing**: Concurrent translation and evaluation for 3x+ speed improvement

**Performance**: Consistent terminology application with 6-8 RAG terms per chapter, ~90s translation time per chapter

---

## Project Vision

Traditional LLM translation suffers from inconsistent terminology and style drift, especially in domain-specific content like cultivation novels. This system builds a living knowledge base that combines:

1. **Style Rules** - Learned patterns for tone, pacing, structure from professional translations
2. **Terminology RAG** - Database ensuring "丹帝 → Pill God" consistency
3. **Pattern Learning** - Automatic extrapolation for new terms (凝气期 → Qi Condensation stage)

## Architecture Overview

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
RAG Terminology Database
    ↓ (Rules + RAG Application)
Final Enhanced Translation 
    ↓ (AI Quality Evaluation)
Quality Improvement Metrics
```

### **Example Translation Flow**

**Input Chinese:** `龙尘突破到金丹期，成为了丹帝传人`

**Step 1 - Baseline Translation:**
```
"Long Chen broke through to the Golden Core stage and became the Alchemy Emperor's successor"
```

**Step 2 - RAG Term Extraction:**
```python
extracted_terms = ["龙尘", "金丹期", "丹帝"]
rag_mappings = {
    "龙尘": "Long Chen",
    "金丹期": "Golden Core stage",  
    "丹帝": "Pill God"  # ← Corrects "Alchemy Emperor"
}
```

**Step 3 - Enhanced Translation (Rules + RAG):**
```
"Long Chen broke through to the Golden Core stage and became the Pill God's successor"
```

**Result:** Perfect terminology consistency with natural English flow

---

## Project Structure

```
/backend
├── data/
│   ├── chapters/
│   │   ├── raw/              # Original scraped content
│   │   ├── clean/            # Processed Chinese text
│   │   └── ground_truth/     # Reference translations
│   ├── rules/
│   │   ├── extracted_raw.json    # Initial rule database
│   │   ├── cleaned.json           # Filtered actionable rules
│   │   └── analysis/              # AI analysis outputs
│   └── terminology/
│       ├── extracted_terminology.json  # Raw terminology differences
│       ├── rag_database.json          # Clean RAG terminology database
│       └── raw_responses/             # AI extraction responses
├── scripts/
│   ├── 1_baseline_translate.py     # Baseline translation pipeline
│   ├── 2_extract_rules.py          # Style rule learning via AI comparison
│   ├── 3_clean_rules.py            # AI-powered rule refinement
│   ├── 4_enhanced_translate.py     # Rules-only translation
│   ├── 5_extract_terminology.py    # Extract terminology differences
│   ├── 6_clean_terms.py            # Build clean RAG database
│   ├── 7_final_translate.py        # Final Rules + RAG translation
│   ├── 8_evaluate.py               # Quality assessment pipeline
│   └── scrape/                     # Data collection tools
├── results/
│   ├── baseline/                   # Initial translation results
│   ├── enhanced/                   # Rules-only translation results
│   ├── final/                      # Rules + RAG translation results
│   └── evaluation_complete/        # Comparative analysis
└── temp/                           # Temporary files and logs
```

## Complete Workflow

### 1. Data Collection
```bash
# Scrape raw Chinese chapters
python scripts/scrape/scrape.py

# Clean and process text  
python scripts/scrape/clean.py
```

### 2. Baseline Translation
```bash
# Translate chapters with DeepSeek API
python scripts/1_baseline_translate.py --start 1 --end 10 --concurrent 5
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
# Re-translate with learned rules only
python scripts/4_enhanced_translate.py --start 1 --end 10 --concurrent 3
```

### 5. Terminology Extraction
```bash
# Extract terminology differences via AI comparison
python scripts/5_extract_terminology.py --start 1 --end 10 --concurrent 2

# Build clean RAG terminology database
python scripts/6_clean_terms.py
```

### 6. Final Translation (Rules + RAG)
```bash
# Final translation with rules + RAG terminology
python scripts/7_final_translate.py --start 1 --end 10 --concurrent 3
```

### 7. Quality Evaluation
```bash
# Comprehensive quality assessment: Baseline vs Final
python scripts/8_evaluate.py --start 1 --end 10 --concurrent 3
```

## Key Innovations

### 1. **Hybrid Rules + RAG Architecture**
```python
# Style rules ensure consistent flow and tone
style_rules = [
    "Use punchy sentences in action scenes",
    "Maintain conversational dialogue style", 
    "Preserve Chinese paragraph breaks"
]

# RAG ensures perfect terminology
rag_lookup = {
    "丹帝": "Pill God",           # Exact mapping
    "金丹期": "Golden Core stage", # Cultivation realm
    "九星霸体诀": "Nine Star Hegemon Body Art"  # Technique name
}

# Combined prompt
prompt = f"""STYLE RULES: {style_rules}
TERMINOLOGY: {rag_lookup}
CHINESE: {input_text}"""
```

### 2. **AI-Powered Learning Pipeline**
```python
# Extract differences by comparing translations
enhanced_vs_ground_truth = ai_compare(enhanced_translation, ground_truth)

# Clean and validate with AI
clean_rules = cerebras_clean(raw_extracted_rules)
clean_terminology = validate_terminology(raw_terminology_differences)

# Apply in final translation
final_translation = translate_with_rules_and_rag(chinese_text, clean_rules, clean_terminology)
```

### 3. **Async Processing Architecture**
```python
# Concurrent processing across multiple chapters
async def process_chapters(chapter_range):
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [process_chapter_async(ch, semaphore) for ch in chapters]
    results = await asyncio.gather(*tasks)
    return results

# Result: 3x+ speed improvement over sequential processing
```

## Tech Stack

### **APIs & Models**
- **DeepSeek V3**: Primary translation engine (baseline, enhanced, final)
- **Cerebras Qwen-3-32B**: Rule extraction and cleaning (free tier)
- **OpenAI SDK**: Unified async interface for both APIs

### **RAG Infrastructure**
- **Simple Term Matching**: Direct Chinese → English term lookups
- **Pattern Recognition**: Cultivation term categorization (realms, techniques, titles)
- **Validation System**: Text verification to prevent AI hallucinations

### **Performance Metrics**
- **Processing Speed**: ~90s per chapter with Rules + RAG
- **Terminology Accuracy**: 6-8 terms applied per chapter
- **Concurrent Processing**: 3x speed improvement via async
- **Quality Evaluation**: 5-category AI scoring system

## Results & Performance

### Translation Quality Improvements
- **Terminology Consistency**: Near complete accuracy for known terms via RAG
- **Style Consistency**: Applied 10 learned rules per translation
- **Pattern Learning**: Automatic categorization of cultivation terms

### System Performance
- **Speed**: ~90s per chapter (Rules + RAG), 30s baseline
- **Accuracy**: RAG terms successfully applied in context
- **Scalability**: Async processing handles 10+ chapters concurrently
- **Cost**: ~$0.005 per chapter with DeepSeek API

### Example Results
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

## Development Roadmap

### Phase 1: Core System (Complete)
- [x] Async translation pipeline
- [x] Rule extraction and application via AI comparison
- [x] Quality evaluation system
- [x] RAG terminology database with validation

### Phase 2: Enhancement & Optimization (Current)
- [x] Terminology extraction pipeline
- [x] AI-powered rule and terminology cleaning
- [x] Complete evaluation pipeline (Baseline vs Final)
- [ ] Context caching optimization for cost reduction

### Phase 3: Production Scaling (Next)
- [ ] Multi-novel terminology sharing
- [ ] Real-time translation API
- [ ] Web interface for translation management
- [ ] Expanded terminology database (1000+ terms)

### Phase 4: Multi-Domain Expansion (Future)
- [ ] Support for other web novel genres  
- [ ] Multi-language support (JP→EN, KR→EN)
- [ ] Integration with reading platforms
- [ ] Community-driven terminology databases

## Cost Analysis

**Per Chapter Costs** (current API usage):
- **Baseline Translation**: ~2K tokens × $0.27/1M = $0.0005
- **Rules + RAG Translation**: ~4K tokens × $0.27/1M = $0.001
- **Evaluation**: ~8K tokens × $0.27/1M = $0.002
- **Total per chapter**: ~$0.004

**At Scale:**
- **100 chapters**: ~$0.40 total cost
- **Time to process**: ~2.5 hours (100 chapters concurrent)
- **Quality**: High terminology consistency

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
cat results/evaluation_complete/reports/simplified_evaluation_report.txt
```

## Why This Approach Works

### vs. Simple LLM Translation
- **Consistency**: RAG ensures "丹帝" always becomes "Pill God"
- **Quality**: Style rules maintain professional translation standards
- **Learning**: System improves by learning from professional translations

### vs. Traditional CAT Tools  
- **Intelligence**: Learns patterns via AI comparison, not just exact matches
- **Adaptability**: Self-improves from ground truth examples
- **Validation**: AI prevents hallucinations through text verification

### vs. Human Translation
- **Speed**: ~90s per chapter vs hours for human translation
- **Availability**: 24/7 operation with consistent quality
- **Consistency**: Never forgets established terminology or style rules
- **Cost**: $0.004/chapter vs significantly more for human translation
- **Learning**: Continuously improves from professional examples

---

## Legacy Setup (Local Infrastructure)

**Original Plan**: Deploy locally with vLLM + CUDA 12.8 on WSL2  
**Current Status**: Moved to API-based development due to driver compatibility issues

<details>
<summary>Click to view original local setup plans</summary>

### Local Environment Setup (WSL2 + CUDA 12.8) - ON HOLD

This project was originally designed for Windows 11 + WSL2 with local GPU acceleration via CUDA 12.8 and NVIDIA RTX 5090. Due to vLLM driver compatibility issues, moved to API-based development for faster iteration.

#### System Requirements (Future)
- Windows 11 with WSL2 enabled
- NVIDIA GPU with latest WSL-compatible drivers  
- Ubuntu 24.04 LTS installed via WSL
- Python 3.12
- CUDA Toolkit 12.8 (WSL version)

#### Local Stack (Planned)
- **LLM**: Qwen2.5-32B (local inference via vLLM)
- **Vector Store**: FAISS for semantic pattern search
- **Framework**: LangChain for RAG orchestration  
- **Backend**: FastAPI for API services
- **Frontend**: Next.js for translation interface

*Note: Will return to local deployment once NVIDIA driver ecosystem stabilizes for WSL2 + vLLM compatibility.*

</details>

---

*This system represents the next evolution in domain-specific translation: combining the consistency of rule-based systems with the precision of retrieval-augmented generation, creating translations that maintain both technical accuracy and artistic flow.*