# Backend Scripts Documentation

Technical reference for translation pipeline scripts.

## Common Patterns

**Standard Args**: `[--start N] [--end N] [--concurrent N]`  
**Chapter Input**: `../data/chapters/clean/chapter_XXXX_cn.txt`  
**Ground Truth**: `../data/chapters/ground_truth/chapter_XXXX_en.txt`

## Script Reference

### 1. `1_baseline_translate.py`
```bash
python 1_baseline_translate.py
```
**Function**: Baseline translation (DeepSeek API)  
**Config**: Hardcoded in script (edit to change range/concurrency)  
**Output**: `../results/baseline/translations/`

---

### 2. `2_parallel_extraction.py`
```bash
python 2_parallel_extraction.py [standard args]
```
**Function**: Runs 2a + 2b in parallel  
**Scripts**: `2a_extract_rules.py` + `2b_extract_terminology.py`

#### 2a. `2a_extract_rules.py`
**Function**: Extract translation rules via AI comparison  
**Model**: DeepSeek V3 | **Processing**: Async  
**Input**: Baseline translations + ground truth  
**Output**: `../data/rules/extracted_raw.json`

#### 2b. `2b_extract_terminology.py`
**Function**: Extract terminology differences via AI comparison  
**Model**: Cerebras Qwen-3-32B | **Processing**: Async  
**Input**: Baseline translations + ground truth
**Output**: `../data/terminology/extracted_terminology.json`

---

### 3. `3_parallel_cleaning.py`
```bash
python 3_parallel_cleaning.py
```
**Function**: Runs 3a in parallel

#### 3a. `3a_clean_rules.py`
**Function**: Clean/filter extracted rules  
**Model**: Cerebras Qwen-3-32B | **Processing**: Sequential  
**Input**: `../data/rules/extracted_raw.json`  
**Output**: `../data/rules/cleaned.json`

---

### 4. `4_build_chromadb.py`
```bash
python 4_build_chromadb.py [--bge|--qwen|--lite]
```
**Function**: Build ChromaDB vector database  
**Embeddings**: BGE-M3 (default) | Qwen3-8B | MPNet  
**Processing**: Sequential, no chapter args  
**Input**: `../data/terminology/extracted_terminology.json`  
**Output**: `../data/terminology/chroma_db_bge/` (or `chroma_db_rag/`, `chroma_db/`)

---

### 5. `5_final_translate.py`
```bash
python 5_final_translate.py [standard args] [--test] [--qwen] [--lite] [--debug] [--dry-run]
```
**Function**: Final translation with rules + RAG terminology  
**Model**: DeepSeek V3 | **Processing**: Async  
**Features**: Semantic chunking + BGE-M3 vector RAG  
**Input**: Chinese chapters + rules + ChromaDB  
**Output**: `../results/final/translations/`

**Special Flags**:
- `--test`: Test RAG system with sample queries
- `--debug`: Save prompts to `../debug/prompts/`
- `--dry-run`: Generate prompts without API calls
- `--qwen`: Use Qwen3-8B embeddings ChromaDB
- `--lite`: Use MPNet embeddings ChromaDB

---

### 6. `6_evaluate.py`
```bash
python 6_evaluate.py [standard args]
```
**Function**: AI-powered quality assessment  
**Model**: DeepSeek V3 | **Processing**: Async  
**Comparison**: Baseline vs Final vs Professional  
**Output**: `../results/evaluation/reports/evaluation_report.txt`

---

### `run_pipeline.py` (Updated)
```bash  
python run_pipeline.py [standard args]
```
---

## Quick Start Commands

```bash
# Run complete streamlined pipeline
python run_streamlined_pipeline.py --start 1 --end 3

# Or run individual parallel steps
python 2_parallel_extraction.py --start 1 --end 3 --concurrent 3
python 3_parallel_cleaning.py  
python 4_build_chromadb.py
python 5_final_translate.py --start 1 --end 3 --concurrent 3
python 6_evaluate.py --start 1 --end 3 --concurrent 3

# Test specific components
python 5_final_translate.py --test           # Test RAG system
python 5_final_translate.py --debug --dry-run --start 1 --end 1  # Debug prompts
```
## Processing Summary

| Script | Type | Args | Model | Embeddings |
|--------|------|------|-------|------------|
| 1 | Async | Hardcoded | DeepSeek | - |
| 2,4,8 | Async | Standard | DeepSeek | - |
| 3 | Sequential | None | Cerebras | - |
| 5 | Async | Standard | Cerebras | - |
| 6 | Sequential | Embedding flags | - | BGE-M3/Qwen3/MPNet |
| 7 | Async | Standard + special | DeepSeek | BGE-M3/Qwen3/MPNet |

## Typical Usage

**Full Pipeline**:
```bash
python 1_baseline_translate.py  # Edit config in script
python 2_extract_rules.py --start 1 --end 3
python 3_clean_rules.py
python 4_enhanced_translate.py --start 1 --end 3
python 5_extract_terminology.py --start 1 --end 3
python 6_clean_terms.py  # Uses BGE-M3 by default
python 7_final_translate.py --start 1 --end 3  # Uses BGE-M3 by default
python 8_evaluate.py --start 1 --end 3
```

**Development/Testing**:
```bash
# Test RAG retrieval
python 7_final_translate.py --test

# Debug prompts without API calls
python 7_final_translate.py --start 1 --end 1 --dry-run --debug

# Live translation with prompt saving
python 7_final_translate.py --start 1 --end 1 --debug
```

**Environment**: `DEEPSEEK_API_KEY`, `CEREBRAS_API_KEY`