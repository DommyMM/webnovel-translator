# Backend Scripts Documentation

Technical reference for streamlined translation pipeline scripts.

## Common Patterns

**Standard Args**: `[--start N] [--end N] [--concurrent N]`  
**Chapter Input**: `../data/chapters/clean/chapter_XXXX_cn.txt`  
**Ground Truth**: `../data/chapters/ground_truth/chapter_XXXX_en.txt`

## Streamlined Pipeline (6 Steps)

### 1. `1_baseline_translate.py`
```bash
python 1_baseline_translate.py
```
**Function**: Baseline translation (DeepSeek API)  
**Config**: Hardcoded in script (edit to change range/concurrency)  
**Model**: DeepSeek V3 | **Processing**: Async  
**Output**: `../results/baseline/translations/`

---

### 2. `2_parallel_extraction.py`
```bash
python 2_parallel_extraction.py [standard args]
```
**Function**: Runs rule extraction + terminology extraction in parallel  
**Scripts**: Coordinates `2a_extract_rules.py` + `2b_extract_terminology.py`

#### 2a. `2a_extract_rules.py`
**Function**: Extract style/structure rules via AI comparison  
**Model**: DeepSeek V3 | **Processing**: Async  
**Input**: Baseline translations + ground truth  
**Output**: `../data/rules/extracted_raw.json`

#### 2b. `2b_extract_terminology.py`
**Function**: Extract terminology differences via AI comparison  
**Model**: Cerebras Qwen-3-32B | **Processing**: Async  
**Input**: Baseline translations + ground truth  
**Output**: `../data/terminology/extracted_terminology.json`

---

### 3. `3_clean_rules.py`
```bash
python 3_clean_rules.py
```
**Function**: Clean/filter extracted rules with AI  
**Model**: Cerebras Qwen-3-32B | **Processing**: Sequential  
**Input**: `../data/rules/extracted_raw.json`  
**Output**: `../data/rules/cleaned.json`

---

### 4. `4_build_chromadb.py`
```bash
python 4_build_chromadb.py [--bge|--qwen|--lite]
```
**Function**: Build ChromaDB vector database from terminology  
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
**Features**: Semantic chunking + BGE-M3 vector RAG + style rules  
**Input**: Chinese chapters + cleaned rules + ChromaDB  
**Output**: `../results/final/translations/`

**Special Flags**:
- `--test`: Test RAG system with sample queries
- `--debug`: Save prompts to `../debug/prompts/`
- `--dry-run`: Generate prompts without API calls
- `--qwen`: Use Qwen3-8B embeddings ChromaDB
- `--lite`: Use MPNet embeddings ChromaDB

---

### 6. `6_evaluate.py` (Optional)
```bash
python 6_evaluate.py [standard args]
```
**Function**: AI-powered quality assessment  
**Model**: DeepSeek V3 | **Processing**: Async  
**Comparison**: Baseline vs Final vs Professional  
**Output**: `../results/evaluation/reports/evaluation_report.txt`

---

## Processing Summary

| Step | Script | Type | Args | Model | Embeddings |
|------|--------|------|------|-------|------------|
| 1 | `1_baseline_translate.py` | Async | Hardcoded | DeepSeek | - |
| 2a | `2a_extract_rules.py` | Async | Standard | DeepSeek | - |
| 2b | `2b_extract_terminology.py` | Async | Standard | Cerebras | - |
| 3 | `3_clean_rules.py` | Sequential | None | Cerebras | - |
| 4 | `4_build_chromadb.py` | Sequential | Embedding flags | - | BGE-M3/Qwen3/MPNet |
| 5 | `5_final_translate.py` | Async | Standard + special | DeepSeek | BGE-M3/Qwen3/MPNet |
| 6 | `6_evaluate.py` | Async | Standard | DeepSeek | - |

## Quick Start Commands

### Full Streamlined Pipeline
```bash
# Run complete pipeline (all steps)
python 1_baseline_translate.py                           # Edit config in script
python 2_parallel_extraction.py --start 1 --end 3       # Steps 2a + 2b in parallel
python 3_clean_rules.py                                  # Step 3
python 4_build_chromadb.py                              # Step 4 (uses BGE-M3 by default)
python 5_final_translate.py --start 1 --end 3           # Step 5 (uses BGE-M3 by default)
python 6_evaluate.py --start 1 --end 3                  # Step 6 (optional)
```

### Individual Steps
```bash
# Run individual parallel steps manually
python 2a_extract_rules.py --start 1 --end 3 --concurrent 3
python 2b_extract_terminology.py --start 1 --end 3 --concurrent 2

# Different embedding models
python 4_build_chromadb.py --qwen                       # Use Qwen3-8B embeddings
python 4_build_chromadb.py --lite                       # Use basic MPNet embeddings
python 5_final_translate.py --qwen --start 1 --end 3    # Use Qwen3 RAG
python 5_final_translate.py --lite --start 1 --end 3    # Use MPNet RAG
```

### Development/Testing
```bash
# Test RAG retrieval system
python 5_final_translate.py --test

# Debug prompts without API calls
python 5_final_translate.py --start 1 --end 1 --dry-run --debug

# Live translation with prompt saving
python 5_final_translate.py --start 1 --end 1 --debug
```

## Environment Variables

Required: `DEEPSEEK_API_KEY`, `CEREBRAS_API_KEY`

```bash
# Set in .env file
echo "DEEPSEEK_API_KEY=your_key_here" > .env
echo "CEREBRAS_API_KEY=your_key_here" >> .env
```