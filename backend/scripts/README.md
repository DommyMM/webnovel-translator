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

### 2. `2_extract_rules.py`
```bash
python 2_extract_rules.py [standard args]
```
**Function**: Extract translation rules via AI comparison  
**Model**: DeepSeek V3 | **Processing**: Async  
**Input**: Baseline translations + ground truth  
**Output**: `../data/rules/extracted_raw.json`

---

### 3. `3_clean_rules.py`
```bash
python 3_clean_rules.py
```
**Function**: Clean/filter extracted rules  
**Model**: Cerebras Qwen-3-32B | **Processing**: Sequential  
**Input**: `../data/rules/extracted_raw.json`  
**Output**: `../data/rules/cleaned.json`

---

### 4. `4_enhanced_translate.py`
```bash
python 4_enhanced_translate.py [standard args]
```
**Function**: Translate with learned rules only  
**Same as step 2**: Model, processing, chapter inputs  
**Input**: + `../data/rules/cleaned.json`  
**Output**: `../results/enhanced/translations/`

---

### 5. `5_extract_terminology.py`
```bash
python 5_extract_terminology.py [standard args]
```
**Function**: Extract terminology differences for RAG  
**Model**: Cerebras Qwen-3-32B | **Processing**: Async  
**Input**: Enhanced translations + ground truth  
**Output**: `../data/terminology/extracted_terminology.json`

---

### 6. `6_clean_terms.py`
```bash
python 6_clean_terms.py [--qwen] [--lite]
```
**Function**: Build ChromaDB vector database  
**Embeddings**: BGE-M3 (default) | Qwen3-8B | MPNet  
**Processing**: Sequential, no chapter args  
**Input**: `../data/terminology/extracted_terminology.json`  
**Output**: `../data/terminology/chroma_db_bge/` (or `chroma_db_rag/`, `chroma_db/`)

---

### 7. `7_final_translate.py`
```bash
python 7_final_translate.py [standard args] [--test] [--qwen] [--lite] [--debug] [--dry-run]
```
**Function**: Final translation with rules + RAG  
**Model**: DeepSeek V3 | **Processing**: Async  
**Embeddings**: BGE-M3 (default) | Qwen3-8B | MPNet  
**Chunking**: Semantic units (lines → sentences)  
**Threshold**: 0.15 similarity for term retrieval  
**Input**: + Clean rules + ChromaDB  
**Output**: `../results/final/translations/`  
**Debug**: `../debug/prompts/` (with --debug or --dry-run)

---

### 8. `8_evaluate.py`
```bash
python 8_evaluate.py [standard args]
```
**Function**: Quality evaluation across all stages  
**Same as step 2**: Model, processing  
**Input**: Baseline + final + ground truth  
**Output**: `../results/evaluation/`

## Embedding Models

| Flag | Model | Use Case |
|------|-------|----------|
| Default | BGE-M3 | Best retrieval quality |
| --qwen | Qwen3-8B | Chinese-specialized |
| --lite | MPNet | Fast/basic embeddings |

## Special Flags

**Step 7 Only**:
- `--test`: Test RAG system without translation
- `--debug`: Save full prompts to debug folder
- `--dry-run`: Build prompts without API calls

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