# Backend Scripts Documentation

Technical reference for translation pipeline scripts.

## Common Patterns

**Standard Args**: `[--start N] [--end N] [--chapter N] [--concurrent N]`  
**Chapter Input**: `../data/chapters/clean/chapter_X.txt`  
**Ground Truth**: `../data/chapters/ground_truth/chapter_X.txt`

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
python 6_clean_terms.py
```
**Function**: Build ChromaDB vector database  
**Same as step 3**: Sequential processing, no args  
**Embeddings**: Qwen3-8B Chinese-specialized  
**Input**: `../data/terminology/extracted_terminology.json`  
**Output**: `../data/terminology/chroma_db_rag/`

---

### 7. `7_final_translate.py`
```bash
python 7_final_translate.py [standard args] [--test] [--no-qwen]
```
**Function**: Final translation with rules + RAG  
**Same as step 2**: Model, processing, chapter inputs  
**Input**: + Clean rules + ChromaDB  
**Output**: `../results/final/translations/`  
**Performance**: ~90s/chapter

---

### 8. `8_evaluate.py`
```bash
python 8_evaluate.py [standard args]
```
**Function**: Quality evaluation across all stages  
**Same as step 2**: Model, processing  
**Input**: Baseline + final + ground truth  
**Output**: `../results/evaluation/`

## Processing Summary

| Script | Type | Args | Model |
|--------|------|------|-------|
| 1 | Async | Hardcoded | DeepSeek |
| 2,4,7,8 | Async | Standard | DeepSeek |
| 3,6 | Sequential | None | Cerebras |
| 5 | Async | Standard | Cerebras |

## Typical Usage

**Full Pipeline**:
```bash
python 1_baseline_translate.py  # Edit config in script
python 2_extract_rules.py --start 1 --end 3
python 3_clean_rules.py
python 4_enhanced_translate.py --start 1 --end 3
python 5_extract_terminology.py --start 1 --end 3
python 6_clean_terms.py
python 7_final_translate.py --start 1 --end 3
python 8_evaluate.py --start 1 --end 3
```

**Environment**: `DEEPSEEK_API_KEY`, `CEREBRAS_API_KEY`