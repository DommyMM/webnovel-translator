# Backend Scripts

Async translation pipeline with rule learning for Chinese cultivation novels.

## Complete Workflow

### 1. Baseline Translation (Async)
```bash
python 1_baseline_translate.py
```
- **Parallel processing**: Translates chapters 1-3 concurrently using DeepSeek API
- **Performance**: ~60s for however many parallel
- **Output**: Raw translations without any learned rules applied
- **Results**: `../results/baseline/translations/` + analytics

**Arguments:**
```bash
python 1_baseline_translate.py --start 1 --end 5 --concurrent 8
```
- `--start N`: Start chapter number (default: 1)
- `--end N`: End chapter number (default: 3)
- `--concurrent N`: Max parallel requests (default: 10)

### 2. Extract Rules (Async)  
```bash
python 2_extract_rules.py
```
- **Analysis**: Compares baseline translations with professional ground truth
- **AI extraction**: Uses DeepSeek to identify improvement patterns
- **Performance**: ~21s concurrent processing
- **Output**: Raw rules database with 15+ extraction rules
- **Results**: `../data/rules/extracted_raw.json`

**Arguments:**
```bash
python 2_extract_rules.py --start 1 --end 5 --chapter 2
```
- `--start N`: Start chapter number (default: 1)
- `--end N`: End chapter number (default: 3) 
- `--chapter N`: Extract rules from single chapter only

### 3. Clean Rules (Cerebras)
```bash
python 3_clean_rules.py  
```
- **AI cleaning**: Uses Cerebras qwen-3-32b to filter and refine rules
- **Quality control**: Removes parsing artifacts and vague descriptions
- **Output**: 10 high-quality, actionable translation rules
- **Categories**: Terminology (2) + Style (3) + Structure (3) + Cultural (2)
- **Results**: `../data/rules/cleaned.json` + readable `.txt`

**No arguments** - uses fixed input/output paths and Cerebras qwen-3-32b model.

### 4. Enhanced Translation (Async)
```bash
python 4_enhanced_translate.py
```
- **Rule application**: Re-translates chapters using the 10 learned rules
- **Parallel processing**: Concurrent translation with rule guidance
- **Quality improvement**: Applies specific terminology, style, and cultural rules
- **Output**: Enhanced translations for comparison
- **Results**: `../results/enhanced/translations/`

**Arguments:**
```bash
python 4_enhanced_translate.py --start 1 --end 5 --chapter 2
```
- `--start N`: Start chapter number (default: 1)
- `--end N`: End chapter number (default: 3)
- `--chapter N`: Translate single chapter only

### 5. Evaluation (Async)
```bash
python 5_evaluate.py
```
- **Comparison**: Evaluates baseline vs enhanced vs professional ground truth
- **AI scoring**: Uses DeepSeek (low temp) for objective quality assessment
- **Metrics**: Flow, character voice, clarity, genre feel, overall enjoyment
- **Performance**: ~22s concurrent evaluation
- **Results**: Detailed analytics + side-by-side comparisons

**Arguments:**
```bash
python 5_evaluate.py --start 1 --end 5 --chapter 2
```
- `--start N`: Start chapter number (default: 1)
- `--end N`: End chapter number (default: 3)
- `--chapter N`: Evaluate single chapter only  

## Key Features

- **Async/Concurrent**: All scripts use parallel processing for 3-6x speed improvement
- **Rule Learning**: Automatically extracts and applies translation improvement patterns
- **Quality Control**: Multi-stage filtering (extraction → cleaning → application)
- **Comprehensive Evaluation**: Objective AI scoring with detailed breakdowns

## Data Flow
```
Chinese Chapters 
    ↓ (Async DeepSeek)
Baseline Translations 
    ↓ (Compare with Ground Truth)
Raw Rules Extraction 
    ↓ (Cerebras Cleaning)
Clean Rules 
    ↓ (Apply During Translation)
Enhanced Translations 
    ↓ (AI Evaluation)
Quality Improvement Metrics
```

## Common Usage Examples

```bash
# Standard workflow (chapters 1-3)
python 1_baseline_translate.py
python 2_extract_rules.py  
python 3_clean_rules.py
python 4_enhanced_translate.py
python 5_evaluate.py

# Process more chapters with higher concurrency
python 1_baseline_translate.py --start 1 --end 10 --concurrent 8
python 2_extract_rules.py --start 1 --end 10 --concurrent 5
python 4_enhanced_translate.py --start 1 --end 10 --concurrent 8
python 5_evaluate.py --start 1 --end 10 --concurrent 5

# Test single chapter
python 4_enhanced_translate.py --chapter 5
python 5_evaluate.py --chapter 5

# Higher quality rule extraction
python 2_extract_rules.py --min-confidence 0.8 --concurrent 3
```

## Current Rules Applied

**Terminology (2 rules)**: Literal cultivation terms, preserve "Qi Condensation" over "Qi Gathering"
**Style (3 rules)**: Neutral tone, punchy fight scenes, preserve conversational voice  
**Structure (3 rules)**: Short paragraphs, preserve scene breaks, simple dialogue tags
**Cultural (2 rules)**: Direct idiom translation, avoid Westernized phrases

Results saved to `results/` with comprehensive analytics and human-readable comparisons.