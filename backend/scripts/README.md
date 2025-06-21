# Backend Scripts

Numbered workflow for the translation pipeline:

## 1. Baseline Translation
```bash
python scripts/1_baseline_translate.py
```
Translates chapters using DeepSeek API without any rules.

## 2. Extract Rules  
```bash
python scripts/2_extract_rules.py
```
Compares baseline translations with ground truth to extract improvement rules.

## 3. Clean Rules
```bash
python scripts/3_clean_rules.py  
```
Uses Cerebras AI to filter and clean extracted rules into actionable format.

## 4. Enhanced Translation
```bash
python scripts/4_enhanced_translate.py
```
Re-translates chapters using the learned rules for improved quality.

## Utilities
- `scripts/scrape/` - Web scraping tools for data collection

## Data Flow
```
Chinese Chapters → Baseline Translation → Rule Extraction → Rule Cleaning → Enhanced Translation
```

Results are saved to `results/` with comparisons and analytics.
