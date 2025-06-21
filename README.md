# RAG-Enhanced Neural Translation System

A self-improving Chinese-to-English translation pipeline specifically designed for web novels, with emphasis on terminology consistency and quality improvement through pattern learning.

## Current Status: API-Based Pipeline (Phase 1)

**Working Components:**
- Rule Extraction: Compares translations to extract improvement patterns
- Rule Cleaning: AI-powered filtering of actionable translation rules  
- Enhanced Translation: Applies learned rules to improve future translations
- Evaluation Pipeline: Measures improvement over baseline translations

**Currently Using**: DeepSeek API + Cerebras API for fast, cost-effective development and testing.

---

## Project Vision

Traditional LLM translation suffers from inconsistent terminology, especially in domain-specific content like cultivation novels. This system builds a living knowledge base that learns optimal translation patterns from existing high-quality translations, then applies those patterns to new content.

## Current Architecture (Phase 1)

### Working Pipeline
```
Chinese Input → Baseline Translation → Compare with Ground Truth → Extract Rules → Enhanced Translation
```

**Example Improvement**:
```
Original Rule: "Use 'Alchemy Emperor' instead of 'Pill God'"
Ground Truth: Actually uses "Pill God" 
Fixed Rule: "Use 'Pill God' instead of 'Alchemy Emperor' to match professional standards"
Result: Better similarity scores and consistency
```

## Project Structure

```
/backend
├── data/
│   ├── chapters/
│   │   ├── raw/              # Original scraped content
│   │   ├── clean/            # Processed Chinese text
│   │   └── ground_truth/     # Reference translations
│   └── rules/
│       ├── extracted_raw.json    # Initial rule database
│       ├── cleaned.json           # Filtered actionable rules
│       └── analysis/              # AI analysis outputs
├── scripts/
│   ├── 1_baseline_translate.py    # Baseline translation pipeline
│   ├── 2_extract_rules.py         # Rule learning from comparisons
│   ├── 3_clean_rules.py           # AI-powered rule refinement
│   ├── 4_enhanced_translate.py    # Rule-enhanced translation
│   ├── 5_evaluate.py              # Quality assessment
│   └── scrape/                    # Data collection tools
├── results/
│   ├── baseline/                  # Initial translation results
│   ├── enhanced/                  # Improved translation results
│   └── analysis/                  # Comparative analysis
└── temp/                          # Temporary files and logs
```

## Current Workflow

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
python scripts/main.py
```

### 3. Rule Learning  
```bash
# Extract rules by comparing with ground truth
python scripts/rules.py

# Clean and filter rules with Cerebras AI
python scripts/clean_rules.py
```

### 4. Enhanced Translation
```bash
# Re-translate with learned rules
python scripts/retranslate.py
```

## Key Innovations

### 1. Pattern-Based Learning
Instead of simple term lookup, retrieves translation patterns with context:
```
Input: "他突破到了金丹期"
Learned Pattern: "突破到 + [realm] → broke through to + [realm]"  
Context: "金丹期 → Golden Core stage (not Gold Core period)"
Output: "He broke through to the Golden Core stage"
```

### 2. Self-Improving Rules Database
```json
{
  "rule_type": "terminology",
  "description": "Use 'Pill God' instead of 'Alchemy Emperor' for consistency",
  "confidence": 0.9,
  "usage_count": 12,
  "success_rate": 0.85
}
```

### 3. Comparative Analysis
- **Baseline Similarity**: ~0.22-0.35 against ground truth
- **Enhanced Similarity**: Target >0.40 with learned rules
- **Continuous Improvement**: Rules updated based on performance

## Tech Stack (Current)

### APIs (Development Phase)
- **DeepSeek API**: Primary translation engine (temp=1.3 for translation tasks)
- **Cerebras API**: Rule extraction and cleaning (free tier: 900 req/hour)
- **OpenAI SDK**: Unified interface for both APIs

### Evaluation Metrics
- **Jaccard Similarity**: Word overlap between translation and ground truth
- **Terminology Consistency**: Track rule application success rates
- **Performance Tracking**: Translation speed, token usage, cost analysis

## Results So Far

### Rule Learning Success
- **Extracted**: 13 raw rules from 3 chapter comparisons
- **Cleaned**: 8 actionable rules (terminology, style, cultural, structure)
- **Quality**: High confidence rules (0.8+) for consistent application

### Translation Improvement (Ongoing)
- **Challenge Identified**: Initial rules learned backwards direction
- **Fix Applied**: Prompt updated to learn FROM ground truth, not away from it
- **Next Test**: Re-run enhanced translation with corrected rule learning

## Development Roadmap

### Phase 1: API-Based Foundation (Current)
- [x] Basic translation pipeline
- [x] Rule extraction and cleaning
- [x] Enhanced translation with rules
- [x] Performance evaluation
- [x] Prompt engineering for better rule learning

### Phase 2: Optimization & Scale (In Progress)
- [x] Fix rule learning direction (learn toward ground truth)
- [ ] Expand to 10+ chapters for better rule diversity
- [ ] Implement rule success tracking and auto-filtering
- [ ] Optimize prompt engineering for consistency

### Phase 3: Local Infrastructure (Future)
- [ ] Local LLM deployment once drivers stabilize
- [ ] FAISS vector store for semantic pattern search  
- [ ] Real-time rule application and learning
- [ ] Web interface for translation management

### Phase 4: Production Features (Planned)
- [ ] Multi-novel rule sharing and adaptation
- [ ] Advanced context understanding (plot awareness)
- [ ] Integration with reading platforms
- [ ] Multi-language support (JP→EN, KR→EN)

## Cost Analysis (API Phase)

**Current Usage** (per chapter):
- **DeepSeek**: ~$0.01-0.02 per chapter translation
- **Cerebras**: Free tier covers rule extraction needs
- **Total**: <$0.05 per chapter for complete pipeline

**Scalability**: At current costs, processing 1000 chapters would cost ~$50, making this approach viable for extensive testing and development.

---

## Why This Approach Works

### vs. Simple LLM Translation
- **Consistency**: Learns and enforces terminology standards
- **Quality**: Builds on proven translation patterns  
- **Context**: Maintains story coherence across chapters

### vs. Traditional CAT Tools  
- **Intelligence**: Understands semantic patterns, not just exact matches
- **Adaptability**: Self-improves from feedback
- **Scale**: Handles novel-length content efficiently

### vs. Human Translation
- **Speed**: 150+ chapters/hour vs 1-2 chapters/day (target)
- **Availability**: 24/7 operation
- **Consistency**: Never forgets established terminology
- **Cost**: <$0.05/chapter vs $50-200/chapter for human translation

## Contributing

1. **Test Current Pipeline**: Run the 4-step workflow on sample chapters
2. **Improve Rule Learning**: Help refine prompts for better rule extraction
3. **Expand Coverage**: Add more chapter pairs for training data
4. **Evaluate Quality**: Compare results against human translations

## Quick Start

```bash
# 1. Set up environment
echo "DEEPSEEK_API_KEY=your_key_here" > .env
echo "CEREBRAS_API_KEY=your_key_here" >> .env

# 2. Run baseline translation
python scripts/main.py

# 3. Extract and clean rules  
python scripts/rules.py
python scripts/clean_rules.py

# 4. Enhanced translation
python scripts/retranslate.py

# 5. Check results
cat results/enhanced_results/analytics/enhanced_analytics.json
```

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

*This system represents the next evolution in domain-specific translation: not just converting languages, but understanding and maintaining the artistic and cultural nuances that make web novels engaging.*