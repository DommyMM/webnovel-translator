# RAG-Enhanced Neural Translation System

A self-improving Chinese-to-English translation pipeline specifically designed for web novels, with emphasis on terminology consistency and quality improvement through pattern learning.

## Project Vision

Traditional LLM translation suffers from inconsistent terminology, especially in domain-specific content like cultivation novels. This system builds a living knowledge base that learns optimal translation patterns from existing high-quality translations, then applies those patterns to new content.

## Core Architecture

### Stage 1: Pattern-Enhanced Translation
```
Chinese Input → Semantic Analysis → Pattern Retrieval → LLM Translation → English Output
```

Instead of simple term lookup, we retrieve translation patterns with context.

Example:
```
Input: "他突破到了金丹期"
Pattern Retrieved: "突破到 + [realm] → broke through to + [realm]"
Context: "金丹期 → Golden Core stage (not Gold Core period)"
Output: "He broke through to the Golden Core stage"
```

### Stage 2: Self-Learning Proofreader
```
My Translation → Compare with Ground Truth → Extract Patterns → Update Knowledge Base
```

The system automatically learns rules like:
- "丹神 → Pill Sovereign (preferred over Dan God)"
- "境界 → realm (in cultivation context, not boundary)"
- "强者 → expert (formal) vs powerhouse (casual)"

## Knowledge Base Structure

### Pattern Database
```json
{
  "pattern_id": "cultivation_breakthrough",
  "chinese_pattern": "[subject] 突破到了 [realm]",
  "english_pattern": "[subject] broke through to [realm]",
  "context_tags": ["cultivation", "advancement"],
  "confidence_score": 0.95,
  "usage_count": 247
}
```

### Terminology Registry
```json
{
  "term": "金丹期",
  "preferred_translation": "Golden Core stage",
  "alternatives": ["Golden Core period", "Gold Core stage"],
  "context": "cultivation_realm",
  "frequency": 1834,
  "quality_score": 0.88
}
```

### Style Rules
```json
{
  "rule": "realm_formality",
  "description": "Use 'stage' for cultivation realms, not 'period' or 'level'",
  "examples": [
    {"bad": "Foundation period", "good": "Foundation stage"},
    {"bad": "Core Formation level", "good": "Core Formation stage"}
  ]
}
```

## Implementation Plan

### Phase 1: Foundation
- [ ] Set up Qwen2.5-32B local inference with vLLM
- [ ] Build FAISS vector store for pattern storage
- [ ] Create pattern extraction pipeline from existing translations
- [ ] Implement basic RAG retrieval system

### Phase 2: Core Translation
- [ ] Design context-aware pattern matching
- [ ] Implement sliding context windows for chapter coherence
- [ ] Build confidence scoring for pattern selection
- [ ] Optimize KV-cache for throughput (target: 150+ chapters/hour)

### Phase 3: Self-Improvement
- [ ] Build comparison engine (my translation vs ground truth)
- [ ] Implement automatic pattern learning
- [ ] Create rule extraction from differences
- [ ] Build feedback loop for knowledge base updates

### Phase 4: Validation
- [ ] Blind A/B testing framework vs GPT-4.1
- [ ] Quality metrics (BLEU, semantic similarity, human preference)
- [ ] Performance benchmarking
- [ ] Documentation and deployment

## Success Metrics

### Quality Targets
- 75%+ preference rate in blind testing vs GPT-4.1
- 90%+ terminology consistency within novels
- Sub-5% error rate on established pattern applications

### Performance Targets
- 150+ chapters/hour translation throughput
- Sub-second pattern retrieval latency

## Technical Stack

### Core Components
- LLM: Qwen2.5-32B (local inference via vLLM)
- Vector Store: FAISS for semantic pattern search
- Framework: LangChain for RAG orchestration
- Backend: FastAPI for API services
- Frontend: Next.js for translation interface

## Development Environment Setup (WSL2 + CUDA 12.8)

This project is developed on Windows 11 using WSL2 with GPU acceleration via CUDA 12.8 and the NVIDIA RTX 5090. The backend stack runs in a Linux virtual environment, with code stored on the Windows filesystem.

### System Requirements
- Windows 11 with WSL2 enabled
- NVIDIA GPU (e.g., RTX 5090) with latest WSL-compatible drivers
- Ubuntu 24.04 LTS installed via WSL
- Python 3.12
- CUDA Toolkit 12.8 (WSL version)

### Installation Steps
```bash
# 1. Install system dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential curl wget git vim software-properties-common
sudo apt install -y python3.12 python3.12-venv python3.12-dev python3-pip

# 2. Install CUDA 12.8 toolkit for WSL
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8 cuda-compiler-12-8

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 3. Verify GPU access
nvidia-smi
nvcc --version
```

### Virtual Environment and Package Setup
```bash
# 1. Install `uv` (fast package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 2. Create virtual environment in Linux FS
mkdir -p ~/venvs/webnovel && cd ~/venvs/webnovel
uv venv vllm_env --python 3.12
source vllm_env/bin/activate

# 3. Install PyTorch with CUDA 12.8
uv pip install --upgrade pip
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 4. Confirm GPU support
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# 5. Install RAG and backend dependencies
uv pip install faiss-cpu sentence-transformers langchain langchain-community pandas numpy huggingface-hub fastapi uvicorn ipython jupyter vllm

# Note: faiss-gpu is not yet available for Python 3.12.
# Using faiss-cpu for now, can recreate with Python 3.10 if needed.
```

## Project Structure
```
/backend          # FastAPI server, LLM inference, RAG pipeline
/frontend         # Next.js web interface
/data            # Pattern databases, knowledge base
/scripts         # Preprocessing and evaluation tools
```

## Scalability Strategy

### Novel Adaptation
1. Zero-shot: Works on new novels using base wuxia knowledge
2. Few-shot: Rapid adaptation with 10-50 sample chapters
3. Full-adaptation: Complete knowledge base for specific series

### Domain Expansion
- Start with cultivation/xianxia novels
- Expand to historical fiction, system novels, etc.
- Eventually support any Chinese web fiction genre

## Key Challenges & Solutions

### Challenge: Pattern Ambiguity
Problem: "境界" could mean "realm", "boundary", "state of mind"
Solution: Context-weighted retrieval + confidence scoring

### Challenge: Style Consistency
Problem: Different translators have different preferences
Solution: Learn dominant patterns from high-quality sources

### Challenge: Novel Evolution
Problem: New terms appear constantly in web novels
Solution: Automatic term detection + pattern generalization

## Why This Approach Works

### vs. Simple LLM Translation
- Consistency: Learns and enforces terminology standards
- Quality: Builds on proven translation patterns
- Context: Maintains story coherence across chapters

### vs. Traditional CAT Tools
- Intelligence: Understands semantic patterns, not just exact matches
- Adaptability: Self-improves from feedback
- Scale: Handles novel-length content efficiently

### vs. Human Translation
- Speed: 150+ chapters/hour vs 1-2 chapters/day
- Availability: 24/7 operation
- Consistency: Never forgets established terminology

## Future Roadmap

### Short-term Goals
- Multi-language support (JP→EN, KR→EN)
- Real-time web novel translation
- Community feedback integration

### Long-term Vision
- Cross-novel knowledge sharing
- Advanced context understanding (plot awareness)
- Integration with popular reading platforms

---

This system represents the next evolution in domain-specific translation: not just converting languages, but understanding and maintaining the artistic and cultural nuances that make web novels engaging.