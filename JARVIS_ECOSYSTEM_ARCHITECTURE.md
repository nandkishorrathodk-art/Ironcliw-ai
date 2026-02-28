# Ironcliw Ecosystem Architecture

**Three Independent Projects Working Together**

## Overview

The Ironcliw ecosystem has been modernized into three separate but connected projects:

```
┌─────────────────┐
│  Reactor Core   │  ← AI/ML Training Engine (Python + MLForge C++)
└────────┬────────┘
         │ uses
         ▼
┌─────────────────┐
│  Ironcliw Prime   │  ← Specialized PRIME Models
└────────┬────────┘
         │ imports
         ▼
┌─────────────────┐
│     Ironcliw      │  ← AI Assistant (Runtime)
└─────────────────┘
```

## 1. Reactor Core

**Repository:** https://github.com/drussell23/reactor-core
**Version:** v1.0.0
**Purpose:** AI/ML Training Engine with Python bindings to MLForge C++ core

### Key Features
- ✅ Environment auto-detection (M1 Mac vs GCP VM)
- ✅ GCP Spot VM checkpoint/resume functionality
- ✅ LoRA/QLoRA support for memory-efficient training
- ✅ PyTorch-first with async-safe training loops
- ✅ Modular architecture: `/training`, `/data`, `/eval`, `/serving`, `/gcp`, `/utils`

### Installation
```bash
# For local development (M1 Mac)
pip install reactor-core[local]

# For GCP training
pip install reactor-core[gcp]
```

### Usage Example
```python
from reactor_core import Trainer, TrainingConfig

config = TrainingConfig(
    model_name="llama-2-7b",
    use_lora=True,
    lora_rank=16,
    num_epochs=3,
)

trainer = Trainer(config)
trainer.train("./data/train.jsonl")
```

### Environment Detection
| Environment | Mode | Features |
|-------------|------|----------|
| M1 Mac 16GB | Lightweight | Inference-only, quantized models |
| GCP 32GB VM | Full Training | LoRA, DPO, FSDP, auto-resume |

---

## 2. Ironcliw Prime

**Repository:** https://github.com/drussell23/jarvis-prime
**Version:** v0.6.0
**Purpose:** Specialized PRIME models for Ironcliw AI Assistant

### Key Features
- ✅ Simple `PrimeModel.from_pretrained()` API
- ✅ Automatic quantization (4-bit, 8-bit) for M1 Mac
- ✅ Pre-configured PRIME models (chat, vision, reasoning)
- ✅ Uses Reactor Core for training
- ✅ Production-ready for Ironcliw integration

### Available Models
| Model | Size | Use Case | M1 Compatible |
|-------|------|----------|---------------|
| `prime-7b-chat-v1` | 7B | Chat, Q&A | ✅ (quantized) |
| `prime-7b-vision-v1` | 7B | Vision + Text | ✅ (quantized) |
| `prime-13b-reasoning-v1` | 13B | Advanced reasoning | ⚠️ (slow) |

### Installation
```bash
# For Ironcliw runtime (inference only)
pip install jarvis-prime

# For model training
pip install jarvis-prime[training]
```

### Usage Example
```python
from jarvis_prime import PrimeModel

# Load quantized model for M1 Mac
model = PrimeModel.from_pretrained(
    "prime-7b-chat-v1",
    quantization="8bit"
)

# Generate response
response = model.generate("What is machine learning?")
```

---

## 3. Ironcliw

**Repository:** https://github.com/drussell23/Ironcliw-AI-Agent
**Version:** v2.0.0+ (current)
**Purpose:** AI Assistant with reasoning, chat, and multimodal capabilities

### Changes Required
Remove training logic from Ironcliw backend and import Ironcliw Prime:

#### Before
```python
# backend/chatbots/claude_chatbot.py
from transformers import AutoModelForCausalLM

class ClaudeChatbot:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("llama-2-7b")
```

#### After
```python
# backend/chatbots/claude_chatbot.py
from jarvis_prime import PrimeModel

class ClaudeChatbot:
    def __init__(self):
        self.model = PrimeModel.from_pretrained("prime-7b-chat-v1", quantization="8bit")
```

### Files to Modify
1. Remove `backend/models/training_pipeline.py` → Use Ironcliw Prime
2. Remove `backend/models/training_interface.py` → Use Ironcliw Prime
3. Update `backend/chatbots/claude_chatbot.py` → Import from `jarvis_prime`
4. Update `backend/requirements.txt`:
   - Remove: `torch`, `transformers`, `peft` (direct deps)
   - Add: `jarvis-prime>=0.6.0`

---

## Compute Environment Strategy

### Local Development (M1 Mac 16GB)
- **Reactor Core:** Disabled (inference-only mode)
- **Ironcliw Prime:** Loads quantized models (8-bit)
- **Ironcliw:** Runs normally with lightweight models

### Remote Training (GCP 32GB VM)
- **Reactor Core:** Full training mode with GCP Spot VM support
- **Ironcliw Prime:** Trains PRIME models using Reactor Core
- **Ironcliw:** Deployed separately (runtime only)

---

## Dependency Graph

```
Ironcliw (runtime)
  │
  └─> jarvis-prime>=0.6.0
        │
        └─> reactor-core>=1.0.0 (training only)
              │
              └─> MLForge (C++ core, optional bindings)
```

---

## Next Steps

### Reactor Core
- [ ] Add pybind11 bindings for MLForge C++ core
- [ ] Implement DPO (Direct Preference Optimization)
- [ ] Add FSDP (Fully Sharded Data Parallel) support
- [ ] Build data preprocessing pipeline
- [ ] Add model serving utilities

### Ironcliw Prime
- [ ] Train `prime-7b-chat-v1` on Ironcliw conversation data
- [ ] Add multimodal support for `prime-7b-vision-v1`
- [ ] Create quantized exports for all models
- [ ] Add model evaluation benchmarks
- [ ] Publish models to Hugging Face Hub

### Ironcliw
- [ ] Refactor backend to use Ironcliw Prime
- [ ] Remove training logic from `backend/models/`
- [ ] Update chatbot integrations
- [ ] Test with quantized models on M1
- [ ] Update documentation

---

## Version Compatibility Matrix

| Component | Reactor Core | Ironcliw Prime | Ironcliw |
|-----------|--------------|--------------|--------|
| Current | v1.0.0 | v0.6.0 | v2.0.0+ |
| Min Required | - | ≥ 1.0.0 | ≥ 0.6.0 |

---

## Summary

✅ **Reactor Core** - Live on GitHub
✅ **Ironcliw Prime** - Live on GitHub
🔄 **Ironcliw** - Needs refactoring to use Ironcliw Prime

**Benefits:**
1. **Separation of Concerns:** Training (Reactor Core) vs Models (Ironcliw Prime) vs Runtime (Ironcliw)
2. **Environment Awareness:** Auto-detect M1 vs GCP and configure accordingly
3. **GCP Spot VM Support:** Save costs with preemptible VMs + auto-resume
4. **Modular Architecture:** Each project is independently testable and deployable
5. **Future-Proof:** Easy to add new models, training methods, and features

---

**Links:**
- Reactor Core: https://github.com/drussell23/reactor-core
- Ironcliw Prime: https://github.com/drussell23/jarvis-prime
- Ironcliw: https://github.com/drussell23/Ironcliw-AI-Agent
- MLForge (C++ Core): https://github.com/drussell23/MLForge
