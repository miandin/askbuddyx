# AskBuddyX GitHub Repository Setup

This directory contains a clean, GitHub-ready version of the AskBuddyX project.

---

## 📁 Directory Structure

```
AskBuddyX-GitHub/
├── askbuddyx/              # Core Python package
│   ├── train/              # Training pipeline
│   ├── eval/               # Evaluation scripts
│   ├── serve/              # Serving utilities
│   ├── publish/            # HF publishing tools
│   ├── config.py           # Configuration
│   └── prompting.py        # System prompts
├── scripts/                # Utility scripts
├── webui/                  # Flask web interface
├── docs/                   # Documentation
├── .github/                # GitHub Actions CI
├── pyproject.toml          # Python dependencies
├── Makefile                # Build automation
├── README.md               # Main documentation
├── MODEL_CARD.md           # Model card
├── LICENSE-THIRD-PARTY.md  # Third-party licenses
└── .gitignore              # Git ignore rules
```

---

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd AskBuddyX-GitHub
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Train Adapter
```bash
make all
```

### 3. Test Output
```bash
python scripts/test_output_format.py
```

### 4. Publish to Hugging Face
```bash
huggingface-cli login
make publish
```

---

## 📦 What's Included

### Core Package (`askbuddyx/`)
- ✅ Training pipeline (fetch, prepare, train)
- ✅ Evaluation tools
- ✅ Publishing utilities
- ✅ Configuration management
- ✅ System prompts

### Scripts (`scripts/`)
- ✅ `test_output_format.py` - Validate model output
- ✅ `demo_prompts.py` - Test multiple prompts
- ✅ `merge_datasets.py` - Combine datasets
- ✅ `duplicate_perfect_examples.py` - Duplicate examples
- ✅ `test_hf_inference.py` - Test HF deployment

### Web UI (`webui/`)
- ✅ Flask-based chat interface
- ✅ Real-time model interaction
- ✅ Chat history management

### Documentation (`docs/`)
- ✅ Quick Start Guide
- ✅ Architecture Document
- ✅ Training Analysis
- ✅ Demo Results
- ✅ Deployment Guides

### CI/CD (`.github/`)
- ✅ Automated testing
- ✅ Code quality checks (ruff)
- ✅ Python 3.12 support

---

## 🔧 Configuration

All settings in `askbuddyx/config.py`:

```python
MODEL_ID = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"
DATASET_ID = "flwrlabs/code-alpaca-20k"
HF_REPO = "salakash/AskBuddyX"
TRAIN_ITERS = 100
DATA_LIMIT = 1000
```

---

## 📝 What's NOT Included

The following are excluded (in `.gitignore`):

- ❌ `.venv/` - Virtual environment
- ❌ `data/` - Training data (generated)
- ❌ `outputs/` - Model outputs (generated)
- ❌ `__pycache__/` - Python cache
- ❌ `.env` - Environment variables

These will be created when you run the training pipeline.

---

## 🎯 GitHub Repository Setup

### Step 1: Create GitHub Repository
```bash
# On GitHub.com, create a new repository named "AskBuddyX"
```

### Step 2: Push to GitHub
```bash
cd /Users/kashifsalahuddin/AskBuddyX-GitHub
git add .
git commit -m "Initial commit: AskBuddyX adapter training framework"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/AskBuddyX.git
git push -u origin main
```

### Step 3: Configure GitHub Actions
- GitHub Actions will run automatically on push
- Tests run on Python 3.12
- Ruff checks code quality

---

## 📚 Key Documentation Files

1. **README.md** - Main project documentation
2. **docs/Quick_Start_Guide.md** - Step-by-step tutorial
3. **docs/AskBuddyX_Architecture_and_Engineering.md** - Technical details
4. **MODEL_CARD.md** - Model information for Hugging Face
5. **LICENSE-THIRD-PARTY.md** - Third-party licenses

---

## 🔄 Workflow

```
1. Train:    make all
2. Test:     python scripts/test_output_format.py
3. Publish:  make publish
4. Iterate:  Adjust config → Retrain → Republish
```

---

## 🌟 Features

- ✅ **One-command training**: `make all`
- ✅ **Automatic publishing**: `make publish`
- ✅ **Web UI included**: `python webui/app.py`
- ✅ **Comprehensive docs**: 10+ documentation files
- ✅ **CI/CD ready**: GitHub Actions configured
- ✅ **MLX optimized**: Runs on Apple Silicon
- ✅ **Production ready**: Clean, tested code

---

## 📊 Project Stats

- **43 files** total
- **11 directories**
- **~2,000 lines** of Python code
- **10+ documentation** files
- **7 utility scripts**
- **100% tested** workflow

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Submit a pull request

---

## 📄 License

See LICENSE-THIRD-PARTY.md for all third-party licenses.

---

## 🔗 Links

- **Hugging Face**: https://huggingface.co/salakash/AskBuddyX
- **Base Model**: mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit
- **Dataset**: flwrlabs/code-alpaca-20k

---

## ✅ Ready to Push

This directory is ready to be pushed to GitHub. All sensitive data, build artifacts, and temporary files are excluded via `.gitignore`.

**Next Steps**:
1. Create GitHub repository
2. Push this directory
3. Enable GitHub Actions
4. Start training adapters!