# Hugging Face Download Tracking Analysis

## Executive Summary

**Current Status**: Downloads are NOT tracked for AskBuddyX  
**Reason**: Missing required query files  
**Solution**: Add `config.json` to enable tracking  
**Impact**: After fix, all downloads (script, web UI, manual) will be counted

---

## Problem Analysis

### Why Downloads Aren't Tracked

Your model shows **"Downloads are not tracked for this model"** because Hugging Face requires specific "query files" to count downloads, and your repository doesn't contain any of them.

### How HF Counts Downloads

According to [HF's official documentation](https://huggingface.co/docs/hub/models-download-stats):

1. **Server-Side Counting**: HF counts HTTP requests (GET/HEAD) to specific files
2. **No Client Tracking**: No information is sent from users; counting happens on HF's servers
3. **Query Files**: HF looks for specific files to avoid double-counting

### Default Query Files (checked in order)

```
1. config.json          ← Primary (most common)
2. config.yaml
3. hyperparams.yaml
4. params.json
5. meta.yaml
```

### Your Current Repository Structure

```
salakash/AskBuddyX/
├── adapters.safetensors      ❌ Not counted
├── adapter_config.json       ❌ Not counted (wrong name)
├── run_meta.json             ❌ Not counted (wrong name)
├── README.md                 ❌ Not counted
├── MODEL_CARD.md             ❌ Not counted
└── LICENSE-THIRD-PARTY.md    ❌ Not counted
```

**Result**: No query files → No download tracking

---

## Will Your Downloads Be Counted?

### Current Implementation Analysis

#### 1. Inference Testing Script (`scripts/test_hf_inference.py`)

```python
snapshot_download(
    repo_id="salakash/AskBuddyX",
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.md"],
    force_download=True,
    local_dir=temp_dir
)
```

**Current Status**: ❌ NOT counted  
**Reason**: Downloads `adapter_config.json` but HF doesn't recognize it as a query file

#### 2. Web UI (`webui/app.py`)

```python
snapshot_download(
    repo_id="salakash/AskBuddyX",
    allow_patterns=["*.safetensors", "*.json"],
    local_dir=adapter_cache_dir
)
```

**Current Status**: ❌ NOT counted  
**Reason**: Same issue - no recognized query files

#### 3. Manual Downloads

Users downloading via:
- HF website UI
- `git clone`
- `huggingface-cli download`

**Current Status**: ❌ NOT counted  
**Reason**: No query files in repository

---

## Solution: Enable Download Tracking

### Option 1: Add config.json (Recommended) ✅

**Implementation**: Already completed in `askbuddyx/publish/make_bundle.py`

The bundle creation script now automatically generates `config.json`:

```json
{
  "model_type": "qwen2",
  "adapter_type": "lora",
  "base_model": "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
  "base_model_reference": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
  "task": "text-generation",
  "framework": "mlx",
  "lora_rank": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "trained_on": "flwrlabs/code-alpaca-20k",
  "training_samples": 2000,
  "training_iterations": 50,
  "model_name": "AskBuddyX",
  "description": "LoRA adapter for Qwen2.5-Coder-0.5B-Instruct trained on code-alpaca-20k dataset. Provides runnable-first coding assistance.",
  "license": "apache-2.0"
}
```

**Benefits**:
- ✅ Enables download tracking
- ✅ Provides useful metadata
- ✅ Standard format recognized by HF
- ✅ No breaking changes to existing code

### Option 2: Register Custom Library (Advanced)

You could register a custom library filter by opening a PR to:
https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/model-libraries.ts

Example for counting `adapter_config.json`:
```typescript
{
  id: "mlx-lora",
  displayName: "MLX LoRA",
  countDownloads: ["adapter_config.json"]
}
```

**Not recommended** because:
- Requires external PR approval
- Takes time to implement
- `config.json` is simpler and standard

---

## After Fix: What Gets Counted?

Once `config.json` is added to your repository, **ALL** of these will be counted:

### 1. Script Downloads ✅
```bash
python scripts/test_hf_inference.py --prompt "test"
```
- Downloads all files including `config.json`
- HF sees GET request to `config.json`
- **Download count +1**

### 2. Web UI Downloads ✅
```bash
python webui/app.py
# User sends message → downloads adapter
```
- Downloads all files including `config.json`
- HF sees GET request to `config.json`
- **Download count +1**

### 3. Manual Downloads ✅
```bash
# Via CLI
huggingface-cli download salakash/AskBuddyX

# Via Python
from huggingface_hub import snapshot_download
snapshot_download("salakash/AskBuddyX")

# Via git
git clone https://huggingface.co/salakash/AskBuddyX
```
All methods download `config.json` → **Download count +1**

### 4. Programmatic Access ✅
```python
from huggingface_hub import hf_hub_download
hf_hub_download("salakash/AskBuddyX", "config.json")
```
Direct file access → **Download count +1**

---

## Implementation Steps

### Step 1: Rebuild Bundle with config.json

```bash
# The make_bundle.py script now automatically creates config.json
make bundle
```

**Output**:
```
Creating config.json for download tracking...
  Created: config.json
```

### Step 2: Verify config.json Exists

```bash
ls -la outputs/hf_bundle/config.json
```

**Expected**:
```
-rw-r--r--  1 user  staff  567 Dec 30 15:22 outputs/hf_bundle/config.json
```

### Step 3: Republish to Hugging Face

```bash
make publish
```

**Output**:
```
Publishing to Hugging Face...
Uploading to: salakash/AskBuddyX
  Uploading config.json...
  ✓ Upload complete
```

### Step 4: Verify on Hugging Face

1. Visit: https://huggingface.co/salakash/AskBuddyX
2. Check "Files and versions" tab
3. Confirm `config.json` is present
4. Wait 24-48 hours for download stats to appear

---

## Download Counting Rules

### What Counts as a Download

✅ **Counted**:
- GET requests to `config.json`
- HEAD requests to `config.json`
- Downloads via `snapshot_download()` (includes config.json)
- Downloads via `hf_hub_download()` for config.json
- Git clones (downloads all files including config.json)
- Manual downloads from HF website

❌ **Not Counted**:
- Viewing files in browser (no download)
- API calls that don't download files
- Cached downloads (unless `force_download=True`)

### Avoiding Double Counting

HF's system is designed to avoid double counting:

1. **Single Query File**: Only counts requests to `config.json`, not every file
2. **Per-Download Basis**: One download = one request to query file
3. **Sharded Models**: Multiple weight files don't multiply the count

### Special Cases

#### GGUF Files
- All GGUF files are counted individually
- May double-count if user clones entire repo
- Most users download single GGUF file

#### Diffusers Models
- Special filter for diffusers library
- Counts both library downloads and manual downloads
- Avoids double counting nested files

---

## Verification & Testing

### Test 1: Local Download Test

```bash
# Test script download
python scripts/test_hf_inference.py --prompt "test" --runs 1

# Check if config.json was downloaded
ls -la /tmp/askbuddyx_*/config.json
```

**Expected**: File exists in temp directory

### Test 2: Check HF Repository

```bash
# List files in HF repo
huggingface-cli repo-files salakash/AskBuddyX
```

**Expected Output**:
```
config.json                    ← Must be present
adapter_config.json
adapters.safetensors
run_meta.json
README.md
MODEL_CARD.md
LICENSE-THIRD-PARTY.md
USAGE.md
```

### Test 3: Monitor Download Stats

1. Visit: https://huggingface.co/salakash/AskBuddyX
2. Look for download badge/counter
3. Perform test download
4. Wait 24-48 hours
5. Check if counter increased

---

## FAQ

### Q: Why wasn't this included initially?

**A**: The original implementation focused on adapter functionality. Download tracking requires specific HF metadata files that weren't part of the core adapter bundle.

### Q: Will old downloads be counted retroactively?

**A**: No. HF only counts downloads going forward from when `config.json` is added. Historical downloads before the fix are not tracked.

### Q: Does force_download=True affect counting?

**A**: Yes, positively. `force_download=True` ensures fresh downloads every time, so each run counts as a new download. Without it, cached downloads wouldn't trigger new requests to `config.json`.

### Q: Can I see who downloaded my model?

**A**: No. HF only provides aggregate download counts. No user information is collected or shared.

### Q: How often are download stats updated?

**A**: Download stats are typically updated every 24-48 hours. Real-time counting is not available.

### Q: What if I have multiple query files?

**A**: HF will count requests to any of the query files. Having multiple query files doesn't multiply the count - it just provides more ways for downloads to be tracked.

### Q: Does this affect model functionality?

**A**: No. `config.json` is purely metadata for HF's tracking system. Your model, scripts, and web UI will work exactly the same way.

---

## Monitoring & Analytics

### Where to View Download Stats

1. **Model Page**: https://huggingface.co/salakash/AskBuddyX
   - Download badge (if available)
   - Model card statistics

2. **Analytics Dashboard** (if you have PRO/Enterprise):
   - Detailed download metrics
   - Geographic distribution
   - Time-series data

3. **API Access**:
   ```python
   from huggingface_hub import HfApi
   api = HfApi()
   info = api.model_info("salakash/AskBuddyX")
   print(f"Downloads: {info.downloads}")
   ```

### Expected Timeline

- **Day 0**: Publish with `config.json`
- **Day 1-2**: First downloads may not show immediately
- **Day 3+**: Download counter should be visible
- **Week 1+**: Consistent tracking of all downloads

---

## Conclusion

### Summary

✅ **Problem Identified**: No query files in repository  
✅ **Solution Implemented**: Added `config.json` generation to bundle script  
✅ **Next Steps**: Republish to Hugging Face with `make publish`  
✅ **Expected Result**: All future downloads will be tracked

### Action Items

1. ✅ Update `make_bundle.py` to create `config.json` (DONE)
2. ⏳ Run `make bundle` to regenerate bundle with config.json
3. ⏳ Run `make publish` to upload to Hugging Face
4. ⏳ Verify `config.json` appears in HF repository
5. ⏳ Wait 24-48 hours for download stats to activate
6. ⏳ Test with script/web UI to confirm downloads are counted

### References

- [HF Download Stats Documentation](https://huggingface.co/docs/hub/models-download-stats)
- [HF Model Libraries](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/model-libraries.ts)
- [HF Integration Guide](https://huggingface.co/docs/hub/models-adding-libraries)

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-30  
**Status**: Solution Implemented, Awaiting Republish