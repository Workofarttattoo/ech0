# Ollama Integration for ech0 (Optional)

## Overview

ech0's wisdom ingestion system supports **optional** Ollama integration to generate richer, more diverse training examples using local LLMs. **By default, ech0 uses grounded datasets** (pre-defined verified knowledge) which requires no external dependencies.

## Quick Start (Grounded Data Only - Default)

**No setup needed!** Just run:

```bash
python3 generate_1m_dataset.py
```

This uses pre-defined, verified knowledge and:
- ✅ No external dependencies
- ✅ No API calls
- ✅ Fast and reliable
- ✅ Automatically streams to external drive when connected

## Optional: Enable Ollama for Enhanced Generation

If you want **richer, AI-generated examples**, you can optionally enable Ollama:

### 1. Install Ollama

**macOS/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Or download from**: https://ollama.com/download

### 2. Start Ollama

```bash
ollama serve
```

Keep this running in a separate terminal.

### 3. Pull a Model

```bash
# Recommended models:
ollama pull mistral      # Best balance (7B params)
ollama pull llama2       # Alternative (7B params)
ollama pull codellama    # For code-heavy domains (7B params)
ollama pull mixtral      # More powerful (47B params, slower)
```

### 4. Enable in Configuration

Edit `ech0_finetune_config.yaml`:

```yaml
data_sources:
  grounded:
    enabled: true
    use_only_grounded: false  # Allow Ollama enhancement

  ollama:
    enabled: true  # Enable Ollama
    model: "mistral"  # Which model to use
    timeout: 300  # 5 minutes (increased from default)
    max_retries: 3
```

### 5. Generate Enhanced Dataset

```bash
python3 generate_1m_dataset.py
```

Now it will use Ollama to generate richer examples while still grounding them in verified knowledge.

## Configuration Options

All settings in `ech0_finetune_config.yaml` under `data_sources.ollama`:

```yaml
ollama:
  enabled: false  # Set to true to enable
  base_url: "http://localhost:11434"
  model: "mistral"  # Model to use
  timeout: 300  # Timeout in seconds (5 minutes)
  max_retries: 3  # Number of retries on failure
  temperature: 0.7  # Creativity (0.0-1.0, higher = more creative)
```

## Troubleshooting

### "Cannot connect to Ollama" Error

```bash
# Check if Ollama is running:
curl http://localhost:11434/api/tags

# If not, start it:
ollama serve
```

### Timeout Errors

The default timeout is 300 seconds (5 minutes). If you still get timeouts:

1. **Use a smaller model**: `mistral` (7B) instead of `mixtral` (47B)
2. **Increase timeout** in config:
   ```yaml
   ollama:
     timeout: 600  # 10 minutes
   ```
3. **Check system resources**: Ollama needs adequate RAM/GPU

### Model Not Found

```bash
# List installed models:
ollama list

# Pull the model you need:
ollama pull mistral
```

## Performance Comparison

| Mode | Speed | Quality | Dependencies |
|------|-------|---------|--------------|
| **Grounded Only** (default) | ⚡ Fast | ✓ Verified | None |
| **Grounded + Ollama** | 🐢 Slower | ✨ Enhanced | Ollama required |

## Recommendations

**Start with grounded-only mode** (default) and only enable Ollama if you need:
- More diverse phrasing and examples
- Creative variations on concepts
- Larger dataset sizes with unique examples

For most users, the grounded dataset provides excellent quality without any setup.

## Testing Ollama Integration

Test if Ollama is working:

```bash
python3 ech0_ollama_integration.py
```

This will:
1. Check if Ollama is running
2. List available models
3. Generate a test example

## Technical Details

### How It Works

1. **Grounded templates** provide the knowledge base
2. **Ollama** (if enabled) enhances examples with:
   - More natural phrasing
   - Additional context
   - Varied question formats
   - Creative reformulations

### Fallback Behavior

If Ollama is enabled but fails:
- System automatically falls back to grounded-only mode
- Retries with exponential backoff
- Logs warnings but continues generation
- No data loss or interruption

### API Calls

- **Grounded mode**: 0 API calls (all local)
- **Ollama mode**: Local API calls to Ollama (http://localhost:11434)
- No external internet APIs required in either mode

## FAQ

**Q: Do I need Ollama?**
A: No! The default grounded mode works great without it.

**Q: Will enabling Ollama slow down generation?**
A: Yes, significantly. Ollama generation is ~10-50x slower than grounded templates.

**Q: Can I use Ollama on external drive?**
A: Yes! External drive detection works with or without Ollama.

**Q: Which model should I use?**
A: Start with `mistral` - good balance of speed and quality.

**Q: Can I switch models mid-generation?**
A: No, restart generation with new model in config.

**Q: Does Ollama require internet?**
A: Only to download models initially. Generation is fully offline.
