# Document to Dataset Generator (General Corpus → Q&A Dataset)

A flexible Python tool that converts **any document corpus** (PDF, TXT, CSV, Markdown, etc.) into high-quality **question-answer datasets** for training or fine-tuning AI models.

Unlike traditional approaches that rely on static keyword lists, this tool introduces **automatic keyword extraction using NLP techniques (TF-IDF / NMF)**—making it adaptable to **any domain** (technical docs, legal text, novels, research papers, internal knowledge bases, etc.).

---

## Key Features

### Multi-Format Document Support
- PDF (`.pdf`)
- Text (`.txt`)
- Markdown (`.md`)
- CSV (`.csv`)
- Fallback support for unknown formats

---

### Intelligent Keyword Extraction

Choose how the system understands your domain:

- **Static (default)**  
  Uses predefined + custom keywords from `keywords.txt`

- **TF-IDF (Recommended for general corpora)**  
  Automatically extracts high-signal terms from your document

- **NMF (Topic Modeling)**  
  Extracts topic-based keywords for broader thematic understanding

This removes the need to manually curate keywords for every dataset.

---

### AI-Powered Q&A Generation
- Uses **Ollama local models** (e.g., `llama3.1`, `mistral`)
- Generates **1–12 high-quality Q&A pairs per chunk**
- Enforces structured JSON output

---

### Smart Text Chunking
- Semantic chunking based on:
  - Extracted keywords
  - Named entities (people, locations, dates, etc.)
- Falls back to sentence-level chunking if needed

---

### Data Quality Controls
- Deduplicates similar Q&A pairs (cosine similarity)
- Filters:
  - Low-quality answers
  - Non-domain content
  - Meta/document structure questions (e.g., “who wrote this”)

---

### Performance & Reliability
- Multi-threaded processing
- Checkpointing (resume after interruption)
- Failed chunk retry system
- System monitoring:
  - CPU
  - Memory
  - GPU (if available)

---

### Custom Prompting
- Provide your own prompt template:
  - File-based (`--prompt-template`)
  - Inline (`--prompt-string`)
- Enables domain-specific dataset shaping

---

## Prerequisites

### 1. Install Ollama

Ollama is required for local model inference.

**macOS/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
````

**Windows:**
Download from: [https://ollama.ai/download](https://ollama.ai/download)

---

### Start Ollama + Pull Model

```bash
ollama serve

# In another terminal:
ollama pull llama3.1
ollama run llama3.1
```

Verify:

```
http://localhost:11434
```

---

### 2. Python Setup

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install jsonlines requests tqdm PyPDF2 psutil textacy scikit-learn spacy pynvml
python -m spacy download en_core_web_sm
```

---

## Usage

### Basic Usage

```bash
python script.py <input_document> <output_dataset.jsonl>
```

Example:

```bash
python script.py "book.pdf" "dataset.jsonl"
```

---

### Using Automatic Keyword Extraction (Recommended)

```bash
python script.py "research_paper.pdf" "dataset.jsonl" \
    --keyword-method tfidf
```

Or with topic modeling:

```bash
python script.py "technical_docs.txt" "dataset.jsonl" \
    --keyword-method nmf
```

---

### Using Static Keywords (Manual Control)

```bash
python script.py "history_book.pdf" "dataset.jsonl" \
    --keyword-method static \
    --keywords-file keywords.txt
```

---

### Advanced Usage

```bash
python script.py <input> <output> \
    --model-name llama3.1 \
    --keyword-method tfidf \
    --max-workers 8 \
    --start-chunk 50
```

---

### Custom Prompt Template

#### Option 1: File

```bash
python script.py "doc.txt" "dataset.jsonl" \
    --prompt-template custom_prompt.txt
```

#### Option 2: Inline

```bash
python script.py "doc.txt" "dataset.jsonl" \
    --prompt-string "Generate detailed technical Q&A from the following text: {chunk}"
```

---

## Output Format

Each line in the output JSONL file:

```json
{
  "instruction": "What is X?",
  "input": "Relevant extracted context...",
  "output": "Detailed answer..."
}
```

---

## How It Works (High-Level)

1. Read document
2. Extract keywords (static / TF-IDF / NMF)
3. Chunk text based on semantic + entity relevance
4. Generate Q&A via Ollama
5. Filter and clean results
6. Deduplicate
7. Write JSONL dataset

---

## Command Line Arguments

### Required

* `document_path` – Input file
* `output_path` – Output dataset

---

### Optional

| Argument            | Description                      |
| ------------------- | -------------------------------- |
| `--keyword-method`  | `static`, `tfidf`, `nmf`         |
| `--keywords-file`   | Path to custom keywords          |
| `--prompt-template` | Path to prompt file              |
| `--prompt-string`   | Inline prompt override           |
| `--model-name`      | Ollama model (default: llama3.1) |
| `--max-workers`     | Parallel threads (default: 4)    |
| `--start-chunk`     | Resume from chunk index          |

---

## Performance Tips

* Increase `--max-workers` for faster processing (if CPU allows)
* Use GPU-enabled Ollama for large documents
* TF-IDF is faster; NMF is more semantic but heavier
* Monitor system logs for bottlenecks

---

## Resume & Checkpoints

* Progress saved to `checkpoint.json`
* Partial output saved to `temp_<document>.jsonl`
* Re-run the same command to resume

---

## Logging

Generated files:

* `dataset_generation.log` – main log
* `raw_responses.log` – raw model outputs

---

## Example Use Cases

This tool is domain-agnostic:

* Academic / research datasets
* Internal knowledge base transformation
* Legal or compliance document structuring
* Technical documentation Q&A generation
* Book or narrative comprehension datasets

---

## Troubleshooting

### Ollama not reachable

Ensure:

```bash
ollama serve
```

---

### No Q&A pairs generated

Try:

* `--keyword-method tfidf`
* Different model (e.g., `mistral`)
* Adjusting the prompt

---

### Memory issues

Reduce:

```bash
--max-workers
```

---

### Poor output quality

* Use a custom prompt template
* Switch keyword extraction method

---

## Contributing

Contributions are welcome:

* Additional keyword extraction strategies
* Improved filtering heuristics
* Prompt engineering improvements

---

## License

Open source – see license file for details.

