import abc
import json
import logging
import os
import re
import time
import argparse
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Set

import jsonlines
import requests
import PyPDF2
import psutil
import textacy.preprocessing as preprocessing
import spacy
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('dataset_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load spaCy for named entity recognition and sentence segmentation
nlp = spacy.load("en_core_web_sm")

# Default domain keywords for a history-oriented dataset; method-specific extracts can replace this.
DEFAULT_DOMAIN_KEYWORDS = {
    'war', 'battle', 'conflict', 'treaty', 'empire', 'kingdom', 'dynasty', 'ruler',
    'king', 'queen', 'emperor', 'chief', 'leader', 'revolution', 'independence',
    'colonial', 'colony', 'settlement', 'migration', 'civilization', 'culture',
    'tradition', 'ritual', 'ceremony', 'religion', 'missionary', 'trade',
    'conquest', 'invasion', 'rebellion', 'uprising', 'movement', 'reform',
    'ancient', 'medieval', 'renaissance', 'industrial', 'modern', 'contemporary',
    'century', 'decade', 'era', 'period', 'age', 'epoch'
}

DEFAULT_PROMPT_TEMPLATE = '''
You are a domain expert tasked with generating 1-12 high-quality question-answer pairs from a given text passage for a fine-tuned question-answering model.
Your output MUST be a valid JSON array containing 1-12 objects, each with the fields "instruction", "input", and "output".
Do NOT include any text outside the JSON array.

Generate Q&A pairs based only on the passage below.

Passage:
<START_PASSAGE>
{chunk}
<END_PASSAGE>
'''

class KeywordExtractor(abc.ABC):
    def __init__(self, custom_keywords: Optional[Iterable[str]] = None):
        self.custom_keywords: Set[str] = set(x.lower() for x in custom_keywords or [])

    @abc.abstractmethod
    def extract(self, text: str) -> Set[str]:
        raise NotImplementedError

    def get_keywords(self, text: str) -> Set[str]:
        extracted = self.extract(text)
        return self.custom_keywords | extracted

class StaticKeywordExtractor(KeywordExtractor):
    def __init__(self, keywords_file: str = "keywords.txt", default_keywords: Optional[Set[str]] = None):
        self.keywords_file = keywords_file
        self.default_keywords = default_keywords or set()
        custom_keywords = self._load_keywords_from_file()
        super().__init__(custom_keywords=custom_keywords)

    def _load_keywords_from_file(self) -> Set[str]:
        if not os.path.exists(self.keywords_file):
            return set()
        try:
            with open(self.keywords_file, 'r', encoding='utf-8') as f:
                return {
                    line.strip().lower()
                    for line in f
                    if line.strip() and not line.strip().startswith('#')
                }
        except Exception as e:
            logger.warning(f"Unable to load keywords file {self.keywords_file}: {e}")
            return set()

    def extract(self, text: str) -> Set[str]:
        return self.default_keywords | self.custom_keywords

class TfIdfKeywordExtractor(KeywordExtractor):
    def __init__(self, custom_keywords: Optional[Iterable[str]] = None, top_n: int = 50,
                 ngram_range=(1, 2), stop_words='english'):
        super().__init__(custom_keywords=custom_keywords)
        self.top_n = top_n
        self.ngram_range = ngram_range
        self.stop_words = stop_words

    def extract(self, text: str) -> Set[str]:
        try:
            vectorizer = TfidfVectorizer(stop_words=self.stop_words, ngram_range=self.ngram_range, max_features=2000)
            matrix = vectorizer.fit_transform([text])
            if matrix.shape[1] == 0:
                return set()
            scores = matrix.toarray()[0]
            terms = vectorizer.get_feature_names_out()
            top_terms = sorted(
                ((term, score) for term, score in zip(terms, scores) if len(term) > 2),
                key=lambda item: item[1],
                reverse=True
            )[:self.top_n]
            return {term for term, _ in top_terms}
        except Exception as e:
            logger.warning(f"TF-IDF keyword extraction failed: {e}")
            return set()

class NmfKeywordExtractor(KeywordExtractor):
    def __init__(self, custom_keywords: Optional[Iterable[str]] = None, n_topics: int = 4, top_n: int = 50,
                 ngram_range=(1, 2), stop_words='english'):
        super().__init__(custom_keywords=custom_keywords)
        self.n_topics = n_topics
        self.top_n = top_n
        self.ngram_range = ngram_range
        self.stop_words = stop_words

    def extract(self, text: str) -> Set[str]:
        try:
            vectorizer = CountVectorizer(stop_words=self.stop_words, ngram_range=self.ngram_range, max_features=3000)
            matrix = vectorizer.fit_transform([text])
            if matrix.shape[1] == 0:
                return set()
            n_components = min(self.n_topics, matrix.shape[1])
            nmf_model = NMF(n_components=n_components, random_state=42, max_iter=200)
            nmf_model.fit(matrix)
            feature_names = vectorizer.get_feature_names_out()
            topics = nmf_model.components_
            keyword_scores: Dict[str, float] = {}
            for topic in topics:
                for term_index, score in enumerate(topic):
                    keyword_scores[feature_names[term_index]] = keyword_scores.get(feature_names[term_index], 0.0) + score
            most_common = sorted(keyword_scores.items(), key=lambda item: item[1], reverse=True)[:self.top_n]
            return {term for term, _ in most_common if len(term) > 2}
        except Exception as e:
            logger.warning(f"NMF keyword extraction failed: {e}")
            return set()


def create_keyword_extractor(method: str, keywords_file: str = "keywords.txt",
                             custom_keywords: Optional[Iterable[str]] = None) -> KeywordExtractor:
    method = method.lower()
    if method == 'static':
        return StaticKeywordExtractor(keywords_file=keywords_file, default_keywords=DEFAULT_DOMAIN_KEYWORDS)
    if method == 'tfidf':
        return TfIdfKeywordExtractor(custom_keywords=custom_keywords)
    if method == 'nmf':
        return NmfKeywordExtractor(custom_keywords=custom_keywords)
    raise ValueError(f"Unknown keyword extraction method: {method}")

def check_ollama_health():
    """Check if Ollama server is running."""
    try:
        response = requests.get("http://localhost:11434", timeout=5)
        if response.ok:
            print("Ollama server is running")
            logger.info("Ollama server is running")
            return True
        else:
            print(f"Ollama server error: {response.text}")
            logger.error(f"Ollama server responded with error: {response.text}")
            return False
    except requests.RequestException as e:
        print(f"Ollama server not reachable: {str(e)}")
        logger.error(f"Ollama server not reachable: {str(e)}")
        return False

def log_system_metrics(chunk_index):
    """Log CPU, memory, and GPU usage every 10 chunks."""
    if chunk_index % 10 != 0:
        return
    cpu_percent = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    mem_usage = mem.used / (1024 ** 3)  # GB
    mem_total = mem.total / (1024 ** 3)  # GB
    print(f"System metrics: CPU {cpu_percent:.1f}%, Memory {mem_usage:.1f}/{mem_total:.1f} GB")
    logger.info(f"System metrics: CPU {cpu_percent:.1f}%, Memory {mem_usage:.1f}/{mem_total:.1f} GB")
    try:
        import pynvml
        pynvml.nvmlInit()
        device = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(device)
        gpu_usage = gpu_mem.used / (1024 ** 3)  # GB
        gpu_total = gpu_mem.total / (1024 ** 3)  # GB
        print(f"GPU metrics: VRAM {gpu_usage:.1f}/{gpu_total:.1f} GB")
        logger.info(f"GPU metrics: VRAM {gpu_usage:.1f}/{gpu_total:.1f} GB")
    except Exception as e:
        print(f"Could not retrieve GPU metrics: {str(e)}")
        logger.warning(f"Could not retrieve GPU metrics: {str(e)}")

def read_text_file(path: str) -> str:
    """Read text from a plain text file."""
    start_time = time.time()
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as file:
            text = file.read()
        print(f"Read text file: {len(text)} characters in {time.time() - start_time:.2f}s")
        logger.info(f"Read text file: {len(text)} characters in {time.time() - start_time:.2f}s")
        return text
    except Exception as e:
        print(f"Error reading text file {path}: {str(e)}")
        logger.error(f"Error reading text file {path}: {str(e)}")
        return ""


def read_document_text(path: str) -> str:
    """Read text from a supported document type."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pdf':
        try:
            with open(path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
            print(f"Read PDF: {len(text)} characters")
            logger.info(f"Read PDF: {len(text)} characters")
            return text
        except Exception as e:
            print(f"Error reading PDF {path}: {str(e)}")
            logger.error(f"Error reading PDF {path}: {str(e)}")
            return ""
    if ext in {'.txt', '.md', '.csv'}:
        return read_text_file(path)
    print(f"Unknown extension {ext}. Trying to read as text.")
    logger.info(f"Unknown extension {ext}. Trying to read as text.")
    return read_text_file(path)


def chunk_text(text: str, keyword_set: Set[str], max_chars=800, overlap=200, min_chunks=3) -> List[str]:
    """Split text into semantically meaningful chunks using domain keywords or named entities."""
    doc = nlp(text)
    chunks: List[str] = []
    current_chunk = ""
    current_length = 0

    # Try splitting by paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_doc = nlp(para)
        has_domain_content = (
            not keyword_set or
            any(keyword in para.lower() for keyword in keyword_set) or
            any(ent.label_ in {'PERSON', 'GPE', 'ORG', 'DATE', 'EVENT'} for ent in para_doc.ents)
        )
        if not has_domain_content:
            print(f"Discarded paragraph (no domain content): {para[:100]}...")
            logger.debug(f"Discarded paragraph (no domain content): {para[:100]}...")
            continue
        para_length = len(para)

        if current_length + para_length <= max_chars:
            current_chunk += para + "\n\n"
            current_length += para_length + 2
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para[-overlap:] if len(para) > overlap else para
            current_length = len(current_chunk)

    if current_chunk:
        chunks.append(current_chunk.strip())

    # If too few chunks, split by sentences
    if len(chunks) < min_chunks and text:
        chunks = []
        current_chunk = ""
        current_length = 0
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            sent_length = len(sent_text)
            has_domain_content = (
                not keyword_set or
                any(keyword in sent_text.lower() for keyword in keyword_set) or
                any(ent.label_ in {'PERSON', 'GPE', 'ORG', 'DATE', 'EVENT'} for ent in sent.ents)
            )
            if not has_domain_content:
                print(f"Discarded sentence (no domain content): {sent_text[:100]}...")
                logger.debug(f"Discarded sentence (no domain content): {sent_text[:100]}...")
                continue
            if current_length + sent_length <= max_chars:
                current_chunk += sent_text + " "
                current_length += sent_length + 1
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sent_text[-overlap:] if len(sent_text) > overlap else sent_text
                current_length = len(current_chunk)
        if current_chunk:
            chunks.append(current_chunk.strip())

    print(f"Split text into {len(chunks)} chunks (text length: {len(text)} chars)")
    logger.info(f"Split text into {len(chunks)} chunks (text length: {len(text)} chars)")
    return chunks

def normalize_text(text):
    """Normalize text to handle case sensitivity and clean up."""
    #return preprocessing.normalize.unicode(text.lower())
    return preprocessing.normalize.unicode(text)


def is_domain_qa(question: str, answer: str, keyword_set: Set[str]) -> bool:
    """Check if a Q&A pair belongs to the document domain using named entities or keywords."""
    doc_q = nlp(question)
    doc_a = nlp(answer)
    has_entities = any(ent.label_ in {'PERSON', 'GPE', 'ORG', 'DATE', 'EVENT'} for ent in doc_q.ents + doc_a.ents)
    if not keyword_set:
        return has_entities or len(answer) >= 100
    has_keywords = any(keyword in question.lower() or keyword in answer.lower() for keyword in keyword_set)
    return has_entities or has_keywords

def deduplicate_qa_pairs(pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove duplicate Q&A pairs based on semantic similarity."""
    if not pairs:
        return pairs

    texts = [pair["instruction"] + " " + pair["output"] for pair in pairs]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    keep_indices = []
    for i in range(len(pairs)):
        if any(similarity_matrix[i][j] > 0.9 for j in keep_indices):
            continue
        keep_indices.append(i)

    deduped_pairs = [pairs[i] for i in keep_indices]
    print(f"Deduplicated {len(pairs)} to {len(deduped_pairs)} Q&A pairs")
    logger.info(f"Deduplicated {len(pairs)} to {len(deduped_pairs)} Q&A pairs")
    return deduped_pairs[:12]  # Limit to 12 pairs

def fix_json_string(json_str):
    """Attempt to fix common JSON errors (trailing commas, invalid escapes)."""
    try:
        # Remove trailing commas
        json_str = re.sub(r',\s*]', ']', json_str)
        json_str = re.sub(r',\s*}', '}', json_str)
        # Fix invalid escapes
        json_str = re.sub(r'\\[^\\bfnrtu"]', r'\\', json_str)
        return json_str
    except Exception as e:
        print(f"Failed to fix JSON string: {str(e)}")
        logger.error(f"Failed to fix JSON string: {str(e)}")
        return json_str

def extract_relevant_input(chunk: str, question: str, full_text: str, keyword_set: Set[str]) -> str:
    """Extract relevant sentences from the chunk or full text for the input field."""
    doc = nlp(chunk)
    relevant_sentences = []
    question_keywords = set(normalize_text(question).split()) & keyword_set

    # First, try to find relevant sentences in the chunk
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if any(keyword in normalize_text(sent_text) for keyword in question_keywords) or \
           any(ent.label_ in {'PERSON', 'GPE', 'ORG', 'DATE', 'EVENT'} for ent in sent.ents):
            relevant_sentences.append(sent_text)

    # If no relevant sentences found, search the full text
    if not relevant_sentences:
        doc_full = nlp(full_text)
        for sent in doc_full.sents:
            sent_text = sent.text.strip()
            if any(keyword in normalize_text(sent_text) for keyword in question_keywords) or \
               any(ent.label_ in {'PERSON', 'GPE', 'ORG', 'DATE', 'EVENT'} for ent in sent.ents):
                relevant_sentences.append(sent_text)
                if len(' '.join(relevant_sentences)) >= 800:
                    break

    input_text = ' '.join(relevant_sentences)[:800]
    if not input_text:
        input_text = chunk[:800]  # Fallback to original chunk if no relevant sentences found
    return input_text


def extract_json_array(text: str, chunk: str, full_text: str, keyword_set: Set[str]) -> List[Dict[str, str]]:
    """Extract JSON array from text, handling malformed cases."""
    exclude_patterns = [
        r'who.*wrote', r'who.*authored', r'what.*title', r'what.*published',
        r'what.*topic.*passage', r'what.*debate', r'what.*focus.*research',
        r'who.*supervisor', r'what.*permits', r'what.*financial', r'what.*table of contents',
        r'who.*mentioned', r'what.*orthography', r'who.*provided', r'what.*list',
        r'what.*abbreviations', r'who.*assisted', r'who.*intellectually',
        r'what.*mean', r'define\s+', r'what.*structure'
    ]
    try:
        start = text.find('[')
        end = text.rfind(']') + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON array found")
        json_str = text[start:end]
        json_str = fix_json_string(json_str)
        parsed = json.loads(json_str)
        if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
            raise ValueError("Invalid JSON structure: not a list of dictionaries")

        filtered_pairs: List[Dict[str, str]] = []
        for item in parsed:
            question = normalize_text(item["instruction"])
            answer = normalize_text(item["output"])
            if (not any(re.search(pattern, question) for pattern in exclude_patterns) and
                len(answer) >= 100 and is_domain_qa(question, answer, keyword_set)):
                item["input"] = extract_relevant_input(chunk, question, full_text, keyword_set)
                item["instruction"] = question.capitalize()
                item["output"] = item["output"].strip() # Keep original formatting for answer
                filtered_pairs.append(item)
            else:
                print(f"Filtered out Q&A pair: Q: {question}, A: {answer[:50]}... (non-domain or too short)")
                logger.debug(f"Filtered out Q&A pair: Q: {question}, A: {answer[:50]}... (non-domain or too short)")

        print(f"Parsed {len(filtered_pairs)} valid JSON Q&A pairs")
        logger.info(f"Parsed {len(filtered_pairs)} valid JSON Q&A pairs")
        return deduplicate_qa_pairs(filtered_pairs)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"JSON parsing failed: {str(e)}. Non-JSON response:\n{text[:500]}...")
        logger.warning(f"JSON parsing failed: {str(e)}. Non-JSON response:\n{text[:500]}...")
        return parse_qa_pairs(text, chunk, full_text, keyword_set)


def parse_qa_pairs(text: str, chunk: str, full_text: str, keyword_set: Set[str]) -> List[Dict[str, str]]:
    """Parse Q: A: pairs from non-JSON response, focusing on domain content."""
    qa_pairs: List[Dict[str, str]] = []
    qa_pattern = re.compile(r'Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:|$)', re.DOTALL)
    matches = qa_pattern.findall(text)
    
    exclude_patterns = [
        r'who.*wrote', r'who.*authored', r'what.*title', r'what.*published',
        r'what.*topic.*passage', r'what.*debate', r'what.*focus.*research',
        r'who.*supervisor', r'what.*permits', r'what.*financial', r'what.*table of contents',
        r'who.*mentioned', r'what.*orthography', r'who.*provided', r'what.*list',
        r'what.*abbreviations', r'who.*assisted', r'who.*intellectually',
        r'what.*mean', r'define\s+', r'what.*structure'
    ]
    
    for question, answer in matches:
        question = normalize_text(question.strip())
        answer = normalize_text(answer.strip())
        if (not any(re.search(pattern, question) for pattern in exclude_patterns) and
            len(answer) >= 100 and is_domain_qa(question, answer, keyword_set)):
            qa_pairs.append({
                "instruction": question.capitalize(),
                "input": extract_relevant_input(chunk, question, full_text, keyword_set),
                "output": answer
            })
            print(f"Accepted Q&A pair: Q: {question.capitalize()}, A: {answer[:50]}...")
            logger.info(f"Accepted Q&A pair: Q: {question.capitalize()}, A: {answer[:50]}...")
        else:
            print(f"Filtered out Q&A pair: Q: {question}, A: {answer[:50]}... (non-domain or too short)")
            logger.debug(f"Filtered out Q&A pair: Q: {question}, A: {answer[:50]}... (non-domain or too short)")
    
    if not qa_pairs:
        print(f"No valid Q&A pairs found in output for chunk:\n{chunk[:500]}...")
        logger.warning(f"No valid Q&A pairs found in output for chunk:\n{chunk[:500]}...")
    else:
        print(f"Extracted {len(qa_pairs)} Q&A pairs via fallback")
        logger.info(f"Extracted {len(qa_pairs)} Q&A pairs via fallback")
    
    return deduplicate_qa_pairs(qa_pairs)

def generate_questions_answers(chunk, full_text, model_name="llama3.1", prompt_template: Optional[str] = None, keyword_set: Optional[Set[str]] = None, max_retries=5):
    """Generate 1-12 Q&A pairs from a document chunk using a generic domain prompt."""
    print(f"Processing chunk (first 200 chars): {chunk[:200]}...")
    logger.info(f"Processing chunk (first 200 chars): {chunk[:200]}...")
    start_time = time.time()
    try:
        prompt = prompt_template.format(chunk=chunk)
    except KeyError:
        prompt = (
            prompt_template.rstrip()
            + "\n\nPassage:\n"
            + chunk
        )

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7
                },
                timeout=180
            )
            if not response.ok:
                print(f"LLaMA request failed (attempt {attempt+1}/{max_retries}): {response.text}")
                logger.error(f"LLaMA request failed (attempt {attempt+1}/{max_retries}): {response.text}")
                time.sleep(5)
                continue

            raw = response.json()["response"]
            with open("raw_responses.log", "a") as f:
                f.write(f"Chunk response at {datetime.now()}:\n{raw}\n{'='*50}\n")
            print(f"Response time: {time.time() - start_time:.2f}s")
            logger.info(f"Response time: {time.time() - start_time:.2f}s")
            return extract_json_array(raw, chunk, full_text, keyword_set)
        except Exception as e:
            print(f"Error processing chunk (attempt {attempt+1}/{max_retries}): {str(e)}")
            logger.error(f"Error processing chunk (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                print(f"Failed to process chunk after {max_retries} attempts:\n{chunk[:500]}...")
                logger.error(f"Failed to process chunk after {max_retries} attempts:\n{chunk[:500]}...")
                return []
            time.sleep(5)
    return []

def process_chunk(i: int, chunk: str, full_text: str, model_name: str, prompt_template: Optional[str], keyword_set: Set[str]):
    """Helper function to process a single chunk and return results."""
    log_system_metrics(i)
    items = generate_questions_answers(chunk, full_text, model_name, prompt_template, keyword_set)
    return i, chunk, items

def generate_dataset_from_document(document_path: str, output_path: str, model_name: str = "llama3.1", keyword_method: str = "static", keywords_file: str = "keywords.txt", prompt_template_path: Optional[str] = None, prompt_string: Optional[str] = None, start_chunk: int = 0, temp_path: Optional[str] = None, checkpoint_path: str = "checkpoint.json", max_workers: int = 4):
    """Generate a dataset from a document and save to JSONL, processing chunks in parallel."""
    if not check_ollama_health():
        print("Aborting: Ollama server not available")
        logger.error("Aborting: Ollama server not available")
        return

    if temp_path is None:
        bookname = os.path.splitext(os.path.basename(document_path))[0]
        temp_path = f"temp_{bookname}.jsonl"
    
    start_time = time.time()
    full_text = read_document_text(document_path)
    print(f"First 500 chars of full text: {full_text[:500]}")
    logger.info(f"First 500 chars of full text: {full_text[:500]}")
    if len(full_text) < 200:
        print(f"Warning: Input text is too short ({len(full_text)} chars). Consider augmenting with additional sources.")
        logger.warning(f"Input text is too short ({len(full_text)} chars). Consider augmenting with additional sources.")

    keyword_extractor = create_keyword_extractor(keyword_method, keywords_file=keywords_file)
    keyword_set = keyword_extractor.get_keywords(full_text)
    print(keyword_set)
    print(f"Keyword extraction method: {keyword_method}. Keywords loaded: {len(keyword_set)}")
    logger.info(f"Keyword extraction method: {keyword_method}. Keywords loaded: {len(keyword_set)}")

    prompt_template = DEFAULT_PROMPT_TEMPLATE
    if prompt_string:
        prompt_template = prompt_string
        print("Using prompt template from command-line argument")
    elif prompt_template_path and os.path.exists(prompt_template_path):
        with open(prompt_template_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        print(f"Loaded prompt template from {prompt_template_path}")

    chunks = chunk_text(full_text, keyword_set)
    
    if start_chunk < 0 or start_chunk >= len(chunks):
        print(f"Invalid start_chunk {start_chunk}; must be between 0 and {len(chunks)-1}")
        logger.error(f"Invalid start_chunk {start_chunk}; must be between 0 and {len(chunks)-1}")
        return

    all_data = []
    stats = {"successful_chunks": 0, "failed_chunks": 0, "total_pairs": 0}
    failed_chunks = []
    lock = threading.Lock()

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        all_data = checkpoint.get("data", [])
        stats = checkpoint.get("stats", {"successful_chunks": 0, "failed_chunks": 0, "total_pairs": 0})
        failed_chunks = checkpoint.get("failed_chunks", [])
        checkpoint_last = checkpoint.get("last_chunk", -1)
        if checkpoint_last >= start_chunk:
            start_chunk = checkpoint_last + 1
            print(f"Resumed from checkpoint: {len(all_data)} pairs, starting at chunk {start_chunk}")
            logger.info(f"Resumed from checkpoint: {len(all_data)} pairs, starting at chunk {start_chunk}")
        else:
            print(f"Ignoring checkpoint (last_chunk {checkpoint_last} < start_chunk {start_chunk})")
            logger.info(f"Ignoring checkpoint (last_chunk {checkpoint_last} < start_chunk {start_chunk})")

    try:
        with tqdm(total=len(chunks), initial=start_chunk, desc="Processing chunks", unit="chunk") as pbar:
            for batch_start in range(start_chunk, len(chunks), max_workers):
                batch = [(i, chunk) for i, chunk in enumerate(chunks[batch_start:batch_start + max_workers], start=batch_start)]
                futures = []
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for i, chunk in batch:
                        futures.append(executor.submit(process_chunk, i, chunk, full_text, model_name, prompt_template, keyword_set))
                    
                    for future in as_completed(futures):
                        try:
                            i, chunk, items = future.result()
                            with lock:
                                if items:
                                    all_data.extend(items)
                                    stats["successful_chunks"] += 1
                                    stats["total_pairs"] += len(items)
                                    print(f"Chunk {i}/{len(chunks)}: Generated {len(items)} Q&A pairs")
                                    logger.info(f"Chunk {i}/{len(chunks)}: Generated {len(items)} Q&A pairs")
                                    if stats["successful_chunks"] % 3 == 0:
                                        with jsonlines.open(temp_path, mode='w') as writer:
                                            writer.write_all(all_data)
                                        print(f"Saved partial results to {temp_path}")
                                        logger.info(f"Saved partial results to {temp_path}")
                                else:
                                    stats["failed_chunks"] += 1
                                    failed_chunks.append((i, chunk))
                                    print(f"Chunk {i}/{len(chunks)}: No Q&A pairs generated, added to failed_chunks")
                                    logger.warning(f"Chunk {i}/{len(chunks)}: No Q&A pairs generated, added to failed_chunks")
                                
                                with open(checkpoint_path, 'w') as f:
                                    json.dump({
                                        "data": all_data,
                                        "stats": stats,
                                        "last_chunk": i,
                                        "failed_chunks": failed_chunks
                                    }, f)
                                
                                elapsed = time.time() - start_time
                                processed_chunks = i - start_chunk + 1 if i >= start_chunk else 1
                                avg_time_per_chunk = elapsed / processed_chunks
                                remaining_chunks = len(chunks) - i
                                eta_seconds = remaining_chunks * avg_time_per_chunk
                                eta = str(timedelta(seconds=int(eta_seconds)))
                                pbar.set_postfix({"ETA": eta, "Pairs": stats["total_pairs"]})
                                pbar.update(1)
                        except Exception as e:
                            print(f"Error processing future for chunk {i}: {str(e)}")
                            logger.error(f"Error processing future for chunk {i}: {str(e)}")
                
                time.sleep(2)

        if failed_chunks:
            print(f"Retrying {len(failed_chunks)} failed chunks sequentially")
            logger.info(f"Retrying {len(failed_chunks)} failed chunks sequentially")
            for i, chunk in failed_chunks:
                log_system_metrics(i)
                print(f"Retrying chunk {i}/{len(chunks)}")
                logger.info(f"Retrying chunk {i}/{len(chunks)}")
                items = generate_questions_answers(chunk, full_text, model_name, prompt_template, keyword_set)
                with lock:
                    if items:
                        all_data.extend(items)
                        stats["successful_chunks"] += 1
                        stats["total_pairs"] += len(items)
                        print(f"Retry chunk {i}/{len(chunks)}: Generated {len(items)} Q&A pairs")
                        logger.info(f"Retry chunk {i}/{len(chunks)}: Generated {len(items)} Q&A pairs")
                        if stats["successful_chunks"] % 3 == 0:
                            with jsonlines.open(temp_path, mode='w') as writer:
                                writer.write_all(all_data)
                            print(f"Saved partial results to {temp_path}")
                            logger.info(f"Saved partial results to {temp_path}")
                    else:
                        print(f"Retry chunk {i}/{len(chunks)}: No Q&A pairs generated:\n{chunk[:500]}...")
                        logger.warning(f"Retry chunk {i}/{len(chunks)}: No Q&A pairs generated:\n{chunk[:500]}...")
                    with open(checkpoint_path, 'w') as f:
                        json.dump({
                            "data": all_data,
                            "stats": stats,
                            "last_chunk": i,
                            "failed_chunks": failed_chunks
                        }, f)

    except Exception as e:
        print(f"Processing interrupted: {str(e)}. Saving partial results")
        logger.error(f"Processing interrupted: {str(e)}. Saving partial results")
        with lock:
            if all_data:
                with jsonlines.open(temp_path, mode='w') as writer:
                    writer.write_all(all_data)
                print(f"Saved partial results to {temp_path}")
                logger.info(f"Saved partial results to {temp_path}")

    print("Generated Q&A pairs (sample of up to 10):")
    logger.info("Generated Q&A pairs (sample of up to 10):")
    for i, item in enumerate(all_data[:10], 1):
        print(f"Pair {i}:")
        print(f"  Instruction: {item['instruction']}")
        print(f"  Input: {item['input'][:200]}...")
        print(f"  Output: {item['output'][:100]}...")
        print(f"  Input length: {len(item['input'])} chars")
        logger.info(f"Pair {i}:")
        logger.info(f"  Instruction: {item['instruction']}")
        logger.info(f"  Input: {item['input'][:200]}...")
        logger.info(f"  Output: {item['output'][:100]}...")
        logger.info(f"  Input length: {len(item['input'])} chars")

    if all_data:
        with jsonlines.open(output_path, mode='w') as writer:
            writer.write_all(all_data)
        print(f"Dataset written to {output_path} with {len(all_data)} items")
        logger.info(f"Dataset written to {output_path} with {len(all_data)} items")
    else:
        print(f"No data written to {output_path}: No Q&A pairs generated")
        logger.warning(f"No data written to {output_path}: No Q&A pairs generated")

    elapsed_time = str(timedelta(seconds=int(time.time() - start_time)))
    print(f"Processing complete in {elapsed_time}")
    print(f"Summary: {stats['successful_chunks']} successful chunks, "
          f"{stats['failed_chunks']} failed chunks, {stats['total_pairs']} total Q&A pairs")
    logger.info(f"Processing complete in {elapsed_time}")
    logger.info(f"Summary: {stats['successful_chunks']} successful chunks, "
                f"{stats['failed_chunks']} failed chunks, {stats['total_pairs']} total Q&A pairs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Q&A dataset from a document")
    parser.add_argument("document_path", help="Path to the source document file")
    parser.add_argument("output_path", help="Path for the output JSONL file")
    parser.add_argument("--keyword-method", type=str, default="static", choices=["static", "tfidf", "nmf"], help="Keyword extraction method to use")
    parser.add_argument("--keywords-file", type=str, default="keywords.txt", help="Path to a custom keyword file")
    parser.add_argument("--prompt-template", type=str, default=None, help="Path to a custom prompt template file")
    parser.add_argument("--prompt-string", type=str, default=None, help="Custom prompt template string")
    parser.add_argument("--start-chunk", type=int, default=0, help="Starting chunk index (0-based, default 0)")
    parser.add_argument("--model-name", type=str, default="llama3.1", help="Ollama model name (e.g., llama3.1, mistral)")
    parser.add_argument("--max-workers", type=int, default=4, help="Number of parallel workers (default 4)")
    args = parser.parse_args()

    generate_dataset_from_document(
        args.document_path,
        args.output_path,
        model_name=args.model_name,
        keyword_method=args.keyword_method,
        keywords_file=args.keywords_file,
        prompt_template_path=args.prompt_template,
        prompt_string=args.prompt_string,
        start_chunk=args.start_chunk,
        max_workers=args.max_workers
    )
