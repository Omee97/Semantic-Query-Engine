from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List
import os
import tempfile
import google.generativeai as genai
# ---------- Pinecone imports changed ----------
from pinecone import Pinecone, ServerlessSpec
# ----------------------------------------------
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import uvicorn
import re
import logging
import time
import spacy
from concurrent.futures import ThreadPoolExecutor
import asyncio

# ---------------- Logging & Threadpool ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure parallel processing
PROCESSING_THREADS = 4
executor = ThreadPoolExecutor(max_workers=PROCESSING_THREADS)

# ---------------- Load spaCy ----------------
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Loaded spaCy model for text processing")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {str(e)}")
    raise SystemExit(1)

# ---------------- Load environment variables ----------------
load_dotenv()

def get_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        logger.error(f"Missing required environment variable: {name}")
        raise SystemExit(1)
    return value

GEMINI_API_KEY = get_env_var("GEMINI_API_KEY")
PINECONE_API_KEY = get_env_var("PINECONE_API_KEY")
PINECONE_INDEX_NAME = get_env_var("PINECONE_INDEX_NAME")

EMBEDDING_DIMENSION = 768  # Gemini embedding dimension

# ---------------- Initialize Gemini ----------------
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-pro")
    logger.info("Gemini initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini: {str(e)}")
    raise SystemExit(1)

# ---------------- Initialize Pinecone (NEW STYLE) ----------------
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # list existing indexes (the Pinecone object returns a structure with .names())
    existing_names = []
    try:
        # pc.list_indexes() in some SDKs returns object with .names() as in docs; handle robustly
        li = pc.list_indexes()
        if hasattr(li, "names"):
            existing_names = li.names()
        elif isinstance(li, (list, tuple)):
            existing_names = list(li)
        else:
            # try to coerce
            existing_names = list(li)
    except Exception:
        # fallback to empty list on unexpected shape
        existing_names = []

    recreate_index = False
    if PINECONE_INDEX_NAME in existing_names:
        try:
            index_info = pc.describe_index(PINECONE_INDEX_NAME)
            # index_info may be dict-like with 'dimension'
            idx_dim = None
            if isinstance(index_info, dict):
                idx_dim = index_info.get("dimension")
            elif hasattr(index_info, "dimension"):
                idx_dim = getattr(index_info, "dimension")
            if idx_dim and int(idx_dim) != EMBEDDING_DIMENSION:
                logger.warning(f"Existing index has wrong dimension ({idx_dim} vs {EMBEDDING_DIMENSION}). Deleting...")
                pc.delete_index(PINECONE_INDEX_NAME)
                recreate_index = True
                time.sleep(5)
        except Exception:
            # If describe fails, attempt recreate
            recreate_index = True

    if PINECONE_INDEX_NAME not in existing_names or recreate_index:
        logger.info(f"Creating index {PINECONE_INDEX_NAME} with dimension {EMBEDDING_DIMENSION}...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        logger.info("Waiting briefly for index initialization...")
        time.sleep(5)

    # Create an Index client from the Pinecone client
    index = pc.Index(PINECONE_INDEX_NAME)

    try:
        stats = index.describe_index_stats()
        logger.info(f"Using Pinecone index: {PINECONE_INDEX_NAME}")
        logger.info(f"Index stats - vector count: {stats.get('total_vector_count', 0)}")
    except Exception:
        logger.info("Index ready (stats unreadable).")

except Exception as e:
    logger.error(f"Pinecone initialization failed: {str(e)}")
    raise SystemExit(1)

# ---------------- FastAPI setup ----------------
app = FastAPI(title="PDF Question Answering Platform")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------------- Utility functions (original logic preserved) ----------------
def truncate_by_bytes(text: str, max_bytes: int = 40000) -> str:
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode('utf-8', errors='ignore').rstrip()

async def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        logger.info(f"Fast text extraction from: {pdf_path}")
        text = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: "\n".join(
                page.extract_text() for page in PdfReader(pdf_path).pages if page.extract_text()
            )
        )

        if not text.strip():
            raise ValueError("PDF contains no extractable text")

        logger.info(f"Extracted {len(text.split())} words")
        return text
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        raise HTTPException(400, f"PDF processing error: {str(e)}")

async def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    def process_chunk(sentences):
        chunks = []
        current_chunk = []
        current_size = 0

        for sent in sentences:
            word_count = len(sent.split())
            if current_size + word_count > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
            current_chunk.append(sent)
            current_size += word_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    try:
        doc = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: nlp(text[:1000000])
        )

        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        if not sentences:
            return []

        batch_size = max(1, len(sentences) // PROCESSING_THREADS + 1)
        batches = [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]

        tasks = [
            asyncio.get_event_loop().run_in_executor(executor, process_chunk, batch)
            for batch in batches
        ]

        results = await asyncio.gather(*tasks)
        chunks = [chunk for sublist in results for chunk in sublist]

        logger.info(f"Created {len(chunks)} chunks using {PROCESSING_THREADS} threads")
        return chunks

    except Exception as e:
        logger.error(f"Chunking error: {str(e)}")
        raise HTTPException(500, f"Text processing error: {str(e)}")

def preprocess_text(text: str) -> str:
    """Clean and preprocess text for better chunking"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers and headers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Clean up common PDF artifacts
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', ' ', text)
    
    # Ensure proper sentence endings
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    
    return text.strip()

def create_semantic_chunks(sentences: List[str], chunk_size: int) -> List[str]:
    """Create chunks that respect semantic boundaries"""
    chunks = []
    current_chunk = []
    current_size = 0
    
    for i, sentence in enumerate(sentences):
        sentence_words = len(sentence.split())
        
        # Check if adding this sentence would exceed chunk size
        if current_size + sentence_words > chunk_size and current_chunk:
            # Look ahead to see if we should break here or continue
            if i < len(sentences) - 1:
                next_sentence = sentences[i + 1]
                next_words = len(next_sentence.split())
                
                # If next sentence is short and would fit, continue
                if current_size + sentence_words + next_words <= chunk_size * 1.2:
                    current_chunk.append(sentence)
                    current_size += sentence_words
                    continue
            
            # Create chunk and reset
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_words
        else:
            current_chunk.append(sentence)
            current_size += sentence_words
    
    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def create_fallback_response(question: str, chunks: List[str]) -> str:
    """Create a meaningful response when AI generation fails"""
    if not chunks:
        return "No relevant information found in the document."
    
    # Analyze the question type
    question_lower = question.lower()
    
    # Enhanced fallback with programming examples
    if any(word in question_lower for word in ["c++", "cpp", "programming", "code", "project"]):
        return create_programming_fallback_response(question, chunks)
    
    elif any(word in question_lower for word in ["who", "person", "individual", "someone"]):
        # Look for names, titles, or person references
        person_info = extract_person_info(chunks)
        if person_info:
            return f"Based on the document, here's what I found about the person/individual:\n\n{person_info}"
    
    elif any(word in question_lower for word in ["what", "information", "details", "facts"]):
        # Provide key information from chunks
        key_info = extract_key_information(chunks)
        return f"Here's the relevant information from the document:\n\n{key_info}"
    
    elif any(word in question_lower for word in ["when", "time", "date", "period"]):
        # Look for temporal information
        time_info = extract_temporal_info(chunks)
        if time_info:
            return f"Temporal information found:\n\n{time_info}"
    
    elif any(word in question_lower for word in ["where", "location", "place", "area"]):
        # Look for location information
        location_info = extract_location_info(chunks)
        if location_info:
            return f"Location information found:\n\n{location_info}"
    
    else:
        # Generic response with relevant chunks - more conversational
        relevant_text = "\n\n".join([f"**From your document:** {chunk[:300]}..." for i, chunk in enumerate(chunks[:3])])
        return f"Here's what I found in your document that might help answer your question:\n\n{relevant_text}\n\n**Quick tip:** This is a fallback response while the AI is busy. For a more detailed, personalized answer with examples and explanations, try asking again in a few minutes!"

def create_programming_fallback_response(question: str, chunks: List[str]) -> str:
    """Create a programming-focused fallback response"""
    question_lower = question.lower()
    
    # Extract relevant programming information from chunks
    programming_info = extract_programming_info(chunks)
    
    # Create a natural, conversational response
    response = "Hey! Let me help you with some great C++ project ideas. "
    
    if programming_info:
        response += f"Looking at your document, I can see some relevant programming background: {programming_info.lower()}\n\n"
    else:
        response += "While your document doesn't have specific C++ details, I've got some awesome project ideas for you!\n\n"
    
    response += "**Here are 5 beginner-friendly C++ projects to get you started:**\n\n"
    
    # Add basic C++ project examples with more natural language
    projects = [
        "**Calculator** - Build a simple calculator that handles basic math operations. Great for learning input/output and functions!",
        "**Number Guessing Game** - Create a game where the computer picks a random number and the user tries to guess it. Perfect for practicing loops and conditionals.",
        "**Student Grade Manager** - Build a system to store and calculate student grades. Excellent for learning arrays, file handling, and data structures.",
        "**Bank Account System** - Design a simple banking system with classes and objects. Great introduction to object-oriented programming!",
        "**Library Management** - Create a system to manage books and borrowers. Perfect for practicing data structures and file I/O operations."
    ]
    
    for i, project in enumerate(projects, 1):
        response += f"{i}. {project}\n\n"
    
    response += "**Pro Tips for Getting Started:**\n"
    response += "• Start with console applications - they're perfect for learning the basics\n"
    response += "• Focus on one concept at a time (variables, loops, functions, then classes)\n"
    response += "• Practice with small, manageable pieces before building bigger projects\n"
    response += "• Don't worry about making it perfect - focus on getting it working first!\n\n"
    
    response += "**Ready to dive in?** Pick one project that sounds interesting and start coding! If you need help with specific parts, just ask.\n\n"
    response += "*Note: This is a fallback response while the AI is busy. For more detailed examples and code snippets, try asking again in a few minutes!*"
    
    return response

def extract_programming_info(chunks: List[str]) -> str:
    """Extract programming-related information from chunks"""
    programming_keywords = [
        "c++", "cpp", "programming", "code", "algorithm", "data structure",
        "software", "development", "computer science", "coding", "project"
    ]
    
    relevant_info = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        if any(keyword in chunk_lower for keyword in programming_keywords):
            # Extract sentences containing programming keywords
            sentences = chunk.split('. ')
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in programming_keywords):
                    relevant_info.append(sentence.strip())
    
    if relevant_info:
        return "**Programming-related information found:**\n" + "\n".join([f"• {info}" for info in relevant_info[:3]])
    return ""

def is_programming_question(question: str) -> bool:
    """Check if the question is programming-related"""
    programming_keywords = [
        "c++", "cpp", "programming", "code", "project", "algorithm", "data structure",
        "software", "development", "coding", "program", "function", "class", "object",
        "loop", "array", "string", "pointer", "memory", "file", "database"
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in programming_keywords)

def build_enhanced_programming_prompt(question: str, contexts: List[str]) -> str:
    """Build an enhanced prompt specifically for programming questions"""
    context_text = "\n\n".join(f"### Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts))
    
    prompt = f"""
You are a friendly, expert programming instructor who loves helping people learn to code. You're here to give practical, actionable advice with working code examples.

## Your Style:
- **Be conversational and encouraging** - like a helpful coding mentor
- **Focus on being practical** - give answers people can use immediately
- **Use the document context naturally** - only mention it if it adds real value
- **Write clean, working code** - examples that actually run
- **Write naturally** - don't use formal headers or rigid structure

## Document Context (Optional Reference):
{context_text}

## User Question:
{question}

## Response Approach:

### **Main Answer (95% of response):**
- Give a direct, helpful answer to the programming question
- Include working code examples with explanations
- Provide practical implementation steps
- Make it easy to follow and implement
- Write in a natural, conversational way

### **Context Enhancement (5% of response):**
- If the document has relevant programming info, mention it briefly and naturally
- Connect it seamlessly to your advice
- Don't force it if it's not helpful

### **Code Examples:**
- Provide complete, runnable code
- Include helpful comments
- Show modern best practices
- Make it copy-paste ready

## Response Style:
- Start with a friendly, direct answer
- Include code examples naturally in the flow
- Mention relevant document context only if it adds real value
- End with helpful tips and next steps
- Keep the tone conversational and encouraging

## Important Notes:
- **Be helpful first, RAG second**
- Write like you're teaching a friend to code
- Include complete, working examples
- Focus on practical implementation
- Keep it conversational and engaging
- Don't use formal headers or rigid structure
- Don't over-emphasize the document context
- Make the response flow naturally

Please provide a helpful, practical programming answer in a natural, conversational style:
"""
    return prompt

def extract_person_info(chunks: List[str]) -> str:
    """Extract person-related information from chunks"""
    person_patterns = [
        r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Full names
        r'\b[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+\b',  # Names with initials
        r'\b[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+\b',  # Three-part names
    ]
    
    person_info = []
    for chunk in chunks:
        for pattern in person_patterns:
            matches = re.findall(pattern, chunk)
            for match in matches:
                if match not in person_info:
                    person_info.append(match)
    
    if person_info:
        return f"**People mentioned:** {', '.join(person_info[:5])}"
    return "No specific person names found in the relevant sections."

def extract_key_information(chunks: List[str]) -> str:
    """Extract key information from chunks"""
    key_info = []
    for i, chunk in enumerate(chunks[:3]):  # Limit to first 3 chunks
        # Extract first few sentences as key information
        sentences = chunk.split('. ')
        if sentences:
            key_sentences = sentences[:2]  # First 2 sentences
            key_info.append(f"**Section {i+1}:** {'. '.join(key_sentences)}.")
    
    return '\n\n'.join(key_info)

def extract_temporal_info(chunks: List[str]) -> str:
    """Extract temporal information from chunks"""
    time_patterns = [
        r'\b\d{4}\b',  # Years
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b',
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # Dates
        r'\b\d{1,2}-\d{1,2}-\d{4}\b',  # Dates with dashes
    ]
    
    time_info = []
    for chunk in chunks:
        for pattern in time_patterns:
            matches = re.findall(pattern, chunk, re.IGNORECASE)
            for match in matches:
                if match not in time_info:
                    time_info.append(match)
    
    if time_info:
        return f"**Temporal references:** {', '.join(time_info[:5])}"
    return "No specific temporal information found in the relevant sections."

def extract_location_info(chunks: List[str]) -> str:
    """Extract location information from chunks"""
    location_patterns = [
        r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)* (?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Place|Pl|Court|Ct)\b',
        r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)* (?:City|Town|Village|County|State|Country|Province)\b',
        r'\b[A-Z]{2}\b',  # State abbreviations
    ]
    
    location_info = []
    for chunk in chunks:
        for pattern in location_patterns:
            matches = re.findall(pattern, chunk)
            for match in matches:
                if match not in location_info:
                    location_info.append(match)
    
    if location_info:
        return f"**Location references:** {', '.join(location_info[:5])}"
    return "No specific location information found in the relevant sections."

def analyze_question_type(question: str) -> str:
    """Analyze the type of question being asked"""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ["who", "person", "individual", "someone"]):
        return "PERSON_QUERY"
    elif any(word in question_lower for word in ["what", "information", "details", "facts", "describe", "explain"]):
        return "INFORMATION_QUERY"
    elif any(word in question_lower for word in ["when", "time", "date", "period", "duration"]):
        return "TEMPORAL_QUERY"
    elif any(word in question_lower for word in ["where", "location", "place", "area", "address"]):
        return "LOCATION_QUERY"
    elif any(word in question_lower for word in ["how", "method", "process", "procedure", "steps"]):
        return "PROCESS_QUERY"
    elif any(word in question_lower for word in ["why", "reason", "cause", "purpose", "motivation"]):
        return "REASONING_QUERY"
    elif any(word in question_lower for word in ["calculate", "compute", "total", "sum", "percentage"]):
        return "CALCULATION_QUERY"
    elif any(word in question_lower for word in ["compare", "difference", "similar", "versus", "vs"]):
        return "COMPARISON_QUERY"
    else:
        return "GENERAL_QUERY"

def get_embedding(text: str) -> List[float]:
    try:
        if not text.strip():
            raise ValueError("Empty text provided")

        res = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document",
            title="PDF Content"
        )

        vector = res.get("embedding") or res.get("data")
        if isinstance(vector, dict) and "embedding" in vector:
            vector = vector["embedding"]
        if not isinstance(vector, list):
            raise ValueError("Invalid vector format")
        # optionally warn if len != EMBEDDING_DIMENSION
        if len(vector) != EMBEDDING_DIMENSION:
            logger.warning(f"Embedding dimension mismatch: got {len(vector)} (expected {EMBEDDING_DIMENSION})")
        return vector
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise HTTPException(500, f"Embedding generation failed: {str(e)}")

async def index_chunks(chunks: List[str]):
    try:
        logger.info(f"Background indexing started for {len(chunks)} chunks")

        def process_batch(batch):
            vectors = []
            for i, chunk in enumerate(batch):
                try:
                    vector = get_embedding(chunk)
                    vectors.append({
                        "id": f"chunk-{time.time_ns()}-{i}",
                        "values": vector,
                        "metadata": {"text": truncate_by_bytes(chunk)}
                    })
                except Exception as e:
                    logger.warning(f"Skipping chunk {i}: {str(e)}")
            return vectors

        batch_size = 50
        all_vectors = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vectors = await asyncio.get_event_loop().run_in_executor(
                executor,
                process_batch,
                batch
            )
            all_vectors.extend(vectors)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")

        if all_vectors:
            index.upsert(vectors=all_vectors)
            logger.info(f"Indexed {len(all_vectors)} vectors")
            
            # Wait a moment for index to be ready
            await asyncio.sleep(2)

    except Exception as e:
        logger.error(f"Background indexing failed: {str(e)}")

def is_complex_question(question: str) -> bool:
    complexity_triggers = [
        "calculate", "how much", "total", "combined",
        "over time", "after", "before", "difference",
        "compared to", "step by step", "process"
    ]
    return any(trigger in question.lower() for trigger in complexity_triggers)

def search_similar_chunks(query: str, top_k: int = 7) -> List[str]:
    try:
        # Check index stats first
        try:
            stats = index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            logger.info(f"Index contains {total_vectors} vectors")
            if total_vectors == 0:
                logger.warning("Index is empty - no chunks have been indexed yet")
                return []
        except Exception as e:
            logger.warning(f"Could not get index stats: {str(e)}")
        
        # Enhanced query preprocessing
        enhanced_query = enhance_query_for_search(query)
        logger.info(f"Enhanced query: '{enhanced_query[:100]}...'")
        
        query_vector = get_embedding(enhanced_query)
        logger.info(f"Searching with query: '{query[:100]}...'")
        
        # Try multiple search strategies
        chunks = []
        
        # Strategy 1: Direct semantic search
        try:
            results = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
            chunks.extend(extract_chunks_from_results(results, top_k))
        except Exception as e:
            logger.warning(f"Direct search failed: {str(e)}")
        
        # Strategy 2: If not enough results, try with broader search
        if len(chunks) < top_k // 2:
            try:
                # Use a more general query
                general_query = extract_key_concepts(query)
                general_vector = get_embedding(general_query)
                
                results = index.query(
                    vector=general_vector,
                    top_k=top_k,
                    include_metadata=True,
                    include_values=False
                )
                additional_chunks = extract_chunks_from_results(results, top_k)
                
                # Merge without duplicates
                existing_texts = set(chunk for chunk, _ in chunks)
                for chunk, score in additional_chunks:
                    if chunk not in existing_texts:
                        chunks.append((chunk, score))
                        if len(chunks) >= top_k:
                            break
            except Exception as e:
                logger.warning(f"General search failed: {str(e)}")
        
        # Strategy 3: If still not enough, try fuzzy matching on text
        if len(chunks) < top_k // 2:
            try:
                fuzzy_chunks = fuzzy_text_search(query, top_k)
                chunks.extend(fuzzy_chunks)
            except Exception as e:
                logger.warning(f"Fuzzy search failed: {str(e)}")
        
        # Sort by relevance score and return top chunks
        chunks.sort(key=lambda x: x[1] or 0, reverse=True)
        final_chunks = chunks[:top_k]
        
        logger.info(f"Final chunks found: {len(final_chunks)}")
        return [chunk for chunk, _ in final_chunks]

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        logger.error(f"Search error details: {type(e).__name__}: {str(e)}")
        return []

def extract_chunks_from_results(results, top_k: int) -> List[tuple]:
    """Extract chunks from Pinecone query results"""
    matches = []
    if hasattr(results, 'matches'):
        matches = results.matches
    elif isinstance(results, dict):
        matches = results.get("matches", [])
    else:
        matches = results if isinstance(results, list) else []
    
    chunks_with_scores = []
    for match in matches:
        try:
            if hasattr(match, 'metadata'):
                text = match.metadata.get("text", "")
                score = getattr(match, 'score', None) or getattr(match, 'similarity', None)
            elif isinstance(match, dict):
                text = match.get("metadata", {}).get("text", "")
                score = match.get("score") or match.get("similarity")
            else:
                continue
            
            if text and len(text.strip()) > 10:  # Only add substantial chunks
                chunks_with_scores.append((text, score))
        except Exception as e:
            logger.warning(f"Error processing match: {str(e)}")
            continue
    
    return chunks_with_scores

def enhance_query_for_search(query: str) -> str:
    """Enhance query with synonyms and context"""
    # Add common synonyms and related terms
    synonyms = {
        "who": "person individual someone",
        "what": "information details facts",
        "when": "time date period",
        "where": "location place area",
        "how": "method process procedure",
        "why": "reason cause purpose",
        "this": "document text content",
        "that": "information content",
        "it": "document content information"
    }
    
    enhanced = query.lower()
    for word, synonym in synonyms.items():
        if word in enhanced:
            enhanced += " " + synonym
    
    return enhanced

def extract_key_concepts(query: str) -> str:
    """Extract key concepts from query for broader search"""
    # Remove common question words and focus on key concepts
    question_words = ["who", "what", "when", "where", "how", "why", "is", "are", "was", "were", "do", "does", "did"]
    
    words = query.lower().split()
    key_words = [word for word in words if word not in question_words and len(word) > 2]
    
    return " ".join(key_words) if key_words else query

def fuzzy_text_search(query: str, top_k: int) -> List[tuple]:
    """Fallback to fuzzy text search when semantic search fails"""
    try:
        # Get all chunks from the index
        stats = index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        
        if total_vectors == 0:
            return []
        
        # This is a simplified approach - in production you might want to store text separately
        # For now, we'll return empty list and let the main function handle it
        return []
    except Exception as e:
        logger.warning(f"Fuzzy search not implemented: {str(e)}")
        return []

def build_prompt(question: str, contexts: List[str]) -> str:
    context_text = "\n\n".join(f"### Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts))

    prompt = f"""
You are a helpful, knowledgeable AI assistant with expertise in programming, technology, and many other fields. You're here to provide clear, helpful answers to users' questions.

## Your Approach:
- **Primary Goal**: Give a clean, direct, helpful answer to the user's question
- **RAG Context**: Use the document context as background reference to enhance your response
- **Style**: Be conversational, clear, and practical - like a helpful expert friend
- **Focus**: Answer the user's question first, then enhance with relevant context if helpful

## Document Context (Reference Only):
{context_text}

## User Question:
{question}

## Response Guidelines:

### **Main Answer (80% of response):**
- Give a direct, helpful answer to the user's question
- Be conversational and engaging
- Include practical examples, code snippets, or explanations
- Make it immediately useful and actionable

### **Context Enhancement (20% of response):**
- If the document context adds value, mention it briefly
- Connect it naturally to your main answer
- Don't force it if it's not relevant

### **Response Style:**
- Start with a clear, direct answer
- Use natural language and conversational tone
- Include practical examples and actionable advice
- Be comprehensive but not overwhelming
- Focus on being helpful rather than just informative

## Example Response Structure:
[Direct, helpful answer to the question with practical examples]

[Brief mention of relevant document context if it adds value]

[Additional helpful tips or next steps]

## Important Notes:
- **Be helpful first, RAG second**
- Write like you're talking to a friend who needs help
- Include code examples for programming questions
- Provide practical, actionable advice
- Make the response natural and engaging
- Don't over-emphasize the document context

Please provide a helpful, conversational answer:
"""
    logger.debug(f"Generated clean, Gemini-like prompt: {prompt[:500]}...")
    return prompt

def ask_gemini(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        text = getattr(response, "text", None) or (response.get("text") if isinstance(response, dict) else None)
        if not text:
            raise ValueError("Empty response from Gemini")
        
        # Clean up the response to make it more natural
        cleaned_text = clean_response_format(text.strip())
        return cleaned_text
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Gemini error: {error_msg}")
        
        # Handle specific API errors
        if "429" in error_msg or "quota" in error_msg.lower():
            return """I'm currently experiencing high demand and have hit my API rate limits. 

**What this means:**
- The system is working correctly, but I've exceeded my free tier limits
- This is a temporary limitation, not a system error

**Solutions:**
1. **Wait a few minutes** and try again (rate limits reset periodically)
2. **Upgrade your Gemini API plan** for higher limits
3. **Use the system during off-peak hours**

**Alternative approach:**
You can still use the document search functionality - the PDF has been processed and indexed successfully. The system can find relevant text chunks even when the AI generation is rate-limited."""
        
        elif "timeout" in error_msg.lower():
            return "The request took too long to process. Please try again with a shorter question or wait a moment and retry."
        
        elif "invalid" in error_msg.lower() or "malformed" in error_msg.lower():
            return "There was an issue with the request format. Please try rephrasing your question."
        
        else:
            return f"I encountered an error while processing your request: {error_msg}. Please try again or contact support if the issue persists."

def clean_response_format(text: str) -> str:
    """Clean up response format to make it more natural and conversational"""
    # Remove formal headers and structure markers
    text = re.sub(r'^\s*#+\s*[A-Za-z\s:]+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\*\*[^*]+\*\*:\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[A-Za-z\s]+:\s*$', '', text, flags=re.MULTILINE)
    
    # Remove excessive line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Clean up bullet points to be more natural
    text = re.sub(r'^\s*[-•]\s*', '• ', text, flags=re.MULTILINE)
    
    # Remove formal section separators
    text = re.sub(r'^\s*[-=]{3,}\s*$', '', text, flags=re.MULTILINE)
    
    # Make sure the response flows naturally
    text = text.strip()
    
    return text

# ---------------- API Endpoints (original ones preserved) ----------------

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        logger.info(f"Processing: {file.filename} (size: {getattr(file, 'size', 0)/1024:.1f}KB)")

        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(400, "Only PDF files are allowed")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            text = await extract_text_from_pdf(tmp_path)
            chunks = await chunk_text(text)

            if not chunks:
                raise HTTPException(400, "No valid text chunks created")

            chunks = chunks[:1000] if len(chunks) > 1000 else chunks
            
            # Wait for indexing to complete instead of background processing
            await index_chunks(chunks)

            return {"success": True, "message": f"PDF processed successfully! {len(chunks)} text chunks indexed and ready for questions."}

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(500, f"Processing error: {str(e)}")

@app.post("/api/ask")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"Received question: '{question}'")

        if not question.strip():
            raise HTTPException(400, "Question cannot be empty")

        # Enhanced question analysis
        question_type = analyze_question_type(question)
        top_k = 10 if is_complex_question(question) else 5
        logger.info(f"Question type: {question_type}, Using top_k={top_k} for {'complex' if top_k == 10 else 'simple'} question")

        relevant_chunks = search_similar_chunks(question, top_k=top_k)
        logger.info(f"Found {len(relevant_chunks)} relevant chunks")

        if not relevant_chunks:
            return {"answer": "I couldn't find any relevant information in the document to answer your question. This might happen if:\n\n1. The document hasn't been fully processed yet - try waiting a moment and asking again\n2. Your question is too specific or uses different terminology than what's in the document\n3. The document content doesn't contain information related to your question\n\nTry rephrasing your question or asking something more general about the document content."}

        # Log chunk previews for debugging
        for i, chunk in enumerate(relevant_chunks[:3]):
            logger.info(f"Chunk {i+1} preview: {chunk[:100]}...")

        # Create a fallback response using just the chunks (when AI is rate-limited)
        fallback_response = create_fallback_response(question, relevant_chunks)
        
        # Check if this is a programming question that needs enhanced LLM response
        if is_programming_question(question):
            enhanced_prompt = build_enhanced_programming_prompt(question, relevant_chunks)
            try:
                logger.info(f"Using enhanced programming prompt, length: {len(enhanced_prompt)} characters")
                answer = ask_gemini(enhanced_prompt)
                logger.info(f"Generated enhanced answer length: {len(answer)} characters")
                return {"answer": answer}
            except Exception as e:
                logger.warning(f"Enhanced AI generation failed, using fallback: {str(e)}")
                return {"answer": fallback_response}
        
        try:
            prompt = build_prompt(question, relevant_chunks)
            logger.info(f"Generated prompt length: {len(prompt)} characters")
            answer = ask_gemini(prompt)
            logger.info(f"Generated answer length: {len(answer)} characters")
            return {"answer": answer}
        except Exception as e:
            logger.warning(f"AI generation failed, using fallback: {str(e)}")
            return {"answer": fallback_response}

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Question error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error answering question: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



# ---------------- Run ----------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
