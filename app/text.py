import io
from pathlib import Path

try:
    import pypdf
except ImportError:
    pypdf = None


def extract_text(filename: str, content: bytes) -> str:
    """Extract text from uploaded file content.
    
    Supports: PDF, Markdown, and plain text files.
    """
    extension = Path(filename).suffix.lower()
    
    if extension == ".pdf":
        if pypdf is None:
            raise ValueError("pypdf is required for PDF support")
        try:
            reader = pypdf.PdfReader(io.BytesIO(content))
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {e}") from e
    
    elif extension in (".md", ".markdown", ".txt"):
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode text file: {e}") from e
    
    else:
        raise ValueError(f"Unsupported file type: {extension}")


def split_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    if not text or not text.strip():
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        
        if chunk:
            chunks.append(chunk)
        
        # Move start position, accounting for overlap
        start = end - chunk_overlap
        if start <= 0 or end >= len(text):
            break
    
    return chunks
