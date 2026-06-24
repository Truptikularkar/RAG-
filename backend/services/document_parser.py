import os
from pypdf import PdfReader

class DocumentParser:
    @staticmethod
    def parse_file(file_path: str) -> str:
        """Parses a text, markdown, or PDF file and returns its plaintext content."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".pdf":
            return DocumentParser._parse_pdf(file_path)
        elif ext in [".txt", ".md", ".json", ".py", ".csv", ".html"]:
            return DocumentParser._parse_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    @staticmethod
    def _parse_pdf(file_path: str) -> str:
        text = []
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        except Exception as e:
            raise RuntimeError(f"Error parsing PDF file: {str(e)}")
            
        return "\n\n".join(text)

    @staticmethod
    def _parse_text(file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Error reading text file: {str(e)}")
