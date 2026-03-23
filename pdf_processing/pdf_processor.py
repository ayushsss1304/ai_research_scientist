"""
PDF Processing Module
Handles text extraction from PDFs with OCR support
Complete implementation for the AI Research Scientist system
"""

import PyPDF2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import io
import os
from typing import List, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Process PDFs with OCR support"""
    
    def __init__(self, use_ocr: bool = True):
        """
        Initialize PDF processor
        
        Args:
            use_ocr: Whether to use OCR for scanned documents
        """
        self.use_ocr = use_ocr
    
    def extract_text_from_pdf(self, pdf_path: str, use_ocr: Optional[bool] = None) -> str:
        """
        Extract text from PDF with OCR fallback
        
        Args:
            pdf_path: Path to PDF file
            use_ocr: Whether to use OCR (overrides instance setting)
        
        Returns:
            Extracted text as string
        """
        if use_ocr is None:
            use_ocr = self.use_ocr
        
        text = ""
        
        try:
            # Try regular text extraction first
            logger.info(f"Extracting text from {pdf_path}")
            text = self._extract_text_native(pdf_path)
            
            # If no text extracted or very little text, use OCR
            if use_ocr and len(text.strip()) < 100:
                logger.info("Native extraction yielded little text, trying OCR...")
                text = self._extract_text_with_ocr(pdf_path)
        
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            if use_ocr:
                logger.info("Attempting OCR as fallback...")
                text = self._extract_text_with_ocr(pdf_path)
        
        logger.info(f"Extracted {len(text)} characters from {pdf_path}")
        return text
    
    def _extract_text_native(self, pdf_path: str) -> str:
        """
        Extract text using PyPDF2
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        text = ""
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                logger.info(f"PDF has {len(pdf_reader.pages)} pages")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {e}")
        
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
        
        return text
    
    def _extract_text_with_ocr(self, pdf_path: str) -> str:
        """
        Extract text using OCR (Tesseract)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        text = ""
        
        try:
            # Convert PDF to images
            logger.info("Converting PDF to images for OCR...")
            images = convert_from_path(pdf_path, dpi=300)
            
            logger.info(f"Processing {len(images)} pages with OCR")
            
            # OCR each page
            for i, image in enumerate(images):
                logger.info(f"OCR processing page {i+1}/{len(images)}")
                page_text = pytesseract.image_to_string(image, lang='eng')
                text += page_text + "\n\n"
        
        except Exception as e:
            logger.error(f"OCR error: {e}")
            logger.error("Make sure Tesseract and Poppler are installed")
            logger.error("Ubuntu/Debian: sudo apt-get install tesseract-ocr poppler-utils")
            logger.error("macOS: brew install tesseract poppler")
        
        return text
    
    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from image using OCR
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text
        """
        try:
            logger.info(f"Extracting text from image: {image_path}")
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang='eng')
            logger.info(f"Extracted {len(text)} characters from image")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, 
                   overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks
        
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to end at a sentence boundary
            if end < len(text):
                # Look for sentence ending in the last 100 characters
                last_period = text[max(0, end-100):end].rfind('.')
                last_newline = text[max(0, end-100):end].rfind('\n')
                
                boundary = max(last_period, last_newline)
                if boundary != -1:
                    end = max(0, end - 100) + boundary + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start += chunk_size - overlap
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def get_pdf_metadata(self, pdf_path: str) -> dict:
        """
        Extract metadata from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with metadata
        """
        metadata = {}
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                info = pdf_reader.metadata
                
                metadata = {
                    'num_pages': len(pdf_reader.pages),
                    'author': info.get('/Author', '') if info else '',
                    'title': info.get('/Title', '') if info else '',
                    'subject': info.get('/Subject', '') if info else '',
                    'creator': info.get('/Creator', '') if info else '',
                    'producer': info.get('/Producer', '') if info else '',
                    'creation_date': str(info.get('/CreationDate', '')) if info else '',
                }
        
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
        
        return metadata
    
    def is_scanned_pdf(self, pdf_path: str, sample_pages: int = 3) -> bool:
        """
        Determine if PDF is scanned (image-based) or native text
        
        Args:
            pdf_path: Path to PDF file
            sample_pages: Number of pages to sample
        
        Returns:
            True if PDF appears to be scanned
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                total_pages = len(pdf_reader.pages)
                pages_to_check = min(sample_pages, total_pages)
                
                total_text_length = 0
                
                for i in range(pages_to_check):
                    page = pdf_reader.pages[i]
                    text = page.extract_text()
                    total_text_length += len(text.strip())
                
                # If average text per page is less than 100 chars, likely scanned
                avg_text_per_page = total_text_length / pages_to_check
                
                is_scanned = avg_text_per_page < 100
                
                logger.info(f"PDF analysis: {'Scanned' if is_scanned else 'Native text'} "
                          f"(avg {avg_text_per_page:.0f} chars/page)")
                
                return is_scanned
        
        except Exception as e:
            logger.error(f"Error checking if PDF is scanned: {e}")
            return False
    
    def process_document(self, file_path: str) -> dict:
        """
        Process a document and return structured data
        
        Args:
            file_path: Path to document
        
        Returns:
            Dictionary with text, chunks, and metadata
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        result = {
            'file_path': file_path,
            'file_type': ext,
            'text': '',
            'chunks': [],
            'metadata': {},
            'error': None
        }
        
        try:
            if ext == '.pdf':
                result['text'] = self.extract_text_from_pdf(file_path)
                result['metadata'] = self.get_pdf_metadata(file_path)
                result['metadata']['is_scanned'] = self.is_scanned_pdf(file_path)
            
            elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
                result['text'] = self.extract_text_from_image(file_path)
            
            elif ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    result['text'] = f.read()
            
            else:
                result['error'] = f"Unsupported file type: {ext}"
                logger.error(result['error'])
                return result
            
            # Create chunks
            result['chunks'] = self.chunk_text(result['text'])
            
            logger.info(f"Document processed successfully: {file_path}")
            logger.info(f"  - Text length: {len(result['text'])} characters")
            logger.info(f"  - Chunks: {len(result['chunks'])}")
        
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error processing document: {e}")
        
        return result
    
    def extract_pages(self, pdf_path: str, page_numbers: List[int]) -> str:
        """
        Extract text from specific pages
        
        Args:
            pdf_path: Path to PDF file
            page_numbers: List of page numbers (0-indexed)
            
        Returns:
            Extracted text from specified pages
        """
        text = ""
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in page_numbers:
                    if 0 <= page_num < len(pdf_reader.pages):
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
                    else:
                        logger.warning(f"Page {page_num} out of range")
        
        except Exception as e:
            logger.error(f"Error extracting pages: {e}")
        
        return text
    
    def get_page_count(self, pdf_path: str) -> int:
        """
        Get number of pages in PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Number of pages
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                return len(pdf_reader.pages)
        except Exception as e:
            logger.error(f"Error getting page count: {e}")
            return 0


class OCRProcessor:
    """Specialized OCR processor for images and scanned documents"""
    
    def __init__(self, language: str = 'eng'):
        """
        Initialize OCR processor
        
        Args:
            language: Tesseract language code
        """
        self.language = language
    
    def process_image(self, image_path: str) -> str:
        """
        Process single image with OCR
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text
        """
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang=self.language)
            return text
        except Exception as e:
            logger.error(f"OCR error on image: {e}")
            return ""
    
    def process_pdf_with_ocr(self, pdf_path: str, dpi: int = 300) -> str:
        """
        Process entire PDF with OCR
        
        Args:
            pdf_path: Path to PDF file
            dpi: DPI for image conversion
            
        Returns:
            Extracted text
        """
        text = ""
        
        try:
            images = convert_from_path(pdf_path, dpi=dpi)
            
            for i, image in enumerate(images):
                logger.info(f"OCR processing page {i+1}/{len(images)}")
                page_text = pytesseract.image_to_string(image, lang=self.language)
                text += f"\n\n--- Page {i+1} ---\n\n{page_text}"
        
        except Exception as e:
            logger.error(f"OCR error on PDF: {e}")
        
        return text
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        image = image.convert('L')
        
        # Can add more preprocessing here:
        # - Increase contrast
        # - Denoise
        # - Binarize
        
        return image