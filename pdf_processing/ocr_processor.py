"""
OCR Processor for handling scanned documents and images
Uses Tesseract OCR for text extraction
"""

import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import os
from typing import List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRProcessor:
    """Specialized OCR processor for images and scanned documents"""
    
    def __init__(self, language: str = 'eng', dpi: int = 300):
        """
        Initialize OCR processor
        
        Args:
            language: Tesseract language code (default: 'eng')
            dpi: DPI for PDF to image conversion (default: 300)
        """
        self.language = language
        self.dpi = dpi
        self._verify_tesseract()
    
    def _verify_tesseract(self):
        """Verify Tesseract is installed and accessible"""
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR is available")
        except Exception as e:
            logger.warning(f"Tesseract OCR not found: {e}")
            logger.warning("OCR functionality will not be available")
    
    def process_image(self, image_path: str, preprocess: bool = True) -> str:
        """
        Process single image with OCR
        
        Args:
            image_path: Path to image file
            preprocess: Whether to preprocess image for better OCR
            
        Returns:
            Extracted text
        """
        try:
            logger.info(f"Processing image: {image_path}")
            image = Image.open(image_path)
            
            if preprocess:
                image = self.preprocess_image(image)
            
            text = pytesseract.image_to_string(image, lang=self.language)
            logger.info(f"Extracted {len(text)} characters from image")
            
            return text
        
        except Exception as e:
            logger.error(f"OCR error on image {image_path}: {e}")
            return ""
    
    def process_pdf_with_ocr(self, pdf_path: str, 
                             page_range: Optional[Tuple[int, int]] = None) -> str:
        """
        Process entire PDF with OCR
        
        Args:
            pdf_path: Path to PDF file
            page_range: Optional tuple of (start_page, end_page) for partial processing
            
        Returns:
            Extracted text from all pages
        """
        text = ""
        
        try:
            logger.info(f"Converting PDF to images: {pdf_path}")
            
            # Convert PDF to images
            if page_range:
                first_page, last_page = page_range
                images = convert_from_path(
                    pdf_path, 
                    dpi=self.dpi,
                    first_page=first_page,
                    last_page=last_page
                )
            else:
                images = convert_from_path(pdf_path, dpi=self.dpi)
            
            logger.info(f"Processing {len(images)} pages with OCR")
            
            # OCR each page
            for i, image in enumerate(images, 1):
                logger.info(f"OCR processing page {i}/{len(images)}")
                
                # Preprocess image for better OCR
                preprocessed = self.preprocess_image(image)
                
                # Extract text
                page_text = pytesseract.image_to_string(
                    preprocessed, 
                    lang=self.language
                )
                
                text += f"\n\n--- Page {i} ---\n\n{page_text}"
            
            logger.info(f"OCR complete: extracted {len(text)} characters")
        
        except Exception as e:
            logger.error(f"OCR error on PDF {pdf_path}: {e}")
            logger.error("Make sure Poppler is installed for PDF conversion")
        
        return text
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed image
        """
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Increase contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Increase sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            # Denoise
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Binarization (convert to black and white)
            threshold = 128
            image = image.point(lambda p: 255 if p > threshold else 0)
            
            return image
        
        except Exception as e:
            logger.warning(f"Image preprocessing error: {e}")
            return image
    
    def extract_text_from_images(self, image_paths: List[str]) -> str:
        """
        Extract text from multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Combined extracted text
        """
        text = ""
        
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"Processing image {i}/{len(image_paths)}")
            page_text = self.process_image(image_path)
            text += f"\n\n--- Image {i} ---\n\n{page_text}"
        
        return text
    
    def get_ocr_confidence(self, image: Image.Image) -> float:
        """
        Get OCR confidence score for an image
        
        Args:
            image: PIL Image object
            
        Returns:
            Confidence score (0-100)
        """
        try:
            data = pytesseract.image_to_data(
                image, 
                lang=self.language, 
                output_type=pytesseract.Output.DICT
            )
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            
            if confidences:
                return sum(confidences) / len(confidences)
            
            return 0.0
        
        except Exception as e:
            logger.error(f"Error calculating OCR confidence: {e}")
            return 0.0
    
    def save_preprocessed_image(self, image_path: str, output_path: str):
        """
        Save preprocessed version of image
        
        Args:
            image_path: Input image path
            output_path: Output image path
        """
        try:
            image = Image.open(image_path)
            preprocessed = self.preprocess_image(image)
            preprocessed.save(output_path)
            logger.info(f"Saved preprocessed image to {output_path}")
        
        except Exception as e:
            logger.error(f"Error saving preprocessed image: {e}")
    
    def batch_process_images(self, image_dir: str, output_file: str):
        """
        Process all images in a directory and save to text file
        
        Args:
            image_dir: Directory containing images
            output_file: Output text file path
        """
        # Get all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}
        image_files = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
        
        logger.info(f"Found {len(image_files)} images in {image_dir}")
        
        # Extract text from all images
        text = self.extract_text_from_images(image_files)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logger.info(f"Saved extracted text to {output_file}")


class DocumentScanner:
    """Helper class for scanning document images"""
    
    def __init__(self, ocr_processor: Optional[OCRProcessor] = None):
        """
        Initialize document scanner
        
        Args:
            ocr_processor: OCRProcessor instance (creates new if None)
        """
        self.ocr = ocr_processor or OCRProcessor()
    
    def scan_document(self, image_path: str) -> Dict:
        """
        Scan a document image and return structured data
        
        Args:
            image_path: Path to document image
            
        Returns:
            Dictionary with text, confidence, and metadata
        """
        image = Image.open(image_path)
        preprocessed = self.ocr.preprocess_image(image)
        
        text = pytesseract.image_to_string(preprocessed, lang=self.ocr.language)
        confidence = self.ocr.get_ocr_confidence(preprocessed)
        
        return {
            'text': text,
            'confidence': confidence,
            'image_path': image_path,
            'image_size': image.size,
            'character_count': len(text),
            'word_count': len(text.split())
        }
    
    def detect_orientation(self, image_path: str) -> int:
        """
        Detect document orientation
        
        Args:
            image_path: Path to image
            
        Returns:
            Rotation angle (0, 90, 180, 270)
        """
        try:
            image = Image.open(image_path)
            osd = pytesseract.image_to_osd(image)
            
            # Parse rotation from OSD output
            for line in osd.split('\n'):
                if 'Rotate:' in line:
                    angle = int(line.split(':')[1].strip())
                    return angle
            
            return 0
        
        except Exception as e:
            logger.error(f"Error detecting orientation: {e}")
            return 0
    
    def auto_rotate(self, image_path: str, output_path: str):
        """
        Automatically rotate image to correct orientation
        
        Args:
            image_path: Input image path
            output_path: Output image path
        """
        angle = self.detect_orientation(image_path)
        
        if angle != 0:
            image = Image.open(image_path)
            rotated = image.rotate(angle, expand=True)
            rotated.save(output_path)
            logger.info(f"Rotated image by {angle} degrees")
        else:
            logger.info("No rotation needed")