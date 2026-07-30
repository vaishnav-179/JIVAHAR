import unittest
from pathlib import Path
from pypdf import PdfReader

class TestKnowledgeBase(unittest.TestCase):
    """
    Validates that the PDF document files in the knowledge base exist
    and are structurally valid and readable by the pypdf library.
    """
    
    def setUp(self):
        # Locate data/knowledge_base relative to test location
        self.kb_dir = Path(__file__).resolve().parent.parent / "data" / "knowledge_base"
        self.expected_files = [
            "food_safety_manual.pdf",
            "government_regulations.pdf",
            "ngo_policies.pdf",
            "platform_faqs.pdf"
        ]

    def test_pdf_files_exist(self):
        # Verify all expected documents exist
        for filename in self.expected_files:
            filepath = self.kb_dir / filename
            self.assertTrue(filepath.exists(), f"Expected knowledge base PDF file is missing: {filename}")

    def test_pdf_parsing_readable(self):
        # Verify pypdf can read pages and extract text from each file
        for filename in self.expected_files:
            filepath = self.kb_dir / filename
            try:
                reader = PdfReader(str(filepath))
                
                # 1. Page count verification
                self.assertGreater(len(reader.pages), 0, f"PDF '{filename}' is empty (has 0 pages)")
                
                # 2. Text extraction verification
                extracted_text = reader.pages[0].extract_text()
                self.assertIsNotNone(extracted_text, f"Text extraction returned None for '{filename}'")
                
                # Remove whitespace to ensure there's actual content
                cleaned_text = extracted_text.strip()
                self.assertTrue(len(cleaned_text) > 0, f"Extracted text from '{filename}' page 1 is empty")
                
                # Verify that key terms exist inside the file to ensure the content compiled correctly
                if filename == "food_safety_manual.pdf":
                    self.assertIn("Hygiene", cleaned_text)
                    self.assertIn("Temperature Danger Zone", cleaned_text)
                elif filename == "government_regulations.pdf":
                    self.assertIn("Good Samaritan Protection Act", cleaned_text)
                elif filename == "ngo_policies.pdf":
                    self.assertIn("NGO Food Distribution Guidelines", cleaned_text)
                elif filename == "platform_faqs.pdf":
                    self.assertIn("FAQs", cleaned_text)
                    
            except Exception as e:
                self.fail(f"Structure verification failed for PDF '{filename}': {e}")

if __name__ == "__main__":
    unittest.main()
