import PyPDF2
import sys

pdf_path = r"c:\Users\budhi\Documents\Budhil\Programming\Projects\AI For Bharat Hackathon\Idea Submission _ AWS AI for Bharat Hackathon(1) (2).pdf"

try:
    reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    print(text)
except Exception as e:
    print(f"Error reading PDF: {e}", file=sys.stderr)
    sys.exit(1)
