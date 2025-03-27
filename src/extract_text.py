'''import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path, output_folder):
    """Extract text from a PDF file and save it as a text file."""
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text("text") + "\n"

    # Save extracted text
    output_file = os.path.join(output_folder, os.path.basename(pdf_path).replace(".pdf", ".txt"))
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Text extracted and saved at: {output_file}")
    return output_file

if __name__ == "__main__":
    #pdf_path = "data/resume.pdf"  # Path to input PDF
    pdf_path = "data/Test.pdf"  # Path to input PDF
    output_folder = "processed_text/"
    os.makedirs(output_folder, exist_ok=True)
    
    extract_text_from_pdf(pdf_path, output_folder)'''


import fitz  # PyMuPDF
import os
import re

'''def clean_text(text):
    """Remove bullets and unwanted symbols from extracted text."""
    text = re.sub(r"•|◦|●|o|▪|‣|‿|⁃|–|—|→|⇒|›|»|☛|✔|▶|▷|☑|☐|□|○|◘|◙|♦|◊|★|☆", "", text)  # Remove bullets
    # text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces/newlines
    text = re.sub(r"\s+", " ", text)
    return text'''
def clean_text(text):
    """Remove bullets and unwanted symbols from extracted text while preserving line structure."""
    text = re.sub(r"•|◦|●|▪|‣|‿|⁃|–|—|→|⇒|›|»|☛|✔|▶|▷|☑|☐|□|○|◘|◙|♦|◊|★|☆", "", text)  # Remove bullets
    lines = text.split("\n")  # Split text into lines
    cleaned_lines = [line for line in lines if line.strip()]  # Remove empty lines but keep structure
    return "\n".join(cleaned_lines)

def extract_text_from_pdf(pdf_path, output_folder):
    """Extract and clean text from a PDF file, then save it as a text file."""
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        raw_text = page.get_text("text")
        cleaned_text = clean_text(raw_text)
        text += cleaned_text + "\n"

    # Save extracted text
    output_file = os.path.join(output_folder, os.path.basename(pdf_path).replace(".pdf", ".txt"))
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"✅ Text extracted and cleaned. Saved at: {output_file}")
    return output_file

if __name__ == "__main__":
    pdf_path = "data/Test.pdf"  # Path to input PDF
    output_folder = "processed_text/"
    os.makedirs(output_folder, exist_ok=True)
    
    extract_text_from_pdf(pdf_path, output_folder)

