import io
import re
from typing import List

from fastapi import APIRouter, File, UploadFile
import fitz  # PyMuPDF
import docx

router = APIRouter(
    prefix="/api/text-extraction",
    tags=["Text Extraction"],
)


def clean_extracted_text(text: str) -> str:
    """
    Safety-net normalizer for extracted text. Even with PyMuPDF
    (which preserves reading order far better than PyPDF2), some
    PDFs still leave irregular runs of spaces/tabs between words
    or whitespace-only lines around page boundaries. This normalizes:
      - collapses runs of spaces/tabs into a single space
      - treats whitespace-only lines as blank lines
      - collapses 3+ blank lines into a single blank line
    """
    if not text:
        return text

    text = re.sub(r"[ \t]+", " ", text)

    # only spaces (e.g. leftover from a page break) becomes truly empty
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    # Collapse 3+ consecutive newlines into just 2 (one blank line)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

@router.post("/extract")
async def extract_text(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        filename = file.filename
        content = await file.read()
        raw_text = ""

        content_type = file.content_type or ""

        try:
            if filename.lower().endswith(".pdf") or content_type == "application/pdf":
                pdf_doc = fitz.open(stream=content, filetype="pdf")
                text_pages = []
                for page in pdf_doc:
                 
                    page_text = page.get_text(sort=True)
                    if page_text:
                        text_pages.append(page_text)
                pdf_doc.close()
                raw_text = "\n".join(text_pages)

            elif filename.lower().endswith(".docx") or "wordprocessingml" in content_type:
                doc = docx.Document(io.BytesIO(content))
                text_paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                raw_text = "\n".join(text_paragraphs)
                
            else:
                raw_text = "Unsupported file format. Please upload PDF or DOCX."

            # إضافة النتيجة للقائمة
            results.append({
                "raw_text": clean_extracted_text(raw_text),
                "original_file_name": filename
            })

        except Exception as e:
            # في حالة وجود ملف تالف أو مشفر
            results.append({
                "raw_text": f"Error extracting text: {str(e)}",
                "original_file_name": filename
            })
    
    # إرجاع البيانات بنفس الهيكل الذي يتوقعه تطبيق فلاتر
    return {"files": results}