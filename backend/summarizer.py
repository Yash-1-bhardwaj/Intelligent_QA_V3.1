
import requests
from page_text_extractor import PageTextExtractor


class DocumentSummarizer:

    def __init__(self, model="qwen2.5:7b"):
        self.model = model
        self.extractor = PageTextExtractor()
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt):
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2}
            },
            timeout=600
        )

        response.raise_for_status()
        return response.json()["response"].strip()

    def summarize(self, pdf_path):
        pages = self.extractor.extract(str(pdf_path))

        # Keep page numbers attached to the extracted content.
        page_texts = [
            f"Page {page['page_number']}:\n{page['text'].strip()}"
            for page in pages
            if page["text"].strip()
        ]

        if not page_texts:
            return (
                "No extractable text was found in this PDF. "
                "It may contain scanned pages or images."
            )

        # Build chunks without splitting individual page text
        # unnecessarily. Limit each chunk to about 6000 characters.
        chunks = []
        current_chunk = []
        current_length = 0

        for page_text in page_texts:
            if len(page_text) > 6000:
                # Flush the existing chunk before splitting a long page.
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                chunks.extend(
                    page_text[i:i + 6000]
                    for i in range(0, len(page_text), 6000)
                )
                continue

            if current_chunk and current_length + len(page_text) > 6000:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(page_text)
            current_length += len(page_text)

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        # First pass: summarize each chunk.
        chunk_summaries = []

        for index, chunk in enumerate(chunks, start=1):
            prompt = f"""
Summarize this section of a PDF for a student.

Section {index} of {len(chunks)}:
{chunk}

Instructions:
- Preserve the source's important concepts and terminology.
- Include important definitions, formulas, methods, and examples.
- Keep the summary concise but informative.
- Do not invent details absent from the section.
- Preserve page references when useful.
- Return only the section summary.
"""
            chunk_summaries.append(self.generate(prompt))

        # Second pass: combine section summaries.
        combined = "\n\n".join(
            f"Section {i}:\n{summary}"
            for i, summary in enumerate(chunk_summaries, start=1)
        )

        final_prompt = f"""
Create a structured study summary of the PDF using
ONLY the section summaries below.

{combined}

Organize the result into:
1. Document overview
2. Main topics and concepts
3. Important definitions and formulas
4. Key examples or methods
5. Final takeaways

Instructions:
- Combine related topics logically.
- Remove repetition without dropping important ideas.
- Do not invent missing information.
- Preserve relevant page references from the source.
- If a detail is unavailable in the section summaries,
  do not make it up.
"""

        return self.generate(final_prompt)
