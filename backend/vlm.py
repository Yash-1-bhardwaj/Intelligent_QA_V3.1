import base64
from urllib import response
import requests


class VLM:

    def __init__(self, model="qwen2.5vl:7b"):
        self.model = model
        self.url = "http://localhost:11434/api/chat"


    def answer(self, question, image_paths, page_numbers):

        images = []

        for image_path in image_paths:

            with open(image_path, "rb") as image_file:

                image_base64 = base64.b64encode(
                    image_file.read()
                ).decode("utf-8")

                images.append(image_base64)


        page_list = ", ".join(
            str(page) for page in page_numbers
        )


        prompt = f"""
You are answering a question about a PDF document.

Question:
{question}

The following PDF pages were retrieved because they are
visually relevant to the question:

Pages: {page_list}

Carefully examine the provided page images.

Answer the question using ONLY information visible
in the provided pages.

If the answer is not present in the pages, say:
"I could not find the answer in the retrieved pages."

Mention the relevant PDF page number in your answer.

Do not invent information.
"""


        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": images
                    }
                ],
                "stream": False
            }
        )


        if not response.ok:
            print("\nOLLAMA ERROR:")
            print(response.text)
            raise RuntimeError("Ollama API request failed")

        return response.json()["message"]["content"]
    
    def describe_page(self, question, image_path):
        with open(image_path, "rb") as image_file:
            image = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

        prompt = f"""
You are analyzing a PDF page for a question-answering system.

Question:
{question}

Carefully examine the page image.

Extract ONLY information from the page that could help answer
the question.

Focus on:
- relevant concepts
- important text
- diagrams
- equations
- relationships
- processes
- solutions or explanations

If the page does not contain useful information for the question,
say:

NOT_RELEVANT

Do not invent information.
Do not answer the question.
Return only the relevant evidence/description.
"""

        response = requests.post(
        self.url,
        json={
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image]
                }
            ],
            "stream": False
        }
    )

        if not response.ok:
            print("\nOLLAMA ERROR:")
            print(response.text)
            raise RuntimeError(
            "Ollama API request failed"
        )

        return response.json()["message"]["content"]
    
    def select_best_page(self, question, image_paths, page_numbers):
        images = []

        for image_path in image_paths:
            with open(image_path, "rb") as image_file:
                images.append(
                base64.b64encode(image_file.read()).decode("utf-8")
            )

        page_labels = ", ".join(str(p) for p in page_numbers)

        prompt = f"""
You are selecting evidence pages from a PDF for question answering.

Question:
{question}

The images are provided in this exact order:
{page_labels}

Choose the page that contains the most direct evidence
needed to answer the question.

Rules:
- Examine the actual text, diagrams, headings, and equations.
- Prefer a page that explicitly explains the requested concept.
- Do not choose a page merely because it discusses a related topic.
- If multiple pages are equally necessary, choose the one with
  the clearest direct explanation.
- Only select from the provided page numbers.

Return ONLY the selected page number.
Do not include explanations or other text.
"""

        response = requests.post(
        self.url,
        json={
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": images
                }
            ],
            "stream": False
        },
        timeout=600
    )

        response.raise_for_status()

        return response.json()["message"]["content"].strip()

    def answer_from_text(self, question, page_texts, page_numbers):
        page_evidence = "\n\n".join(
        f"PDF Page {page_number}:\n{text}"
        for page_number, text in zip(page_numbers, page_texts)
    )

        prompt = f"""
You are an academic document question-answering assistant.

USER QUESTION:
{question}

EVIDENCE EXTRACTED FROM THE PDF:
{page_evidence}

Instructions:
1. Answer the exact question directly.
2. Write a clear, complete explanation, not just a phrase.
3. Explain the key concept and how it works.
4. Use numbered steps or bullet points when helpful.
5. Include relevant examples, formulas, or terminology
   only when supported by the supplied evidence.
6. Do not repeat the question unnecessarily.
7. Do not invent facts or claim the PDF says something
   that is absent from the evidence.
8. If the evidence is insufficient, explicitly say so.
9. Mention the relevant PDF page numbers.

Write a useful answer for a student preparing for an exam.
"""

        response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen2.5:7b",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        },
            timeout=600
        )

        if not response.ok:
            print("LLM ERROR:", response.text)
            raise RuntimeError("LLM answer generation failed")

        return response.json()["message"]["content"].strip()
