import pymupdf


class PageTextExtractor:

    def extract(self, pdf_path):

        document = pymupdf.open(pdf_path)

        pages = []

        for page_number, page in enumerate(document, start=1):

            text = page.get_text().strip()

            pages.append({
                "page_number": page_number,
                "text": text
            })

        document.close()

        return pages


if __name__ == "__main__":

    extractor = PageTextExtractor()

    pages = extractor.extract("C:\\Users\\Tanu Bhardwaj\\Downloads\\Intelligent_QA_V3\\document\\Transformer.pdf")

    for page in pages[:5]:
        print("\n====================")
        print("PAGE:", page["page_number"])
        print(page["text"][:500])