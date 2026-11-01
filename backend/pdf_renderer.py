import pymupdf
from pathlib import Path


def render_pdf(pdf_path, output_dir="page_images", dpi=150):

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open(pdf_path)

    for page_number, page in enumerate(document, start=1):

        pixmap = page.get_pixmap(dpi=dpi)

        output_path = output_dir / f"page_{page_number:03d}.png"

        pixmap.save(output_path)

        print(f"Rendered page {page_number}: {output_path}")

    document.close()


if __name__ == "__main__":
    render_pdf("C:\\Users\\Tanu Bhardwaj\\Downloads\\Intelligent_QA_V3\\document\\Transformer.pdf")