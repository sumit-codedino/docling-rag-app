import os
from docling.document_converter import DocumentConverter

def document_processor(pdf_dir, md_dir):
	if not os.path.exists(md_dir):
		os.makedirs(md_dir)

	pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
	for pdf_file in pdf_files:
		pdf_path = os.path.join(pdf_dir, pdf_file)
		md_path = os.path.join(md_dir, f"{os.path.splitext(pdf_file)[0]}.md")

		if not os.path.exists(md_path):
			print(f"Converting `{pdf_file}` to Markdown ...")

			doc_converter = DocumentConverter()
			result = doc_converter.convert(source=pdf_path)
			
			with open(md_path, 'w', encoding='utf-8') as md_file:
				md_file.write(result.document.export_to_markdown())