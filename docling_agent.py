from docling.document_converter import DocumentConverter
from pathlib import Path
import logging
import os
import pdfplumber
import sys
from datetime import datetime
import re
import pandas as pd
from PyPDF2 import PdfReader, PdfWriter
import json
from dotenv import load_dotenv

load_dotenv()

# Create output directory
OUTPUT_DIR = Path("markdown_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Create logs directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

def setup_logger(name: str = __name__) -> logging.Logger:
    """Setup logger with both file and console handlers"""
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console Handler (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # File Handler (DEBUG level)
    log_file = LOGS_DIR / f"docling_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logger()

# NEW FUNCTION AND MAIN
def extract_page_with_keywords(pdf_path: str, keywords: list, output_pdf: str = "temp.pdf"):
    """
    Extract the page containing all specified keywords and save it as a new PDF.

    Args:
        pdf_path (str): Path to the original PDF file.
        keywords (list): List of keywords to search for.
        output_pdf (str): Output path for the new PDF containing the matching page.
    """
    try:
        logger.info(f"Opening PDF: {pdf_path}")
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if all(keyword.lower() in text.lower() for keyword in keywords):
                    logger.info(f"Found matching page: {i}")
                    from PyPDF2 import PdfReader, PdfWriter

                    reader = PdfReader(pdf_path)
                    writer = PdfWriter()
                    writer.add_page(reader.pages[i - 1])

                    with open(output_pdf, "wb") as f_out:
                        writer.write(f_out)

                    logger.info(f"Saved matching page to: {output_pdf}")
                    print(f"✅ Matching page extracted to {output_pdf}")
                    return

            logger.warning("No page matched all keywords.")
            print("❌ No page matched all keywords.")

    except Exception as e:
        logger.error(f"Failed to extract page: {e}", exc_info=True)
        print(f"❌ Error: {e}")

def extract_tables_with_camelot(pdf_path: str, output_md: str = "tables_output.txt"):
    """
    Extract tables from a PDF using Camelot and save them in markdown format.
    """
    try:
        import camelot
    except ImportError:
        print("❌ Camelot is not installed. Install with: pip install camelot-py[cv]")
        return

    logger.info(f"Extracting tables from: {pdf_path}")
    markdown_output = ""
    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        if not tables:
            print("❌ No tables found in the PDF.")
            return

        for i, table in enumerate(tables, start=1):
            df = table.df
            if df.shape[0] > 1:
                df.columns = df.iloc[0]
                df = df[1:]
            md = df.to_markdown(index=False)
            markdown_output += f"\n### Table {i}:\n{md}\n"

        with open(output_md, "w", encoding="utf-8") as f:
            f.write(markdown_output)
        logger.info(f"Tables saved to markdown file: {output_md}")

    except Exception as e:
        logger.error(f"Error extracting tables: {e}", exc_info=True)
        print(f"❌ Failed to extract tables: {e}")

def generate_structured_output_from_gemini(markdown_file_path: str, company_name: str, quarter_year: str) -> dict:
    """
    Send the markdown table content to Gemini with a predefined TypedDict schema and return structured output.

    Args:
        markdown_file_path (str): Path to the markdown file containing the financial table.
        company_name (str): Name of the company.
        quarter_year (str): Quarter and year, e.g., '4QFY25'.

    Returns:
        dict: Parsed structured financial summary as JSON.
    """
    from typing import TypedDict
    from google import genai
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    class QuarterResults(TypedDict):
        revenue: float
        cost_of_materials_consumed: float
        purchase_of_stock_in_trade: float
        changes_in_inventories: float
        profit_before_tax: float
        finance_costs: float
        depreciation_and_amortisation: float
        total_tax_expenses: float
        other_income: float

    class FinancialData(TypedDict):
        unit_of_measure: str
        currency: str
        _31_03_25_quarter_results: QuarterResults
        _31_03_24_quarter_results: QuarterResults

    class OutputSchema(TypedDict):
        company_name: str
        quarter_year: str
        data: FinancialData

    # Load markdown table content
    with open(markdown_file_path, "r", encoding="utf-8") as f:
        markdown_table = f.read()

    prompt = f"""
    Extract financial data from the table below. 
    Ensure the JSON keys match the schema exactly including date suffixes like '31.03.25_quarter_results'.

    Company: {company_name}
    Quarter: {quarter_year}

    Markdown Table:
    {markdown_table}
    """

    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": OutputSchema
        }
    )

    # Store response data in output file
    output_file = OUTPUT_DIR / "gemini_output.json"
    response_data = json.loads(response.text)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(response_data, f, indent=2)
    
    return response_data


def main():
    
    pdf_path = "dhanuka.pdf"
    keywords = ["Revenue from Operations", "31.03.2025", "31.03.2024", "31.12.2024"]
    extract_page_with_keywords(pdf_path, keywords)
    extract_tables_with_camelot("temp.pdf", "markdown_outputs/temp_tables.txt")
    result = generate_structured_output_from_gemini("markdown_outputs/temp_tables.txt", "Emami", "4QFY25")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()