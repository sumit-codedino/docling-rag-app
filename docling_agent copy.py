from docling.document_converter import DocumentConverter
from pathlib import Path
import logging
import os
import pdfplumber
import sys
from datetime import datetime
import re
import pandas as pd

# Try to import camelot (optional dependency)
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    print("camelot not available. Install with: pip install camelot-py[cv]")

# Try to import img2table (optional dependency)
try:
    from img2table.document import PDF
    from img2table.ocr import TesseractOCR
    IMG2TABLE_AVAILABLE = True
except ImportError:
    IMG2TABLE_AVAILABLE = False
    print("img2table not available. Install with: pip install img2table")

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

def extract_tables_with_img2table(pdf_path: str) -> str:
    """
    Extract tables from PDF using img2table (most accurate method).
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Markdown formatted tables
    """
    if not IMG2TABLE_AVAILABLE:
        logger.warning("img2table not available, falling back to pdfplumber")
        return extract_tables_with_pdfplumber(pdf_path)
    
    try:
        logger.info(f"Extracting tables with img2table from: {pdf_path}")
        
        # Initialize OCR (requires Tesseract to be installed)
        ocr = TesseractOCR(n_threads=1, lang='eng')
        
        # Load PDF
        pdf = PDF(pdf_path)
        
        # Extract tables
        tables = pdf.extract_tables(ocr=ocr)
        
        markdown_tables = []
        for page_num, page_tables in tables.items():
            logger.info(f"Found {len(page_tables)} tables on page {page_num}")
            
            for table_idx, table in enumerate(page_tables, 1):
                try:
                    # Convert table to DataFrame
                    df = table.to_pandas()
                    
                    # Clean up the DataFrame
                    df = df.dropna(how='all')  # Remove completely empty rows
                    df = df.dropna(axis=1, how='all')  # Remove completely empty columns
                    
                    if len(df) > 0 and len(df.columns) > 0:
                        # Convert to markdown
                        md_table = df.to_markdown(index=False)
                        markdown_tables.append(f"\n### Page {page_num} Table {table_idx}\n{md_table}\n")
                        logger.debug(f"Successfully extracted table {table_idx} from page {page_num}")
                    else:
                        logger.debug(f"Skipping empty table {table_idx} from page {page_num}")
                        
                except Exception as e:
                    logger.warning(f"Failed to process table {table_idx} on page {page_num}: {e}")
        
        result = "\n".join(markdown_tables)
        logger.info(f"img2table extraction completed. Total tables: {len(markdown_tables)}")
        return result
        
    except Exception as e:
        logger.error(f"img2table extraction failed: {e}", exc_info=True)
        logger.info("Falling back to pdfplumber")
        return extract_tables_with_pdfplumber(pdf_path)

def extract_tables_with_pdfplumber(pdf_path: str) -> str:
    """
    Extract tables from a PDF using pdfplumber and return as markdown string.
    """
    logger.info("Extracting tables with pdfplumber...")
    markdown_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue
                try:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    md = df.to_markdown(index=False)
                    markdown_tables.append(f"\n### Page {i} Table\n{md}\n")
                except Exception as e:
                    logger.warning(f"Failed to parse table on page {i}: {e}")

    return "\n".join(markdown_tables)


# New function: extract_tables_with_camelot
def extract_tables_with_camelot(pdf_path: str) -> str:
    """
    Extract tables from a PDF using Camelot and return as markdown string.
    """
    if not CAMELOT_AVAILABLE:
        logger.warning("Camelot not available, skipping.")
        return ""
    
    logger.info("Extracting tables with Camelot...")
    markdown_tables = []

    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")

        for i, table in enumerate(tables, start=1):
            try:
                df = table.df
                if df.empty:
                    continue
                # Promote first row to header if it looks like one
                if df.shape[0] > 1:
                    df.columns = df.iloc[0]
                    df = df[1:]
                md = df.to_markdown(index=False)
                markdown_tables.append(f"\n### Camelot Table {i}\n{md}\n")
            except Exception as e:
                logger.warning(f"Failed to parse Camelot table {i}: {e}")

    except Exception as e:
        logger.error(f"Camelot extraction failed: {e}", exc_info=True)

    return "\n".join(markdown_tables)

def save_pdf_to_markdown_with_tables(pdf_path, output_dir="markdown_outputs", return_content=False, use_img2table=True, use_camelot=False):
    """
    Convert PDF to markdown using Docling and table extraction, and save to file.

    Args:
        pdf_path (str): Path to the PDF file.
        output_dir (str): Output directory.
        return_content (bool): Return combined markdown content.
        use_img2table (bool): Use img2table for table extraction (if available).

    Returns:
        str or (str, str): Path to saved markdown file, or (path, content) if return_content=True.
    """
    try:
        logger.info(f"Starting conversion for: {pdf_path}")
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            logger.error(f"File not found: {pdf_path}")
            return None if not return_content else (None, "")

        # Use Docling for narrative markdown
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path_obj))
        narrative_md = result.document.export_to_markdown()

        # Use img2table, camelot, or pdfplumber for accurate tables
        if use_img2table and IMG2TABLE_AVAILABLE:
            tables_md = extract_tables_with_img2table(str(pdf_path_obj))
        elif use_camelot and CAMELOT_AVAILABLE:
            tables_md = extract_tables_with_camelot(str(pdf_path_obj))
        else:
            tables_md = extract_tables_with_pdfplumber(str(pdf_path_obj))

        # Combine both
        combined_md = f"# Document: {pdf_path_obj.name}\n\n"
        combined_md += narrative_md.strip() + "\n\n"
        if tables_md:
            combined_md += "\n---\n# Extracted Tables\n" + tables_md

        # Save to markdown file
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_path = output_dir_path / (pdf_path_obj.stem + ".md")
        output_path.write_text(combined_md, encoding="utf-8")

        logger.info(f"Markdown file saved: {output_path}")
        return (str(output_path), combined_md) if return_content else str(output_path)

    except Exception as e:
        logger.error(f"Failed to convert PDF to markdown: {e}", exc_info=True)
        return None if not return_content else (None, "")
    
def save_table_as_markdown(df, output_path):
    """
    Save a pandas DataFrame as a markdown-formatted table.

    Args:
        df (pandas.DataFrame): The table to save.
        output_path (str): Path to the output .txt file.
    """
    try:
        logger.info(f"Starting to save table as markdown: {output_path}")
        logger.debug(f"DataFrame shape: {df.shape}")
        logger.debug(f"DataFrame columns: {list(df.columns)}")
        
        # Ensure output directory exists
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        markdown_table = df.to_markdown(index=False)
        logger.debug(f"Generated markdown table length: {len(markdown_table)} characters")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_table)
        
        logger.info(f"Table successfully saved to {output_path}")
        logger.debug(f"Output file size: {output_path_obj.stat().st_size} bytes")
        
    except Exception as e:
        logger.error(f"Error saving markdown table to {output_path}: {e}", exc_info=True)
        print(f"Error saving markdown table: {e}")
    
def is_md_separator_row(row):
    """
    Check if a markdown table row is a separator row like ['----', '------', ':----:', '---:']
    """
    logger.debug(f"Checking if row is separator: {row}")
    is_separator = all(re.match(r'^[:\- ]+$', cell) for cell in row)
    logger.debug(f"Row is separator: {is_separator}")
    return is_separator

def extract_relevant_table(md_file_path, required_keywords=None):
    """
    Extract the first markdown table containing all required keywords.
    
    Args:
        md_file_path (str): Path to the markdown (.md) file.
        required_keywords (list): List of strings that must appear in the table.
        
    Returns:
        pandas.DataFrame or None
    """
    try:
        logger.info(f"Starting table extraction from: {md_file_path}")
        
        if required_keywords is None:
            required_keywords = ["Revenue from Operations", "31.03.2025", "31.03.2024", "31.12.2024"]
        
        logger.info(f"Searching for keywords: {required_keywords}")
        
        # Check if markdown file exists
        md_path_obj = Path(md_file_path)
        if not md_path_obj.exists():
            logger.error(f"Markdown file not found: {md_file_path}")
            return None
            
        logger.debug(f"Markdown file found, size: {md_path_obj.stat().st_size} bytes")

        with open(md_file_path, 'r', encoding='utf-8') as f:
            md_text = f.read()
        
        logger.debug(f"Read markdown content, length: {len(md_text)} characters")

        # Regex to find all markdown tables
        table_regex = re.compile(r'((?:\|.*?\n)+)', re.MULTILINE)
        table_matches = table_regex.findall(md_text)
        
        logger.info(f"Found {len(table_matches)} potential tables in markdown")
        
        for i, match in enumerate(table_matches):
            logger.debug(f"Processing table {i+1}/{len(table_matches)}")
            
            lines = [line.strip() for line in match.strip().split("\n") if line.strip()]
            if len(lines) < 2:
                logger.debug(f"Table {i+1}: Skipping - insufficient lines ({len(lines)})")
                continue  # Not a valid table

            # Try pandas read_csv approach first (more reliable for well-formed tables)
            try:
                # Convert markdown table to CSV-like format
                csv_lines = []
                for line in lines:
                    # Remove leading/trailing | and split by |
                    cells = line.strip('|').split('|')
                    # Clean each cell and join with comma
                    cleaned_cells = [cell.strip() for cell in cells]
                    csv_lines.append(','.join(cleaned_cells))
                
                # Join lines and create a StringIO object
                from io import StringIO
                csv_content = '\n'.join(csv_lines)
                
                # Try to read with pandas
                df = pd.read_csv(StringIO(csv_content), skip_blank_lines=True)
                
                # Remove separator row if present (rows with only dashes and colons)
                df = df[~df.apply(lambda row: row.astype(str).str.match(r'^[:\- ]+$').all(), axis=1)]
                
                logger.debug(f"Table {i+1}: Successfully parsed with pandas, shape: {df.shape}")
                
            except Exception as pandas_error:
                logger.debug(f"Table {i+1}: Pandas parsing failed, trying manual approach: {pandas_error}")
                
                # Fallback to manual parsing
                # Split each line into columns
                rows = [line.strip('|').split('|') for line in lines]
                rows = [[cell.strip() for cell in row] for row in rows]
                
                logger.debug(f"Table {i+1}: {len(rows)} rows, {len(rows[0]) if rows else 0} columns")

                # Ensure all rows have the same number of columns
                if not rows:
                    logger.debug(f"Table {i+1}: Skipping - no rows")
                    continue
                    
                # Find the maximum number of columns and ensure consistent structure
                max_cols = max(len(row) for row in rows)
                logger.debug(f"Table {i+1}: Maximum columns found: {max_cols}")
                
                # Pad shorter rows with empty strings to match the longest row
                padded_rows = []
                for row in rows:
                    if len(row) < max_cols:
                        padded_row = row + [''] * (max_cols - len(row))
                        logger.debug(f"Table {i+1}: Padded row from {len(row)} to {len(padded_row)} columns")
                    else:
                        padded_row = row
                    padded_rows.append(padded_row)
                
                # Verify all rows now have the same number of columns
                if not all(len(row) == max_cols for row in padded_rows):
                    logger.warning(f"Table {i+1}: Column count mismatch after padding, skipping")
                    continue
                
                header = padded_rows[0]
                logger.debug(f"Table {i+1}: Header columns: {len(header)}")
                
                # Check for separator row and determine data rows
                if len(padded_rows) > 2 and is_md_separator_row(padded_rows[1]):
                    data = padded_rows[2:]
                    logger.debug(f"Table {i+1}: Found separator row, data rows: {len(data)}")
                else:
                    data = padded_rows[1:]
                    logger.debug(f"Table {i+1}: No separator row, data rows: {len(data)}")

                # Create DataFrame with explicit column names
                df = pd.DataFrame(data, columns=header)
                logger.debug(f"Table {i+1}: Created DataFrame with shape {df.shape}")
                logger.debug(f"Table {i+1}: Column names: {list(df.columns)}")
                
                # Log first few rows for debugging
                if len(df) > 0:
                    logger.debug(f"Table {i+1}: First row data: {df.iloc[0].tolist()}")

            # Continue with keyword matching
            try:
                # Flatten table to string for keyword searching
                flat_text = df.to_string(index=False).lower()
                logger.debug(f"Table {i+1}: Flattened text length: {len(flat_text)}")
                
                # Check if all keywords are present
                keyword_matches = []
                for keyword in required_keywords:
                    is_present = keyword.lower() in flat_text
                    keyword_matches.append(is_present)
                    logger.debug(f"Table {i+1}: Keyword '{keyword}' present: {is_present}")
                
                if all(keyword_matches):
                    logger.info(f"Found matching table {i+1} with all required keywords")
                    logger.debug(f"Table {i+1} columns: {list(df.columns)}")
                    return df  # Return the first matching table
                else:
                    logger.debug(f"Table {i+1}: Missing keywords, continuing search")

            except Exception as e:
                logger.error(f"Failed to process table {i+1}: {e}", exc_info=True)
                print(f"Failed to process table: {e}")

        logger.warning("No matching table found with all required keywords")
        return None  # No matching table found
        
    except Exception as e:
        logger.error(f"Error in extract_relevant_table: {e}", exc_info=True)
        return None

def main():
    """Main function to run the PDF to Markdown converter and table extraction"""
    logger.info("Starting main execution")
    
    try:
        pdf_path = 'dhanuka.pdf'
        save_pdf_to_markdown_with_tables(pdf_path, use_img2table=False, use_camelot=True)
        # Check if markdown file exists
        md_file_path = f'markdown_outputs/{pdf_path.split(".")[0]}.md'
        md_path_obj = Path(md_file_path)
        
        if not md_path_obj.exists():
            logger.error(f"Markdown file not found: {md_file_path}")
            print(f"Error: Markdown file '{md_file_path}' not found!")
            print("Please run PDF conversion first or check the file path.")
            return
        
        logger.info(f"Found markdown file: {md_file_path}")
        
        # Create output directory for tables
        output_path = "output/relevant_financial_table.txt"
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory prepared: {output_dir}")
        
        # Extract relevant table
        logger.info("Starting table extraction process")
        relevant_table = extract_relevant_table(md_file_path)

        if relevant_table is not None:
            logger.info("Relevant table found, saving to markdown format")
            save_table_as_markdown(relevant_table, output_path)
            print(f"✅ Successfully extracted and saved table to: {output_path}")
        else:
            logger.warning("No matching table found")
            print("❌ No matching table found with the required keywords.")
            print("Check the markdown file content and keyword requirements.")
            
    except Exception as e:
        logger.error(f"Error in main function: {e}", exc_info=True)
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()