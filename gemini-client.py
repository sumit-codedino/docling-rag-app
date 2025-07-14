import os
import logging
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, Literal, Annotated
import asyncio

class FinancialMetrics(BaseModel):
    revenue: float = Field(..., description="Revenue from Operations (INR mn/bn)", ge=0)
    profit_before_tax: float = Field(..., description="Profit Before Tax")
    cost_of_materials_consumed: Optional[float] = Field(None, description="Cost of Materials Consumed")
    purchase_of_stock_in_trade: Optional[float] = Field(None, description="Purchase of Stock in Trade")
    changes_in_inventories: Optional[float] = Field(None, description="Changes in inventories of finished goods, WIP, and Stock in Trade")
    finance_costs: Optional[float] = Field(None, description="Finance Costs (Interest Cost)")
    depreciation_and_amortisation: Optional[float] = Field(None, description="Depreciation and Amortisation Expense")
    total_tax_expenses: Optional[float] = Field(None, description="Total Tax Expenses")
    other_income: Optional[float] = Field(None, description="Other Income")


class QuarterData(BaseModel):
    unit_of_measure: Literal["INR mn", "INR bn", "INR cr", "INR"]
    currency: Literal["INR"] = "INR"
    current_quarter: FinancialMetrics
    prior_year_quarter: FinancialMetrics


class FinancialDocument(BaseModel):
    company_name: str = Field(..., min_length=1, description="Name of the company")
    quarter_year: Annotated[str, Field(pattern=r"^[1-4]QFY[0-9]{2}$")] = Field(..., description="Quarter and year of the report (e.g., 4QFY25)")
    data: QuarterData

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-2.0-flash-001"
TEMPERATURE = 0.7
MAX_TOKENS = 2048

def get_gemini_client(gemini_api_key: str):
    """Initialize and return Gemini client"""
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not provided")
    try:
        return genai.Client(api_key=gemini_api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {str(e)}")
        raise Exception(f"Failed to initialize Gemini client: {str(e)}")

def create_prompt(table_content: str) -> str:
    """Create prompt for financial data extraction"""
    return f"""
The following markdown table contains quarterly financial results for a company.

Extract and return the financial data using this structure:
- company_name: Name of the company
- quarter_year: Quarter and year in format like "4QFY25"
- data: Object containing:
  - unit_of_measure: Unit of measure (INR mn, INR bn, INR cr, or INR)
  - currency: Currency (default: INR)
  - current_quarter: Financial data for current quarter including revenue, profit_before_tax, etc.
  - prior_year_quarter: Financial data for same quarter last year

Extract all available financial metrics from the table. For missing values, use null or omit the field.

Table:
{table_content}
"""

async def generate_estimates_text(txt_path: str) -> dict:
    """
    Extract structured financial data from a markdown table using Gemini.

    Args:
        txt_path (str): Path to the .txt file containing markdown table
        gemini_api_key (str): Your Gemini API key

    Returns:
        dict: Structured financial data
    """
    try:
        # Validate input file
        if not os.path.exists(txt_path):
            logger.error(f"File not found: {txt_path}")
            raise FileNotFoundError(f"File not found: {txt_path}")

        # Read table content
        with open(txt_path, "r", encoding="utf-8") as f:
            table_content = f.read()
        
        if not table_content.strip():
            logger.error(f"File is empty: {txt_path}")
            raise ValueError(f"File is empty: {txt_path}")

        logger.info(f"Processing table from: {txt_path}")

        # Create prompt and initialize client
        prompt = create_prompt(table_content)
        client = get_gemini_client(GEMINI_API_KEY)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": FinancialDocument,
            },
        )

        return response.text

    except FileNotFoundError:
        logger.error(f"File not found: {txt_path}")
        raise
    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Gemini generation failed: {str(e)}")
        raise

async def main():
    txt_path = "output/relevant_financial_table.txt"
    response = await generate_estimates_text(txt_path)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())