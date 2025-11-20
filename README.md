# Financial Analysis RAG Application

A comprehensive stock analysis application that combines Retrieval-Augmented Generation (RAG) from financial analysis textbooks with real-time stock data from Yahoo Finance to provide detailed investment analysis reports.

## Features

- 📚 **RAG-Powered Analysis**: Uses embeddings from financial analysis textbooks in `data/FinAnalysisTexts/` to provide context-aware analysis
- 📊 **Real-Time Stock Data**: Fetches current financial data from Yahoo Finance
- 📈 **Visualizations**: Generates charts for price history, valuation metrics, and profitability analysis
- 📋 **Comprehensive Reports**: Creates detailed reports with tables showing:
  - Key financial metrics (PE ratio, market cap, etc.)
  - Profitability metrics (margins, ROE, ROA)
  - Financial health metrics (current ratio, debt-to-equity, etc.)
  - Analyst estimates and recommendations
  - Financial statements (income statement, balance sheet, cash flow)
- 🤖 **AI-Powered Insights**: Uses LLM to generate investment analysis based on financial principles

## Setup

### Prerequisites

- Python 3.8 or higher
- GROQ API key (for LLM)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd FinAnalysisRAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

4. **Manually ingest the financial analysis texts** (required before first use):
```bash
python utils/embeddings.py
```
This will process all PDFs from `data/FinAnalysisTexts/` and load them into the knowledge base. This may take several minutes depending on the number of documents.

## Usage

### Starting the Application

Run the Chainlit application:
```bash
chainlit run app.py
```

**Note:** Make sure you've run the ingestion script first (see step 4 above), otherwise the chatbot will warn you that the knowledge base is empty.

The application will open in your browser automatically.

### Using the Application

1. **Stock Analysis**: Simply enter a stock ticker symbol (e.g., `AAPL`, `MSFT`, `TSLA`) to get a comprehensive analysis report.

2. **General Questions**: Ask questions about financial concepts (e.g., "What is the PE ratio?", "Explain debt-to-equity ratio") and the system will answer using the knowledge base.

### Example Queries

- `AAPL` - Get full stock analysis for Apple
- `analyze MSFT` - Analyze Microsoft stock
- `What is the PE ratio for TSLA?` - Get specific metric
- `Explain how to analyze financial statements` - Learn financial concepts

## Report Contents

When you analyze a stock, you'll receive:

1. **Key Financial Metrics Table**: Current price, market cap, PE ratios, valuation metrics
2. **Profitability Metrics Table**: Profit margins, ROE, ROA
3. **Financial Health Table**: Liquidity ratios, debt metrics, revenue
4. **Analyst Estimates Table**: Target prices, recommendations, analyst count
5. **Financial Statements**: Income statement, balance sheet, cash flow
6. **Price Chart**: Historical price and volume data
7. **Valuation Metrics Chart**: Visual comparison of key ratios
8. **Profitability Chart**: Visual representation of profitability metrics
9. **AI Analysis**: Comprehensive investment analysis with buy/hold/sell recommendation

## Project Structure

```
FinAnalysisRAG/
├── app.py                      # Main Chainlit application
├── requirements.txt            # Python dependencies
├── data/
│   └── FinAnalysisTexts/       # Financial analysis PDFs
├── agents/
│   └── financial_agent.py     # Financial analysis agent (handles all data fetching, visualizations, and analysis)
├── utils/
│   └── embeddings.py          # RAG embedding and retrieval functions
└── chroma_db/                  # Persistent vector database (created automatically)
```

## Technologies Used

- **Chainlit**: Chat interface
- **ChromaDB**: Vector database for RAG
- **Sentence Transformers**: Text embeddings
- **LangChain**: LLM orchestration
- **Groq**: Fast LLM inference
- **yfinance**: Yahoo Finance data
- **Matplotlib**: Data visualization
- **Pandas**: Data manipulation

## Disclaimer

This tool provides educational and informational analysis only. It is not financial advice. Always consult with a qualified financial advisor before making investment decisions.

## Deployment

For beta testing and production deployment, see [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

**Quick Deploy Options:**
- **Railway**: Easiest option, free tier available
- **Render**: Free tier with automatic deployments
- **Fly.io**: Global deployment, good free tier
- **Docker**: Works on any cloud platform

See DEPLOYMENT.md for step-by-step guides.

## License

[Add your license here]

