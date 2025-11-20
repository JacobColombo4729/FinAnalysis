"""
Financial Analysis RAG Application

This application provides comprehensive stock analysis using:
- RAG (Retrieval-Augmented Generation) from financial analysis textbooks
- Real-time stock data from Yahoo Finance
- Visualizations and tables for financial reports
"""
import os
import re
from typing import Optional

import chainlit as cl
import chromadb
from dotenv import load_dotenv
from langchain_community.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

from agents.financial_agent import FinancialAnalysisAgent

# Load environment variables
load_dotenv()


def extract_ticker(message: str) -> Optional[str]:
    """
    Extract stock ticker from user message.
    
    Args:
        message: User's message
    
    Returns:
        str: Ticker symbol if found, None otherwise
    """
    # Common question words and words to exclude
    common_words = {
        'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 
        'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 
        'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'WAY', 'USE', 'SHE', 'MAN', 'HAD', 
        'DID', 'LET', 'PUT', 'SAY', 'TOO', 'WHY', 'WHAT', 'WHEN', 'WHERE', 'WHICH',
        'WITH', 'FROM', 'THAT', 'THIS', 'THEY', 'THEM', 'THEN', 'THAN', 'WILL',
        'WOULD', 'COULD', 'SHOULD', 'MIGHT', 'MUST', 'SHALL', 'EXPLAIN', 'TELL',
        'ABOUT', 'BETWEEN', 'AMONG', 'DURING', 'AFTER', 'BEFORE', 'UNDER', 'OVER'
    }
    
    message_upper = message.upper().strip()
    
    # Only extract ticker if message is very short (likely just a ticker) or has explicit ticker patterns
    # More conservative patterns that require context indicating it's a ticker
    patterns = [
        r'\$([A-Z]{1,5})\b',  # $AAPL format - most reliable
        r'analyze\s+([A-Z]{1,5})\b',  # analyze AAPL
        r'report\s+on\s+([A-Z]{1,5})\b',  # report on AAPL
        r'stock\s+([A-Z]{1,5})\b',  # stock AAPL
        r'([A-Z]{1,5})\s+stock\b',  # AAPL stock
        r'([A-Z]{1,5})\s+analysis\b',  # AAPL analysis
    ]
    
    # Check explicit patterns first
    for pattern in patterns:
        match = re.search(pattern, message_upper)
        if match:
            ticker = match.group(1)
            if ticker not in common_words and len(ticker) >= 1 and len(ticker) <= 5:
                return ticker
    
    # If message is very short (1-5 chars) and all caps, might be just a ticker
    # But exclude if it's a common word
    if len(message_upper) <= 5 and message_upper.isalpha() and message_upper.isupper():
        if message_upper not in common_words:
            return message_upper
    
    return None


@cl.on_chat_start
async def start_chat():
    """
    Initialize the chat session and set up the financial analysis system.
    
    This function:
    - Initializes the embedding model and ChromaDB
    - Creates the financial analysis agent
    - Displays the welcome message
    
    Note: Document embeddings should be loaded manually using utils/embeddings.py
    """
    # ====================================================================
    # STEP 1: Initialize Embedding Model and ChromaDB
    # ====================================================================
    # Initialize SentenceTransformer for generating embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Initialize ChromaDB client for persistent vector storage
    # Use CHROMA_DB_PATH env var if set (for persistent volumes in deployment), otherwise use ./chroma_db
    chroma_db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    try:
        chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        
        # Get or create ChromaDB collection for financial analysis texts
        financial_collection = chroma_client.get_or_create_collection(
            name="financial-analysis-texts",
            metadata={"description": "Financial analysis textbooks and resources"}
        )
        
        # ====================================================================
        # STEP 2: Check Financial Analysis Texts Collection
        # ====================================================================
        # Note: Document ingestion should be run manually using utils/embeddings.py
        # Only show warning if collection is empty
        collection_count = financial_collection.count()
        if collection_count == 0:
            await cl.Message(
                content=(
                    "**Knowledge base is empty!**\n\n"
                    "Please run the ingestion script manually before using the chatbot:\n"
                    "```bash\n"
                    "python utils/embeddings.py\n"
                    "```\n\n"
                    "This will load all financial analysis texts from `data/FinAnalysisTexts/` into the knowledge base."
                )
            ).send()
    except Exception as e:
        # Handle ChromaDB corruption or initialization errors
        error_msg = str(e)
        if "PanicException" in error_msg or "range start index" in error_msg:
            await cl.Message(
                content=(
                    "**ChromaDB Database Error**\n\n"
                    "The ChromaDB database appears to be corrupted. To fix this:\n\n"
                    "1. **Backup** (optional): If you want to keep the old data, rename the directory:\n"
                    "   ```bash\n"
                    "   mv chroma_db chroma_db_backup\n"
                    "   ```\n\n"
                    "2. **Recreate the database** by running the ingestion script:\n"
                    "   ```bash\n"
                    "   python utils/embeddings.py\n"
                    "   ```\n\n"
                    "3. **Restart the chatbot** after re-ingestion.\n\n"
                    "**Note:** The chatbot will work without RAG capabilities until the database is fixed."
                )
            ).send()
            # Set collection to None so the agent can still work (without RAG)
            financial_collection = None
        else:
            # Re-raise other exceptions
            raise
    
    # ====================================================================
    # STEP 3: Initialize LLM
    # ====================================================================
    llm = ChatGroq(
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
    )
    
    # ====================================================================
    # STEP 4: Initialize Memory
    # ====================================================================
    memory = ConversationBufferWindowMemory(
        k=5, input_key="input", memory_key="history"
    )

    # ====================================================================
    # STEP 5: Initialize Financial Analysis Agent
    # ====================================================================
    # Create progress callback function for sending loading updates
    async def send_progress_update(message: str):
        """Send progress update message to the chat."""
        await cl.Message(content=message).send()
    
    financial_agent = FinancialAnalysisAgent(
        model=model,
        llm=llm,
        memory=memory,
        financial_collection=financial_collection,
        progress_callback=send_progress_update
    )
    
    # ====================================================================
    # STEP 6: Store in Session
    # ====================================================================
    cl.user_session.set("financial_agent", financial_agent)
    cl.user_session.set("financial_collection", financial_collection)

    # ====================================================================
    # STEP 7: Welcome Message
    # ====================================================================
    await cl.Message(
        content=(
            "### Welcome to Financial Analysis\n\n"
            "I'm your AI financial analyst powered by:\n"
            "- **RAG (Retrieval-Augmented Generation)** from financial analysis textbooks\n"
            "- **Real-time stock data** from Yahoo Finance\n"
            "- **Comprehensive visualizations** and analysis reports\n\n"
            
            "#### How to Use\n"
            "Simply enter a stock ticker symbol (e.g., **AAPL**, **MSFT**, **TSLA**) and I'll generate a comprehensive analysis report including:\n\n"
            "- **Stock Price Charts** - Historical price and volume data\n"
            "- **Financial Metrics** - PE ratio, valuation metrics, profitability ratios\n"
            "- **Financial Statements** - Income statement, balance sheet, cash flow\n"
            "- **Analysis Tables** - Key metrics in organized tables\n"
            "- **Investment Analysis** - AI-powered analysis using financial principles\n"
            "- **Risk Assessment** - Financial health and risk factors\n\n"
            
            "#### Example Queries\n"
            "- `AAPL` or `analyze AAPL` - Get full stock analysis\n"
            "- `What is the PE ratio for MSFT?` - Ask specific questions\n"
            "- `Explain debt-to-equity ratio` - Learn financial concepts\n"
            "- `Compare valuation metrics for TSLA` - Get comparative analysis\n\n"
            
            "#### Disclaimer\n"
            "This tool provides educational and informational analysis only. "
            "It is not financial advice. Always consult with a qualified financial advisor before making investment decisions.\n\n"
            
            "**Ready to analyze a stock? Just enter a ticker symbol!**"
        )
    ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    """
    Handle incoming user messages.
    
    This function:
    1. Extracts ticker symbols from user messages
    2. Routes to financial analysis agent
    3. Generates comprehensive reports with tables and visualizations
    4. Answers general financial questions using RAG
    """
    user_input = message.content.strip()
    
    # Prevent duplicate processing - check if we've already processed this exact message
    processed_messages = cl.user_session.get("processed_messages", set())
    message_hash = hash(f"{user_input}_{message.created_at if hasattr(message, 'created_at') else ''}")
    
    if message_hash in processed_messages:
        # Already processed this message, skip to prevent duplicates
        print(f"Skipping duplicate message: {user_input[:50]}...")
        return
    
    # Mark as processed (keep only last 10 to avoid memory issues)
    processed_messages.add(message_hash)
    if len(processed_messages) > 10:
        processed_messages = set(list(processed_messages)[-10:])
    cl.user_session.set("processed_messages", processed_messages)
    
    # Get financial agent from session
    financial_agent = cl.user_session.get("financial_agent")
    
    if not financial_agent:
        await cl.Message(
            content="System not properly initialized. Please refresh the chat."
        ).send()
        return

    try:
        # Check if user is asking for stock screening/comparison
        if financial_agent.is_screening_query(user_input):
            await cl.Message(content=f"**Starting stock screening analysis...**\n\nThis may take a moment as I analyze multiple stocks.").send()
            try:
                response = financial_agent.screen_stocks(user_input)
                await cl.Message(content=response).send()
            except Exception as e:
                await cl.Message(
                    content=f"Error screening stocks: {str(e)}\n\nPlease try again or be more specific."
                ).send()
        # Check if user is asking for a specific stock analysis
        elif extract_ticker(user_input):
            ticker = extract_ticker(user_input)
            # Generate comprehensive stock analysis
            await cl.Message(content=f"**Starting comprehensive analysis of {ticker}...**\n\nGathering all the data needed for your report.").send()
            
            try:
                # Agent handles all data fetching, report building, and visualization preparation
                analysis = financial_agent.analyze_stock(ticker)
                
                # Send the complete report (only once)
                await cl.Message(content=analysis['report']).send()
                
                # Send visualizations
                for chart_key, chart_info in analysis['visualization_files'].items():
                    try:
                        await cl.Message(
                            content=chart_info['title'],
                            elements=[
                                cl.Image(
                                    name=chart_key,
                                    path=chart_info['path'],
                                    display="inline"
                                )
                            ]
                        ).send()
                    except Exception as e:
                        print(f"Error displaying {chart_key}: {e}")
                
            except Exception as e:
                await cl.Message(
                    content=f"Error analyzing {ticker}: {str(e)}\n\nPlease check that the ticker symbol is correct and try again."
                ).send()
        else:
            # Answer general financial question using RAG
            await cl.Message(content=f"Searching knowledge base and generating answer...").send()
            response = financial_agent.answer_question(user_input)
            await cl.Message(content=response).send()
    
    except Exception as e:
        error_msg = f"Oops! Something went wrong: {e}"
        await cl.Message(content=error_msg).send()
        print(f"Error handling message: {e}")
