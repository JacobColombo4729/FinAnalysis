"""
Financial Analysis Agent

This agent uses RAG to answer questions about financial analysis using the knowledge base,
and integrates with Yahoo Finance to provide real-time stock data.
Handles all financial data fetching, visualizations, and analysis.
"""
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Optional
import chromadb
import os
import pandas as pd
import base64
import tempfile
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import yfinance as yf

from utils.embeddings import retrieve_relevant_chunks


class FinancialAnalysisAgent:
    """
    Agent that provides financial analysis using RAG and real-time stock data.
    Handles all financial data fetching, visualizations, and analysis.
    """
    
    # Cache for tickers (refresh daily)
    _tickers_cache = None
    _tickers_cache_time = None
    CACHE_DURATION = 24 * 60 * 60  # 24 hours in seconds
    
    # Mapping of filenames to MLA citations
    SOURCE_CITATIONS = {
        "CorporateFinance.pdf": "Berk, Jonathan, and Peter DeMarzo. Corporate Finance. Pearson, 2020.",
        "CorporateFinanceforDummies.pdf": "Taillard, Michael. Corporate Finance For Dummies. For Dummies, 2012.",
        "CorporateFinancialAnalysiswithMicrosoftExcel.pdf": "Fairhurst, Danielle Stein. Corporate Financial Analysis with Microsoft Excel. Wiley, 2015.",
        "Financial Intelligence.pdf": "Berman, Karen, and Joe Knight. Financial Intelligence: A Manager's Guide to Knowing What the Numbers Really Mean. Harvard Business Review Press, 2013.",
        "Financial Planning & Analysis and Performance Management ( PDFDrive ).pdf": "Nayar, Jack. Financial Planning & Analysis and Performance Management. Wiley, 2018.",
        "Financial-Shenanigans-How-to-Detect-Accounting-Gimmicks-and-Fraud-in-Financial-Reports-Fourth-Edition-pages-1-160.pdf": "Schilit, Howard, Jeremy Perler, and Yoni Engelhart. Financial Shenanigans: How to Detect Accounting Gimmicks and Fraud in Financial Reports. 4th ed., McGraw-Hill Education, 2018.",
        "financial-statement-analysis-lifa.pdf": "Robinson, Thomas R., et al. International Financial Statement Analysis. 3rd ed., CFA Institute, 2015.",
        "FinancialIntelligenceRevisedEdition.pdf": "Berman, Karen, and Joe Knight. Financial Intelligence: A Manager's Guide to Knowing What the Numbers Really Mean. Revised ed., Harvard Business Review Press, 2013.",
        "FinancialModeling.pdf": "Benninga, Simon. Financial Modeling. 4th ed., MIT Press, 2014.",
        "FinancialModelinginExcelforDummies.pdf": "Fairhurst, Danielle Stein. Financial Modeling in Excel For Dummies. For Dummies, 2017.",
        "International-financial-statement-analysis-CFA-Institute.pdf": "Robinson, Thomas R., et al. International Financial Statement Analysis. 3rd ed., CFA Institute, 2015.",
        "ReadingFinancialReportsforDummies.pdf": "Epstein, Lita. Reading Financial Reports For Dummies. 3rd ed., For Dummies, 2013.",
        "The Intelligent Investor - BENJAMIN GRAHAM.pdf": "Graham, Benjamin. The Intelligent Investor: The Definitive Book on Value Investing. Revised ed., HarperBusiness, 2006.",
        "Warren-Buffett-and-the-Interpretation-of-Financial-Statements.pdf": "Buffett, Mary, and David Clark. Warren Buffett and the Interpretation of Financial Statements: The Search for the Company with a Durable Competitive Advantage. Scribner, 2008."
    }
    
    def __init__(
        self,
        model: SentenceTransformer,
        llm: ChatGroq,
        memory: ConversationBufferWindowMemory,
        financial_collection: chromadb.Collection,
        progress_callback=None
    ):
        self.model = model
        self.llm = llm
        self.memory = memory
        self.financial_collection = financial_collection
        self.progress_callback = progress_callback  # Callback for sending progress updates
    
    def _get_rag_context(self, query: str, k: int = 5) -> tuple:
        """Retrieve relevant context from the financial analysis knowledge base.
        
        Returns:
            tuple: (formatted_context_string, sources_list)
        """
        if self.financial_collection is None:
            return ("", [])  # No RAG context available if collection is not initialized
        
        try:
            chunks = retrieve_relevant_chunks(query, self.financial_collection, k=k, include_metadata=True)
            if not chunks:
                return ("", [])
            
            context_parts = []
            sources = []
            seen_sources = {}  # Track unique sources to avoid duplicates
            source_counter = 1
            
            for chunk, chunk_id, metadata, distance in chunks:
                source_filename = metadata.get('source', 'Unknown')
                
                # Get MLA citation for this source
                if source_filename in self.SOURCE_CITATIONS:
                    mla_citation = self.SOURCE_CITATIONS[source_filename]
                else:
                    # Fallback: format filename nicely if not in mapping
                    source_clean = source_filename.replace('.pdf', '').replace('_', ' ').title()
                    mla_citation = f"{source_clean}. PDF."
                
                # Use source number if we've seen this source before, otherwise assign new number
                if source_filename not in seen_sources:
                    seen_sources[source_filename] = source_counter
                    sources.append(mla_citation)
                    source_num = source_counter
                    source_counter += 1
                else:
                    source_num = seen_sources[source_filename]
                
                context_parts.append(f"[Source {source_num}] {mla_citation}\n{chunk}\n")
            
            formatted_context = "\n---\n".join(context_parts)
            return (formatted_context, sources)
        except Exception as e:
            print(f"Error retrieving RAG context: {e}")
            return ("", [])
    
    def _send_progress(self, message: str):
        """Send progress update if callback is available."""
        if self.progress_callback:
            import asyncio
            try:
                # Try to get the running event loop
                loop = asyncio.get_running_loop()
                # Schedule the coroutine to run
                loop.create_task(self.progress_callback(message))
            except RuntimeError:
                # No event loop running, create a new one
                asyncio.run(self.progress_callback(message))
    
    def analyze_stock(self, ticker: str) -> Dict:
        """
        Perform comprehensive stock analysis.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Dict containing all analysis data including text report, tables, and visualizations
        """
        try:
            self._send_progress(f"Fetching financial data and market information for **{ticker}**...")
            # Fetch all stock data
            stock_info = self._get_stock_info(ticker)
            financial_statements = self._get_financial_statements(ticker)
            historical_data = self._get_historical_data(ticker, period="1y")
            analyst_data = self._get_analyst_data(ticker)
            
            # Generate RAG context for analysis
            analysis_query = f"Analyze {ticker} stock: {stock_info.get('company_name', '')}. "
            analysis_query += f"Key metrics: PE ratio {stock_info.get('pe_ratio')}, "
            analysis_query += f"profit margin {stock_info.get('profit_margin')}, "
            analysis_query += f"ROE {stock_info.get('roe')}, debt to equity {stock_info.get('debt_to_equity')}"
            
            rag_context, rag_sources = self._get_rag_context(analysis_query, k=5)
            
            # Create visualizations
            self._send_progress(f"Creating charts and visualizations...")
            price_chart = self._create_price_chart(historical_data, ticker)
            metrics_chart = self._create_metrics_comparison_chart(stock_info, ticker)
            profitability_chart = self._create_profitability_chart(stock_info)
            
            # Create financial statement charts
            revenue_chart = None
            if not financial_statements['income_statement'].empty:
                revenue_chart = self._create_financial_statement_chart(
                    financial_statements['income_statement'],
                    "Revenue Trend",
                    "Total Revenue"
                )
            
            # Generate text analysis using LLM
            analysis_prompt = self._create_analysis_prompt(stock_info, financial_statements, analyst_data, rag_context)
            analysis_text = self.llm.invoke(analysis_prompt).content
            
            # Build complete report
            report = self._build_report(ticker, stock_info, financial_statements, analyst_data, analysis_text, rag_sources)
            
            # Prepare visualizations with file paths
            visualization_files = self._prepare_visualizations(
                ticker,
                price_chart,
                metrics_chart,
                profitability_chart,
                revenue_chart
            )
            
            return {
                'ticker': ticker,
                'report': report,  # Complete formatted report text
                'visualization_files': visualization_files,  # Dict with file paths for Chainlit
            }
        except Exception as e:
            raise Exception(f"Error analyzing stock {ticker}: {str(e)}")
    
    def _build_report(self, ticker: str, stock_info: Dict, financial_statements: Dict, 
                     analyst_data: Dict, analysis_text: str, sources: list = None) -> str:
        """Build the complete formatted report with all tables and analysis."""
        report_parts = []
        
        # Header
        report_parts.append(f"# Comprehensive Stock Analysis: {stock_info['ticker']}\n")
        report_parts.append(f"**{stock_info['company_name']}**\n")
        report_parts.append(f"*{stock_info['sector']} - {stock_info['industry']}*\n")
        report_parts.append("---\n")
        
        # Key Metrics Table
        report_parts.append("## Key Financial Metrics\n")
        metrics_data = {
            'Metric': [
                'Current Price',
                'Market Cap',
                '52 Week High',
                '52 Week Low',
                'PE Ratio',
                'Forward PE',
                'PEG Ratio',
                'Price to Book',
                'Beta',
                'EPS',
                'Dividend Yield',
            ],
            'Value': [
                f"${stock_info.get('current_price', 'N/A')}",
                self._format_currency(stock_info.get('market_cap', 'N/A')),
                f"${stock_info.get('52_week_high', 'N/A')}",
                f"${stock_info.get('52_week_low', 'N/A')}",
                stock_info.get('pe_ratio', 'N/A'),
                stock_info.get('forward_pe', 'N/A'),
                stock_info.get('peg_ratio', 'N/A'),
                stock_info.get('price_to_book', 'N/A'),
                stock_info.get('beta', 'N/A'),
                stock_info.get('eps', 'N/A'),
                f"{stock_info.get('dividend_yield', 0):.2f}%",
            ]
        }
        metrics_df = pd.DataFrame(metrics_data)
        report_parts.append(metrics_df.to_markdown(index=False))
        report_parts.append("\n")
        
        # Profitability Metrics Table
        report_parts.append("## Profitability Metrics\n")
        profitability_data = {
            'Metric': [
                'Profit Margin',
                'Operating Margin',
                'ROE (Return on Equity)',
                'ROA (Return on Assets)',
            ],
            'Value': [
                f"{float(stock_info.get('profit_margin', 0)) * 100:.2f}%" if stock_info.get('profit_margin') != 'N/A' else 'N/A',
                f"{float(stock_info.get('operating_margin', 0)) * 100:.2f}%" if stock_info.get('operating_margin') != 'N/A' else 'N/A',
                f"{float(stock_info.get('roe', 0)) * 100:.2f}%" if stock_info.get('roe') != 'N/A' else 'N/A',
                f"{float(stock_info.get('roa', 0)) * 100:.2f}%" if stock_info.get('roa') != 'N/A' else 'N/A',
            ]
        }
        profitability_df = pd.DataFrame(profitability_data)
        report_parts.append(profitability_df.to_markdown(index=False))
        report_parts.append("\n")
        
        # Financial Health Table
        report_parts.append("## Financial Health Metrics\n")
        health_data = {
            'Metric': [
                'Current Ratio',
                'Quick Ratio',
                'Debt to Equity',
                'Revenue',
                'Enterprise Value',
            ],
            'Value': [
                stock_info.get('current_ratio', 'N/A'),
                stock_info.get('quick_ratio', 'N/A'),
                stock_info.get('debt_to_equity', 'N/A'),
                self._format_currency(stock_info.get('revenue', 'N/A')),
                self._format_currency(stock_info.get('enterprise_value', 'N/A')),
            ]
        }
        health_df = pd.DataFrame(health_data)
        report_parts.append(health_df.to_markdown(index=False))
        report_parts.append("\n")
        
        # Analyst Data Table
        if analyst_data:
            report_parts.append("## Analyst Estimates\n")
            analyst_data_table = {
                'Metric': [
                    'Target Price',
                    'Target High',
                    'Target Low',
                    'Recommendation',
                    'Number of Analysts',
                ],
                'Value': [
                    f"${analyst_data.get('target_price', 'N/A')}",
                    f"${analyst_data.get('target_high', 'N/A')}",
                    f"${analyst_data.get('target_low', 'N/A')}",
                    analyst_data.get('recommendation', 'N/A'),
                    analyst_data.get('number_of_analysts', 'N/A'),
                ]
            }
            analyst_df = pd.DataFrame(analyst_data_table)
            report_parts.append(analyst_df.to_markdown(index=False))
            report_parts.append("\n")
        
        # Financial Statements Tables
        if not financial_statements['income_statement'].empty:
            report_parts.append("## Income Statement (Annual)\n")
            income_stmt = financial_statements['income_statement']
            # Show top 10 rows
            income_stmt_display = income_stmt.head(10).T
            report_parts.append(income_stmt_display.to_markdown())
            report_parts.append("\n")
        
        # AI Analysis
        report_parts.append("## AI-Powered Analysis\n")
        report_parts.append(analysis_text)
        report_parts.append("\n")
        
        # Citations
        if sources:
            report_parts.append("## Works Cited\n")
            for source in sources:
                # Sources are already in MLA format from SOURCE_CITATIONS
                report_parts.append(f"{source}\n")
            report_parts.append("\n")
        
        return "\n".join(report_parts)
    
    def _prepare_visualizations(self, ticker: str, price_chart: str, metrics_chart: str, 
                                profitability_chart: str, revenue_chart: str) -> Dict:
        """Prepare visualization files for Chainlit display."""
        visualization_files = {}
        
        # Create temp directory for charts
        temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp_charts")
        os.makedirs(temp_dir, exist_ok=True)
        
        def save_chart(base64_data, chart_name):
            """Save base64 chart data to a file."""
            if not base64_data:
                return None
            try:
                img_data = base64.b64decode(base64_data)
                temp_path = os.path.join(temp_dir, f"{ticker}_{chart_name}.png")
                with open(temp_path, 'wb') as f:
                    f.write(img_data)
                return temp_path
            except Exception as e:
                print(f"Error saving {chart_name} chart: {e}")
                return None
        
        if price_chart:
            path = save_chart(price_chart, "price")
            if path:
                visualization_files['price_chart'] = {
                    'path': path,
                    'title': '## Price Chart'
                }
        
        if metrics_chart:
            path = save_chart(metrics_chart, "metrics")
            if path:
                visualization_files['metrics_chart'] = {
                    'path': path,
                    'title': '## Valuation Metrics Chart'
                }
        
        if profitability_chart:
            path = save_chart(profitability_chart, "profitability")
            if path:
                visualization_files['profitability_chart'] = {
                    'path': path,
                    'title': '## Profitability Metrics Chart'
                }
        
        if revenue_chart:
            path = save_chart(revenue_chart, "revenue")
            if path:
                visualization_files['revenue_chart'] = {
                    'path': path,
                    'title': '## Revenue Trend Chart'
                }
        
        return visualization_files
    
    def _create_analysis_prompt(self, stock_info: Dict, financial_statements: Dict, analyst_data: Dict, rag_context: str) -> str:
        """Create a prompt for LLM to generate stock analysis."""
        
        prompt = f"""You are a financial analyst providing a comprehensive stock analysis report.

STOCK INFORMATION:
Company: {stock_info.get('company_name', 'N/A')} ({stock_info.get('ticker', 'N/A')})
Sector: {stock_info.get('sector', 'N/A')}
Industry: {stock_info.get('industry', 'N/A')}
Current Price: ${stock_info.get('current_price', 'N/A')}
Market Cap: {self._format_currency(stock_info.get('market_cap', 'N/A'))}

KEY METRICS:
- PE Ratio: {stock_info.get('pe_ratio', 'N/A')}
- Forward PE: {stock_info.get('forward_pe', 'N/A')}
- PEG Ratio: {stock_info.get('peg_ratio', 'N/A')}
- Price to Book: {stock_info.get('price_to_book', 'N/A')}
- Beta: {stock_info.get('beta', 'N/A')}
- EPS: {stock_info.get('eps', 'N/A')}
- Dividend Yield: {stock_info.get('dividend_yield', 0):.2f}%

PROFITABILITY:
- Profit Margin: {stock_info.get('profit_margin', 'N/A')}
- Operating Margin: {stock_info.get('operating_margin', 'N/A')}
- ROE (Return on Equity): {stock_info.get('roe', 'N/A')}
- ROA (Return on Assets): {stock_info.get('roa', 'N/A')}

FINANCIAL HEALTH:
- Current Ratio: {stock_info.get('current_ratio', 'N/A')}
- Quick Ratio: {stock_info.get('quick_ratio', 'N/A')}
- Debt to Equity: {stock_info.get('debt_to_equity', 'N/A')}

ANALYST DATA:
- Target Price: ${analyst_data.get('target_price', 'N/A')}
- Recommendation: {analyst_data.get('recommendation', 'N/A')}
- Number of Analysts: {analyst_data.get('number_of_analysts', 'N/A')}

FINANCIAL ANALYSIS KNOWLEDGE BASE CONTEXT:
{rag_context if rag_context else "No specific context available from knowledge base."}

Based on the above information and financial analysis principles from the knowledge base, provide a comprehensive analysis report covering:
1. Company Overview
2. Valuation Analysis (is the stock overvalued, undervalued, or fairly valued?)
3. Financial Health Assessment
4. Profitability Analysis
5. Risk Factors
6. Investment Recommendation (Buy/Hold/Sell) with reasoning

IMPORTANT: When referencing information from the knowledge base sources, cite them using in-text citations in the format (Source 1), (Source 2), etc., corresponding to the source numbers provided in the context above. Be specific, use the metrics provided, and reference financial analysis principles where relevant. Format the response in clear sections with headers.
"""
        return prompt
    
    def answer_question(self, question: str, ticker: str = None) -> str:
        """
        Answer a financial analysis question using RAG.
        
        Args:
            question: User's question
            ticker: Optional stock ticker if question is about a specific stock
        
        Returns:
            str: Answer to the question
        """
        # Get RAG context
        rag_context, rag_sources = self._get_rag_context(question, k=5)
        
        # If ticker is provided, get stock data
        stock_context = ""
        if ticker:
            try:
                stock_info = self._get_stock_info(ticker)
                stock_context = f"\n\nCurrent Stock Data for {ticker}:\n"
                stock_context += f"Company: {stock_info.get('company_name')}\n"
                stock_context += f"Price: ${stock_info.get('current_price')}\n"
                stock_context += f"PE Ratio: {stock_info.get('pe_ratio')}\n"
                stock_context += f"Market Cap: {self._format_currency(stock_info.get('market_cap'))}\n"
            except:
                pass
        
        prompt = f"""You are a financial analysis expert. Answer the following question using the provided context from financial analysis textbooks and resources.

QUESTION: {question}

FINANCIAL ANALYSIS KNOWLEDGE BASE:
{rag_context if rag_context else "No specific context available from the knowledge base."}
{stock_context}

Provide a clear, accurate, and comprehensive answer based on the knowledge base. When referencing information from the knowledge base sources, cite them using in-text citations in the format (Source 1), (Source 2), etc., corresponding to the source numbers provided in the context above. If the knowledge base provides relevant information, use it to give a detailed explanation with proper citations. If the knowledge base doesn't contain specific information about this topic, you may use your general knowledge of financial analysis to provide a helpful answer, but mention that the information may not be from the knowledge base. If real-time stock data is provided, you may reference it in your answer.
"""
        
        response = self.llm.invoke(prompt)
        answer = response.content
        
        # Append formal references if sources were used (only if not already present)
        if rag_sources and "## Works Cited" not in answer and "Works Cited" not in answer:
            answer += "\n\n## Works Cited\n"
            for source in rag_sources:
                # Sources are already in MLA format from SOURCE_CITATIONS
                answer += f"{source}\n"
        
        return answer
    
    def screen_stocks(self, query: str, criteria: str = None, limit: int = 10) -> str:
        """
        Screen and analyze stocks based on user criteria.
        
        Args:
            query: User's query about stock screening (e.g., "best fundamentals", "high ROE")
            criteria: Specific criteria to screen for
            limit: Maximum number of stocks to analyze
        
        Returns:
            str: Analysis of stocks matching the criteria
        """
        try:
            # Extract year from query if present (e.g., "2026", "2025")
            import re
            year_match = re.search(r'\b(20\d{2})\b', query)
            target_year = int(year_match.group(1)) if year_match else None
            is_future_query = target_year is not None and target_year > 2024
            
            # Get tickers from major indices (S&P 500, NASDAQ 100)
            self._send_progress(f"Analyzing stocks from major indices...")
            all_tickers = self._get_popular_tickers()
            # For screening, we can analyze a larger sample but limit to reasonable number
            # to avoid timeout. Sample randomly or take first N for consistency
            max_tickers_to_analyze = min(50, len(all_tickers))  # Analyze up to 50 stocks
            tickers_to_analyze = all_tickers[:max_tickers_to_analyze]
            
            # Analyze stocks and collect data
            stocks_data = []
            valid_count = 0
            for i, ticker in enumerate(tickers_to_analyze):
                if (i + 1) % 20 == 0:  # Update every 20 stocks instead of 10
                    self._send_progress(f"Processed {i + 1}/{len(tickers_to_analyze)} stocks...")
                try:
                    stock_info = self._get_stock_info(ticker)
                    analyst_data = self._get_analyst_data(ticker)
                    
                    # Only include stocks with valid numeric data
                    pe = stock_info.get('pe_ratio')
                    if pe != 'N/A' and pe is not None and isinstance(pe, (int, float)) and pe > 0:
                        # Calculate a fundamentals score (enhanced for future queries)
                        fundamentals_score = self._calculate_fundamentals_score(stock_info, analyst_data, is_future_query)
                        
                        stocks_data.append({
                            'ticker': ticker,
                            'company_name': stock_info.get('company_name', 'N/A'),
                            'pe_ratio': pe,
                            'forward_pe': stock_info.get('forward_pe', 'N/A'),
                            'peg_ratio': stock_info.get('peg_ratio', 'N/A'),
                            'profit_margin': stock_info.get('profit_margin', 'N/A'),
                            'roe': stock_info.get('roe', 'N/A'),
                            'roa': stock_info.get('roa', 'N/A'),
                            'debt_to_equity': stock_info.get('debt_to_equity', 'N/A'),
                            'current_ratio': stock_info.get('current_ratio', 'N/A'),
                            'market_cap': stock_info.get('market_cap', 'N/A'),
                            'price': stock_info.get('current_price', 'N/A'),
                            'target_price': analyst_data.get('target_price', 'N/A'),
                            'earnings_growth': analyst_data.get('earnings_growth', 'N/A'),
                            'revenue_growth': analyst_data.get('revenue_growth', 'N/A'),
                            'recommendation': analyst_data.get('recommendation', 'N/A'),
                            'sector': stock_info.get('sector', 'N/A'),
                            'fundamentals_score': fundamentals_score
                        })
                        valid_count += 1
                except Exception as e:
                    print(f"Error analyzing {ticker}: {e}")
                    continue
            
            if not stocks_data:
                return "I couldn't retrieve stock data at this time. Please try again later."
            
            # Sort by fundamentals score (if query is about best fundamentals or stocks to watch)
            if 'fundamental' in query.lower() or 'best' in query.lower() or 'watch' in query.lower():
                stocks_data.sort(key=lambda x: x.get('fundamentals_score', 0), reverse=True)
            
            # Get RAG context about fundamentals and forward-looking analysis
            rag_query = f"What are the best fundamental metrics for stock analysis? {query}"
            if is_future_query:
                rag_query += f" How to evaluate stocks for future investment in {target_year}?"
            rag_context, rag_sources = self._get_rag_context(rag_query, k=5)
            
            # Create detailed stocks summary with forward-looking metrics
            stocks_summary = "\n".join([
                f"{i+1}. {s['ticker']} - {s['company_name']} (Sector: {s['sector']})\n"
                f"   PE Ratio: {s['pe_ratio']}, Forward PE: {s['forward_pe']}, PEG: {s['peg_ratio']}\n"
                f"   Profit Margin: {s['profit_margin']}, ROE: {s['roe']}, ROA: {s['roa']}\n"
                f"   Debt/Equity: {s['debt_to_equity']}, Current Ratio: {s['current_ratio']}\n"
                f"   Earnings Growth: {s['earnings_growth']}, Revenue Growth: {s['revenue_growth']}\n"
                f"   Target Price: ${s['target_price']}, Recommendation: {s['recommendation']}\n"
                f"   Market Cap: {self._format_currency(s['market_cap'])}, Price: ${s['price']}\n"
                for i, s in enumerate(stocks_data[:limit])
            ])
            
            # Build prompt with forward-looking context
            year_context = ""
            if is_future_query:
                year_context = f"\n\nIMPORTANT: The user is asking about stocks to watch for {target_year}. Focus on:\n"
                year_context += "- Forward-looking metrics (Forward PE, PEG ratio, earnings/revenue growth)\n"
                year_context += "- Analyst recommendations and target prices\n"
                year_context += "- Growth potential and future prospects\n"
                year_context += "- Companies with strong fundamentals that are positioned for future growth\n"
            
            prompt = f"""You are a financial analyst helping to screen stocks based on fundamentals and future prospects.

USER QUERY: {query}
{year_context}

STOCKS DATA (analyzed {valid_count} stocks, showing top {min(limit, len(stocks_data))}):
{stocks_summary}

FINANCIAL ANALYSIS KNOWLEDGE BASE:
{rag_context if rag_context else "No specific context available."}

Based on the user's query and financial analysis principles from the knowledge base, identify and rank the top stocks that best match the criteria. 

For queries about "best fundamentals" or "stocks to watch", prioritize:
- Strong profitability (high profit margins, ROE, ROA)
- Reasonable valuation (not overvalued PE ratios, reasonable PEG, attractive Forward PE)
- Good financial health (low debt-to-equity, good current ratio > 1.0)
- Growth potential (positive earnings growth, revenue growth)
- Analyst support (favorable recommendations, target prices above current price)
- Consistent performance and market position

{"For future-year queries (like 2026), emphasize forward-looking metrics and growth potential." if is_future_query else ""}

IMPORTANT: When referencing information from the knowledge base sources, cite them using in-text citations in the format (Source 1), (Source 2), etc., corresponding to the source numbers provided in the context above.

Provide:
1. Top 5-10 stocks that best match the criteria (ranked)
2. Brief explanation of why each stock was selected
3. Key metrics for each recommended stock in a clear format
4. Forward-looking analysis if relevant (growth prospects, analyst outlook)
5. Any important considerations or risks
6. Sector diversification if relevant

Format the response clearly with rankings, use markdown tables where helpful, and provide actionable insights.
"""
            
            response = self.llm.invoke(prompt)
            answer = response.content
            
            # Append formal references if sources were used (only if not already present)
            if rag_sources and "## Works Cited" not in answer and "Works Cited" not in answer:
                answer += "\n\n## Works Cited\n"
                for source in rag_sources:
                    # Sources are already in MLA format from SOURCE_CITATIONS
                    answer += f"{source}\n"
            
            return answer
            
        except Exception as e:
            return f"Error screening stocks: {str(e)}"
    
    def _calculate_fundamentals_score(self, stock_info: Dict, analyst_data: Dict = None, is_future_query: bool = False) -> float:
        """
        Calculate a fundamentals score for ranking stocks.
        Higher score = better fundamentals.
        
        Args:
            stock_info: Dictionary of stock information
            analyst_data: Dictionary of analyst estimates and recommendations
            is_future_query: Whether this is a forward-looking query
        
        Returns:
            float: Fundamentals score
        """
        score = 0.0
        
        # Profitability metrics (higher is better)
        profit_margin = stock_info.get('profit_margin', 0)
        if profit_margin != 'N/A' and isinstance(profit_margin, (int, float)):
            score += profit_margin * 30  # Weight profit margin
        
        roe = stock_info.get('roe', 0)
        if roe != 'N/A' and isinstance(roe, (int, float)) and roe > 0:
            score += min(roe * 20, 20)  # Cap ROE contribution
        
        roa = stock_info.get('roa', 0)
        if roa != 'N/A' and isinstance(roa, (int, float)) and roa > 0:
            score += min(roa * 15, 15)  # Cap ROA contribution
        
        # Valuation metrics (lower PE is better, but not too low)
        pe = stock_info.get('pe_ratio', 999)
        if pe != 'N/A' and isinstance(pe, (int, float)) and 5 < pe < 50:
            score += 10  # Reasonable PE range
        
        # Forward PE (important for future queries)
        forward_pe = stock_info.get('forward_pe', 999)
        if forward_pe != 'N/A' and isinstance(forward_pe, (int, float)):
            if 5 < forward_pe < 50:
                score += 10
            if is_future_query and forward_pe < pe:  # Forward PE lower than trailing = positive signal
                score += 5
        
        # PEG ratio (lower is better, indicates growth at reasonable price)
        peg = stock_info.get('peg_ratio', 999)
        if peg != 'N/A' and isinstance(peg, (int, float)) and peg > 0:
            if peg < 1.0:
                score += 15  # Excellent PEG
            elif peg < 2.0:
                score += 10  # Good PEG
            elif peg < 3.0:
                score += 5  # Acceptable PEG
        
        # Financial health (lower debt is better)
        debt_equity = stock_info.get('debt_to_equity', 999)
        if debt_equity != 'N/A' and isinstance(debt_equity, (int, float)):
            if debt_equity < 50:
                score += 10
            elif debt_equity < 100:
                score += 5
        
        # Liquidity (higher current ratio is better)
        current_ratio = stock_info.get('current_ratio', 0)
        if current_ratio != 'N/A' and isinstance(current_ratio, (int, float)):
            if current_ratio > 1.5:
                score += 10
            elif current_ratio > 1.0:
                score += 5
        
        # Forward-looking metrics (important for future queries)
        if analyst_data and is_future_query:
            # Earnings growth
            earnings_growth = analyst_data.get('earnings_growth', 0)
            if earnings_growth != 'N/A' and isinstance(earnings_growth, (int, float)):
                if earnings_growth > 0.2:  # >20% growth
                    score += 15
                elif earnings_growth > 0.1:  # >10% growth
                    score += 10
                elif earnings_growth > 0:  # Positive growth
                    score += 5
            
            # Revenue growth
            revenue_growth = analyst_data.get('revenue_growth', 0)
            if revenue_growth != 'N/A' and isinstance(revenue_growth, (int, float)):
                if revenue_growth > 0.2:  # >20% growth
                    score += 10
                elif revenue_growth > 0.1:  # >10% growth
                    score += 7
                elif revenue_growth > 0:  # Positive growth
                    score += 3
            
            # Target price vs current price
            target_price = analyst_data.get('target_price', 'N/A')
            current_price = stock_info.get('current_price', 'N/A')
            if (target_price != 'N/A' and current_price != 'N/A' and 
                isinstance(target_price, (int, float)) and isinstance(current_price, (int, float)) and
                current_price > 0):
                upside = (target_price - current_price) / current_price
                if upside > 0.2:  # >20% upside
                    score += 10
                elif upside > 0.1:  # >10% upside
                    score += 7
                elif upside > 0:  # Positive upside
                    score += 5
        
        return score
    
    def is_screening_query(self, query: str) -> bool:
        """
        Determine if the query is asking for stock screening/comparison.
        Uses LLM for intelligent intent detection.
        
        Args:
            query: User's query
        
        Returns:
            bool: True if query is about screening stocks
        """
        try:
            prompt = f"""
                Determine whether the following user query is requesting *stock screening*, *stock recommendations*, or *comparison of multiple stocks*.

                A query IS a screening/recommendation query if it does ANY of the following:
                - Asks for multiple stocks (e.g., “best stocks”, “top companies”, “which stocks should I buy”)
                - Requests stock suggestions or ideas
                - Asks to compare two or more stocks
                - Seeks a list of stocks based on criteria (e.g., “cheap AI stocks”, “stocks with high dividend yield”)

                A query is NOT a screening/recommendation query (respond “NO”) if it is:
                - About financial concepts, definitions, or terminology (e.g., “what is short selling?”, “explain PE ratio”)
                - About how markets, trading, or metrics work
                - About a single stock or ticker (e.g., “analyze AAPL”, “is TSLA a buy?”)
                - A general investing or finance question not requesting multiple stocks

                USER QUERY: "{query}"

                Respond with ONLY:
                YES — if the user is asking for multiple stocks, stock screening, stock recommendations, or stock comparisons.
                NO — if the query is about a single stock, financial concepts, definitions, or general financial questions.

                Your answer:
            """
            
            response = self.llm.invoke(prompt)
            result = response.content.strip().upper()
            
            # Check if LLM said yes
            return result.startswith("YES") or result.startswith("TRUE") or "YES" in result
        except Exception as e:
            # If LLM call fails, default to False (not a screening query)
            print(f"Error in LLM screening detection: {e}")
            return False
    
    # ====================================================================
    # Financial Data Fetching Methods
    # ====================================================================
    
    def _get_stock_info(self, ticker: str) -> Dict:
        """Get comprehensive stock information for a given ticker."""
        try:
            # Validate ticker format
            if not ticker or len(ticker) > 5 or not ticker.isalpha():
                raise Exception(f"Invalid ticker symbol format: {ticker}")
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check if we got valid data (yfinance returns empty dict for invalid tickers)
            if not info or len(info) < 5:
                raise Exception(f"No data found for ticker: {ticker}. Please verify the ticker symbol is correct.")
            
            stock_info = {
                'ticker': ticker.upper(),
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'current_price': info.get('currentPrice', info.get('regularMarketPrice', 'N/A')),
                'previous_close': info.get('previousClose', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'volume': info.get('volume', 'N/A'),
                'average_volume': info.get('averageVolume', 'N/A'),
                '52_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
                '52_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'forward_pe': info.get('forwardPE', 'N/A'),
                'peg_ratio': info.get('pegRatio', 'N/A'),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'beta': info.get('beta', 'N/A'),
                'eps': info.get('trailingEps', 'N/A'),
                'revenue': info.get('totalRevenue', 'N/A'),
                'profit_margin': info.get('profitMargins', 'N/A'),
                'operating_margin': info.get('operatingMargins', 'N/A'),
                'roe': info.get('returnOnEquity', 'N/A'),
                'roa': info.get('returnOnAssets', 'N/A'),
                'debt_to_equity': info.get('debtToEquity', 'N/A'),
                'current_ratio': info.get('currentRatio', 'N/A'),
                'quick_ratio': info.get('quickRatio', 'N/A'),
                'book_value': info.get('bookValue', 'N/A'),
                'price_to_book': info.get('priceToBook', 'N/A'),
                'enterprise_value': info.get('enterpriseValue', 'N/A'),
                'ev_to_revenue': info.get('enterpriseToRevenue', 'N/A'),
                'ev_to_ebitda': info.get('enterpriseToEbitda', 'N/A'),
                'description': info.get('longBusinessSummary', 'N/A'),
                'website': info.get('website', 'N/A'),
                'employees': info.get('fullTimeEmployees', 'N/A'),
            }
            
            return stock_info
        except Exception as e:
            raise Exception(f"Error fetching stock info for {ticker}: {str(e)}")
    
    def _get_financial_statements(self, ticker: str) -> Dict:
        """Get financial statements (income statement, balance sheet, cash flow) for a ticker."""
        try:
            stock = yf.Ticker(ticker)
            
            return {
                'income_statement': stock.financials,
                'balance_sheet': stock.balance_sheet,
                'cashflow': stock.cashflow,
                'income_statement_quarterly': stock.quarterly_financials,
                'balance_sheet_quarterly': stock.quarterly_balance_sheet,
                'cashflow_quarterly': stock.quarterly_cashflow,
            }
        except Exception as e:
            raise Exception(f"Error fetching financial statements for {ticker}: {str(e)}")
    
    def _get_historical_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Get historical stock price data."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            return hist
        except Exception as e:
            raise Exception(f"Error fetching historical data for {ticker}: {str(e)}")
    
    def _get_analyst_data(self, ticker: str) -> Dict:
        """Get analyst estimates and recommendations."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'target_price': info.get('targetMeanPrice', 'N/A'),
                'target_high': info.get('targetHighPrice', 'N/A'),
                'target_low': info.get('targetLowPrice', 'N/A'),
                'recommendation': info.get('recommendationKey', 'N/A'),
                'number_of_analysts': info.get('numberOfAnalystOpinions', 'N/A'),
                'earnings_growth': info.get('earningsQuarterlyGrowth', 'N/A'),
                'revenue_growth': info.get('revenueGrowth', 'N/A'),
            }
        except Exception as e:
            return {}
    
    def _format_currency(self, value) -> str:
        """Format a number as currency."""
        if value == 'N/A' or value is None:
            return 'N/A'
        try:
            if isinstance(value, (int, float)):
                if abs(value) >= 1e9:
                    return f"${value/1e9:.2f}B"
                elif abs(value) >= 1e6:
                    return f"${value/1e6:.2f}M"
                elif abs(value) >= 1e3:
                    return f"${value/1e3:.2f}K"
                else:
                    return f"${value:,.2f}"
            return str(value)
        except:
            return str(value)
    
    def _format_number(self, value) -> str:
        """Format a number with commas."""
        if value == 'N/A' or value is None:
            return 'N/A'
        try:
            if isinstance(value, (int, float)):
                return f"{value:,.2f}"
            return str(value)
        except:
            return str(value)
    
    def _get_tickers_from_sp500(self) -> list:
        """Fetch S&P 500 tickers from Wikipedia."""
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url, attrs={'class': 'wikitable sortable'})
            sp500_table = tables[0]
            if 'Symbol' in sp500_table.columns:
                tickers = sp500_table['Symbol'].tolist()
                tickers = [str(ticker).replace('.', '-').strip() for ticker in tickers if pd.notna(ticker)]
                return tickers
            else:
                print("Warning: Could not find 'Symbol' column in S&P 500 table")
                return []
        except Exception as e:
            print(f"Error fetching S&P 500 tickers: {e}")
            return []
    
    def _get_tickers_from_nasdaq100(self) -> list:
        """Fetch NASDAQ 100 tickers from Wikipedia."""
        try:
            url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
            tables = pd.read_html(url)
            for table in tables:
                if 'Ticker' in table.columns or 'Symbol' in table.columns:
                    ticker_col = 'Ticker' if 'Ticker' in table.columns else 'Symbol'
                    tickers = table[ticker_col].tolist()
                    tickers = [str(ticker).strip() for ticker in tickers if pd.notna(ticker)]
                    return tickers
            print("Warning: Could not find ticker table in NASDAQ 100 page")
            return []
        except Exception as e:
            print(f"Error fetching NASDAQ 100 tickers: {e}")
            return []
    
    def _get_all_major_tickers(self, use_cache: bool = True) -> list:
        """Get tickers from major indices (S&P 500, NASDAQ 100) with caching."""
        # Check cache
        if use_cache and FinancialAnalysisAgent._tickers_cache is not None and FinancialAnalysisAgent._tickers_cache_time is not None:
            if time.time() - FinancialAnalysisAgent._tickers_cache_time < FinancialAnalysisAgent.CACHE_DURATION:
                return FinancialAnalysisAgent._tickers_cache
        
        all_tickers = []
        
        # Get S&P 500 tickers
        try:
            sp500_tickers = self._get_tickers_from_sp500()
            if sp500_tickers:
                all_tickers.extend(sp500_tickers)
                print(f"✅ Fetched {len(sp500_tickers)} S&P 500 tickers")
        except Exception as e:
            print(f"Warning: Could not fetch S&P 500 tickers: {e}")
        
        # Get NASDAQ 100 tickers
        try:
            nasdaq_tickers = self._get_tickers_from_nasdaq100()
            if nasdaq_tickers:
                all_tickers.extend(nasdaq_tickers)
                print(f"✅ Fetched {len(nasdaq_tickers)} NASDAQ 100 tickers")
        except Exception as e:
            print(f"Warning: Could not fetch NASDAQ 100 tickers: {e}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tickers = []
        for ticker in all_tickers:
            ticker_upper = str(ticker).upper().strip()
            if ticker_upper and ticker_upper not in seen:
                seen.add(ticker_upper)
                unique_tickers.append(ticker_upper)
        
        print(f"✅ Total unique tickers: {len(unique_tickers)}")
        
        # Update cache
        FinancialAnalysisAgent._tickers_cache = unique_tickers
        FinancialAnalysisAgent._tickers_cache_time = time.time()
        
        return unique_tickers
    
    def _get_popular_tickers(self) -> list:
        """Get a list of stock tickers from major indices for screening."""
        try:
            tickers = self._get_all_major_tickers()
            if tickers:
                return tickers
        except Exception as e:
            print(f"Warning: Could not fetch tickers from indices: {e}")
            print("Falling back to curated list...")
        
        # Fallback to curated list if web scraping fails
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
            'V', 'JNJ', 'WMT', 'PG', 'JPM', 'MA', 'UNH', 'HD', 'DIS', 'BAC',
            'XOM', 'CVX', 'ABBV', 'PFE', 'AVGO', 'COST', 'MRK', 'PEP', 'TMO',
            'ABT', 'CSCO', 'ACN', 'ADBE', 'NFLX', 'CMCSA', 'NKE', 'PM', 'LIN',
            'TXN', 'QCOM', 'INTU', 'AMGN', 'HON', 'ISRG', 'AMAT', 'VZ', 'RTX'
        ]
    
    # ====================================================================
    # Visualization Methods
    # ====================================================================
    
    def _create_price_chart(self, historical_data: pd.DataFrame, ticker: str) -> str:
        """Create a price chart for the stock."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
        
        # Price chart
        ax1.plot(historical_data.index, historical_data['Close'], label='Close Price', linewidth=2)
        ax1.fill_between(historical_data.index, historical_data['Low'], historical_data['High'], 
                         alpha=0.3, label='High-Low Range')
        ax1.set_title(f'{ticker} Stock Price History', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # Volume chart
        ax2.bar(historical_data.index, historical_data['Volume'], alpha=0.6, color='orange')
        ax2.set_title('Trading Volume', fontsize=12)
        ax2.set_ylabel('Volume', fontsize=10)
        ax2.set_xlabel('Date', fontsize=10)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return img_base64
    
    def _create_metrics_comparison_chart(self, metrics: dict, ticker: str) -> Optional[str]:
        """Create a bar chart comparing key financial metrics."""
        metric_labels = []
        metric_values = []
        
        metrics_to_show = {
            'PE Ratio': metrics.get('pe_ratio'),
            'Forward PE': metrics.get('forward_pe'),
            'PEG Ratio': metrics.get('peg_ratio'),
            'Price to Book': metrics.get('price_to_book'),
            'Beta': metrics.get('beta'),
        }
        
        for label, value in metrics_to_show.items():
            if value != 'N/A' and value is not None:
                try:
                    # Ensure value is numeric before conversion
                    if isinstance(value, (int, float, str)):
                        float_val = float(value)
                        # Check for valid numeric values
                        if isinstance(float_val, (int, float)) and np.isfinite(float_val):
                            metric_labels.append(label)
                            metric_values.append(float_val)
                except (ValueError, TypeError, OverflowError):
                    pass
        
        if not metric_values:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(metric_labels, metric_values, color='steelblue', alpha=0.7)
        ax.set_title(f'{ticker} - Key Valuation Metrics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return img_base64
    
    def _create_financial_statement_chart(self, financial_data: pd.DataFrame, title: str, metric_name: str) -> Optional[str]:
        """Create a chart for financial statement data."""
        if financial_data is None or financial_data.empty:
            return None
        
        try:
            cols = financial_data.columns[:4]  # Last 4 years
            if len(cols) == 0:
                return None
            
            if metric_name in financial_data.index:
                values = financial_data.loc[metric_name, cols]
            else:
                values = financial_data.iloc[0, :len(cols)]
            
            values = pd.to_numeric(values, errors='coerce')
            values = values.dropna()
            
            if len(values) == 0:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(range(len(values)), values, color='green', alpha=0.7)
            ax.set_title(f'{title} - {metric_name}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Amount ($)', fontsize=12)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels([str(col)[:10] for col in values.index], rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Format y-axis as currency
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e9:.1f}B' if abs(x) >= 1e9 else f'${x/1e6:.1f}M'))
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if abs(height) >= 1e9:
                    label = f'${height/1e9:.2f}B'
                elif abs(height) >= 1e6:
                    label = f'${height/1e6:.2f}M'
                else:
                    label = f'${height/1e3:.2f}K'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        label,
                        ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            return img_base64
        except Exception as e:
            print(f"Error creating financial statement chart: {e}")
            return None
    
    def _create_profitability_chart(self, metrics: dict) -> Optional[str]:
        """Create a chart showing profitability metrics."""
        profitability_metrics = {
            'Profit Margin': metrics.get('profit_margin'),
            'Operating Margin': metrics.get('operating_margin'),
            'ROE': metrics.get('roe'),
            'ROA': metrics.get('roa'),
        }
        
        labels = []
        values = []
        
        for label, value in profitability_metrics.items():
            if value != 'N/A' and value is not None:
                try:
                    # Ensure value is numeric before conversion
                    if isinstance(value, (int, float, str)):
                        float_val = float(value) * 100  # Convert to percentage
                        # Check for valid numeric values
                        if isinstance(float_val, (int, float)) and np.isfinite(float_val):
                            labels.append(label)
                            values.append(float_val)
                except (ValueError, TypeError, OverflowError):
                    pass
        
        if not values:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, values, color='green', alpha=0.7)
        ax.set_title('Profitability Metrics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return img_base64

