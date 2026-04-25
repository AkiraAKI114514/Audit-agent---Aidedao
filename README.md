# AI-Powered Multi-Agent Financial Auditing System

![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![AI Framework](https://img.shields.io/badge/Agent-Multi--Agent--Architecture-red)
![UI Framework](https://img.shields.io/badge/GUI-CustomTkinter-orange)

An advanced financial risk auditing platform that leverages a **Multi-Agent Systems (MAS)** architecture to automate the parsing, analysis, and risk assessment of corporate financial reports (PDFs).

---

## 📑 Overview

This system automates the traditional auditing workflow by deploying specialized AI agents. It can process complex financial documents, calculate key financial indicators, perform industry benchmarking (via Tavily search), and generate comprehensive risk assessment reports with anomaly detection.

###  The Agent Team
- **Data Agent**: Extracts and cleans structured financial data from raw PDF documents.
- **Analysis Agent**: Calculates financial ratios (Liquidity, Debt, Profitability) and identifies trends.
- **Risk Agent**: Evaluates risk levels using statistical methods (Z-Scores) and qualitative logic.
- **Audit Agent**: Finalizes the audit opinion and provides strategic recommendations.
- **Benchmarker**: Fetches industry-wide data for horizontal comparison.

---

##  Key Features

- **Automated PDF Parsing**: Handles complex financial tables and text using `pdfplumber`.
- **Statistical Anomaly Detection**: Calculates Z-Scores to identify data outliers compared to industry standards.
- **Real-time Web Research**: Integrated with **Tavily API** for live industry benchmarking.
- **Modern GUI**: A user-friendly console built with `CustomTkinter` for managing audit tasks, selecting files, and monitoring real-time logs.
- **LLM Agnostic**: Optimized for high-performance models like DeepSeek (via SiliconFlow) and OpenAI.

---

##  Project Structure

```text
.
├── t1.py                # Core Multi-Agent logic and Pipeline definition
├── gui_runner.py        # GUI interface and thread management
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```
## Installations
- **pip install request tavity pdfplumber openai

## Run code
- ** Engine test: ```python t1.py "company_name" --pdf --peer-pdf --ai-api-url --ai-api-key --model --tc-api-key```
- ** Complete application(+gui): ```python gui_runner.py```
