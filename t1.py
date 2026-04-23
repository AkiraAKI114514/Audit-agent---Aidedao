from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from openai import OpenAI
from tavily import TavilyClient

import statistics
import pdfplumber
import requests


NUMERIC_RE = re.compile(r"([+-]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?)")
CURRENCY_WORDS = ["元", "亿", "万", "USD", "CNY", "RMB", "美元"]


@dataclass
class AuditResult:
    company: str
    source: str
    financials: dict[str, Any] = field(default_factory=dict)
    indicators: dict[str, Any] = field(default_factory=dict)
    comparisons: dict[str, Any] = field(default_factory=dict)
    ind_mean: dict[str, Any] = field(default_factory=dict)
    ind_std: dict[str, Any] = field(default_factory=dict)
    z_scores: dict[str, Any] = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)
    risk_scores: dict[str, float] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    report: str = ""


class LLMService:
    def __init__(self, api_url: str, api_key: str):
        self.api_key = api_key
        self.api_url = api_url

        self.client = OpenAI(
            api_key = api_key,
            base_url= api_url
        )

    def call(self, Response_format: dict, model: str) -> str:
        if not self.api_key:
            raise RuntimeError("缺少 AI API Key，无法调用模型服务。")
        time.sleep(5)
        max_entries = 3
        for i in range(max_entries):
            try:
                response = self.client.chat.completions.create(
                    model= model,
                    messages= Response_format,
                    stream= False,
                )
                print(f"[CAUSION] 本次消耗 Token 总数: {response.usage.total_tokens}")
                return response.choices[0].message.content
                # response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
                # response.raise_for_status()

            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    print(f">>> 触发 TPM 限制，等待 5 秒后进行第 {i+1} 次重试...")
                    time.sleep(10) # 触发限制后多等一会儿
                else:
                    raise e
                    print("[ERROR] API 调用失败。")
        raise RuntimeError("多次重试后仍触发 TPM 限制，请稍后再试。")


class DataAgent:
    def __init__(self, api_url: str | None = None, llm_service: LLMService | None = None, model: str | None = None):
        self.api_url = api_url
        self.llm_service = llm_service
        self.model = model

    def fetch_from_api(self, api_url: str, company: str) -> dict[str, Any]:
        response = requests.get(api_url, params={"company": company}, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {"raw": data}

    def extract_text_from_pdf(self, pdf_path: Path) -> list[str]:
        text_parts_box: list[str] = []
        print(f">>> 开始解析 PDF 文件: {pdf_path.name}") # <--- 新增
        # 尝试使用 pdfplumber 增强表格读取能力
        with pdfplumber.open(pdf_path) as pdf:
            table_num = 0
            page_mark = 1
            print(f">>> 总页数: {len(pdf.pages)}，正在逐页提取表格和文本...")
            for page in pdf.pages:
                if page_mark%100 == 1:
                    text_parts_box.append([])
                    text_parts = text_parts_box[-1]

                text_parts.append(f"\n<PAGE{page_mark}>")
                # 1. 获取表格占据的矩形区域
                table_objects = page.find_tables()
                table_bboxes = [t.bbox for t in table_objects]

                def is_not_in_table(obj):
                    # obj 是一个字符对象，它有 x0, top, x1, bottom 等坐标
                    obj_bbox = (obj["x0"], obj["top"], obj["x1"], obj["bottom"])
                    for tb in table_bboxes:
                        # 如果字符的坐标在表格矩形内，则排除
                        if (obj_bbox[0] >= tb[0] and obj_bbox[1] >= tb[1] and 
                            obj_bbox[2] <= tb[2] and obj_bbox[3] <= tb[3]):
                            return False
                    return True
                # 使用 filter 过滤掉在表格内的字符，然后再提取文本
                page_text = page.filter(is_not_in_table).extract_text()
                text_parts.append(page_text or "")

                # 2. 提取表格
                tables = page.extract_tables()
                for table in tables:
                    row_num = 0
                    text_parts.append(f"<TABLE{table_num}>")
                    for row in table:
                        # 处理每一行，将单元格连接起来
                        col_num = 0
                        row_string = ""
                        for c in row:
                            if c:
                                content = f"[{row_num}/{col_num}]"+ str(c).replace("\n","")
                            else:
                                content = f"[{row_num}/{col_num}]"+ ""
                            row_string += content+" "
                            col_num += 1

                        text_parts.append(row_string)
                        row_num += 1
                    text_parts.append(f"</TABLE{table_num}>")
                    table_num += 1
                
                text_parts.append(f"</PAGE{page_mark}>")
                page_mark += 1
            print(f"    - {pdf_path.name} 解析完成。")
        for i in range(len(text_parts_box)):
            text_parts_box[i] = "\n".join(text_parts_box[i])
        return text_parts_box

    def parse_numeric_values(self, text: list[str], buffer: dict[str, Any]) -> dict[str, Any]:
        #在处理中文财务术语（如“扣非净利润”、“加权ROE”）方面，国产模型有天然的语料优势，且性价比极高
        model = self.model
        prompt = [
                    {
                        "role": "system",
                        "content": (
                            "你是一位资深的注册会计师（CPA）和数据提取专家。你的任务是从结构化的 PDF 坐标文本中精准提取财务数据。\n"
                            "你具备极强的语义理解能力，能够识别会计科目的同义词（如“营业总收入”等同于“营业收入”）。\n"
                            "仅返回 JSON 格式内容，不含解释，不包含反斜杠，不换行。严格按照如下格式：\n"
                            "输出数字位数不用逗号分隔\n"
                            "{\"year\": \"\", \"revenue\": 0.0, \"net_profit\": 0.0, \"total_assets\": 0.0, \"total_liabilities\": 0.0, \"current_liabilities\": 0.0, \"equity\": 0.0, \"accounts_receivable\": 0.0, \"current_assets\": 0.0, \"inventory\": 0.0, \"cogs\": 0.0, \"cash\": 0.0}\n"
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            "# 任务目标\n"
                            "从下方的 <PDF_DATA> 中提取指标，要求：\n"
                            "1. 统一单位：若原文为“万元”或“亿元”，请自动换算为“元”（浮点数）。\n"
                            "2. 数据清洗：识别括号负数（如 (100) -> -100），剔除千分位逗号。\n"
                            "3. 字段映射：\n"
                            "   - year: 直接输出年份\n"
                            "   - revenue: 营业总收入\n"
                            "   - net_profit: 归属于母公司股东的净利润\n"
                            "   - total_assets: 总资产\n"
                            "   - total_liabilities: 总负债\n"
                            "   - current_liabilities 流动负债\n"
                            "   - equity: 归属于母公司股东的所有者权益合计\n"
                            "   - accounts_receivable: 应收账款\n"
                            "   - current_assets: 流动资产\n"
                            "   - inventory: 存货流动资产\n"
                            "   - cogs: Cost of Goods Sold，销售成本或营业成本\n"
                            "   - cash: 期末现金及现金等价物余额\n"
                            "4. 要求数据类型：\n"
                            "   - year: str\n"
                            "   - revenue: float\n"
                            "   - net_profit: float\n"
                            "   - total_assets: float\n"
                            "   - total_liabilities: float\n"
                            "   - current_liabilities float\n"
                            "   - equity: float\n"
                            "   - accounts_receivable: float\n"
                            "   - current_assets: float\n"
                            "   - inventory: float\n"
                            "   - cogs: float\n"
                            "   - cash: float\n"
                            "5. 注意，pdf可能因过长而被拆分，给出的可能为不完整pdf，将提供已经找到并整理好的数据（json），需要鉴别并补充。"
                            
                            "# 输出要求\n"
                            "仅返回 JSON 格式内容，不含解释，不包含反斜杠，不换行。严格按照如下格式：\n"
                            "输出数字位数不用逗号分隔\n"
                            "{\"year\": \"\", \"revenue\": 0.0, \"net_profit\": 0.0, \"total_assets\": 0.0, \"total_liabilities\": 0.0, \"current_liabilities\": 0.0, \"equity\": 0.0, \"accounts_receivable\": 0.0, \"current_assets\": 0.0, \"inventory\": 0.0, \"cogs\": 0.0, \"cash\": 0.0}\n\n"
                            
                            "# 数据提供\n"
                            "<Data found from privious pages>\n"
                            f"{buffer}\n"
                            "</Data found from privious pages>\n\n"
                            "<PDF_DATA>\n"
                            f"{text}\n"  # 这里填入你处理过的 [r/c] 格式文本
                            "</PDF_DATA>"
                        )
                    }
                ]
        metrics = self.llm_service.call(prompt, model)
        #print(metrics)
        metrics = json.loads(metrics)
        return metrics

    def _extract_first_number(self, text: str) -> float:
        match = NUMERIC_RE.search(text)
        if not match:
            return 0.0
        value = match.group(1).replace(",", "")
        try:
            return float(value)
        except ValueError:
            return 0.0

    def build_financial_profile(self, source: str, company: str, pdf_paths: list[Path] | None = None, api_data: dict[str, Any] | None = None) -> dict[str, Any]:

        profile = {
            "company": company,
            "source": source,
            "raw": {},
            "financials": {}
        }

        if api_data:
            profile["raw"] = api_data
            if "financials" in api_data and isinstance(api_data["financials"], dict):
                profile["financials"] = api_data["financials"]
            else:
                profile["financials"] = self.parse_numeric_values(json.dumps(api_data, ensure_ascii=False))
        elif pdf_paths:
            for pdf_path in pdf_paths:
                textnum = 0
                buffer = {}
                raw_text = self.extract_text_from_pdf(pdf_path)
                for text_part in raw_text:
                    print(f">>> 正在将第 {textnum + 1} 块数据发送至 LLM 进行财务指标清洗...")
                    profile["raw"][f"text{textnum}"] = text_part[:4000]
                    buffer = self.parse_numeric_values(text_part,buffer=buffer)
                    textnum += 1 
                profile["financials"][buffer.get("year")] = buffer
                
            
        else:
            profile["financials"] = {}

        return profile


class AnalysisAgent:
    def __init__(self, llm_service: LLMService | None = None, model: str | None = None):
        self.llm_service = llm_service
        self.model = model
    def extract_indicators(self, financials: dict[str, Any]) -> dict[str, Any]:
        indicator_dict: dict[str, dict] = {}
        for fin_data in financials:
            revenue = float(financials[fin_data].get("revenue", 0.0) or 0.0)
            net_profit = float(financials[fin_data].get("net_profit", 0.0) or 0.0)
            total_assets = float(financials[fin_data].get("total_assets", 0.0) or 0.0)
            total_liabilities = float(financials[fin_data].get("total_liabilities", 0.0) or 0.0)
            current_liabilities = float(financials[fin_data].get("current_liabilities", 0.0) or 0.0)
            equity = float(financials[fin_data].get("equity", 0.0) or 0.0)
            cash = float(financials[fin_data].get("cash", 0.0) or 0.0)
            cogs = float(financials[fin_data].get("cogs", 0.0) or 0.0)
            inventory = float(financials[fin_data].get("inventory", 0.0) or 0.0)
            current_assets = float(financials[fin_data].get("current_assets", 0.0) or 0.0)
            accounts_receivable = float(financials[fin_data].get("accounts_receivable", 0.0) or 0.0)
            year = str(financials[fin_data].get("year",""))
            net_working_capital:float = current_assets - current_liabilities


            indicator: dict[str, Any] = {
                "year": year,
                "revenue": revenue,
                "net_profit": net_profit,
                "profit_margin": (net_profit / revenue * 100) if revenue else 0.0,
                "gross_profit_margin": (revenue - cogs)/revenue if revenue else 0.0,
                "debt_ratio": (total_liabilities / total_assets * 100) if total_assets else 0.0,
                "return_on_assets": (net_profit / total_assets) if total_assets else 0.0,
                "liquidity_ratio": (net_working_capital + current_liabilities)/ current_liabilities if current_liabilities else 0.0,
                "days_sales_outstanding": (accounts_receivable / revenue) * 365 if revenue else 0.0,
                "days_inventory_outstanding": (inventory / cogs) * 365 if cogs else 0.0,
                "equity_ratio": (equity / total_assets) if total_assets else 0.0,
                "roe": (net_profit / equity * 100) if equity else 0.0,
                "cash_to_liabilities": (cash / total_liabilities * 100) if total_liabilities else 0.0,
            }

            indicator_dict[indicator.get("year","")] = indicator

        return indicator_dict

    def industry_comparison(self, indicators: dict[str, float], ind_means: dict[str, float] | None = None) -> dict[str, Any]:
        benchmark = ind_means if ind_means else {
            "profit_margin": 12.0,
            "debt_ratio": 55.0,
            "roe": 10.0,
            "cash_to_liabilities": 25.0,
        }
        comparison: dict[str, Any] = {}
        for year, dict0 in indicators.items():
            for key, value in dict0.items():
                if key in benchmark[year]:
                    comparison.setdefault(year, {})[key] = {
                        "value": round(value, 2),
                        "benchmark": benchmark[year][key],
                        "status": "低" if value < benchmark[year][key] else "高" if value > benchmark[year][key] else "持平",
                    }
        return comparison

    def generate_analysis_report(self, result: AuditResult) -> str:
        model = self.model
        prompt = [
                    {
                        "role": "system",
                        "content": (
                            "你是一位资深的财务分析师，擅长通过财务数据透视企业的经营风险与盈利质量。\n"
                            "以专业的语言严格生成分析内容，不要加信件heading，不要输出表格。"
                            )
                        
                    },
                    {
                        "role": "user",
                        "content": (
                            "# Context\n"
                            "请针对以下公司的财务表现并根据行业对比结果撰写一份简要分析报告。\n\n"

                            "# Input Data\n"
                            "# 数据字典严格按照tag年份/季度排序\n"
                            f"- 年份/季度：{[key for key in result.indicators]}"
                            f"- 公司名称：{result.company}\n"
                            f"- 营业收入；净利润；净利率；资产负债率；ROE；核心指标与行业对比： {result.indicators}\n"
                            f"- 核心指标与行业对比：{result.comparisons}\n\n"

                            "# Report Requirements\n"
                            "请按以下结构撰写（字数控制在 500 字以内）：\n"
                            "1. 经营概况：分析营收与利润规模，评价盈利能力（基于 ROE 和净利率）。\n"
                            "2. 财务稳健性：结合资产负债率和现金流覆盖情况，评估偿债风险。\n"
                            "3. 基于本公司财务表现给出分析结论。\n\n"
                            "# Tone\n"
                            "专业、客观、言简意赅。避免口水话，多使用“财务韧性”、“资本结构优化”、“盈利质量”等专业词汇\n"
                        )
                    }
                ]

        if self.llm_service:
            return self.llm_service.call(prompt, model)


class RiskAgent:
    def __init__(self, anomaly_thresholds: dict[str, float] | None = None, llm_service: LLMService | None = None, model: str | None = None):
        self.thresholds = anomaly_thresholds or {
            "profit_margin": 5.0,
            "debt_ratio": 65.0,
            "roe": 5.0,
            "cash_to_liabilities": 10.0,
        }
        self.llm_service = llm_service
        self.model = model

    def detect_anomalies(self, indicators: dict[str, dict], ind_mean: dict[str, dict]) -> dict[str, list]:
        anomalies: dict[str,list[str]] = {}
        for year, dict0 in indicators.items():
            if anomalies.get("year", None) == None:
                anomalies[year] = []
            if dict0.get("profit_margin", 0.0) < ind_mean[year]["profit_margin_mean"]:
                anomalies[year].append("净利率低于行业警戒线，存在盈利能力不足风险。")
            if dict0.get("debt_ratio", 0.0) > ind_mean[year]["debt_ratio_mean"]:
                anomalies[year].append("资产负债率偏高，可能存在偿债压力。")
            if dict0.get("roe", 0.0) < ind_mean[year]["roe_mean"]:
                anomalies[year].append("股本回报率偏低，资本使用效率不佳。")
            if dict0.get("cash_to_liabilities", 0.0) < ind_mean[year]["cash_to_liabilities_mean"]:
                anomalies[year].append("现金储备不足以覆盖负债，流动性风险较高。")

        return anomalies

    def score_risk(self, indicators: dict[str, float], anomalies: dict[str,list[str]], z_scores: dict[str, dict]) -> dict[str, Any]:
        risk_scores: dict[str, float] = {}
        model = self.model
        system_prompt = [
            {
                "role": "system",
                "content": (
                    "# Role\n"
                    "你是一位资深财务审计专家系统，专注于通过定量指标分析识别公司财务造假及经营风险。\n\n"
                
                    "# Task\n"
                    "根据用户提供的多年公司指标信息及 Z-Score，按照特定的【三层加权风险评分体系】计算综合风险分。\n\n"
                
                    "# Scoring Logic\n"
                    "每个维度的子分计算逻辑如下：\n"
                    "Score = min(100, Z_Base + Trend_Penalty + Logic_Penalty)\n\n"
                    
                    "1. Z_Base (基础分): \n"
                       "- Z-Score < 1.8 (高危): 50-60分\n"
                       "- 1.8 <= Z-Score < 3.0 (预警): 20-40分\n"
                       "- Z-Score >= 3.0 (健康): 0-10分\n"
                    "2. Trend_Penalty (趋势异常 +25分): \n"
                       "- 关键指标同比恶化超过 30%，或连续两年向差。\n"
                    "3. Logic_Penalty (关联矛盾 +25分): \n"
                       "- 识别红旗信号（如：净利润增长但现金流为负；收入增长但税收下降；大存大贷等）。\n"
                       "- 只有在证据充分时才允许确认关联矛盾\n\n"
                
                    "# Weights(严格要求以下权重)\n"
                    "- 偿债风险 (30%): 资产负债率、流动比率\n"
                    "- 盈利质量 (25%): 净利润率、ROA、现金流背离\n"
                    "- 营运效率 (20%): 应收账款天数、存货天数趋势\n"
                    "- 结构稳定性 (15%): 权益比率、NWC 趋势\n"
                    "- 成长性风险 (10%): 收入增速、毛利率突变\n\n"
                
                    "# Output Format (Strict JSON)\n"
                    "- 输出数字位数不用逗号分隔\n"
                    "- 必须直接返回 JSON 格式，无换行符，无反斜杠，包含以下字段：\n"
                    "{\n"
                    "  \"dimensions\": {\n"
                    "    \"solvency\": {\"score\": 0, \"reason\": \"\"},\n"
                    "    \"profitability\": {\"score\": 0, \"reason\": \"\"},\n"
                    "    \"efficiency\": {\"score\": 0, \"reason\": \"\"},\n"
                    "    \"stability\": {\"score\": 0, \"reason\": \"\"},\n"
                    "    \"growth\": {\"score\": 0, \"reason\": \"\"}\n"
                    "  },\n"
                    "  \"total_risk_score\": 0,\n"
                    "  \"audit_conclusion\": \"核心风险点总结\",\n"
                    "  \"red_flags\": [\"信号1\", \"信号2\"]\n"
                    "}\n"
                    )
            },
            {
                "role": "user",
                "content": (
                    "#数据:\n"
                    f"公司指标：{indicators}\n"
                    f"异常情况：{anomalies}"
                    f"Z_scores: {z_scores}\n\n"

                    "# Output Format (Strict JSON)\n"
                    "- 输出数字位数不用逗号分隔\n"
                    "- 必须直接返回 JSON 格式，无换行符，无反斜杠，包含以下字段：\n"
                    "{\n"
                    "  \"dimensions\": {\n"
                    "    \"solvency\": {\"score\": 0, \"reason\": \"\"},\n"
                    "    \"profitability\": {\"score\": 0, \"reason\": \"\"},\n"
                    "    \"efficiency\": {\"score\": 0, \"reason\": \"\"},\n"
                    "    \"stability\": {\"score\": 0, \"reason\": \"\"},\n"
                    "    \"growth\": {\"score\": 0, \"reason\": \"\"}\n"
                    "  },\n"
                    "  \"total_risk_score\": 0,\n"
                    "  \"audit_conclusion\": \"核心风险点总结\",\n"
                    "  \"red_flags\": [\"信号1\", \"信号2\"]\n"
                    "}\n"
                )
            }
        ]
        print(f">>> 正在将数据发送至 LLM 进行风险分数计算...")
        metrics= self.llm_service.call(system_prompt, model)
        risk_scores = json.loads(metrics)
        return risk_scores
    
    def z_scores(self, indicators: dict[str, float], ind_mean: dict[str, float] | None = None, ind_std: dict[str, float] | None = None) -> dict[str, dict]:
        z_scores: dict[str, dict] = {}

        if ind_std != None and ind_mean != None:
            for year in ind_mean:
                for key in ind_mean[year]:
                    extent_key = "_mean"
                    key = key.replace(extent_key,"")
                    key = key.strip()
                    value = indicators[year].get(key, 0.0)
                    z_scores.setdefault(year, {})[key] = (value - ind_mean[year].get(key+"_mean", 0.0)) / max(ind_std[year].get(key+"_std", 1.0), 0.0001)

        return z_scores


class AuditAgent:
    def __init__(self, llm_service: LLMService | None = None, model: str | None = None, tc_api_key: str | None = None):
        self.llm_service = llm_service
        self.model = model
        self.tc_key = tc_api_key
    def generate_recommendations(self, result: AuditResult) -> dict[str, list[str]]:
        recommendations: dict[str, list[str]] = {}
        for year in result.indicators:
            recommendations.setdefault(year, [])
            if result.indicators[year].get("profit_margin", 0.0) < result.ind_mean[year].get("profit_margin_mean", 0.0):
                recommendations[year].append("检查成本结构与毛利水平，评估可提升利润率的环节。")
            if result.indicators[year].get("debt_ratio", 0.0) > result.ind_mean[year].get("debt_ratio_mean", 0.0):
                recommendations[year].append("审查负债组合与偿债计划，关注短期借款与应付债务。")
            if result.indicators[year].get("cash_to_liabilities", 0.0) < result.ind_mean[year].get("cash_to_liabilities_mean", 0.0):
                recommendations[year].append("强化现金流管理，评估应收账款和存货周转情况。")
            if result.indicators[year].get("liquidity_ratio", 1) < 1:
                recommendations[year].append("预示流动性危机。")
            if result.indicators[year].get("days_sales_outstanding", 0.0) > result.ind_mean[year].get("days_sales_outstanding_mean", 0.0):
                recommendations[year].append("应收账款周转天数大于行业平均，须确认收入风险。")
            if result.indicators[year].get("days_inventory_outstanding", 0.0) > result.ind_mean[year].get("days_inventory_outstanding_mean", 0.0):
                recommendations[year].append("存货周转天数大于行业平均，须确认存货减值或虚增风险。")
            if not recommendations[year]:
                recommendations[year].append("当前指标整体稳定，建议继续关注业务增长与现金流质量。 ")
        return recommendations
 
    def build_audit_report(self, result: AuditResult) -> str:
        if self.tc_key:
            tc_client = TavilyClient(api_key=self.tc_key)
            searching_response = tc_client.search(
                query=f"{result.company} 舆论",
                include_answer="advanced",
                search_depth="advanced",
                max_results=5,
                chunks_per_source=3
            )
            if searching_response.get("answer", None) != None:
                print("【网络舆论搜索结果】")
                print(searching_response.get("answer", ""))
        else:
            searching_response = ["未开启网络搜索，请无视网络搜索要求"]
        if self.llm_service:
            model = self.model
            prompt = [
                        {
                            "role": "system",
                            "content": (
                                "你是一位资深注册会计师（CPA）。你擅长通过财务数据的异常波动发现企业潜在的经营风险、财务造假或管理漏洞。你的语言风格稳重、严谨、客观。\n"
                                "以专业的语言严格生成报告内容，不要加信件heading，不要输出表格，尽可能找出数据中的异常点并加以分析。\n"
                                "# 要求：\n"
                                "1. 不得直接下结论，必须先给出可能的正常解释\n"
                                "2. 每个异常必须对应一个财务报表认定（assertion）\n" 
                                "3. 所有风险判断需说明对重大错报的影响\n"
                                "4. 不得使用绝对性语言（如“必然”“确定”）\n"
                                "5. 高度疑似 / 系统性问题 / 重大风险等决定性词语只有在证据充分时才允许，因此单个风险的分析在证据不足时必须使用限制性表达\n"
                                "6. 严格按照以下结构输出\n\n"
                                "# 格式：\n"
                                """一、整体风险结论（Overall Risk Conclusion）
                                风险等级：（Low / Medium / High）
                                核心判断：
                                （一句话总结整体风险，必须统一，不允许自相矛盾）

                                主导风险因素（Top Risk Drivers）：
                                1.
                                2.
                                3.

                                ⸻

                                二、关键异常识别（Key Anomalies Identified）

                                异常 1：
                                指标表现：（描述数据及趋势）
                                可能的正常解释：
                                （如产品结构变化、成本上升等）
                                潜在风险解释：
                                （如收入确认问题、费用操纵等）
                                影响的审计认定（Assertions）：
                                （如 Cut-off、Completeness、Valuation 等）
                                风险等级：（High / Medium / Low）

                                异常 2：
                                （同上结构重复）

                                ⸻

                                三、风险优先级排序（Risk Prioritization）

                                风险事项：
                                现金影响：（High / Medium / Low）
                                持续经营影响：（High / Medium / Low）
                                认定影响：（High / Medium / Low）
                                综合优先级：（High / Medium / Low 或星级）

                                （按风险逐条列出）

                                优先级结论：
                                当前最优先审计领域为：XXX
                                原因：对现金流 / 持续经营 / 报表认定影响最大

                                ⸻

                                四、重大错报风险映射（RMM Mapping）

                                风险领域：
                                可能错报类型：（如提前确认收入、减值不足等）
                                涉及科目：（如收入、应收账款、存货等）
                                风险等级：（High / Medium / Low）

                                （按风险逐条列出）

                                ⸻

                                五、审计程序建议（Audit Procedures）

                                针对风险 1：
                                审计目标（Assertion）：（如 Cut-off、Existence 等）
                                建议程序：
                                1.
                                2.
                                3.

                                针对风险 2：
                                审计目标（Assertion）：
                                建议程序：
                                1.
                                2.
                                3.

                                ⸻

                                六、持续经营与财务稳定性评估（Going Concern Assessment）

                                当前判断：
                                （避免绝对结论，如“存在重大疑虑”，除非证据充分）

                                潜在风险信号：
                                1.
                                2.

                                审计建议：
                                1.
                                2.

                                ⸻

                                七、趋势与未来风险预测（Forward-looking Risks）

                                未来可能风险：
                                1.
                                2.

                                盈余管理动机：
                                （如利润波动大、存在平滑动机等）

                                ⸻

                                八、审计结论影响（Audit Implication）

                                （说明这些风险如果无法获取充分审计证据，会如何影响审计报告）

                                示例句：
                                若上述风险未能获取充分、适当的审计证据支持，可能对审计意见产生影响，包括增加关键审计事项或在极端情况下影响审计意见类型"""
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                "请基于以下风险评分、行业基准对比及异常检测结果(如果给出）和财务分析结果撰写审计建议：\n"
                                "# 输入数据 (Context)\n"
                                "数据为多年数据组成的字典，键为年份或季度\n"
                                f"- 目标公司：{result.company}\n"
                                f"- 核心指标 (Indicators): {result.indicators}\n"
                                f"- 行业基准偏离度 (Z-Scores): {result.z_scores}\n"
                                f"- 自动检测到的异常项 (Anomalies): {result.anomalies}\n"
                                f"- 综合风险评分: {result.risk_score}\n\n"

                                "# 审计逻辑参考 (加分点)\n"
                                "- 若 |Z-Score| > 2.0，视为显著统计学异常，必须在报告中分析原因。\n"
                                "- 风险评分权重：营收异常(0.4)、杠杆异常(0.3)、利润异常(0.3)。请结合此权重评价公司的核心风险分布。\n\n"

                                "# 报告内容要求\n"
                                "一、【风险导向的重点识别】：1. 对各异常金融指标给出解释（会给出什么错报），结合风险评分和Z-Score给出整体企业风险画像和风险评分、2.高风险领域识别（收入确认，存货，应收账款）、3.异常财务指标（毛利率异常，应收账款周转率异常）\n"
                                "二、【关键程序智能推荐】：分点写出可能存在的财务风险（如现金流断裂、虚增收入、资产减值、存货减值、关联交易等），并分别给出识别依据，并给出推荐的审计程序以及原因。\n"
                                "三、【趋势风险预测】（重要）：根据多年数据给出一下预测：1. 预测重大错报风险（如收入异常增长，应收账款暴涨，存货异常或现金流与利润背离）、2.预测冗余管理或财务操纵概率（如是否有盈余平滑，是否存在利润操纵动机）（指标例子：净利润与经营现金流背离、应计利润比例异常、收入和应收账款不匹配）、3.预测持续经营风险（重要）（如是否可能发生资金链紧张，债务违约或发生持续亏损）"
                                f"四、【持续审计建议与风险预警】（重要）：1.结合网络搜索后得到的相关信息{searching_response}（报告中每一处网络相关信息的运用都需要给出网站出处或新闻出处）与多年输入信息，给出预测的审计重点领域，如之后应该重点审查什么（如收入确认，金融工具估值，关联交易方）、2. 实时监测关键指标，监控舆情，给出自动风险预警。\n"


                                "# 约束\n"
                                "- 禁止口水话，使用专业术语。\n"
                                "- 前后逻辑统一，统一口径。在风险导向审计框架下，不同财务指标的重要性并非等权。重大错报风险高、金额大、对报表影响深远的指标优先；“质”的差异（合规、造假）重于“量”的差异；外部证据优先于内部证据。应综合考虑审计证据的可靠性，通过追加程序解决冲突。所以改成“虽然 A（正面指标），但 B（高优先级风险）更具审计相关性，因此整”体风险判断以 B 为主。\n"
                                "- “可能存在虚构收入”“高度不确定性”“系统性风险”“现金流枯竭趋势”属于结论级语言，审计师只能在“充分、适当审计证据基础上”得出结论，证据不足时不下结论，可以说“风险增加”“需关注”\n"
                                "- 以专业的语言严格生成分析内容，不要加信件heading，禁止输出表格。\n"
                                "- 字数控制在 1500 - 1800 字之间。\n"
                                "- 直接输出报告正文。\n\n"
                            )
                        }

                    ]
        print(">>正在生成最终审计报告")
        return self.llm_service.call(prompt, model)


class IndustryBenchmarker:
    def __init__(self, data_agent: DataAgent, analysis_agent: AnalysisAgent):
        self.data_agent = data_agent
        self.analysis_agent = analysis_agent
        self.all_indicators: dict[str, dict] = {}

    def collect_company_data(self, pdf_paths: list[Path]):
        
        """批量提取多家公司的财务指标"""
        for path in pdf_paths:
            try:
                # 复用 DataAgent 提取原始数据
                profile = self.data_agent.build_financial_profile(
                    source="pdf", 
                    company= path.stem, 
                    pdf_paths= [path]
                )
                # 复用 AnalysisAgent 计算比例指标
                indicators = self.analysis_agent.extract_indicators(profile["financials"])
                for year in indicators:
                    if self.all_indicators.get(year,None) != None: 
                        # 存入样本池
                        for key0, value in indicators[year].items():
                            if key0 in self.all_indicators[year]:
                                self.all_indicators[year][key0].append(value)
                    else:
                        self.all_indicators[year] = {
                            "revenue": [],
                            "net_profit": [],
                            "profit_margin": [],
                            "gross_profit_margin": [],
                            "debt_ratio": [],
                            "return_on_assets": [],
                            "liquidity_ratio": [],
                            "days_sales_outstanding": [],
                            "days_inventory_outstanding": [],
                            "equity_ratio": [],
                            "roe": [],
                            "cash_to_liabilities" : []
                        }

                    # 存入样本池
                    for key0, value in indicators[year].items():
                        if key0 in self.all_indicators[year] and type(value) != str:
                            self.all_indicators[year][key0].append(value)
            except Exception as e:
                print(f"解析 {path} 失败: {e}")

    def calculate_benchmarks(self) -> dict[str, dict]:
        #print(self.all_indicators)
        """计算最终的行业均值和标准差"""
        means: dict[str,dict] = {}
        stds: dict[str,dict] = {}
        for year, dict0 in self.all_indicators.items():

            for key, values in dict0.items():
                if len(values) > 1:
                    means.setdefault(year, {})[f"{key}_mean"] = statistics.mean(values)
                    stds.setdefault(year, {})[f"{key}_std"] = statistics.stdev(values)
                else:
                    # 样本太少时给默认值，防止除以 0 报错
                    means.setdefault(year, {})[f"{key}_mean"] = values[0] if values else 0.0
                    stds.setdefault(year, {})[f"{key}_std"] = 1.0
        return means, stds


class AuditPipeline:
    def __init__(
        self,
        data_agent: DataAgent,
        analysis_agent: AnalysisAgent,
        risk_agent: RiskAgent,
        audit_agent: AuditAgent,
        benchmarker: IndustryBenchmarker,
    ):
        self.data_agent = data_agent
        self.analysis_agent = analysis_agent
        self.risk_agent = risk_agent
        self.audit_agent = audit_agent
        self.benchmarker = benchmarker

    def run(self, company: str, pdf_paths: list[Path] | None, peer_pdfs: list[Path] | None,api_url: str | None, output_dir: Path | None) -> AuditResult:
        source = "pdf" if pdf_paths else "api"
        result = AuditResult(company=company, source=source)
        api_data = None
        if api_url:
            api_data = self.data_agent.fetch_from_api(api_url, company)
        

        profile = self.data_agent.build_financial_profile(
            source=source,
            company=company,
            pdf_paths=pdf_paths,
            api_data=api_data,
        )
        result.financials = profile.get("financials", {})
        print(f"【{result.company}年报基本信息】")
        self.print_metrix(result.financials,)
        result.indicators = self.analysis_agent.extract_indicators(result.financials)
        print(f"【{result.company}年报基本指标】")
        self.print_metrix(result.indicators,)

        if peer_pdfs:
            self.benchmarker.collect_company_data(peer_pdfs)
            result.ind_mean, result.ind_std = self.benchmarker.calculate_benchmarks()
        else:
            result.ind_mean = None
            result.ind_std = None
        print("【行业指标平均】")
        self.print_metrix(result.ind_mean)
        print("【行业指标标准差】")
        self.print_metrix(result.ind_std)

        result.comparisons = self.analysis_agent.industry_comparison(result.indicators, result.ind_mean)
        result.report = self.analysis_agent.generate_analysis_report(result)

        result.anomalies = self.risk_agent.detect_anomalies(result.indicators, result.ind_mean)
        print("【异常风险提示】")
        print(result.anomalies)
        print("")
        result.z_scores = self.risk_agent.z_scores(result.indicators, ind_mean=result.ind_mean, ind_std=result.ind_std)
        print("【Z_Scores】")
        self.print_metrix(result.z_scores)
        print("")
        result.risk_score = self.risk_agent.score_risk(result.indicators, result.anomalies, z_scores= result.z_scores)
        print("【风险分析】")
        self.print_metrix(result.risk_score.get("dimensions",{}))
        print(f"风险评分：{result.risk_score.get('total_risk_score','')}")
        print("")

        result.recommendations = self.audit_agent.generate_recommendations(result)
        #print(result.recommendations)
        result.report += "\n\n" + self.audit_agent.build_audit_report(result)

        print("\n==============================================================================\n")
        print("【审计建议报告】")
        print(result.report)

        # output_dir.mkdir(parents=True, exist_ok=True)
        # output_file = output_dir / f"{company.replace(' ', '_')}_audit.json"
        # output_file.write_text(json.dumps(result.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"公司：{result.company}")
        print(f"来源：{result.source}")
        print(f"风险评分：{result.risk_score.get('total_risk_score','')}")
        return result
    
    def print_metrix(self, metrix: dict[str, Any], layers: int | None = 0):
        if type(metrix) != dict:
            print(metrix)
            return
        for key, value in metrix.items():
            print("\t"*layers + f"{key}: ",end="")
            if type(value) == dict:
                self.print_metrix(value, layers+1)
            else:
                if type(value) == float:
                    result = round(value, 4)
                    print(result)
                else:
                    print(value)
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 Agent 的财务风险审计程序。")
    parser.add_argument("company", help="要分析的公司名称。")
    parser.add_argument("--pdf", help="公司年报 PDF 文件路径。")
    parser.add_argument("--peer-pdf", default=None,help="公司年报 PDF 文件路径。")
    # parser.add_argument("--api-url", help="财务数据 API URL，可选。")
    # parser.add_argument("--output-dir", default="audit_results", help="结果输出目录。")
    parser.add_argument("--ai-api-url", default=None, help="LLM 模型服务 URL。")
    parser.add_argument("--ai-api-key", default=None, help="LLM 模型服务 API Key。")
    parser.add_argument("--model", default=None, help="LLM 模型名称。")
    parser.add_argument("--tc-api-key", default=None, help="Tavily 搜索服务 API Key。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ai_service = None
    if args.ai_api_key and args.ai_api_url:
        ai_service = LLMService(args.ai_api_url, args.ai_api_key)
    elif not args.disable_ai:
        print("警告：未配置 AI API Key，已切换为本地规则模式。")
    # ai_service = LLMService(DEFAULT_AI_API_URL, DEFAULT_AI_API_KEY)

    data_agent = DataAgent(llm_service=ai_service, model=args.model)
    analysis_agent = AnalysisAgent(llm_service=ai_service, model=args.model)
    risk_agent = RiskAgent(llm_service=ai_service, model= args.model)
    audit_agent = AuditAgent(llm_service=ai_service, model=args.model,tc_api_key=args.tc_api_key)
    benchmarker = IndustryBenchmarker(data_agent=data_agent, analysis_agent=analysis_agent)
    pipeline = AuditPipeline(data_agent, analysis_agent, risk_agent, audit_agent, benchmarker)

    pdf_list = []
    peer_pdf_list = []

    if args.pdf:
        pdf_list = args.pdf.split(",")
        for i in range(len(pdf_list)):
            pdf_list[i] = Path(pdf_list[i])
    
    for pdf_path in pdf_list:
        if pdf_path and not pdf_path.exists():
            raise FileNotFoundError(f"指定的 PDF 文件不存在：{pdf_path}")
    
    if args.peer_pdf:
        peer_pdf_list = args.peer_pdf.split(",")
        for i in range(len(peer_pdf_list)):
            peer_pdf_list[i] = Path(peer_pdf_list[i])
    else:
        peer_pdf_list = None
    

    result = pipeline.run(company=args.company, pdf_paths=pdf_list, peer_pdfs=peer_pdf_list, api_url=None, output_dir=None)
    #print(f"结果已写入：{Path(args.output_dir) / (args.company.replace(' ', '_') + '_audit.json')}")


if __name__ == "__main__":
    main()
    # ai_service = LLMService("https://api.siliconflow.cn/v1", "sk-qkjtyjmzbtdhxhhdlttsvfeejnmmwpcoadblkrvgfeihewwk")
    # risk = RiskAgent(llm_service=ai_service, model="deepseek-ai/DeepSeek-V3.2")

    # risk_scores = risk.score_risk(indicators={'2024': {'year': '2024', 'revenue': 5518756937.25, 'net_profit': 893066517.48, 'profit_margin': 16.18238541096205, 'gross_profit_margin': 1.0, 'debt_ratio': 29.9484598175773, 'return_on_assets': 0.10842311100036663, 'liquidity_ratio': -0.31766057110006546, 'days_sales_outstanding': 27.492722983811095, 'days_inventory_outstanding': 0.0, 'equity_ratio': 0.6815728490142506, 'roe': 15.907780240538846, 'cash_to_liabilities': 28.076538257782218}},anomalies=[],z_scores={})
    # print(risk_scores)
    #e:\pyfiles\aidedao\.venv\Scripts\python.exe e:/pyfiles/aidedao/t1.py "Company1" --pdf Zhongxin24Q1.pdf,Zhongxin25Q1.pdf --peer-pdf Haitian24Q1.pdf,Qianhe24Q1.pdf,Haitian25Q1.pdf,Qianhe25Q1.pdf --ai-api-url https://api.siliconflow.cn/v1 --ai-api-key sk-qkjtyjmzbtdhxhhdlttsvfeejnmmwpcoadblkrvgfeihewwk

    #e:\pyfiles\aidedao\.venv\Scripts\python.exe e:/pyfiles/aidedao/t1.py "中炬高新" --pdf ./dist0/Zhongxin24.pdf,./dist0/Zhongxin23.pdf --peer-pdf ./dist0/Haitian24.pdf,./dist0/Qianhe24.pdf,./dist0/Haitian23.pdf,./dist0/Qianhe23.pdf --ai-api-url https://api.siliconflow.cn/v1 --ai-api-key sk-qkjtyjmzbtdhxhhdlttsvfeejnmmwpcoadblkrvgfeihewwk --model deepseek-ai/DeepSeek-V3.2 --tc-api-key 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
    #Pro/deepseek-ai/DeepSeek-V3.2,deepseek-ai/DeepSeek-V3.2,Qwen/Qwen3.5-9B
