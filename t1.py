from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from openai import OpenAI

import statistics
import pdfplumber
import requests



#DEFAULT_AI_MODEL = os.getenv("SILICON_FLOW_MODEL", "gpt-4-silicon")

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
    risk_score: float = 0.0
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
        try:
            response = self.client.chat.completions.create(
                model= model,
                messages= Response_format,
                stream= False,
            )
            result_json = response.choices[0].message.content
            # response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            # response.raise_for_status()
            print(f"[CAUSION] 本次消耗 Token 总数: {response.usage.total_tokens}")
        except Exception as e:
            print("[ERROR] API 调用失败。")
            response = self.client.chat.completions.create(
                model= model,
                messages= Response_format,
                stream= False,
            )
        return result_json


class DataAgent:
    def __init__(self, api_url: str | None = None, llm_service: LLMService | None = None):
        self.api_url = api_url
        self.llm_service = llm_service

    def fetch_from_api(self, api_url: str, company: str) -> dict[str, Any]:
        response = requests.get(api_url, params={"company": company}, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {"raw": data}

    def extract_text_from_pdf(self, pdf_path: Path) -> list[str]:
        text_parts_box: list[str] = []
        # 尝试使用 pdfplumber 增强表格读取能力
        with pdfplumber.open(pdf_path) as pdf:
            table_num = 0
            page_mark = 1
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
        
        for i in range(len(text_parts_box)):
            text_parts_box[i] = "\n".join(text_parts_box[i])
        return text_parts_box

    def parse_numeric_values(self, text: list[str], buffer: dict[str, Any]) -> dict[str, Any]:
        #在处理中文财务术语（如“扣非净利润”、“加权ROE”）方面，国产模型有天然的语料优势，且性价比极高
        model = "deepseek-ai/DeepSeek-V3.2"
        prompt = [
                    {
                        "role": "system",
                        "content": (
                            "你是一位资深的注册会计师（CPA）和数据提取专家。你的任务是从结构化的 PDF 坐标文本中精准提取财务数据。"
                            "你具备极强的语义理解能力，能够识别会计科目的同义词（如“营业总收入”等同于“营业收入”）。"
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
                            "   - year: 年份，年度/季度(若为年度报告则直接输出年份，若为季度报告则格式为[year](Quarter)）\n"
                            "   - revenue: 营业收入 / 营业总收入\n"
                            "   - net_profit: 归属于母公司股东的净利润\n"
                            "   - total_assets: 资产总额 / 总资产\n"
                            "   - total_liabilities: 负债总额 / 总负债\n"
                            "   - equity: 归属于母公司股东的所有者权益 / 股东权益合计\n"
                            "   - cash: 期末现金及现金等价物余额\n"
                            "4. 要求数据类型：\n"
                            "   - year: str\n"
                            "   - revenue: float\n"
                            "   - net_profit: float\n"
                            "   - total_assets: float\n"
                            "   - total_liabilities: float\n"
                            "   - equity: float\n"
                            "   - cash: float\n"
                            "5. 注意，pdf可能因过长而被拆分，给出的可能为不完整pdf，将提供已经找到并整理好的数据（json），需要鉴别并补充。"
                            
                            "# 输出要求\n"
                            "仅返回 JSON 格式内容，不含解释，不含引号，不换行。严格按照如下格式：\n"
                            "{\"year\": \"2026(Quarter1)\", \"revenue\": 0.0, \"net_profit\": 0.0, ...}\n\n"
                            
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
            textnum = 0
            for pdf_path in pdf_paths:
                buffer = {}
                raw_text = self.extract_text_from_pdf(pdf_path)
                for text_part in raw_text:
                    profile["raw"][f"text{textnum}"] = text_part[:4000]
                    buffer = self.parse_numeric_values(text_part,buffer=buffer)
                profile["financials"][buffer.get("year")] = buffer
                
            
        else:
            profile["financials"] = {}

        return profile


class AnalysisAgent:
    def __init__(self, llm_service: LLMService | None = None):
        self.llm_service = llm_service

    def extract_indicators(self, financials: dict[str, Any]) -> dict[str, Any]:
        indicator_dict: dict[str, dict] = {}
        for fin_data in financials:
            revenue = float(financials[fin_data].get("revenue", 0.0) or 0.0)
            net_profit = float(financials[fin_data].get("net_profit", 0.0) or 0.0)
            total_assets = float(financials[fin_data].get("total_assets", 0.0) or 0.0)
            total_liabilities = float(financials[fin_data].get("total_liabilities", 0.0) or 0.0)
            equity = float(financials[fin_data].get("equity", 0.0) or 0.0)
            cash = float(financials[fin_data].get("cash", 0.0) or 0.0)
            year = str(financials[fin_data].get("year",""))


            indicator: dict[str, Any] = {
                "year": year,
                "revenue": revenue,
                "net_profit": net_profit,
                "profit_margin": (net_profit / revenue * 100) if revenue else 0.0,
                "debt_ratio": (total_liabilities / total_assets * 100) if total_assets else 0.0,
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
        model = "deepseek-ai/DeepSeek-V3.2"
        prompt = [
                    {
                        "role": "system",
                        "content": "你是一位资深的财务分析师，擅长通过财务数据透视企业的经营风险与盈利质量。"
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
                            "3. 根据行业对比结果给出分析结论。\n\n"
                            "# Tone\n"
                            "专业、客观、言简意赅。避免口水话，多使用“财务韧性”、“资本结构优化”、“盈利质量”等专业词汇\n"
                        )
                    }
                ]

        if self.llm_service:
            return self.llm_service.call(prompt, model)


class RiskAgent:
    def __init__(self, anomaly_thresholds: dict[str, float] | None = None):
        self.thresholds = anomaly_thresholds or {
            "profit_margin": 5.0,
            "debt_ratio": 65.0,
            "roe": 5.0,
            "cash_to_liabilities": 10.0,
        }


    def detect_anomalies(self, indicators: dict[str, float]) -> dict[str, list]:
        anomalies: dict[str,list[str]] = {}
        for year, dict0 in indicators.items():
            if anomalies.get("year", None) == None:
                anomalies[year] = []
            if dict0.get("profit_margin", 0.0) < self.thresholds["profit_margin"]:
                anomalies[year].append("净利率低于行业警戒线，存在盈利能力不足风险。")
            if dict0.get("debt_ratio", 0.0) > self.thresholds["debt_ratio"]:
                anomalies[year].append("资产负债率偏高，可能存在偿债压力。")
            if dict0.get("roe", 0.0) < self.thresholds["roe"]:
                anomalies[year].append("股本回报率偏低，资本使用效率不佳。")
            if dict0.get("cash_to_liabilities", 0.0) < self.thresholds["cash_to_liabilities"]:
                anomalies[year].append("现金储备不足以覆盖负债，流动性风险较高。")

        return anomalies

    def score_risk(self, indicators: dict[str, float], anomalies: dict[str,list[str]]) -> dict[str, float]:
        risk_scores: dict[str, float] = {}
        for year in indicators:
            score = 50.0
            score += max(0.0, min(20.0, indicators[year].get("debt_ratio", 0.0) / 5.0))
            score += max(0.0, min(20.0, (20.0 - indicators[year].get("profit_margin", 0.0)) / 1.0))
            score += 10.0 * len(anomalies[year])
            risk_scores[year] = min(100.0, round(score, 1))
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
    def __init__(self, llm_service: LLMService | None = None):
        self.llm_service = llm_service

    def generate_recommendations(self, result: AuditResult) -> dict[str, list[str]]:

        recommendations: dict[str, list[str]] = {}
        for year in result.indicators:
            recommendations.setdefault(year, [])
            if result.indicators[year].get("profit_margin", 0.0) < 10.0:
                recommendations[year].append("检查成本结构与毛利水平，评估可提升利润率的环节。")
            if result.indicators[year].get("debt_ratio", 0.0) > 60.0:
                recommendations[year].append("审查负债组合与偿债计划，关注短期借款与应付债务。")
            if result.indicators[year].get("cash_to_liabilities", 0.0) < 15.0:
                recommendations[year].append("强化现金流管理，评估应收账款和存货周转情况。 ")
            if not recommendations[year]:
                recommendations[year].append("当前指标整体稳定，建议继续关注业务增长与现金流质量。 ")
        return recommendations

    def build_audit_report(self, result: AuditResult) -> str:
        if self.llm_service:
            model = "deepseek-ai/DeepSeek-V3.2"
            prompt = [
                        {
                            "role": "system",
                            "content": "你是一位资深注册会计师（CPA）。你擅长通过财务数据的异常波动发现企业潜在的经营风险、财务造假或管理漏洞。你的语言风格稳重、严谨、客观。\n"
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

                                "# 报告结构要求\n"
                                "1. 【整体评价】：基于风险评分，给出该公司的风险等级（低/中/高）。\n"
                                "2. 【异常穿透分析】：结合 Z-Score，详细分析营收、利润或负债中的离群表现。例如：若营收 Z 值极高但利润 Z 值极低，需质疑盈利质量。\n"
                                "3. 【潜在风险警示】：列出可能存在的财务风险（如现金流断裂、虚增收入、资产减值等）。\n"
                                "4. 【审计改进建议】：针对发现的问题，给出具体的核查建议或管理改进方案。\n\n"

                                "# 约束\n"
                                "- 禁止口水话，使用专业术语。\n"
                                "- 字数控制在 400-600 字之间。\n"
                                "- 直接输出报告正文。\n\n"
                            )
                        }

                    ]
        
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
                            "debt_ratio": [],
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
        print(result.financials)
        result.indicators = self.analysis_agent.extract_indicators(result.financials)
        print(result.indicators)

        if peer_pdfs:
            self.benchmarker.collect_company_data(peer_pdfs)
            result.ind_mean, result.ind_std = self.benchmarker.calculate_benchmarks()
        else:
            result.ind_mean = None
            result.ind_std = None

        result.comparisons = self.analysis_agent.industry_comparison(result.indicators, result.ind_mean)
        result.report = self.analysis_agent.generate_analysis_report(result)

        result.anomalies = self.risk_agent.detect_anomalies(result.indicators)
        result.risk_score = self.risk_agent.score_risk(result.indicators, result.anomalies)
        result.z_scores = self.risk_agent.z_scores(result.indicators, ind_mean=result.ind_mean, ind_std=result.ind_std)

        result.recommendations = self.audit_agent.generate_recommendations(result)
        result.report += "\n\n" + self.audit_agent.build_audit_report(result)

        print(result.ind_mean)
        print(result.ind_std)
        print(result.anomalies)
        print(f"Risk Score:\t{result.risk_score}")
        print(f"Z Scores\t{result.z_scores}")
        print(result.recommendations)
        print(result.report)

        # output_dir.mkdir(parents=True, exist_ok=True)
        # output_file = output_dir / f"{company.replace(' ', '_')}_audit.json"
        # output_file.write_text(json.dumps(result.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 Agent 的财务风险审计程序。")
    parser.add_argument("company", help="要分析的公司名称。")
    parser.add_argument("--pdf", help="公司年报 PDF 文件路径。")
    parser.add_argument("--peer-pdf", default=None,help="公司年报 PDF 文件路径。")
    # parser.add_argument("--api-url", help="财务数据 API URL，可选。")
    # parser.add_argument("--output-dir", default="audit_results", help="结果输出目录。")
    parser.add_argument("--ai-api-url", default=None, help="LLM 模型服务 URL。")
    parser.add_argument("--ai-api-key", default=None, help="LLM 模型服务 API Key。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ai_service = None
    if args.ai_api_key and args.ai_api_url:
        ai_service = LLMService(args.ai_api_url, args.ai_api_key)
    elif not args.disable_ai:
        print("警告：未配置 AI API Key，已切换为本地规则模式。")
    # ai_service = LLMService(DEFAULT_AI_API_URL, DEFAULT_AI_API_KEY)

    data_agent = DataAgent(llm_service=ai_service)
    analysis_agent = AnalysisAgent(llm_service=ai_service)
    risk_agent = RiskAgent()
    audit_agent = AuditAgent(llm_service=ai_service)
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
    print("分析完成：")
    print(f"公司：{result.company}")
    print(f"来源：{result.source}")
    print(f"风险评分：{result.risk_score}")
    #print(f"结果已写入：{Path(args.output_dir) / (args.company.replace(' ', '_') + '_audit.json')}")


if __name__ == "__main__":
    main()
    #e:\pyfiles\aidedao\.venv\Scripts\python.exe e:/pyfiles/aidedao/t1.py "Company1" --pdf Zhongxin24Q1.pdf,Zhongxin25Q1.pdf --peer-pdf Haitian24Q1.pdf,Qianhe24Q1.pdf,Haitian25Q1.pdf,Qianhe25Q1.pdf --ai-api-url https://api.siliconflow.cn/v1 --ai-api-key
