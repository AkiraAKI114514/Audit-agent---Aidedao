from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from openai import OpenAI

import pdfplumber 
import requests


DEFAULT_AI_API_URL = "https://api.siliconflow.cn/v1"
DEFAULT_AI_API_KEY = "sk-qkjtyjmzbtdhxhhdlttsvfeejnmmwpcoadblkrvgfeihewwk"
DEFAULT_AI_MODEL = os.getenv("SILICON_FLOW_MODEL", "gpt-4-silicon")

NUMERIC_RE = re.compile(r"([+-]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?)")
CURRENCY_WORDS = ["元", "亿", "万", "USD", "CNY", "RMB", "美元"]


@dataclass
class AuditResult:
    company: str
    source: str
    financials: dict[str, Any] = field(default_factory=dict)
    indicators: dict[str, float] = field(default_factory=dict)
    comparisons: dict[str, Any] = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)
    risk_score: float = 0.0
    recommendations: list[str] = field(default_factory=list)
    report: str = ""


class DataAgent:
    def __init__(self, api_url: str | None = None, llm_service: LLMService | None = None):
        self.api_url = api_url
        self.llm_service = llm_service

    def fetch_from_api(self, api_url: str, company: str) -> dict[str, Any]:
        response = requests.get(api_url, params={"company": company}, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {"raw": data}

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        text_parts = []
        # 尝试使用 pdfplumber 增强表格读取能力
        with pdfplumber.open(pdf_path) as pdf:
            table_num = 0
            page_mark = 1
            for page in pdf.pages:
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
        return "\n".join(text_parts)

    def parse_numeric_values(self, text: str) -> dict[str, float]:
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
                            "   - revenue: 营业收入 / 营业总收入\n"
                            "   - net_profit: 归属于母公司股东的净利润\n"
                            "   - total_assets: 资产总额 / 总资产\n"
                            "   - total_liabilities: 负债总额 / 总负债\n"
                            "   - equity: 归属于母公司股东的所有者权益 / 股东权益合计\n"
                            "   - cash: 期末现金及现金等价物余额\n"
                            "   - currency type: 币种和单位\n\n"
                            
                            "# 输出要求\n"
                            "仅返回 JSON 格式，不含解释。格式如下：\n"
                            "{\"revenue\": 0.0, \"net_profit\": 0.0, ...}\n\n"
                            
                            "<PDF_DATA>\n"
                            f"{text}\n"  # 这里填入你处理过的 [r/c] 格式文本
                            "</PDF_DATA>"
                        )
                    }
                ]
        metrics = self.llm_service.call(prompt, model)
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

    def build_financial_profile(self, source: str, company: str, pdf_path: Path | None = None, api_data: dict[str, Any] | None = None) -> dict[str, Any]:
        profile = {
            "company": company,
            "source": source,
            "raw": {},
        }

        if api_data:
            profile["raw"] = api_data
            if "financials" in api_data and isinstance(api_data["financials"], dict):
                profile["financials"] = api_data["financials"]
            else:
                profile["financials"] = self.parse_numeric_values(json.dumps(api_data, ensure_ascii=False))
        elif pdf_path:
            raw_text = self.extract_text_from_pdf(pdf_path)
            profile["raw"] = {"text": raw_text[:4000]}
            profile["financials"] = self.parse_numeric_values(raw_text)
        else:
            profile["financials"] = {}

        return profile


class AnalysisAgent:
    def __init__(self, llm_service: LLMService | None = None):
        self.llm_service = llm_service

    def extract_indicators(self, financials: dict[str, Any]) -> dict[str, float]:
        revenue = float(financials.get("revenue", 0.0) or 0.0)
        net_profit = float(financials.get("net_profit", 0.0) or 0.0)
        total_assets = float(financials.get("total_assets", 0.0) or 0.0)
        total_liabilities = float(financials.get("total_liabilities", 0.0) or 0.0)
        equity = float(financials.get("equity", 0.0) or 0.0)
        cash = float(financials.get("cash", 0.0) or 0.0)

        indicators: dict[str, float] = {
            "revenue": revenue,
            "net_profit": net_profit,
            "profit_margin": (net_profit / revenue * 100) if revenue else 0.0,
            "debt_ratio": (total_liabilities / total_assets * 100) if total_assets else 0.0,
            "roe": (net_profit / equity * 100) if equity else 0.0,
            "cash_to_liabilities": (cash / total_liabilities * 100) if total_liabilities else 0.0,
        }
        return indicators

    def industry_comparison(self, indicators: dict[str, float]) -> dict[str, Any]:
        benchmark = {
            "profit_margin": 12.0,
            "debt_ratio": 55.0,
            "roe": 10.0,
            "cash_to_liabilities": 25.0,
        }
        comparison: dict[str, Any] = {}
        for key, value in indicators.items():
            if key in benchmark:
                comparison[key] = {
                    "value": round(value, 2),
                    "benchmark": benchmark[key],
                    "status": "低" if value < benchmark[key] else "高" if value > benchmark[key] else "持平",
                }
        return comparison

    def generate_analysis_report(self, result: AuditResult) -> str:
        prompt = (
            "请基于以下财务指标生成简明分析报告：\n"
            f"公司：{result.company}\n"
            f"营业收入：{result.indicators.get('revenue', 0.0):,.2f}\n"
            f"净利润：{result.indicators.get('net_profit', 0.0):,.2f}\n"
            f"净利率：{result.indicators.get('profit_margin', 0.2):.2f}%\n"
            f"资产负债率：{result.indicators.get('debt_ratio', 0.0):.2f}%\n"
            f"ROE：{result.indicators.get('roe', 0.0):.2f}%\n"
            "请根据行业对比结果给出分析结论。"
        )
        if self.llm_service:
            return self.llm_service.call(prompt, task="analysis")
        return (
            "综合分析：公司营业收入与净利润水平需要进一步确认。"
            " 净利率与行业基准相比存在差异，若资产负债率偏高，则应关注资本结构风险。"
        )


class RiskAgent:
    def __init__(self, anomaly_thresholds: dict[str, float] | None = None):
        self.thresholds = anomaly_thresholds or {
            "profit_margin": 5.0,
            "debt_ratio": 65.0,
            "roe": 5.0,
            "cash_to_liabilities": 10.0,
        }

    def detect_anomalies(self, indicators: dict[str, float]) -> list[str]:
        anomalies: list[str] = []
        if indicators.get("profit_margin", 0.0) < self.thresholds["profit_margin"]:
            anomalies.append("净利率低于行业警戒线，存在盈利能力不足风险。")
        if indicators.get("debt_ratio", 0.0) > self.thresholds["debt_ratio"]:
            anomalies.append("资产负债率偏高，可能存在偿债压力。")
        if indicators.get("roe", 0.0) < self.thresholds["roe"]:
            anomalies.append("股本回报率偏低，资本使用效率不佳。")
        if indicators.get("cash_to_liabilities", 0.0) < self.thresholds["cash_to_liabilities"]:
            anomalies.append("现金储备不足以覆盖负债，流动性风险较高。")
        return anomalies

    def score_risk(self, indicators: dict[str, float], anomalies: list[str]) -> float:
        score = 50.0
        score += max(0.0, min(20.0, indicators.get("debt_ratio", 0.0) / 5.0))
        score += max(0.0, min(20.0, (20.0 - indicators.get("profit_margin", 0.0)) / 1.0))
        score += 10.0 * len(anomalies)
        return min(100.0, round(score, 1))


class AuditAgent:
    def __init__(self, llm_service: LLMService | None = None):
        self.llm_service = llm_service

    def generate_recommendations(self, result: AuditResult) -> list[str]:
        recommendations: list[str] = []
        if result.indicators.get("profit_margin", 0.0) < 10.0:
            recommendations.append("检查成本结构与毛利水平，评估可提升利润率的环节。")
        if result.indicators.get("debt_ratio", 0.0) > 60.0:
            recommendations.append("审查负债组合与偿债计划，关注短期借款与应付债务。")
        if result.indicators.get("cash_to_liabilities", 0.0) < 15.0:
            recommendations.append("强化现金流管理，评估应收账款和存货周转情况。 ")
        if not recommendations:
            recommendations.append("当前指标整体稳定，建议继续关注业务增长与现金流质量。 ")
        return recommendations

    def build_audit_report(self, result: AuditResult) -> str:
        if self.llm_service:
            prompt = (
                "请基于以下风险评分、异常项和财务分析结果撰写审计建议：\n"
                f"风险评分：{result.risk_score}\n"
                f"异常项：{'；'.join(result.anomalies) if result.anomalies else '无'}\n"
                f"建议：{'；'.join(result.recommendations)}\n"
            )
            return self.llm_service.call(prompt, task="audit")

        sections = [
            f"风险评分：{result.risk_score}/100",
            "异常检测：" + ("；".join(result.anomalies) if result.anomalies else "未发现重大异常。"),
            "建议：" + "；".join(result.recommendations),
        ]
        return "\n".join(sections)


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
        return result_json


class AuditPipeline:
    def __init__(
        self,
        data_agent: DataAgent,
        analysis_agent: AnalysisAgent,
        risk_agent: RiskAgent,
        audit_agent: AuditAgent,
    ):
        self.data_agent = data_agent
        self.analysis_agent = analysis_agent
        self.risk_agent = risk_agent
        self.audit_agent = audit_agent

    def run(self, company: str, pdf_path: Path | None, api_url: str | None, output_dir: Path) -> AuditResult:
        source = "pdf" if pdf_path else "api"
        result = AuditResult(company=company, source=source)
        api_data = None
        if api_url:
            api_data = self.data_agent.fetch_from_api(api_url, company)

        profile = self.data_agent.build_financial_profile(
            source=source,
            company=company,
            pdf_path=pdf_path,
            api_data=api_data,
        )
        result.financials = profile.get("financials", {})
        result.indicators = self.analysis_agent.extract_indicators(result.financials)
        result.comparisons = self.analysis_agent.industry_comparison(result.indicators)
        result.report = self.analysis_agent.generate_analysis_report(result)
        result.anomalies = self.risk_agent.detect_anomalies(result.indicators)
        result.risk_score = self.risk_agent.score_risk(result.indicators, result.anomalies)
        result.recommendations = self.audit_agent.generate_recommendations(result)
        result.report += "\n\n" + self.audit_agent.build_audit_report(result)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{company.replace(' ', '_')}_audit.json"
        output_file.write_text(json.dumps(result.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 Agent 的财务风险审计程序。")
    parser.add_argument("company", help="要分析的公司名称。")
    parser.add_argument("--pdf", help="公司年报 PDF 文件路径。")
    parser.add_argument("--api-url", help="财务数据 API URL，可选。")
    parser.add_argument("--output-dir", default="audit_results", help="结果输出目录。")
    parser.add_argument("--disable-ai", action="store_true", help="禁用 AI 模型调用，仅使用本地规则生成分析结果。")
    parser.add_argument("--ai-api-url", default=DEFAULT_AI_API_URL, help="LLM 模型服务 URL。")
    parser.add_argument("--ai-api-key", default=DEFAULT_AI_API_KEY, help="LLM 模型服务 API Key。")
    parser.add_argument("--ai-model", default=DEFAULT_AI_MODEL, help="LLM 模型名称。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ai_service = None
    if not args.disable_ai and args.ai_api_key:
        ai_service = LLMService(args.ai_api_url, args.ai_api_key)
    elif not args.disable_ai:
        print("警告：未配置 AI API Key，已切换为本地规则模式。")

    data_agent = DataAgent(llm_service=ai_service)
    analysis_agent = AnalysisAgent(llm_service=ai_service)
    risk_agent = RiskAgent()
    audit_agent = AuditAgent(llm_service=ai_service)
    pipeline = AuditPipeline(data_agent, analysis_agent, risk_agent, audit_agent)

    pdf_path = Path(args.pdf) if args.pdf else None
    if pdf_path and not pdf_path.exists():
        raise FileNotFoundError(f"指定的 PDF 文件不存在：{pdf_path}")

    result = pipeline.run(args.company, pdf_path, args.api_url, Path(args.output_dir))
    print("分析完成：")
    print(f"公司：{result.company}")
    print(f"来源：{result.source}")
    print(f"风险评分：{result.risk_score}")
    print(f"结果已写入：{Path(args.output_dir) / (args.company.replace(' ', '_') + '_audit.json')}")


if __name__ == "__main__":
    main()
    # ai_service = LLMService(DEFAULT_AI_API_URL, DEFAULT_AI_API_KEY)
    # dataagent = DataAgent(llm_service=ai_service)
    # text = dataagent.build_financial_profile(company="中炬高新", source="2025 年第一季度报告", pdf_path="11062118.pdf")
    # print(text["financials"])
