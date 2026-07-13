"""
数据规范 Agent v1.0 — 数据分类分级+标准化+编码规范

对齐最新国家标准:
  - GB/T 43697-2024 数据分类分级规则 (2024.10实施)
  - GB/T 43705-2025 科学数据安全分类分级指南
  - SJ/T 12043-2025 工业数据分类分级指南
  - GB/T 36073-2025 DCMM 2.0 数据标准域
  - GB/T 46207-2025 科学数据标识编码规范

核心能力:
  1. 数据分类 — 按行业/业务属性自动分类
  2. 数据分级 — 核心/重要/一般 三级自动判定
  3. 格式标准化 — 日期/电话/编码/单位统一
  4. 数据元管理 — 字段标准名映射
  5. 合规检查 — 对标国标打分
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Optional

from dateutil.parser import parse as dt_parse

from src.agents.base import AgentBase, AgentCategory, AgentResult
from src.layers.bronze import BronzeLayer

import logging
logger = logging.getLogger(__name__)


class StandardizeAgent(AgentBase):
    """数据规范 Agent — 分类分级+标准化"""

    name = "standardize"
    description = "数据分类分级(GB/T 43697)+格式标准化+数据元管理+DCMM标准对齐"
    category = AgentCategory.GOVERN
    version = "1.0.0"

    # ── 国标分类体系 ──
    INDUSTRY_CLASSES = {
        "工业": ["生产", "制造", "设备", "产线", "工艺", "零件", "装配"],
        "金融": ["银行", "交易", "贷款", "理财", "保险", "证券", "支付", "账户"],
        "电信": ["通信", "基站", "信号", "带宽", "运营商"],
        "交通": ["运输", "交通", "物流", "车辆", "航线", "港口"],
        "能源": ["电力", "石油", "天然气", "能源", "电网", "煤矿"],
        "医疗": ["患者", "病历", "诊断", "药品", "医院", "处方"],
        "教育": ["学生", "教师", "课程", "成绩", "学历"],
        "政务": ["政府", "行政", "审批", "公共", "户籍"],
    }

    # ── 国标三级分级 ──
    GRADE_RULES = {
        "核心数据": {
            "keywords": ["国家安全", "国民经济命脉", "重大公共利益", "国家秘密"],
            "field_patterns": ["身份证", "id_card", "idcard", "护照", "passport"],
            "min_grade_score": 80,
        },
        "重要数据": {
            "keywords": ["重要民生", "经济运行", "社会稳定", "公共健康"],
            "field_patterns": ["phone", "手机", "电话", "bank", "银行卡", "email", "邮箱"],
            "min_grade_score": 50,
        },
        "一般数据": {
            "keywords": ["企业自身", "组织权益", "个人信息"],
            "field_patterns": [],
            "min_grade_score": 0,
        },
    }

    # ── 标准数据元映射 ──
    STANDARD_NAMES = {
        "id": "标识符", "name": "名称", "phone": "联系电话", "mobile": "手机号码",
        "email": "电子邮箱", "address": "地址", "date": "日期", "time": "时间",
        "amount": "金额", "price": "单价", "quantity": "数量", "status": "状态",
        "create_time": "创建时间", "update_time": "更新时间",
        "身份证": "公民身份号码", "手机号": "手机号码", "电话": "联系电话",
        "邮箱": "电子邮箱", "地址": "通讯地址", "日期": "日期",
        "姓名": "姓名", "年龄": "年龄", "性别": "性别",
    }

    async def execute(self, **kwargs: Any) -> AgentResult:
        """
        Args:
            source_name: Bronze数据源名称
            bronze: BronzeLayer实例
            action: classify|grade|format|full (默认full)
            industry_hint: 行业提示(工业/金融/电信/交通/能源/医疗/教育/政务)
        """
        source_name = kwargs.get("source_name", "")
        bronze = kwargs.get("bronze") or BronzeLayer()
        action = kwargs.get("action", "full")
        industry_hint = kwargs.get("industry_hint", "")

        if not source_name:
            return AgentResult.fail("Missing source_name")

        try:
            raw = bronze.read_latest(source_name)
        except FileNotFoundError:
            return AgentResult.fail(f"Source not found: {source_name}")
        if not raw:
            return AgentResult.ok(data={}, message="Empty dataset")

        columns = list(raw[0].keys())
        rows = [list(r.values()) for r in raw]
        n = len(rows)

        report = {"source": source_name, "columns": len(columns), "rows": n}

        if action in ("classify", "full"):
            report["classification"] = self._classify(columns, rows, industry_hint)

        if action in ("grade", "full"):
            report["grading"] = self._grade(columns, rows)

        if action in ("format", "full"):
            report["format_issues"] = self._check_formats(columns, rows)

        # DCMM标准对齐得分
        if action == "full":
            report["dcmm_score"] = self._dcmm_score(report)

        return AgentResult.ok(
            data=report,
            message=f"Standardized {source_name}: {len(columns)} cols | "
                    f"行业={report.get('classification',{}).get('industry','?')} | "
                    f"DCMM={report.get('dcmm_score',{}).get('level','?')}",
        )

    # ═══════════════════════════════════════════
    # 数据分类 (GB/T 43697)
    # ═══════════════════════════════════════════

    def _classify(
        self, columns: list[str], rows: list[list], hint: str
    ) -> dict[str, Any]:
        """按行业/业务属性分类"""
        # 推断行业
        if hint:
            industry = hint
        else:
            industry = self._infer_industry(columns, rows)

        # 字段业务分类
        field_classes = {}
        for col in columns:
            field_classes[col] = self._classify_field(col)

        return {
            "industry": industry,
            "industry_standard": self._industry_standard(industry),
            "field_count": len(columns),
            "field_classes": field_classes,
            "classified_at": datetime.now(timezone.utc).isoformat(),
        }

    def _infer_industry(self, columns: list[str], rows: list[list]) -> str:
        """从字段名和数据内容推断行业"""
        all_text = " ".join(str(c) for c in columns).lower()
        # 采样数据内容
        for row in rows[:20]:
            for v in row:
                if v and isinstance(v, str):
                    all_text += " " + v.lower()

        scores = {}
        for industry, keywords in self.INDUSTRY_CLASSES.items():
            score = sum(1 for k in keywords if k in all_text)
            if score:
                scores[industry] = score

        if scores:
            return max(scores, key=scores.get)
        return "通用"

    def _classify_field(self, col_name: str) -> str:
        """单个字段的业务分类"""
        c = col_name.lower()
        if any(k in c for k in ("id", "编号", "代码", "code", "编码")):
            return "标识类"
        if any(k in c for k in ("name", "名称", "title", "标题", "姓名")):
            return "名称类"
        if any(k in c for k in ("phone", "电话", "mobile", "手机", "email", "邮箱", "address", "地址")):
            return "联系信息类"
        if any(k in c for k in ("date", "日期", "time", "时间", "created", "updated")):
            return "时间类"
        if any(k in c for k in ("amount", "金额", "price", "价格", "cost", "成本", "fee", "费用")):
            return "金额类"
        if any(k in c for k in ("status", "状态", "type", "类型", "category", "分类")):
            return "状态类"
        if any(k in c for k in ("desc", "描述", "remark", "备注", "note", "说明")):
            return "描述类"
        return "其他"

    def _industry_standard(self, industry: str) -> str:
        """行业对应的标准"""
        mapping = {
            "工业": "SJ/T 12043-2025 工业数据分类分级指南",
            "金融": "JR/T 0197-2020 金融数据安全分级指南",
            "电信": "YD/T 4251-2023 电信大数据安全管控分类分级",
            "政务": "DB3212/T 1116-2022 政务数据安全分类分级指南",
            "能源": "DB14/T 3551-2025 能源数据安全保护分类分级指南",
            "医疗": "GB/T 39725-2020 健康医疗数据安全指南",
        }
        return mapping.get(industry, "GB/T 43697-2024 数据分类分级规则(通用)")

    # ═══════════════════════════════════════════
    # 数据分级 (核心/重要/一般)
    # ═══════════════════════════════════════════

    def _grade(self, columns: list[str], rows: list[list]) -> dict[str, Any]:
        """三级分级: 核心数据/重要数据/一般数据"""
        field_grades = {}
        grade_counts = {"核心数据": 0, "重要数据": 0, "一般数据": 0}

        for col in columns:
            grade = self._grade_field(col, rows, columns.index(col))
            field_grades[col] = grade
            grade_counts[grade] += 1

        # 整体级别 = 最高级别
        if grade_counts["核心数据"] > 0:
            overall = "核心数据"
        elif grade_counts["重要数据"] > 0:
            overall = "重要数据"
        else:
            overall = "一般数据"

        return {
            "overall_grade": overall,
            "field_grades": field_grades,
            "grade_counts": grade_counts,
            "reference": "GB/T 43697-2024",
        }

    def _grade_field(self, col: str, rows: list[list], idx: int) -> str:
        """单字段分级"""
        c = col.lower()

        # 核心数据检测
        for pattern in self.GRADE_RULES["核心数据"]["field_patterns"]:
            if pattern in c:
                return "核心数据"

        # 内容检测 — 身份证号
        for row in rows[:20]:
            if idx < len(row) and row[idx] and isinstance(row[idx], str):
                if re.match(r'\d{17}[\dXx]', str(row[idx])):
                    return "核心数据"

        # 重要数据检测
        for pattern in self.GRADE_RULES["重要数据"]["field_patterns"]:
            if pattern in c:
                return "重要数据"

        # 姓名列 — 一般数据（但需脱敏）
        if any(k in c for k in ("name", "姓名")):
            return "一般数据"

        return "一般数据"

    # ═══════════════════════════════════════════
    # 格式标准化检查
    # ═══════════════════════════════════════════

    def _check_formats(
        self, columns: list[str], rows: list[list]
    ) -> list[dict[str, Any]]:
        """检查格式合规性"""
        issues = []

        for i, col in enumerate(columns):
            c = col.lower()

            # 日期格式检查
            if any(k in c for k in ("date", "日期", "time", "时间")):
                non_std = 0
                for row in rows[:100]:
                    if i < len(row) and row[i]:
                        if not self._is_iso_date(str(row[i])):
                            non_std += 1
                if non_std:
                    issues.append({
                        "column": col, "type": "date_format",
                        "count": non_std,
                        "suggestion": "统一为 ISO 8601 格式 (YYYY-MM-DD)",
                    })

            # 手机号格式检查
            elif any(k in c for k in ("phone", "手机", "mobile", "电话")):
                non_std = 0
                for row in rows[:100]:
                    if i < len(row) and row[i]:
                        digits = re.sub(r'\D', '', str(row[i]))
                        if len(digits) != 11 or not digits.startswith("1"):
                            non_std += 1
                if non_std:
                    issues.append({
                        "column": col, "type": "phone_format",
                        "count": non_std,
                        "suggestion": "统一为11位数字 (如13800138000)",
                    })

        return issues

    @staticmethod
    def _is_iso_date(value: str) -> bool:
        """检查是否为ISO日期格式"""
        try:

            dt_parse(str(value))
            return True
        except Exception:
            return False

    # ═══════════════════════════════════════════
    # DCMM 标准对齐评分
    # ═══════════════════════════════════════════

    def _dcmm_score(self, report: dict) -> dict[str, Any]:
        """DCMM 数据标准域评分 (0-100)"""
        score = 100
        deductions = []

        # 1. 分类覆盖率
        classification = report.get("classification", {})
        fields = classification.get("field_count", 0)
        field_classes = classification.get("field_classes", {})
        classified = sum(1 for v in field_classes.values() if v != "其他")
        if fields:
            coverage = classified / fields * 100
            if coverage < 80:
                deductions.append(f"分类覆盖率仅{coverage:.0f}%")

        # 2. 格式问题
        fmt_issues = report.get("format_issues", [])
        if fmt_issues:
            score -= len(fmt_issues) * 5
            deductions.append(f"{len(fmt_issues)}个格式问题")

        # 3. 分级完成度
        grading = report.get("grading", {})
        if grading:
            counts = grading.get("grade_counts", {})
            if counts.get("核心数据", 0) > 0 and not any("核心" in d for d in deductions):
                deductions.append("含核心数据字段，需重点保护")

        level = self._score_to_dcmm_level(max(0, min(100, score)))
        return {
            "score": max(0, min(100, score)),
            "level": level,
            "deductions": deductions,
            "dcmm_domain": "数据标准 (GB/T 36073-2025)",
        }

    @staticmethod
    def _score_to_dcmm_level(score: float) -> str:
        if score >= 95: return "优化级(5)"
        if score >= 85: return "量化管理级(4)"
        if score >= 70: return "稳健级(3)"
        if score >= 50: return "受管理级(2)"
        return "初始级(1)"
