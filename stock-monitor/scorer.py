#!/usr/bin/env python3
"""
评分模块：对买入信号个股进行量化评分与硬过滤
"""
import pandas as pd
from typing import Dict, Any, Optional


def calculate_score(ticker_data: pd.DataFrame, qqq_data: pd.DataFrame) -> Dict[str, Any]:
    """
    计算个股评分
    
    Args:
        ticker_data: 个股数据（已经过时间切片处理后的 DataFrame）
        qqq_data: QQQ 数据（已经过时间切片处理后的 DataFrame）
    
    Returns:
        包含评分、风险、相对强度和所有计算结果的字典
    """
    # 初始化结果字典
    result = {
        "score": None,
        "risk": None,
        "rs": None,
        "M": None,
        "multiplier": None,
        "daily_return": None,
        "stock_5d_return": None,
        "qqq_5d_return": None,
        "reason": []
    }
    
    try:
        # 确保数据长度足够
        if len(ticker_data) < 6 or len(qqq_data) < 6:
            result["reason"].append("数据长度不足")
            # 将reason列表合并为字符串
            result["reason"] = "; ".join(result["reason"])
            return result
        
        # 获取收盘价序列
        # 处理不同形式的输入：DataFrame 或 Series
        if isinstance(ticker_data, pd.DataFrame):
            # 如果是 DataFrame，尝试获取 'Close' 列
            if 'Close' in ticker_data.columns:
                close_series = ticker_data['Close']
            elif not ticker_data.empty:
                # 如果没有 'Close' 列，但 DataFrame 不为空，假设整个 DataFrame 就是收盘价数据
                close_series = ticker_data.iloc[:, 0]
            else:
                # 如果 DataFrame 为空，返回错误
                result["reason"].append("个股数据为空")
                # 将reason列表合并为字符串
                result["reason"] = "; ".join(result["reason"])
                return result
        elif isinstance(ticker_data, pd.Series):
            # 如果是 Series，直接使用
            close_series = ticker_data
        else:
            result["reason"].append("数据类型错误")
            # 将reason列表合并为字符串
            result["reason"] = "; ".join(result["reason"])
            return result
        
        # 确保 close_series 不为空且长度足够
        if close_series is None or len(close_series) < 6:
            result["reason"].append("个股收盘价数据长度不足")
            # 将reason列表合并为字符串
            result["reason"] = "; ".join(result["reason"])
            return result
        
        # 检查 close_series 是否全为空值
        if close_series.isna().all():
            result["reason"].append("个股收盘价数据全为空")
            # 将reason列表合并为字符串
            result["reason"] = "; ".join(result["reason"])
            return result
        
        # 同样处理 QQQ 数据
        if isinstance(qqq_data, pd.DataFrame):
            if 'Close' in qqq_data.columns:
                qqq_close_series = qqq_data['Close']
            elif not qqq_data.empty:
                # 如果没有 'Close' 列，但 DataFrame 不为空，假设整个 DataFrame 就是收盘价数据
                qqq_close_series = qqq_data.iloc[:, 0]
            else:
                # 如果 DataFrame 为空，返回错误
                result["reason"].append("QQQ数据为空")
                # 将reason列表合并为字符串
                result["reason"] = "; ".join(result["reason"])
                return result
        elif isinstance(qqq_data, pd.Series):
            qqq_close_series = qqq_data
        else:
            result["reason"].append("QQQ数据类型错误")
            # 将reason列表合并为字符串
            result["reason"] = "; ".join(result["reason"])
            return result
        
        # 确保 qqq_close_series 不为空且长度足够
        if qqq_close_series is None or len(qqq_close_series) < 6:
            result["reason"].append("QQQ收盘价数据长度不足")
            # 将reason列表合并为字符串
            result["reason"] = "; ".join(result["reason"])
            return result
        
        # 检查 qqq_close_series 是否全为空值
        if qqq_close_series.isna().all():
            result["reason"].append("QQQ收盘价数据全为空")
            # 将reason列表合并为字符串
            result["reason"] = "; ".join(result["reason"])
            return result
        
        # 确保 close_series 有足够的数据点
        if len(close_series) < 20:
            result["reason"].append("个股数据长度不足20天")
            # 将reason列表合并为字符串
            result["reason"] = "; ".join(result["reason"])
            return result
        
        # 确保 qqq_close_series 有足够的数据点
        if len(qqq_close_series) < 6:
            result["reason"].append("QQQ数据长度不足6天")
            # 将reason列表合并为字符串
            result["reason"] = "; ".join(result["reason"])
            return result
        
        # 计算 EMA20
        ema20 = close_series.ewm(span=20, adjust=False).mean()
        
        # 计算 HHV20 序列
        hhv20_series = close_series.rolling(window=20).max()
        
        # 检查 ema20 是否为空
        if ema20.empty:
            result["reason"].append("EMA20计算失败")
            # 将reason列表合并为字符串
            result["reason"] = "; ".join(result["reason"])
            return result
        
        # 检查 hhv20_series 是否为空
        if hhv20_series.empty:
            result["reason"].append("HHV20计算失败")
            # 将reason列表合并为字符串
            result["reason"] = "; ".join(result["reason"])
            return result
        
        # 获取当前值 (t)
        try:
            c_t = float(close_series.iloc[-1])
            ema20_t = float(ema20.iloc[-1])
        except Exception as e:
            result["reason"].append(f"获取当前值失败: {str(e)}")
            # 将reason列表合并为字符串
            result["reason"] = "; ".join(result["reason"])
            return result
        
        # 获取昨日值 (t-1)
        try:
            c_t1 = float(close_series.iloc[-2])
            hhv20_t1 = float(hhv20_series.iloc[-2])  # HHV20_{t-1} = max(CLOSE_{t-20} ~ CLOSE_{t-1})
        except Exception as e:
            result["reason"].append(f"获取昨日值失败: {str(e)}")
            # 将reason列表合并为字符串
            result["reason"] = "; ".join(result["reason"])
            return result
        
        # 获取前天值 (t-2)
        try:
            hhv20_t2 = float(hhv20_series.iloc[-3])  # HHV20_{t-2} = max(CLOSE_{t-21} ~ CLOSE_{t-2})
        except Exception as e:
            result["reason"].append(f"获取前天值失败: {str(e)}")
            # 将reason列表合并为字符串
            result["reason"] = "; ".join(result["reason"])
            return result
        
        # 获取5日前值 (t-5)
        try:
            c_t5 = float(close_series.iloc[-6])  # 因为 iloc[-1] 是 t，所以 t-5 是 iloc[-6]
            qqq_c_t = float(qqq_close_series.iloc[-1])
            qqq_c_t5 = float(qqq_close_series.iloc[-6])
        except Exception as e:
            result["reason"].append(f"获取5日前值失败: {str(e)}")
            # 将reason列表合并为字符串
            result["reason"] = "; ".join(result["reason"])
            return result
        
        # 计算涨幅限制
        daily_return = (c_t / c_t1 - 1)
        result["daily_return"] = daily_return
        result["reason"].append(f"日涨幅: {daily_return:.2%}")
        
        if daily_return > 0.1:  # 超过 10%
            result["reason"].append("日涨幅超过 10%")
            # 将reason列表合并为字符串
            result["reason"] = "; ".join(result["reason"])
            # 保留计算结果，但将score设为None
            return result
        
        # 计算偏离风险 (RISK)
        risk = (c_t - ema20_t) / ema20_t
        result["risk"] = risk
        result["reason"].append(f"偏离风险: {risk:.2%}")
        
        if risk <= 0 or risk > 0.06:  # 风险不在 0 到 6% 之间
            if risk <= 0:
                result["reason"].append("偏离风险不大于 0")
            else:
                result["reason"].append("偏离风险超过 6%")
            # 将reason列表合并为字符串
            result["reason"] = "; ".join(result["reason"])
            # 保留计算结果，但将score设为None
            return result
        
        # 计算相对强度 (RS)
        stock_5d_return = (c_t / c_t5 - 1)
        qqq_5d_return = (qqq_c_t / qqq_c_t5 - 1)
        result["stock_5d_return"] = stock_5d_return
        result["qqq_5d_return"] = qqq_5d_return
        result["reason"].append(f"个股5日涨幅: {stock_5d_return:.2%}")
        result["reason"].append(f"QQQ5日涨幅: {qqq_5d_return:.2%}")
        
        # 5日动能 短期惯性 动能确认
        if stock_5d_return <= 0:
            result["reason"].append("5日动能不大于 0")
            # 将reason列表合并为字符串
            result["reason"] = "; ".join(result["reason"])
            # 保留计算结果，但将score设为None
            return result
        
        if qqq_5d_return == 0:
            rs = float('inf')  # 避免除零
            result["reason"].append("QQQ5日涨幅为0，RS设为无穷大")
        else:
            rs = stock_5d_return / qqq_5d_return
            result["reason"].append(f"相对强度: {rs:.2f}")
        
        result["rs"] = rs
        
        if rs <= 1.0:
            result["reason"].append("相对强度不大于 1.0")
            # 将reason列表合并为字符串
            result["reason"] = "; ".join(result["reason"])
            # 保留计算结果，但将score设为None
            return result
        
        # 计算动能分 M
        # 动能分M = 突破强度 + 5日动能
        # 突破强度 = CLOSE_t / HHV20_{t-1} − 1
        # 5日动能 = CLOSE_t / CLOSE_{t-5} − 1
        m1 = (c_t / hhv20_t1 - 1)  # 突破强度
        m3 = stock_5d_return  # 5日动能
        M = m1 + m3
        result["M"] = M
        result["reason"].append(f"突破强度: {m1:.2%}")
        result["reason"].append(f"5日动能: {m3:.2%}")
        result["reason"].append(f"动能核心: {M:.2%}")
        
        # 计算系数 Multiplier
        if risk < 0.03:
            multiplier = 1.2
            result["reason"].append("风险系数: 1.2")
        else:  # 3% ≤ RISK ≤ 6%
            multiplier = 1.0
            result["reason"].append("风险系数: 1.0")
        
        result["multiplier"] = multiplier
        
        # 计算最终得分
        score = M * multiplier
        # 删除将 score 转化为百分比的代码
        result["score"] = score
        result["reason"].append(f"原始评分: {score:.4f}")
        result["reason"].append("通过所有硬过滤")
        
        # 将reason列表合并为字符串
        result["reason"] = "; ".join(result["reason"])
        
        return result
        
    except Exception as e:
        result["reason"].append(f"计算错误: {str(e)}")
        # 将reason列表合并为字符串
        result["reason"] = "; ".join(result["reason"])
        return result
