# pip install pandas matplotlib numpy  (可选: scipy 用于 PCHIP 插值)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 可选：PCHIP 插值；若没有 scipy 将退化为线性插值 ---
try:
    from scipy.interpolate import PchipInterpolator
    HAS_PCHIP = True
except Exception:
    HAS_PCHIP = False

def hampel_filter(series, k, n_sigmas=3.0):
    """
    Hampel 滤波（窗口=2k+1）。返回 去尖刺后的数组 与 布尔索引 outliers。
    使用 rolling median + MAD 向量化实现。
    """
    s = pd.Series(series).copy()
    win = 2 * k + 1
    med = s.rolling(win, center=True, min_periods=1).median()
    abs_dev = (s - med).abs()
    mad = abs_dev.rolling(win, center=True, min_periods=1).median()
    sigma = 1.4826 * mad
    outliers = abs_dev > (n_sigmas * sigma)
    s_f = s.copy()
    s_f[outliers] = med[outliers]
    return s_f.to_numpy(), outliers.to_numpy()


def replace_by_interp(x, mask):
    """把 mask=True 的点用相邻非异常点插值替换（优先 PCHIP，其次线性）。"""
    x = np.asarray(x, float).copy()
    idx_bad = np.flatnonzero(mask)
    idx_ok = np.flatnonzero(~mask)
    if len(idx_bad) and len(idx_ok) >= 2:
        if HAS_PCHIP:
            f = PchipInterpolator(idx_ok, x[idx_ok], extrapolate=True)
            x[idx_bad] = f(idx_bad)
        else:
            x[idx_bad] = np.interp(idx_bad, idx_ok, x[idx_ok])
    return x

def denoised_data(csv_file):
    df = pd.read_csv(csv_file,sep=';')
    # t = parse_time(df.iloc[:, 0])
    t = df.iloc[:, 0].to_numpy(float)
    V = df.iloc[:, 1].to_numpy(float)
    I = df.iloc[:, 2].to_numpy(float)
    Zfeq = df.iloc[:, 3].to_numpy(float)
    Zre = df.iloc[:, 4].to_numpy(float)
    Zim = df.iloc[:, 5].to_numpy(float)

    # -------- 去尖刺（Voltage / Current）--------
    win = max(5, round(0.01 * len(V)))   # 窗口 k（两侧各 k 个点）
    thr = 3.0                            # 阈值（倍数 MAD）

    # Voltage：优先用 Hampel；否则用移动中位数+插值
    V_d, idxV = hampel_filter(V, k=win, n_sigmas=thr)
    # 你若想与 MATLAB else 分支一致，用插值替换异常点：
    V_d = replace_by_interp(V, idxV)     # 可注释此行保持“Hampel替换为中位数”

    # Current
    I_d, idxI = hampel_filter(I, k=win, n_sigmas=thr)
    I_d = replace_by_interp(I, idxI)

    new_data=pd.DataFrame({'t':t,'V':V_d,'I':I,'Zfeq':Zfeq,'Zre':Zre,'Zim':Zim})
    new_data.columns=df.columns
    new_data.to_csv('./denoised_data.csv',encoding='utf_8_sig',index=None)