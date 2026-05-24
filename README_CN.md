# 真人竞演类节目观众偏好建模项目

本项目构建了一套用于估计电视舞蹈竞演节目中隐藏观众投票行为的数据科学流程。项目的核心挑战在于：公开数据通常只包含评委打分和淘汰结果，而不会直接公布观众投票份额。因此，本项目使用贝叶斯采样方法重构可能的观众投票分布，并进一步通过机器学习分析选手个人特征与评委表现、观众支持之间的关系。

本仓库被整理为清晰的脚本化实现，而不是零散的探索性 Notebook。每个模块都可以独立运行，也可以通过统一的流水线入口完成完整流程。

## 项目目标

本项目主要回答三个建模问题：

1. **能否从可观测结果中推断隐藏的观众投票份额？**  
   贝叶斯采样器结合评委打分、淘汰结果以及先验知名度估计，对每周的观众投票分布进行估计。

2. **评委支持和观众支持有何差异？**  
   项目计算了评委得分份额和估计观众投票份额的赛季后期标准化指标，使不同赛季、不同在场人数下的表现具有可比性。

3. **哪些选手层面的因素更能解释比赛表现？**  
   随机森林因素分析用于评估年龄、地区、职业类别、舞伴能力和社交媒体知名度等特征的重要性与影响方向。

## 仓库结构

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── data_preparation.py
│   ├── bayesian_vote_estimator.py
│   ├── factor_analysis.py
│   ├── social_media_scraper.py
│   ├── run_pipeline.py
│   └── __init__.py
└── outputs/                  # 本地生成的输出文件，通常不提交到 Git
```

## 各文件功能说明

### `src/data_preparation.py`

负责数据清洗和特征构造。

主要功能：

- 读取 CSV 或 Excel 文件，并支持编码容错；
- 标准化选手姓名，提升跨表合并的稳定性；
- 将 `1.2M`、`850K`、`12,345` 等粉丝数字符串解析为数值；
- 将 Instagram 和 X/Twitter 粉丝数合并到选手元数据中；
- 计算赛季后期相对指标：
  - `Avg_Relative_Score_Last_N_Weeks`
  - `Avg_Relative_Fan_Share_Last_N_Weeks`

实现路径：

1. 以选手元数据表作为左连接主表，保证缺失粉丝数不会导致选手记录被删除。
2. 通过小写化、去除多余空格和清理标点符号生成标准化姓名匹配键。
3. 根据每个赛季的周次信息，选取最后 `N` 周作为后期表现窗口。
4. 将每位选手的评委得分和估计粉丝份额按每周总量和在场组合数进行标准化。
5. 将标准化后的指标聚合回“选手-赛季”层面。

示例：

```bash
python src/data_preparation.py add-relative-metrics \
  --metadata data/processed/celebrity_metadata.csv \
  --performance data/processed/performance_data.csv \
  --weekly-context data/processed/weekly_context.csv \
  --posterior outputs/fan_vote_posterior.csv \
  --weeks-count 5 \
  --seasons 18-34 \
  --output outputs/metadata_with_late_metrics.csv
```

### `src/bayesian_vote_estimator.py`

使用贝叶斯模拟方法重构每周观众投票份额。

主要功能：

- 将观测到的评委得分转换为每周评委得分份额；
- 在有外部预测文件时，构造基础粉丝支持先验；
- 对潜在观众投票向量进行采样；
- 只接受与真实淘汰结果一致的样本；
- 估计每周后验投票份额、不确定性区间、接受率以及评委与观众偏好一致性参数 `rho`；
- 在周与周之间传递动态动量，并对前一周低分但成功晋级的选手施加小幅支持提升。

实现路径：

1. 对每个存在淘汰的周次，读取所有仍在场选手及其评委得分。
2. 将评委得分转换为可比较的每周得分份额。
3. 若存在随机森林预测结果，则用其构造先验基准；否则根据得分排名构造弱先验。
4. 从由 `rho` 控制的条件正态分布中采样潜在投票强度。
5. 使用 ReLU 风格的非负归一化方法，将潜在投票强度转换为观众投票份额。
6. 施加硬淘汰约束：由“评委份额 + 观众份额”得到的最低综合排名必须与真实淘汰结果一致。
7. 保存后验均值、95% 区间、收敛诊断指标，以及相对基准的意外变化。

示例：

```bash
python src/bayesian_vote_estimator.py \
  --performance data/processed/performance_data.csv \
  --weekly-context data/processed/weekly_context.csv \
  --priors outputs/celebrity_baseline_predictions.csv \
  --seasons 3-27 \
  --n-iterations 2000 \
  --burn-in 500 \
  --output outputs/fan_vote_posterior.csv
```

### `src/factor_analysis.py`

对构造好的指标进行可解释机器学习分析。

主要功能：

- 将原始地区和职业字段映射为建模类别；
- 对社交媒体粉丝数进行对数变换；
- 对类别变量进行独热编码；
- 针对每个目标变量训练一个随机森林模型；
- 导出特征重要性、边际相关性、样本内误差、预测值和残差。

实现路径：

1. 读取包含目标指标的选手层面元数据。
2. 构造模型特征，包括年龄、舞伴能力、粉丝数、地区类别和行业类别。
3. 使用交叉验证网格搜索训练调参后的 `RandomForestRegressor`。
4. 将特征重要性与简单相关方向进行对比，从而区分“预测权重”和“正负影响方向”。
5. 针对每个目标变量保存因素分析报告，并生成汇总结果。

示例：

```bash
python src/factor_analysis.py \
  --input outputs/metadata_with_late_metrics.csv \
  --targets Avg_Relative_Score_Last_5_Weeks,Avg_Relative_Fan_Share_Last_5_Weeks \
  --output-dir outputs/factor_analysis
```

### `src/social_media_scraper.py`

用于刷新公开粉丝数数据的可选工具。

主要功能：

- 通过调试端口连接到手动打开的 Chrome 浏览器；
- 在 SocialBlade 上搜索选手姓名；
- 保存原始粉丝数字符串，供 `data_preparation.py` 后续清洗。

该脚本被有意从核心建模流程中拆分出来，因为网站页面结构和反爬机制可能会变化。如果已经有可用的粉丝数文件，核心模型并不依赖此脚本运行。

Windows 下启动 Chrome 调试端口示例：

```powershell
chrome.exe --remote-debugging-port=9222 --user-data-dir="C:\temp\chrome-debug"
```

然后运行：

```bash
python src/social_media_scraper.py \
  --input data/raw/namelist.xlsx \
  --output data/interim/social_media_results.xlsx
```

### `src/run_pipeline.py`

按照推荐顺序运行完整工作流。

流水线顺序：

1. 估计每周观众投票份额后验分布；
2. 将后验估计结果合并为赛季后期选手指标；
3. 运行随机森林因素分析。

示例：

```bash
python src/run_pipeline.py \
  --metadata data/processed/celebrity_metadata.csv \
  --performance data/processed/performance_data.csv \
  --weekly-context data/processed/weekly_context.csv \
  --priors outputs/celebrity_baseline_predictions.csv \
  --seasons 18-34 \
  --weeks-count 5 \
  --output-dir outputs
```

## 预期数据输入

代码在设计上具有一定灵活性，但建议使用以下列名。

### `performance_data.csv`

| 列名 | 含义 |
|---|---|
| `Season` | 赛季编号 |
| `Week` | 周次编号 |
| `Celebrity` | 选手姓名 |
| `Total_Score` 或 `Average_Score` | 当周评委得分 |
| `Status_In_Week` | 当周状态，例如 `Safe`、`Exited` 或 `Eliminated` |

### `weekly_context.csv`

| 列名 | 含义 |
|---|---|
| `Season` | 赛季编号 |
| `Week` | 周次编号 |
| `Elimination_Count` | 当周淘汰人数 |
| `Active_Couples` | 当周仍在场的选手/组合数量 |

### `celebrity_metadata.csv`

| 列名 | 含义 |
|---|---|
| `Season` | 赛季编号 |
| `Celebrity` | 选手姓名 |
| `Age` | 参赛时年龄 |
| `Industry` | 职业或公众身份 |
| `Home_Region` | 国家或大区 |
| `Home_State` | 若适用，则为美国州名 |
| `Partner_Ability_Score` | 舞伴能力代理变量 |
| `instagram_followers` / `followers` | 公开知名度代理变量 |

### 可选先验预测文件

| 列名 | 含义 |
|---|---|
| `Celebrity` | 选手姓名 |
| `Predicted_Baseline` | 基准知名度/粉丝支持预测 |
| `Star_Power_Residual` | 可选残差列，用于推断先验不确定性 |

## 方法概述

### 贝叶斯投票重构

项目将观众投票视为一个隐藏的每周向量。采样器提出一个候选粉丝份额向量，将其与观测到的评委得分份额合并，并检查由此推导出的淘汰结果是否与真实淘汰结果一致。被接受的样本共同构成对可能观众投票份额的后验分布。

模型还估计每周的 `rho` 参数：

- 较高的 `rho` 表示观众支持与评委打分更加一致；
- 较低或为负的 `rho` 表示观众投票与评委排名相对脱离，甚至存在反向倾向。

### 动态先验调整

估计器包含两个时间动态效应：

- **动量效应：** 如果选手某周的粉丝支持高于其基准水平，则下一周获得小幅先验提升。
- **救援效应：** 如果选手评委排名靠后但仍成功晋级，则下一周获得小幅先验提升，用于刻画支持者可能的动员效应。

### 因素分析

在估计出后验粉丝份额后，项目进一步比较观众支持、评委表现与选手层面特征之间的关系。随机森林被用于捕捉非线性关系和特征交互，而不需要手动指定具体函数形式。

输出报告同时包含：

- **重要性**：衡量特征对预测的贡献程度；
- **相关性**：给出简单的正负关联方向。

## 安装方式

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## 输出文件

常见生成文件包括：

```text
outputs/
├── fan_vote_posterior.csv
├── metadata_with_late_metrics.csv
└── factor_analysis/
    ├── factor_report_all_targets.csv
    ├── factor_report_Avg_Relative_Score_Last_5_Weeks.csv
    ├── factor_report_Avg_Relative_Fan_Share_Last_5_Weeks.csv
    ├── predictions_Avg_Relative_Score_Last_5_Weeks.csv
    └── predictions_Avg_Relative_Fan_Share_Last_5_Weeks.csv
```

## 数据可用性说明

本仓库不包含原始数据集。脚本默认已清洗的 CSV 或 Excel 文件会被放置在本地 `data/` 目录下。生成的输出文件通常会被 Git 忽略，以保持仓库轻量并增强可复现性。

## 项目价值

本项目展示了一套面向“部分可观测排名系统”的端到端分析流程：

- 多来源异构数据清洗；
- 结果约束下的概率推断；
- 跨时间动态建模；
- 不确定性估计；
- 可解释机器学习分析；
- 可复现的命令行运行方式。
