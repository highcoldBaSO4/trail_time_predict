# 越野跑比赛时间概率预测系统 V0.2

本项目是一个本地运行的 Python 工具。系统读取个人历史 FIT 活动和比赛 GPX 路线，建立平路、上坡、下坡、持续能力及疲劳画像，并输出带可信度和 P10/P50/P90 区间的完赛时间预测。

所有 FIT 和 GPX 数据仅在当前电脑中处理，不依赖数据库或云端服务。

## 当前功能

- 批量解析历史 FIT 活动并检查数据质量
- 上传后确认活动属于越野或路跑，并选择是否用于建模
- 从越野和路跑样本中建立带可信度的个人能力画像
- 识别 GPX 中的自然平路、连续爬坡和连续下降
- 使用连续坡度能力曲线和分地形连续疲劳曲线预测逐段时间
- 根据目标比赛时长匹配短时、中时、长时和超长持续能力
- 手动设置当前状态、温湿度及相对日常训练的技术、泥泞和负重影响
- 根据 FIT 时间、坐标和海拔建立历史夜间及海拔覆盖画像
- 根据比赛日期、出发时间和 GPX 自动识别夜间路段及海拔影响
- 使用 Monte Carlo 输出 P10、P50、P90 和综合预测可信度
- 在网页中展示能力、路线、分段、时间损耗和完整报告
- 下载 Markdown 报告和预测 JSON

## 环境要求

- Python 3.11+
- Windows、macOS 或 Linux

主要依赖统一记录在 `requirements.txt`，开发测试依赖记录在 `requirements-dev.txt`。

## 安装

进入项目目录后创建独立虚拟环境：

```powershell
python -m venv .venv
```

不需要激活虚拟环境，直接安装依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS 或 Linux 使用：

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

## 网页使用

Windows 可以直接双击：

```text
start_web.bat
```

也可以从命令行启动：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

默认访问地址：

```text
http://localhost:8501
```

网页操作流程：

1. 上传多个历史 FIT。
2. 点击“解析并确认活动”。
3. 确认每个活动属于越野或路跑，并选择是否用于建模。
4. 上传比赛 GPX。
5. 填写比赛状态、条件、日期、出发时间和时区。
6. 点击“开始计算”，查看或下载预测报告。

不建议用于建模的活动默认不会被选中，用户仍可手动调整。

## 命令行使用

默认读取：

```text
data/activities/*.fit
data/races/race.gpx
```

运行：

```powershell
.\.venv\Scripts\python.exe main.py --aid-minutes 20
```

完整示例：

```powershell
.\.venv\Scripts\python.exe main.py `
  --activities D:\runs `
  --race D:\races\race.gpx `
  --output output `
  --segment-distance 100 `
  --current-form slight_fatigue `
  --temperature 28 `
  --humidity 80 `
  --technical-level 2 `
  --mud-level 1 `
  --carried-weight 1.0 `
  --race-start 2026-08-01T06:00:00+08:00 `
  --aid-minutes 20
```

`--race-start` 必须使用包含时区的 ISO 8601 时间。提供比赛出发时间后，系统会结合 GPX 经纬度自动判断每个分段的昼夜状态。

## 输出文件

命令行运行后，`output/` 包含：

| 文件 | 内容 |
| --- | --- |
| `runner_profile.json` | 个人能力、可信度、疲劳、活动质量和历史环境覆盖 |
| `race_segments.json` | 自然地形分段、坡度、坐标和海拔 |
| `prediction.json` | P10/P50/P90、逐段预测和修正明细 |
| `report.md` | 可阅读的完整预测报告 |
| `elevation_profile.png` | 路线相对海拔图 |

## 模型口径

### 活动与地形

- 平路统一定义为 `-2%～+2%`。
- 微坡为大于 2% 至 5%，之后为 5%～10%、10%～15% 和 15% 以上；下坡使用对称范围。
- 越野与路跑平路样本会加权组合，路跑配速先折算为越野条件。
- 上坡 VAM、下坡速度和疲劳系数均使用连续插值，避免档位边界突变。
- 活动时间新旧、活动类型、样本覆盖和数据质量会影响能力权重及可信度。

### 持续能力与疲劳

- 系统区分短时、中时、长时和超长持续能力。
- 平路、上坡和下坡分别建立疲劳曲线。
- 持续能力用于匹配比赛目标时长，疲劳曲线用于修正比赛进行过程中的能力衰减，两者分开计算。
- 缺少对应样本时使用保守默认值，并降低可信度。

### 比赛条件

- 当前状态、温度和湿度由用户填写。
- 技术难度、泥泞和携带重量表示相对于日常训练的额外影响，避免与历史 FIT 中已有影响重复计算。
- 历史夜间比例根据 FIT 时间戳和经纬度按平路、上坡、下坡分别统计；夜间平路不计入夜间损耗，只保留为普通平路能力样本。
- 比赛夜间路段根据比赛时间、预计到达时间和 GPX 经纬度逐段判断；夜间损耗仅应用于上坡和下坡，平路不额外折减。
- 比赛海拔影响根据 GPX 分段海拔相对于历史 FIT 训练海拔计算。
- 当比赛海拔明显超出历史覆盖时，系统会降低可信度并给出风险提示。

### 概率预测

报告明确区分：

- 标准能力移动时间
- 条件修正后移动时间
- 补给停留时间
- 最快合理时间 P10
- 中位预测时间 P50
- 保守预测时间 P90
- 综合预测可信度

时间损耗拆解包括基础地形、目标时长适配、疲劳、状态、温湿度、技术、泥泞、夜间、海拔、负重和补给。

## 配置

经验参数集中在 `config/defaults.yaml`，包括：

- 默认能力与疲劳曲线
- 活动时间和类型权重
- 能力可信度规则
- FIT/GPX 数据质量阈值
- 平路坡度边界
- 状态、技术、泥泞、负重和温湿度系数
- 夜间太阳高度阈值
- 海拔影响及历史覆盖阈值
- Monte Carlo 次数和波动参数

当前默认将太阳高度低于 `-6°` 视为夜间。海拔从 1500 米以上开始修正，每增加 1000 米默认增加约 4% 时间；比赛最高海拔超过历史 P90 海拔 500 米时降低预测可信度。这些都是可校准的 V0.2 经验参数。

## 测试

安装开发依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

当前测试覆盖 FIT/GPX 解析、自然坡分段、活动确认、能力画像、连续插值、疲劳、条件修正、概率区间、类型模型、昼夜判断和海拔修正。当前结果为 `25 passed`。

## 当前开发状态

V0.2 Phase 1 和 Phase 2 的主要功能已经接入本地 CLI、Streamlit 页面、JSON 输出和 Markdown 报告。

尚未实现的主要功能：

- 历史比赛 FIT 与 GPX 对齐及表现解释
- 实际成绩在预测分布中的百分位
- 多场比赛系统性偏差识别与保守校准建议
- 分补给站到达和离开时间
- 目标完赛时间反推
