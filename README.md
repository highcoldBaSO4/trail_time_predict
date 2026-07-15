# 越野跑比赛时间预测系统 V0.1

本地读取多个历史 FIT 文件，建立个人平路、爬坡、下坡和长距离疲劳画像，再识别 GPX 中的自然爬坡、下降和平路并预测完成时间。

## 安装

需要 Python 3.11+：

```powershell
cd trail_predictor
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

开发和测试环境可改用 `pip install -r requirements-dev.txt`。

## 命令行使用

将历史文件放入 `data/activities/`，比赛路线放入 `data/races/race.gpx`，然后运行：

```powershell
python main.py --aid-minutes 20
```

也可以指定路径和分段长度：

```powershell
python main.py --activities D:\runs --race D:\races\race.gpx --output output --segment-distance 100
```

输出包括：

- `runner_profile.json`：个人能力画像
- `race_segments.json`：路线分段
- `prediction.json`：逐段预测明细
- `report.md`：预测报告
- `elevation_profile.png`：相对海拔走势

## Streamlit 界面

```powershell
streamlit run app.py
```

Windows 下也可以直接双击 `start_web.bat`，它会使用项目虚拟环境启动本地网页并自动打开浏览器。网页支持：

- 多选上传历史 Activity FIT
- 上传一个比赛 Race GPX
- 实时查看解析与计算进度
- 在网页中查看预测概览、个人能力、自然坡分段和完整报告
- 下载 Markdown 报告和预测 JSON

运行测试：

```powershell
python -m pytest -q
```

## 模型口径

- 个人地形能力：历史 FIT 与比赛 GPX 使用相同的自然坡算法，按100米窗口识别连续爬坡、下降和平路，再分别计算 VAM、下坡速度和平路配速。
- 坡度档位：上坡分为微坡1%–5%、缓坡5%–10%、中坡10%–15%、陡坡15%以上；下坡使用对称的四档。报告同时展示每档的样本数、距离、垂直高度和能力值。
- 平路：不使用心率，也不再要求连续1公里。越野自然平路样本占70%；路跑自然平路配速先乘1.10折算装备与路面影响，再占30%。仅有一种数据时自动使用对应来源。
- 上坡：按 5%–10%、10%–15%、15% 以上计算 VAM。
- 下坡：按坡度区间计算水平速度，同时记录垂直下降速度。
- 比赛路线：默认按100米采样坡度，平滑海拔后识别连续自然爬坡/下降，并合并坡中不超过200米的短暂起伏；最终预测段长度由地形决定，不再固定500米硬切。
- 疲劳：比较 0–3h、3–5h、5h 以后速度，数值代表能力保留比例；预测耗时按 `基础耗时 / 疲劳比例` 修正。
- 缺少某类有效样本时使用保守默认值，画像中的活动摘要可帮助判断样本是否足够。

V0.1 未纳入天气、路面技术难度、海拔适应、补给点分布与停留行为。首次验收应使用“历史活动训练、已完赛 FIT 建模，另一场已知结果的 GPX 做回测”，并比较预测总时间与真实移动时间。
