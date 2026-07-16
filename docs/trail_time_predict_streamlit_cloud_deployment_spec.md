# Trail Time Predict 在线部署方案（供 Codex 执行）

## 1. 目标

将当前本地运行的 Python + Streamlit 项目部署为可公开访问的网站。

部署架构：

```text
GitHub 仓库
    ↓
Streamlit Community Cloud
    ↓
自动安装依赖
    ↓
运行 app.py
    ↓
生成公开访问地址
```

本阶段不引入：

- 自建服务器
- Docker
- Nginx
- 数据库
- 对象存储
- 独立后端
- 自定义域名

## 2. 部署方式

使用 GitHub + Streamlit Community Cloud。

GitHub 用于保存源码；Streamlit Community Cloud 用于拉取代码、创建 Python 环境、安装依赖、启动应用并提供公开 HTTPS 地址。

## 3. 项目入口

入口文件：

```text
app.py
```

云端启动方式：

```bash
streamlit run app.py
```

云端不依赖 `start_web.bat`。Windows 批处理文件只用于本地运行。

## 4. Codex 执行任务

1. 确保 `app.py` 可以在 Linux 环境运行。
2. 补全 `requirements.txt`。
3. 移除本地绝对路径依赖。
4. 使用临时目录或内存处理上传文件。
5. 增加上传文件数量和大小限制。
6. 增加 FIT / GPX 隐私提示。
7. 增加异常处理和友好错误信息。
8. 增加 Streamlit Cloud 部署说明。
9. 更新 README，增加在线体验入口。
10. 保证本地运行方式不受影响。

## 5. requirements.txt 检查

至少确认包含：

```txt
streamlit
pandas
numpy
scipy
matplotlib
plotly
fitparse
gpxpy
```

如项目实际使用其他第三方依赖，也必须补充。

要求：

- 不要加入 Python 标准库。
- 不要遗漏运行时依赖。
- 避免过度严格且可能冲突的版本锁定。
- 当前版本稳定时，可保留合理版本范围。

示例：

```txt
streamlit>=1.35
pandas>=2.0
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
plotly>=5.20
fitparse>=1.2
gpxpy>=1.6
```

## 6. 跨平台路径处理

禁止使用 Windows 绝对路径，例如：

```python
"D:\\projects\\trail_time_predict\\data"
```

统一使用：

```python
from pathlib import Path
```

示例：

```python
BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
```

上传文件和中间结果不应写入固定项目目录。

## 7. 上传文件处理

FIT 支持多文件上传：

```python
uploaded_fit_files = st.file_uploader(
    "上传 FIT 文件",
    type=["fit"],
    accept_multiple_files=True,
)
```

GPX 支持单文件上传：

```python
uploaded_gpx_file = st.file_uploader(
    "上传 GPX 文件",
    type=["gpx"],
    accept_multiple_files=False,
)
```

## 8. 临时目录处理

用户上传的 FIT、GPX 和中间文件统一放入临时目录。

```python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as temp_dir:
    temp_path = Path(temp_dir)
```

示例：

```python
fit_path = temp_path / uploaded_file.name
fit_path.write_bytes(uploaded_file.getvalue())
```

要求：

- 不依赖 `data/activities` 长期保存用户文件。
- 不依赖 `output/` 长期保存分析结果。
- 云端实例重启后临时文件消失是正常行为。
- 当前产品流程应保持为“上传 → 分析 → 返回结果”。
- 不提供永久保存能力。

## 9. 内存优先处理

如果解析函数支持 bytes 或 file-like object，优先直接使用内存数据：

```python
file_bytes = uploaded_file.getvalue()
```

只有解析库必须要求文件路径时，再写入临时目录。

## 10. 缓存策略

可使用：

```python
@st.cache_data
```

缓存重复解析结果。

推荐使用文件内容哈希或 bytes 作为缓存输入，不要直接使用 `UploadedFile` 对象。

```python
import hashlib

def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
```

注意：

- 不将用户敏感数据写入永久存储。
- 缓存函数返回值应可序列化。
- 大型解析结果应控制缓存规模。

## 11. 上传限制

建议配置：

```python
MAX_FIT_FILES = 30
MAX_SINGLE_FILE_MB = 20
MAX_TOTAL_UPLOAD_MB = 100
```

错误提示示例：

```text
最多上传 30 个 FIT 文件。
单个文件不能超过 20 MB。
本次上传总大小不能超过 100 MB。
```

## 12. 文件类型与异常校验

需要处理：

- 扩展名错误
- 文件损坏
- FIT 无 record 数据
- FIT 缺失距离
- FIT 缺失海拔
- FIT 缺失心率
- FIT 缺失温度
- GPX 无轨迹点
- GPX 无海拔
- GPX 路线过短
- GPX 数据异常
- 单个文件解析失败

原则：

- 单个 FIT 失败时，其他文件继续分析。
- 展示失败文件名和原因。
- 不向普通用户输出 Python 堆栈。
- 详细异常写入服务日志。

## 13. 用户隐私提示

FIT 文件可能包含：

- 精确 GPS 轨迹
- 家庭或工作地址附近的起终点
- 活动日期和时间
- 心率
- 温度
- 训练习惯
- 设备信息

在上传区域附近增加：

```text
隐私提示：

上传的 FIT 和 GPX 文件仅用于本次分析，应用不会主动将文件写入项目仓库或长期保存。

FIT 和 GPX 可能包含精确位置、活动时间、心率等隐私信息。上传前请确认你愿意处理这些数据。
```

禁止：

- 将上传文件提交到 GitHub。
- 将用户文件保存到公开目录。
- 在日志中输出完整轨迹。
- 在错误信息中展示用户完整本地路径。
- 将文件内容发送给无关第三方服务。

## 14. 页面状态与进度提示

建议增加：

```python
with st.spinner("正在解析 FIT 文件并建立能力画像..."):
    ...
```

流程：

```text
1. 校验上传文件
2. 解析 FIT
3. 建立个人能力画像
4. 解析 GPX
5. 生成路线分段
6. 计算预测
7. 生成报告
```

## 15. 资源控制

免费实例资源有限，应避免：

- 一次读取大量超大 FIT。
- 无限 Monte Carlo 次数。
- 重复解析相同文件。
- 长时间阻塞主线程。
- 生成超高分辨率图片。
- DataFrame 无限增长。

建议：

```text
默认 Monte Carlo：3000 次
最大 Monte Carlo：10000 次
最大 FIT 数量：30
最大总上传体积：100 MB
```

## 16. Streamlit 配置

检查或新增：

```text
.streamlit/config.toml
```

建议内容：

```toml
[server]
headless = true
maxUploadSize = 200

[browser]
gatherUsageStats = false

[theme]
base = "light"
```

不得将密码、Token 或密钥写入配置文件。

## 17. 密钥管理

当前项目如果没有外部 API，则不需要密钥。

未来如加入天气、地图、数据库或对象存储，应使用 Streamlit Secrets：

```python
api_key = st.secrets["WEATHER_API_KEY"]
```

真实密钥不得提交到 GitHub。

## 18. Streamlit Cloud 部署步骤

代码调整完成后，由用户执行：

1. 打开 Streamlit Community Cloud。
2. 使用 GitHub 登录。
3. 授权访问仓库。
4. 新建应用。
5. 填写：

```text
Repository:
highcoldBaSO4/trail_time_predict

Branch:
main

Main file path:
app.py
```

6. 设置应用子域名。
7. 点击 Deploy。
8. 查看构建日志。
9. 修复缺失依赖或路径问题。
10. 部署成功后获得公开网址。

## 19. README 更新

README 顶部增加在线体验入口：

```markdown
## 在线体验

[打开 Trail Time Predict](https://YOUR-APP-NAME.streamlit.app)
```

也可增加徽章：

```markdown
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-NAME.streamlit.app)
```

README 还应增加：

- 在线体验说明
- 本地运行说明
- 数据隐私说明
- 上传限制
- 支持的文件格式
- 常见问题

## 20. 本地运行兼容

部署改造不能破坏本地运行：

```bash
pip install -r requirements.txt
streamlit run app.py
```

Windows 用户可以继续使用现有 bat 文件，但核心程序不能依赖它。

## 21. 首页文案

建议首页增加：

```text
上传你的历史 FIT 训练数据和目标比赛 GPX，生成个人越野能力画像与完赛时间预测。
```

简单流程：

```text
第一步：上传历史 FIT 文件
第二步：上传比赛 GPX
第三步：设置比赛条件
第四步：生成预测报告
```

## 22. 错误处理要求

顶层分析入口增加异常保护：

```python
try:
    result = run_prediction(...)
except UserInputError as exc:
    st.warning(str(exc))
except Exception:
    st.error("分析过程中出现异常，请检查上传文件后重试。")
    logger.exception("prediction failed")
```

要求：

- 用户输入问题用 warning。
- 文件解析问题用 error。
- 未知异常记录日志。
- 不向用户展示敏感堆栈信息。

## 23. 日志要求

使用标准 logging：

```python
import logging

logger = logging.getLogger(__name__)
```

允许记录：

- 文件数量
- 文件大小
- 解析成功或失败
- 分析耗时
- 异常类型

禁止记录：

- 完整 GPS 轨迹
- 完整 FIT 内容
- 用户精确位置
- 用户文件二进制内容
- 用户隐私数据

## 24. 测试要求

### 本地测试

验证：

- 单个 FIT
- 多个 FIT
- 单个 GPX
- FIT 缺少心率
- FIT 缺少温度
- GPX 缺少海拔
- 损坏文件
- 超出上传数量
- 超出大小限制

### Linux 兼容测试

检查：

- 无 Windows 路径
- 文件名包含中文
- 文件名包含空格
- 大小写路径问题
- UTF-8 编码

### 部署测试

验证：

- 页面正常打开
- 上传正常
- 分析正常
- 报告下载正常
- 多用户会话互不污染
- 刷新页面不会读取其他用户文件

## 25. 会话隔离

不同用户的上传文件和结果必须隔离。

禁止使用全局共享路径：

```text
output/result.json
data/current.fit
```

应使用：

- 临时目录
- `st.session_state`
- 每次请求独立的内存对象

## 26. 验收标准

完成后必须满足：

- GitHub 仓库可被 Streamlit Cloud 直接部署。
- 入口文件为 `app.py`。
- `requirements.txt` 完整。
- Linux 环境下无路径错误。
- 上传文件使用临时目录或内存处理。
- 不长期保存用户 FIT 和 GPX。
- 页面包含隐私提示。
- 页面包含上传限制提示。
- 单个损坏文件不会导致整个应用崩溃。
- 多用户会话数据隔离。
- 在线应用可以生成预测结果。
- 在线应用可以下载报告。
- README 包含在线体验入口。
- 本地运行方式继续有效。

## 27. Codex 执行顺序

### Phase 1：部署兼容

1. 检查依赖。
2. 清理本地绝对路径。
3. 临时文件改造。
4. Linux 兼容。
5. 本地运行验证。

### Phase 2：安全和稳定

1. 文件大小限制。
2. 文件数量限制。
3. 异常处理。
4. 日志脱敏。
5. 用户隐私提示。
6. 会话隔离。

### Phase 3：部署文件

1. 检查 `.streamlit/config.toml`。
2. 更新 README。
3. 增加部署说明。
4. 增加在线体验占位链接。

### Phase 4：测试

1. 运行单元测试。
2. 本地 Streamlit 测试。
3. 模拟异常文件。
4. 检查 Linux 兼容。
5. 输出部署检查清单。

不要一次性重写整个项目，应尽可能保留现有可工作的模块。
