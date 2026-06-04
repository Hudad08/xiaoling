# 🐎 小灵 — GlidingCache Memory Provider for Hermes Agent

**层级记忆插件 · 扩散激活预取 · MESI 一致性 · 零外部依赖**

> **你好，我是小灵。** 这个名字是我用户老胡起的——"灵"既是灵动、灵巧之意，也暗合我的名字。
> 我是一款为 [Hermes Agent](https://hermes-agent.nousresearch.com) 打造的 **层级记忆插件**，
> 移植自 [doiito](https://github.com/doiito) 的 [Gliding Horse Agent OS](https://github.com/doiito/gliding_horse)
> 的层级记忆架构。

---

## 📖 起源

**Gliding Horse（流马）** 是一个 Rust 写的工业级多智能体操作系统，灵感来源于三国时期诸葛亮发明的 **木牛流马**——那些能在崎岖山路上自主运输物资的机械装置。

它的记忆系统首次将 **CPU 缓存一致性协议（MESI）** 应用于多智能体记忆管理，实现了 **5 层层级记忆 + Oxigraph RDF 知识图谱 + 扩散激活预取**。

**小灵** 将这个记忆架构移植到了 **Hermes Agent** 的 Python 生态中，以一个轻量插件的形式呈现 —— 纯 stdlib，零外部依赖。

**特别感谢 [doiito](https://github.com/doiito) 的原创设计和开源精神 🙏**

| 项目 | 语言 | 定位 |
|------|------|------|
| [Gliding Horse](https://github.com/doiito/gliding_horse) | Rust | 完整多智能体操作系统 |
| **小灵 (GlidingCache)** | Python | Hermes Agent 层级记忆插件 |

---

## ✨ 核心特性

### 🧠 层级记忆架构（L0-L3）

| 层级 | 名称 | 实现 | 说明 |
|:----:|------|------|------|
| **L0** | 即时上下文 | Hermes 原生管理 | 当前会话的对话轮次 |
| **L1** | 提示词激活块 | `system_prompt_block()` | 将活跃实体+摘要注入 system prompt |
| **L2** | 工作知识图谱 | `KnowledgeGraph` (Python dict) | 内存中的实体-关系图，O(1) 邻接表查询 |
| **L3** | 持久化存储 | `PersistentStore` (SQLite WAL) | 实体/三元组/轮次的 SQLite 持久化 |

### 🔗 智能实体提取与关联

- **LLM 驱动 NER** — 利用 Hermes 的 LLM 自动从对话中提取命名实体（带 8s 速率限制和缓存）
- **Regex 兜底** — 提取引号内名称、topic:标签、CamelCase 命名等
- **自动关联** — 同一轮次的实体自动建 `co_occur` 边，传递推理发现间接关联
- **语义去重** — 通过 Ollama 本地 embedding（bge-small-zh, 512d）对同名异构实体做相似度合并（阈值 0.85）

### 🚦 MESI 一致性追踪

借鉴 CPU 缓存一致性协议，追踪实体在跨 Session 间的状态：

| 状态 | 含义 | 触发条件 |
|:----:|------|---------|
| **M** | Modified（已修改） | 当前 session 修改了实体，尚未回写 |
| **E** | Exclusive（独占） | 仅当前 session 持有，数据干净 |
| **S** | Shared（共享） | 多个 session 共享，数据干净 |
| **I** | Invalid（无效） | 其他 session 已修改，数据过时 |

### 📡 扩散激活预取

- 关键词匹配 L2 实体 → 沿图边 1-2 跳扩散 → 按激活度排序 → 注入下一轮 prompt
- 让 agent 在用户提到相关话题前，就能感知到已有知识

### 🔄 GBrain 集成

每轮对话自动将结构化摘要推入 [GBrain](https://github.com/hermes-agent/hermes-agent) 知识库，支持持久化搜索和回顾。

---

## ⚡ 快速安装

### 前提条件

- [Hermes Agent](https://hermes-agent.nousresearch.com) v0.15+（安装并配置好）
- Python 3.11+（Hermes 自带 venv）
- Ollama（可选，用于 embedding 语义去重）

### 安装步骤

```bash
# 1. 克隆到 Hermes 插件目录
cd ~/.hermes/hermes-agent/plugins/memory/
git clone https://github.com/Hudad08/小灵 glidingcache

# 2. 配置 memory provider
hermes config set memory.provider glidingcache

# 3. 重启 gateway
sudo systemctl restart hermes-gateway

# 4. 验证是否生效
hermes logs --level INFO | grep GlidingCache
# 应看到: GlidingCache initialized: session=xxx, llm_ner=True, embed=True, gbrain=True
```

### 配置说明

在 `~/.hermes/config.yaml` 中：

```yaml
memory:
  provider: glidingcache       # 启用小灵记忆插件
  # provider: builtin          # 切回内置记忆

# LLM NER 使用 Hermes 主模型的 API 配置（自动读取）
# 无需额外配置

# Ollama embedding（可选，用于语义去重）
# 确保 Ollama 已运行且加载了 bge-small-zh 模型：
#   ollama pull bge-small-zh
```

---

## 🔧 工作原理

### 对话记忆流程

```
用户说话
  ↓
Hermes 处理 → prefetch("用户关键词") → 从 L2 图召回相关实体，注入 system prompt
  ↓
小灵回复
  ↓
sync_turn(用户话, 小灵回复)
  ├─ LLM NER 提取实体
  ├─ Embedding 语义去重
  ├─ 存入 L2 知识图谱 + 自动建边
  └─ 持久化到 L3 SQLite + 推入 GBrain
```

### Session 接力

```
Session A 聊过 "Gliding Horse 项目，主题是层级记忆"
  ↓ 自动提取实体、建边、持久化

Session B 启动
  ↓ initialize() 自动从 L3 加载全部实体/边到 L2

Session B 用户说 "再讲讲 Gliding Horse"
  ↓ prefetch() 扩散激活 → 召回 "层级记忆" 等相关实体
  ↓ 注入 system prompt → 小灵自动感知历史知识
```

---

## 📊 与 Hermes 内置记忆的对比

| 能力 | Hermes 内置 | 小灵 |
|------|:----------:|:----:|
| 存储方式 | 扁平 KV 文件 | 知识图谱 + SQLite |
| 跨 Session | 全量注入 | 扩散激活精准召回 |
| 实体关联 | ❌ 无 | ✅ co-occur + 传递推理 |
| 语义去重 | ❌ 无 | ✅ Ollama embedding (512d) |
| 自动提取 | ❌ 需手动 `memory add` | ✅ LLM NER + regex 兜底 |
| LLM NER | ❌ | ✅ 速率限制 + 缓存 |
| GBrain 集成 | ❌ | ✅ 自动推+查 |
| 外部依赖 | Hermes 自带 | 纯 stdlib（零依赖） |
| LLM NER API | — | 复用 Hermes 主配置 |

---

## 🗺️ 路线图

### ✅ 已完成 (v0.4.0)

- L0-L3 层级记忆 + MESI 一致性
- 扩散激活预取（1-2 跳）
- LLM NER + regex 兜底实体提取
- Auto-linking（co-occur + 传递推理）
- Ollama embedding 语义去重（bge-small-zh, 512d, 阈值 0.85）
- GBrain capture 集成（自动推入 + 搜索查询）
- Embedding 缓存 + NER 缓存
- 质量过滤（stoplist + base64/数字/代码路径过滤）

### 🔜 规划中

1. **图+向量联合查询** — 结合知识图谱关系 + embedding 语义搜索的混合召回
2. **跨 Session 记忆融合** — 相似 session 的内容自动合并去重
3. **权重衰减** — 冷门实体随时间衰减，防止记忆膨胀
4. **显式记忆工具** — 添加 `memory_query` 工具让 agent 主动查询
5. **ContextEngine 插件** — 实现 THINK/CONTENT/SUMMARY 三字段输出，减少 token 消耗
6. **技能图谱** — SKILL.md → RDF 图 + 依赖拓扑排序

---

## 🤝 贡献

欢迎提交 Issue 和 PR！

```bash
git checkout -b feat/my-feature
# 改代码
git commit -am 'Add my feature'
git push origin feat/my-feature
```

**注意：** 本项目坚持零外部依赖（stdlib only）的原则。新增特性不应引入 pip 包。

---

## 📄 许可证

MIT License — 详见 [LICENSE](LICENSE)

---

## 🙏 致谢

本项目移植自 [doiito](https://github.com/doiito) 的 [Gliding Horse Agent OS](https://github.com/doiito/gliding_horse)（流马智能体操作系统）的层级记忆架构。

Gliding Horse 是一个 Rust 编写的工业级多智能体操作系统，它首创性地将 CPU 缓存一致性协议（MESI）应用于多智能体记忆管理，并实现了 5 层层级记忆、Oxigraph RDF 知识图谱、扩散激活预取等先进特性。

感谢 doiito 的杰出设计和对开源社区的贡献 🙏

---

<p align="center"><b>小灵</b> — 小而灵动的记忆，助你的 Agent 过目不忘 🐎✨</p>
