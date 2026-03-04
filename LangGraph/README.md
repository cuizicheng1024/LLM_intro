# LangGraph 入门案例

这个目录下包含了一个简单的 LangGraph 演示程序 `simple_agent.py`，用于展示如何构建一个基础的对话代理。

## 🎯 核心概念

这个案例演示了 LangGraph 的四个核心要素：

1.  **State (状态)**: `AgentState`
    *   定义了在图结构中流转的数据结构。
    *   本例中只包含 `messages` 列表，使用 `operator.add` 策略，意味着新消息会自动追加到列表中。

2.  **Node (节点)**: `chatbot` 和 `weather_tool`
    *   **Chatbot 节点**: 模拟大模型（LLM）的决策过程。它检查用户输入，如果包含"天气"关键词，就决定调用工具；否则直接回复。
    *   **Tool 节点**: 模拟工具的执行。当被调用时，返回预设的天气信息。

3.  **Edge (边)**: `conditional_edges`
    *   定义了节点之间的流转逻辑。
    *   `should_continue` 函数根据 Chatbot 的输出决定下一步是去执行工具（`continue` -> `action`）还是结束对话（`end` -> `END`）。

4.  **Graph (图)**: `StateGraph`
    *   将上述所有组件组装成一个可运行的应用。

## 🚀 运行方法

在项目根目录下运行：

```bash
python LangGraph/simple_agent.py
```

## 📝 输出示例

程序会演示两种场景：

1.  **普通对话**：用户说"你好"，Bot 直接回复。
2.  **工具调用**：用户问"北京天气"，Bot 识别意图 -> 路由到工具节点 -> 工具返回结果。
