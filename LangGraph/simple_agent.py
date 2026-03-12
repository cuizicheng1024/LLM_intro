"""
LangGraph 简易 Agent 示例

- 展示：状态定义、节点函数、条件路由、图构建与编译、运行演示，以及架构可视化导出
- 状态：AgentState 仅包含消息列表；Annotated[List[BaseMessage], operator.add] 表示节点返回的消息会“追加”到已有列表中
- 节点：
  - chatbot：读取最新用户消息，若包含“天气”则返回带 tool_calls 的 AIMessage，提示下一步调用工具；否则直接回复文本
  - weather_tool：模拟工具执行，返回一个 AIMessage，内容为工具结果
- 路由：should_continue 根据最后一条消息是否包含 tool_calls 决定走向；“continue”去工具节点，“end”结束
- 图：入口为 agent；从 agent 通过条件边去 action 或 END；action 通过普通边到 END
- 架构展示：print_architecture 打印节点与边，并尝试通过 app.get_graph(xray=True).draw_mermaid_png() 导出 PNG
- Notebook：在 Jupyter 中可直接显示 PNG；若 PNG 导出不可用，仍可复制 Mermaid 文本在 Markdown 渲染
"""
from typing import TypedDict, Annotated, List, Union
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from pathlib import Path

# ----------------------------------------------------------------------
# 1. 定义状态 (State)
# ----------------------------------------------------------------------
# 状态是图的核心，在节点之间传递。这里我们定义一个简单的状态，只包含消息列表。
# Annotated[List[BaseMessage], operator.add] 的意思是：
# 当有新消息返回时，不是覆盖旧列表，而是追加 (add) 到列表中。
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


# ----------------------------------------------------------------------
# 2. 定义节点 (Nodes)
# ----------------------------------------------------------------------
# 节点是执行具体逻辑的函数。它们接收当前状态，并返回更新后的状态。

def chatbot(state: AgentState):
    """
    模拟一个简单的聊天机器人节点。
    它查看最新的用户消息，如果包含 '天气'，则决定调用工具。
    否则直接回复。
    """
    # 输入：AgentState（消息列表）；输出：返回 dict，包含需追加的 'messages'
    # 注意：additional_kwargs['tool_calls'] 用于向路由函数传达“调用工具”的意图
    print("--- 进入 Chatbot 节点 ---")
    messages = state['messages']
    last_message = messages[-1]
    
    # 这里我们用简单的 if-else 模拟 LLM 的决策过程
    # 在实际应用中，这里通常会调用 OpenAI 或其他大模型
    if "天气" in last_message.content:
        # 模拟 LLM 决定调用工具，返回一个带 tool_calls 标记的消息
        # 注意：这里为了演示简单，我们手动构造一个特殊的 AIMessage
        return {"messages": [AIMessage(content="", additional_kwargs={"tool_calls": [{"name": "get_weather", "args": {"location": "北京"}}]})]}
    else:
        # 普通回复
        return {"messages": [AIMessage(content=f"你说了: {last_message.content}，但我只关心天气。")]}

def weather_tool(state: AgentState):
    """
    模拟一个工具执行节点。
    当 Chatbot 决定查询天气时，流程会来到这里。
    """
    # 输入：AgentState；真实场景通常携带工具入参
    # 输出：返回 dict，'messages' 中包含工具结果消息（演示用 AIMessage）
    print("--- 进入 Weather Tool 节点 ---")
    # 模拟工具执行结果
    tool_result = "北京今天晴朗，气温 25 度。"
    
    # 返回工具执行结果作为 AIMessage (在真实场景中通常是 ToolMessage)
    return {"messages": [AIMessage(content=f"工具调用结果: {tool_result}")]}


# ----------------------------------------------------------------------
# 3. 定义边 (Edges) 和 条件逻辑
# ----------------------------------------------------------------------
# 边决定了节点之间的流转方向。

def should_continue(state: AgentState):
    """
    条件判断函数：决定下一步是去工具节点，还是结束。
    """
    # 路由规则：
    # - 检查最后一条消息的 additional_kwargs 是否包含 'tool_calls'
    # - 有则返回 'continue' 进入工具节点；否则返回 'end' 结束
    messages = state['messages']
    last_message = messages[-1]
    
    # 检查最后一条消息是否包含工具调用意图
    if "tool_calls" in last_message.additional_kwargs:
        return "continue"
    return "end"


# ----------------------------------------------------------------------
# 4. 构建图 (Graph)
# ----------------------------------------------------------------------
# 初始化图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", chatbot)
workflow.add_node("action", weather_tool)
nodes = ["agent", "action"]

# 设置入口点：图从哪里开始运行
workflow.set_entry_point("agent")

# 添加条件边
# 从 'agent' 节点出发，根据 should_continue 的返回值决定去向
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",  # 如果返回 'continue'，去 'action' 节点
        "end": END             # 如果返回 'end'，结束流程
    }
)

# 添加普通边
# 工具执行完后，通常需要把结果返回给 agent 继续处理，或者直接结束
# 这里我们演示工具执行完直接结束
workflow.add_edge("action", END)
edges = [
    ("START", "agent", None),
    ("agent", "action", "continue"),
    ("agent", "END", "end"),
    ("action", "END", None),
]

def print_architecture(nodes, edges):
    # 作用：在终端打印节点与边，并输出 Mermaid 文本；尝试导出 PNG 文件
    # 注意：PNG 导出依赖 LangGraph 对 Mermaid 的渲染，部分版本可能不可用
    print("=== Agent 架构 ===")
    print("节点:", ", ".join(nodes))
    print("边:")
    for src, dst, label in edges:
        if label:
            print(f"  {src} --[{label}]--> {dst}")
        else:
            print(f"  {src} --> {dst}")
    print("\nMermaid（复制到 Markdown 可渲染）:")
    print("flowchart TD")
    for src, dst, label in edges:
        if label:
            print(f"  {src} -->|{label}| {dst}")
        else:
            print(f"  {src} --> {dst}")
    try:
        out_png = Path(__file__).resolve().parent / "agent_arch.png"
        # 优先使用 LangGraph 的内置导出（如果可用）
        # 在 Jupyter 中可配合 IPython.display.Image 显示
        png_bytes = app.get_graph(xray=True).draw_mermaid_png()  # type: ignore[attr-defined]
        out_png.write_bytes(png_bytes)
        print(f"\nPNG 图已导出: {out_png}")
        print("提示：在 Jupyter 中可使用：")
        print("  from IPython.display import Image, display")
        print("  display(Image(filename=str(out_png)))")
    except Exception:
        # 回退：输出 Mermaid 文本，依然可以在 Markdown 中渲染
        print("\n提示：PNG 导出不可用，已输出 Mermaid 文本，可复制到 Markdown 渲染。")

# 编译图
app = workflow.compile()


# ----------------------------------------------------------------------
# 5. 运行演示
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print_architecture(nodes, edges)
    print("=== 演示 1: 普通对话 ===")
    inputs = {"messages": [HumanMessage(content="你好！")]}
    # stream 方法可以流式输出每一步的状态
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"节点 '{key}' 执行完毕:")
            print(f"  当前最新消息: {value['messages'][-1].content}")
            print("-" * 30)

    print("\n=== 演示 2: 触发工具调用 ===")
    inputs = {"messages": [HumanMessage(content="北京天气怎么样？")]}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"节点 '{key}' 执行完毕:")
            # 打印消息内容（如果是工具调用，content可能为空，我们打印 kwargs）
            msg = value['messages'][-1]
            if msg.content:
                print(f"  当前最新消息: {msg.content}")
            else:
                print(f"  (准备调用工具): {msg.additional_kwargs}")
            print("-" * 30)
