"""
LangGraph 最小可运行示例（Hello World）

- MessagesState：用于保存对话消息（role/content），作为图状态的数据结构
- StateGraph：声明式构建计算图，包含节点（node）与边（edge）
- START/END：图的入口与终止标记
- add_node：注册一个可执行节点（函数名即节点名）
- add_edge：连接节点形成有向边；从 START 进入，执行后流向 END
- compile：将声明式图编译为可运行对象
- invoke：向图传入初始状态并执行一次（非流式）

运行结果：返回包含 AI 回复 “hello world” 的消息。
"""
from langgraph.graph.message import MessagesState


from langgraph.graph import StateGraph, MessagesState, START, END

# 节点函数：模拟一个最简单的“模型”节点
# 输入为当前图状态（包含历史消息）；输出为增量更新（新增一条 AI 消息）
def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world"}]}

# 1) 创建一个基于 MessagesState 的状态图
graph = StateGraph[MessagesState, None, MessagesState, MessagesState](MessagesState)

# 2) 注册节点（函数名作为节点标识 "mock_llm"）
graph.add_node(mock_llm)

# 3) 连接有向边：从图入口 START 进入到节点，再从节点流向 END 结束
graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", END)

# 4) 编译得到可执行的工作流对象
graph = graph.compile()

# 5) 以一次性调用方式运行（非流式），传入初始对话消息
# 提示：如需查看流式过程可使用 app.stream(...)
result = graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
print("Final messages:", result.get("messages"))
msgs = result.get("messages", [])
print("Messages count:", len(msgs))
for m in msgs:
    if hasattr(m, "content"):
        print(f"{getattr(m, 'type', 'msg')}: {m.content}")
    elif isinstance(m, dict):
        print(f"{m.get('role','msg')}: {m.get('content')}")
