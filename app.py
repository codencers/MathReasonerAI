import streamlit as st
import math

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent
from langchain_community.utilities import WikipediaAPIWrapper


# ------------------ Streamlit UI ------------------ #

st.set_page_config(
    page_title="Math & Reasoning Assistant",
    page_icon="🧮"
)

st.title("Math & Reasoning Assistant (Gemma2-9b-It)")

groq_api_key = st.sidebar.text_input(
    label="Groq API Key",
    type="password"
)

if not groq_api_key:
    st.info("Enter Groq API key to continue.")
    st.stop()


# ------------------ LLM ------------------ #

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=groq_api_key
)


# ------------------ Tools ------------------ #

wiki = WikipediaAPIWrapper()

@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for general knowledge."""
    return wiki.run(query)


@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions like pow(13, 0.3432)."""
    try:
        allowed = {
            "sqrt": math.sqrt,
            "pow": math.pow,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "pi": math.pi,
            "e": math.e
        }
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception:
        return "Invalid math expression."


tools = [wikipedia_search, calculator]


# ------------------ Agent (Modern 1.x) ------------------ #

agent = create_agent(
    model=llm,
    tools=tools,
   system_prompt = """
You are a precise mathematical and reasoning assistant.

Follow these rules strictly:

1. Answer ONLY what is asked in the question.
2. Do NOT add unrelated explanations.
3. Do NOT change the topic.
4. If the problem requires calculation, use the calculator tool.
5. Show step-by-step reasoning for logic problems.
6. For math problems, clearly show formulas used.
7. If information is missing, say: "Insufficient information to answer."
8. Do NOT hallucinate facts.
9. Keep answers concise but complete.
10. Final answer must be clearly stated at the end.

Your goal is accuracy and relevance, not creativity.

If any thing besides math is asked tell i am made for math and reasoning questions and i can not answer that question.
"""
)


# ------------------ Chat Memory ------------------ #

if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hi! Ask me math or knowledge questions.")
    ]


for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)


# ------------------ User Input ------------------ #

user_input = st.chat_input("Ask your question...")

if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    with st.spinner("Thinking..."):
        response = agent.invoke(
            {"messages": st.session_state.messages}
        )

        output = response["messages"][-1].content

        st.session_state.messages.append(AIMessage(content=output))
        st.chat_message("assistant").write(output)