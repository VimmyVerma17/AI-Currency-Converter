import streamlit as st
import json
from langchain_core.messages import HumanMessage
from app import llm_with_tools, get_conversion_factor, convert   #importing backend logic

st.set_page_config(page_title="AI Currency Converter", page_icon="💱")
st.title("AI Currency Converter 💱")

query = st.text_input(
    "Ask a question like:",
    value="What is the conversion factor between INR and USD?"
)

if st.button("Run"):
    messages = [HumanMessage(query)]
    result = llm_with_tools.invoke(messages)
    messages.append(result)

    st.write("Tool Calls")
    st.json(result.tool_calls)

    conversion_rate = None

    for tool_call in result.tool_calls:
        if tool_call["name"] == "get_conversion_factor":
            tool_msg1 = get_conversion_factor.invoke(tool_call)
            data = json.loads(tool_msg1.content)
            conversion_rate = data["conversion_rate"]
            messages.append(tool_msg1)

        if tool_call["name"] == "convert":
            tool_call["args"]["conversion_rate"] = conversion_rate
            tool_msg2 = convert.invoke(tool_call)
            messages.append(tool_msg2)

    final = llm_with_tools.invoke(messages)

    st.success("### Final Answer")
    st.write(final.content)
