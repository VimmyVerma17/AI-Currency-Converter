import json
import requests
from typing import Annotated
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage

load_dotenv()

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """
    Fetches the currency conversion factor between a base currency and target currency.
    """
    url = f'https://v6.exchangerate-api.com/v6/c754eab14ffab33112e380ca/pair/{base_currency}/{target_currency}'
    response = requests.get(url)
    return response.json()  

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """
    Given a conversion rate, calculates converted currency.
    """
    return base_currency_value * conversion_rate


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  
    temperature=0.2
)

llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

messages = [
    HumanMessage("What is the conversion factor between INR and USD?")
]

ai_message = llm_with_tools.invoke(messages)  
messages.append(ai_message)

print("\n--- Tool Calls from Model ---")
print(ai_message.tool_calls)


conversion_rate = None

for tool_call in ai_message.tool_calls:
    if tool_call["name"] == "get_conversion_factor":
        tool_msg1 = get_conversion_factor.invoke(tool_call)
        rate_json = json.loads(tool_msg1.content)
        conversion_rate = rate_json["conversion_rate"]
        messages.append(tool_msg1)

    if tool_call["name"] == "convert":
        tool_call["args"]["conversion_rate"] = conversion_rate
        tool_msg2 = convert.invoke(tool_call)
        messages.append(tool_msg2)

final_answer = llm_with_tools.invoke(messages)
print("\n--- Final Answer from Model ---")
print(final_answer.content)
