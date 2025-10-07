from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
import json

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

chain = prompt | chat_model

session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

config = {"configurable": {"session_id": "demo-session"}}
config_for_second_test = {"configurable": {"session_id": "demo-session2"}}

# Interactions
response1 = conversational_chain.invoke({"input": "Hello, my name is Evellyn. how are you?"}, config=config)
print("Assistant: ", response1.content)
print("-"*30)

response2 = conversational_chain.invoke({"input": "Can you repeat my name?"}, config=config)
print("Assistant: ", response2.content)
print("-"*30)

response3 = conversational_chain.invoke({"input": "Can you repeat my name in a motivation phrase?"}, config=config)
print("Assistant: ", response3.content)
print("-"*30)

response4 = conversational_chain.invoke({"input": "My name is ryzen, I am 16 years old"}, config=config_for_second_test)
print("Assistant: ", response4.content)
print("-"*30)

response5 = conversational_chain.invoke({"input": "I am from Brazil and I really like nature and biology"}, config=config_for_second_test)
print("Assistant: ", response5.content)
print("-"*30)

response6 = conversational_chain.invoke({"input": "Create a super hero with my name"}, config=config_for_second_test)
print("Assistant: ", response6.content)
print("-"*30)


simple_dict = {}
for session_id, history in session_store.items():
    simple_dict[session_id] = [msg.content for msg in history.messages]

print(json.dumps(simple_dict, indent=2, ensure_ascii=False))
