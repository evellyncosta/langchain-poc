from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import chain
from dotenv import load_dotenv
load_dotenv()

@chain
def square(input_dict:dict) -> dict:
    x = input_dict["x"]
    return {"square_result": x*x}

question_template2 = PromptTemplate(
    input_variables=["square_result"],
    template = "Tell me about the number {square_result}"
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

chain2 = square | question_template2 | model

result = chain2.invoke({"x":3})
print(result.content)