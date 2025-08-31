from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()


question_template = PromptTemplate(
    input_variables=["name"],
    template = "Hi, I'm {name}! Tell me a joke with my name!"
)

text = question_template.format(name="evellyn")
print(text)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

chain = question_template | model

result = chain.invoke({"name": "evellyn"})
print(result.content)