from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

template = PromptTemplate(
    input_variables=["initial_text"],
    template="Translate the following text to English:\n ```{initial_text}```"
)

template_summary = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in 4 words:\n ```{text}```"

)


llm_en = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

translate = template | llm_en | StrOutputParser()
pipeline = {"text": translate} | template_summary | llm_en | StrOutputParser()

result = pipeline.invoke({"initial_text": "Langchain é um framework para desenvolvimento de aplicações de IA."})
print(result)