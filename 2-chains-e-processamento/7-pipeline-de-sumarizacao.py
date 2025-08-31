from langchain_google_genai import ChatGoogleGenerativeAI
from  langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

long_text = """
of steam curls from manholes, a quiet river.
Coffee steam spirals above a newspaper's pale print.
Pedestrians sketch light on sidewalks, hurried, loud with umbrellas.
Buses swallow the morning with their loud yawns.
A sparrow perches on a steel beam, surveying the grid.
The subway sighs somewhere underground, a heartbeat rising.
Neon still glows in the corners where night refused to retire.
A cyclist cuts through the chorus, bright with chrome and momentum.
The city clears its throat, the air turning a little less electric.
Shoes hiss on concrete, a thousand small verbs of arriving.
Dawn keeps its promises in the quiet rhythm of a waking metropolis.
The morning light cascades through towering windows of steel and glass,
casting geometric shadows on busy streets below.
Traffic flows like rivers of metal and light,
while pedestrians weave through crosswalks with purpose.
Coffee shops exhale warmth and the aroma of fresh bread,
as commuters clutch their cups like talismans against the cold.
Street vendors call out in a symphony of languages,
their voices mixing with the distant hum of construction.
Pigeons dance between the feet of hurried workers,
finding crumbs of breakfast pastries on concrete sidewalks.
The city breathes in rhythm with a million heartbeats,
each person carrying dreams and deadlines in equal measure.
Skyscrapers reach toward clouds that drift like cotton,
while far below, subway trains rumble through tunnels.
This urban orchestra plays from dawn until dusk,
a endless song of ambition, struggle, and hope.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=70
)

parts = splitter.create_documents([long_text])

#for part in parts:
#    print(part.page_content)
#    print("-"*30)

llm =  ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# LCEL mpa stage: summarize each chunk
map_prompt = PromptTemplate.from_template("Write a concise summary of the following text:\n {context}")
map_chain = map_prompt | llm | StrOutputParser()

prepare_map_inputs = RunnableLambda(lambda docs: [{"context": doc.page_content} for doc in docs])
map_stage = prepare_map_inputs | map_chain.map()

# LCEL reduce stage: combine summaries into one final summary
reduce_prompt = PromptTemplate.from_template("Combine the following summaries into a single concise summary:\n {context}")
reduce_chain = reduce_prompt | llm | StrOutputParser()

prepare_reduce_inputs = RunnableLambda(lambda summaries: {"context": "\n".join(summaries)})
pipeline = map_stage | prepare_reduce_inputs | reduce_chain

result = pipeline.invoke(parts)
print(result)
