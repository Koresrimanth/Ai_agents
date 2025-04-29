import streamlit as st
import requests
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv

load_dotenv()

# ===== Document Retrieval Tool =====
vectordb = Chroma(persist_directory='chroma_db', embedding_function=OpenAIEmbeddings())
retriever = vectordb.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name='gpt-4'),
    chain_type='stuff',
    retriever=retriever,
    return_source_documents=True
)

# ===== Wikipedia Tool =====
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# ===== Weather Tool =====
def get_weather(city: str) -> str:
    try:
        response = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude=35&longitude=139&current_weather=true"
        )
        if response.status_code == 200:
            data = response.json()
            weather = data['current_weather']
            return f"The current temperature is {weather['temperature']}°C with wind speed {weather['windspeed']} km/h."
        else:
            return "Could not fetch weather data."
    except Exception as e:
        return str(e)

weather_tool = Tool(
    name="Weather",
    func=lambda q: get_weather(q),
    description="Use this to answer weather-related questions. Input should be a city name."
)

# ===== Custom Document QA Tool =====
document_tool = Tool(
    name="DocumentQA",
    func=lambda q: qa_chain.run(q),
    description="Useful for answering questions from uploaded documents."
)

# ===== LangChain Agent =====
agent_executor = initialize_agent(
    tools=[document_tool, wikipedia_tool, weather_tool],
    llm=ChatOpenAI(model_name="gpt-4"),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ===== Streamlit UI =====
st.title("🧠 GenAI Multi-Tool Agent")
query = st.text_input("Ask anything (Weather, Wikipedia, or Your Docs):")

if query:
    result = agent_executor.run(query)
    st.write("**Answer:**", result)

    if "DocumentQA" in result:
        with st.expander("Source Documents"):
            doc_result = qa_chain({"query": query})
            for doc in doc_result['source_documents']:
                st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                st.write(doc.page_content)
