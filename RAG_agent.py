import pandas as pd

df = pd.read_parquet(r"C:\Users\kores\Downloads\train-00000-of-00001.parquet")

from langchain_core.documents import Document

docs=[]
for _,row in df.iterrows():
    text=f"{row['name']} is {row['relation']}. {row['description']}"
    docs.append(
        Document(
            page_content=text,
            metadata={
                "name":row["name"],
                "relation":row["relation"],
                "email":row["email"]
            }
        )
    )


import os
from dotenv import load_dotenv
load_dotenv()
import chromadb
from openai import AzureOpenAI


client=AzureOpenAI(
api_key=os.getenv('api_key'),
azure_endpoint=os.getenv('endpoint'),
api_version=os.getenv('api_version')
)

path='chromadb1'
db=chromadb.PersistentClient(path=path)
collection=db.get_or_create_collection(name='docs')

for i,doc in enumerate(docs):
    embedding=client.embeddings.create(
        model="text-embedding-ada-002",
        input=doc.page_content
    ).data[0].embedding
    print("indexed "+str(i)+" of",len(docs))
    collection.add(
        ids=[str(i)],
        documents=[doc.page_content],
        embeddings=[embedding],
        metadatas=[doc.metadata]
    )


#only for testing
#testing
# query="Tell me about our guest named 'Lady Ada Lovelace"

# question_embedding=client.embeddings.create(
#     model="text-embedding-ada-002",
#     input=query
# ).data[0].embedding

# results = collection.query(
#     query_embeddings=question_embedding,
#     n_results=2
# )
# results['documents'][0]


#now make this as a function
#create a tool from this
def extract_data_from_db(query:str) -> str:
    question_embedding=client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    ).data[0].embedding

    results=collection.query(
        query_embeddings=question_embedding,
        n_results=2
    )
    #i want to retun both content and metadata.works both type of queries like metadata and description
    # if results:
    #     return "\n\n".join([
    #         f"name:{doc.metadata['name']}\n"
    #         f"relation:{doc.metadata['relation']}\n"
    #         f"email:{doc.metadata['email']}\n"
    #         f"info:{doc.page_content}"
    #         for doc in results
    #     ])
    # else:

    #     return "no matching records found"
    return results
    
# quest="Tell me about our guest named LadyAda Lovelace"
# print(extract_data_from_db(quest))

#make this as a tool
from langchain.tools import Tool

data_base_tool=Tool(
    name="data_retriever",
    func=extract_data_from_db,
    description="Returns detailed info about guests based on query."
)

#llm and binding tools
from langchain_openai import AzureChatOpenAI
import os

llm = AzureChatOpenAI(
    api_key=os.getenv('api_key'),
    azure_endpoint=os.getenv('endpoint'),
    api_version=os.getenv('api_version'),
    deployment_name="gpt-4o-mini"  
)

tools=[data_base_tool]
llm_tools=llm.bind_tools(tools)


#instialize state

from typing import TypedDict,Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import START,StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_core.messages import AnyMessage,HumanMessage,AIMessage

class RagState(TypedDict):
    messages:Annotated[list[AnyMessage],add_messages]

def chatbot(state:RagState):
    return {
        "messages":[llm_tools.invoke(state["messages"])]
    }


#BUILD GRAPH    
graph_builder=StateGraph(RagState)

#nodes
graph_builder.add_node("chatbot",chatbot)
graph_builder.add_node("tools",ToolNode(tools))

#edges
graph_builder.add_edge(START,"chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition
)
graph_builder.add_edge("tools","chatbot")

#compile graph
graph = graph_builder.compile()

messages = [HumanMessage(content="Tell me about our guest named 'Lady Ada Lovelace'.")]
response = graph.invoke({"messages": messages})
response['messages'][-1].content
