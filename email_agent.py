import os
import json
from typing import TypedDict,List,Dict,Any,Optional
from langgraph.graph import StateGraph,START,END
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage


class EmailState(TypedDict):
    messages:list[Dict[str,any]]
    email:Dict[str,any]
    is_spam:Optional[bool]
    spam_reason:Optional[str]
    email_category:Optional[str]
    draft_response_email:Optional[str]   

llm = AzureChatOpenAI(
    azure_deployment='gpt-4o-mini',
    api_key=os.getenv('api_key'),
    azure_endpoint=os.getenv('endpoint'),
    api_version=os.getenv('api_version')
)

def read_email(state:EmailState):
    email=state["email"]
    print(f"this email is from {email['sender']} with subject:{email['subject']}")
    return {}


def classify_email(state:EmailState):
    email=state["email"]
    prompt = f"""
    You are a helpful email assistant that analyzes emails and returns a structured response.

    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}

    First, determine if this email is spam. If it is spam, set "is_spam": true and provide a reason in "reason". If it is not spam, set "is_spam": false and return a category in "category". The category must be one of: "inquiry", "thank you", "complaint", or "information".

    Respond only in the following JSON format:

    {{
  "is_spam": true or false,
  "reason": "Explain why it is spam (if applicable)",
  "email_category": "inquiry / thank you / complaint / information (only if not spam)"
    }}
    """

    messages=[HumanMessage(content=prompt)]
    response=llm.invoke(messages)

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback if LLM fails to return valid JSON
              return {
            "is_spam": None,
            "spam_reason": "Failed to parse LLM response",
            "email_category": None,
            "messages": state.get("messages", []) + [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response.content}
            ]
        }

    is_spam = result.get("is_spam", None)
    spam_reason = result.get("reason", None)
    email_category = result.get("email_category", None)
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]

    return {
      "is_spam":is_spam,
      "spam_reason":spam_reason,
      "email_category":email_category,
      "messages":new_messages
    }

    
def handle_spam(state:EmailState):
    print(f"this email is marked as spam Reason:{state['spam_reason']}")
    print("this email has moved into spam folder")
    return {}
    

def draft_response(state: EmailState):
    """Alfred drafts a preliminary response for legitimate emails"""
    email = state["email"]
    email_category = state.get("email_category", "general")
    
    # Prepare our prompt for the LLM
    prompt = f"""
    As Alfred the butler, draft a polite preliminary response to this email.
    
    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}
    
    This email has been categorized as: {email_category}
    
    Draft a brief, professional response that Mr. Hugg can review and personalize before sending.
    """
    
    # Call the LLM
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    # Update messages for tracking
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]
    
    # Return state updates
    return {
        "draft_response_email": response.content,
        "messages": new_messages
    }


def notify(state:EmailState):
    email=state["email"]

    print(f"Sir you have received an email from {email['sender']}.Regarding the subject {email['subject']} .this email is categorized as {state['email_category']}")

    print("I have prepared an draft response for you")
    print(state['draft_response_email'])
    print('*'*50)

    return  {}


#conditional routing logic
def route_email(state:EmailState):
    if state["is_spam"]:
        return "spam"
    else:
        return "legitimate"

#create a graph

graph_builder = StateGraph(EmailState)

#add all nodes
graph_builder.add_node("read_email",read_email)
graph_builder.add_node("classify_email",classify_email)
graph_builder.add_node("handle_spam",handle_spam)
graph_builder.add_node("draft_response",draft_response)
graph_builder.add_node("notify",notify)

#connect all nodes with also conditional edge with the help of edges
graph_builder.add_edge(START,"read_email")
graph_builder.add_edge("read_email","classify_email")

#add conditional edge
graph_builder.add_conditional_edges("classify_email",route_email,{
    "spam":"handle_spam",
    "legitimate":"draft_response"
})

graph_builder.add_edge("handle_spam",END)
graph_builder.add_edge("draft_response","notify")
graph_builder.add_edge("notify",END)

graph=graph_builder.compile()


email = {
    "sender": "winner@lottery-intl.com",
    "subject": "YOU HAVE WON $5,000,000!!!",
    "body": "CONGRATULATIONS! You have been selected as the winner of our international lottery! To claim your $5,000,000 prize, please send us your bank details and a processing fee of $100."
}


print("\nProcessing  email...")

result = graph.invoke({
    "email": email,
    "messages": []
})