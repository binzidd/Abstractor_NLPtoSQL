import os
import streamlit as st
import tempfile
import sqlite3
import re
import ast
from typing import Annotated, Dict, List, Literal, Optional, Tuple, Any
import numpy as np

# New imports for proper noun extraction and FAISS
import faiss
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from langchain_community.tools.sql_database.tool import (
    QuerySQLDataBaseTool,
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
)
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import sqlalchemy
from pydantic import BaseModel, Field

# Configure page
st.set_page_config(page_title="SQL Agent with Proper Nouns", layout="wide")

# App state
if "db_connection" not in st.session_state:
    st.session_state.db_connection = None
if "db_path" not in st.session_state:
    st.session_state.db_path = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "proper_nouns" not in st.session_state:
    st.session_state.proper_nouns = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

# Tool error handler
def handle_tool_error(state) -> dict:
    """Handle tool errors and provide feedback to the agent."""
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks:
    """Create a ToolNode with a fallback to handle errors."""
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

# Pydantic models for state and responses
class State(BaseModel):
    """Define the state for the SQL agent."""
    messages: Annotated[List[AnyMessage], add_messages]
    proper_nouns: List[str] = Field(default_factory=list)
    context: Optional[List[str]] = None

class SubmitFinalAnswer(BaseModel):
    """Submit the final answer to the user based on the query results."""
    final_answer: str = Field(..., description="The final answer to the user")

class SearchProperNouns(BaseModel):
    """Search for relevant proper nouns in the database."""
    query: str = Field(..., description="The search query to find relevant proper nouns")

# Helper functions for proper noun extraction
def query_as_list(engine, query):
    """Execute a SQL query and extract string values, removing numbers."""
    try:
        with engine.connect() as connection:
            result = connection.execute(sqlalchemy.text(query))
            # Convert result to list of strings
            res = [str(row[0]) for row in result if row[0]]
            # Remove numbers and strip whitespace
            res = [re.sub(r"\b\d+\b", "", string).strip() for string in res if string]
            return res
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return []

def extract_proper_nouns(engine):
    """Extract proper nouns from the database by querying tables and columns."""
    proper_nouns = []
    
    # Get list of tables
    inspector = sqlalchemy.inspect(engine)
    tables = inspector.get_table_names()
    
    for table in tables:
        # Get columns for the table
        columns = inspector.get_columns(table)
        
        # Look for string columns that might contain proper nouns
        for column in columns:
            col_name = column['name']
            col_type = str(column['type']).lower()
            
            # Check if column type is string-like
            if 'char' in col_type or 'text' in col_type or 'varchar' in col_type:
                # Query for distinct values in this column
                try:
                    query = f"SELECT DISTINCT \"{col_name}\" FROM \"{table}\" WHERE \"{col_name}\" IS NOT NULL LIMIT 1000"
                    values = query_as_list(engine, query)
                    proper_nouns.extend(values)
                except:
                    # Skip if there's an error with this column
                    pass
    
    # Remove duplicates and empty strings
    proper_nouns = list(set([noun for noun in proper_nouns if noun]))
    return proper_nouns

def create_faiss_index(proper_nouns, api_key):
    """Create a FAISS index for the proper nouns."""
    if not proper_nouns:
        return None
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(api_key=api_key)
    
    # Create a FAISS index
    try:
        index = FAISS.from_texts(
            proper_nouns, 
            embeddings, 
            distance_strategy=DistanceStrategy.COSINE
        )
        return index
    except Exception as e:
        st.error(f"Error creating FAISS index: {e}")
        return None

# UI Components
st.title("SQL Agent with Proper Noun Enhancement")

# API Key input
api_key = st.text_input("Enter your OpenAI API Key", type="password")
os.environ["OPENAI_API_KEY"] = api_key

# File uploader
uploaded_file = st.file_uploader("Upload a SQLite database file", type=["db", "sqlite", "sqlite3"])

# Process uploaded database
if uploaded_file is not None:
    # Create a temporary file to store the uploaded db
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        db_path = tmp_file.name
    
    # Connect to the database
    try:
        engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
        connection = engine.connect()
        st.session_state.db_connection = engine
        st.session_state.db_path = db_path
        st.success(f"Successfully connected to database: {uploaded_file.name}")
        
        # Extract proper nouns
        with st.spinner("Extracting proper nouns from database..."):
            proper_nouns = extract_proper_nouns(engine)
            st.session_state.proper_nouns = proper_nouns
            st.success(f"Extracted {len(proper_nouns)} proper nouns from the database")
            
            # Create FAISS index
            if api_key and proper_nouns:
                with st.spinner("Creating FAISS index for proper nouns..."):
                    faiss_index = create_faiss_index(proper_nouns, api_key)
                    st.session_state.faiss_index = faiss_index
                    st.success("FAISS index created successfully")
                    
                    # Show sample proper nouns
                    if len(proper_nouns) > 0:
                        st.write("Sample proper nouns:")
                        st.write(proper_nouns[:5])
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")

# Define a tool for searching proper nouns
@tool
def search_proper_nouns(query: str) -> str:
    """Search for relevant proper nouns in the database."""
    if st.session_state.faiss_index is None:
        return "No proper noun index available."
    
    try:
        # Search the FAISS index
        results = st.session_state.faiss_index.similarity_search(query, k=5)
        nouns = [doc.page_content for doc in results]
        return f"Relevant proper nouns: {', '.join(nouns)}"
    except Exception as e:
        return f"Error searching proper nouns: {str(e)}"

# Create SQL tools if database is connected
if st.session_state.db_connection is not None and api_key:
    # Create SQL Agent
    def create_sql_agent():
        # Create SQL database tools
        db_uri = f"sqlite:///{st.session_state.db_path}"
        list_tables_tool = ListSQLDatabaseTool(db_uri=db_uri)
        get_schema_tool = InfoSQLDatabaseTool(db_uri=db_uri)
        db_query_tool = QuerySQLDataBaseTool(db_uri=db_uri)
        
        # Create proper noun search tool
        proper_noun_tool = search_proper_nouns
        
        # LLM instance
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Query checking model
        query_check_system = """You are a SQL expert. Review the provided SQL query for any syntax or logical errors.
        If there are any issues, please fix them and provide the corrected SQL. If the query is correct, confirm it."""
        query_check_prompt = ChatPromptTemplate.from_messages([
            ("system", query_check_system),
            ("human", "{content}")
        ])
        query_check = query_check_prompt | llm
        
        # Graph definition
        workflow = StateGraph(State)
        
        # Define first tool call node
        def first_tool_call(state: State) -> dict:
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "list_tables",
                                "args": {},
                                "id": "tool_list_tables",
                            }
                        ],
                    )
                ]
            }
        
        def model_check_query(state: State) -> dict:
            """Check if a query is correct before executing it."""
            last_message = state["messages"][-1]
            return {"messages": [query_check.invoke(last_message.content)]}
        
        # Create context enhancement node
        def enhance_with_proper_nouns(state: State) -> dict:
            """Enhance the query with relevant proper nouns."""
            last_message = state["messages"][-1]
            query = last_message.content
            
            # Search for relevant proper nouns
            if st.session_state.faiss_index:
                results = st.session_state.faiss_index.similarity_search(query, k=5)
                nouns = [doc.page_content for doc in results]
                
                # Add the proper nouns to the state
                return {
                    "context": nouns,
                    "messages": [
                        AIMessage(
                            content=f"I found some relevant proper nouns that might help with your query: {', '.join(nouns)}. I'll use these to help answer your question."
                        )
                    ]
                }
            else:
                return {"messages": []}
        
        # Create query generator with proper noun context
        query_gen_system = """You are a SQL expert with a strong attention to detail.

Given an input question, output a syntactically correct SQLite query to run, then look at the results of the query and return the answer.

You have access to proper nouns from the database that might be relevant to the query. Use these proper nouns to improve your query accuracy.

DO NOT call any tool besides SubmitFinalAnswer to submit the final answer.

When generating the query:

Output the SQL query that answers the input question without a tool call.

Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.

If you get an error while executing a query, rewrite the query and try again.

If you get an empty result set, you should try to rewrite the query to get a non-empty result set. 
NEVER make stuff up if you don't have enough information to answer the query... just say you don't have enough information.

If you have enough information to answer the input question, simply invoke the appropriate tool to submit the final answer to the user.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database."""
        
        query_gen_prompt = ChatPromptTemplate.from_messages([
            ("system", query_gen_system), 
            ("system", "Relevant proper nouns from the database: {context}"),
            ("placeholder", "{messages}")
        ])
        
        query_gen = query_gen_prompt | llm.bind_tools([SubmitFinalAnswer])
        
        def query_gen_node(state: State) -> dict:
            # Get context from the state
            context = state.get("context", [])
            if not context:
                context = ["No specific proper nouns found"]
            
            # Invoke the query generator with context
            message = query_gen.invoke({"messages": state["messages"], "context": context})
            
            # Check for hallucinated tool calls
            tool_messages = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    if tc["name"] != "SubmitFinalAnswer":
                        tool_messages.append(
                            ToolMessage(
                                content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
                                tool_call_id=tc["id"],
                            )
                        )
            return {"messages": [message] + tool_messages}
        
        # Add nodes to the graph
        workflow.add_node("greeting", lambda state: {
            "messages": [AIMessage(content="Hello! I'm your SQL assistant with proper noun enhancement. How can I help you with your database today?")]
        })
        workflow.add_node("first_tool_call", first_tool_call)
        workflow.add_node("list_tables_tool", create_tool_node_with_fallback([list_tables_tool]))
        workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))
        workflow.add_node("enhance_with_proper_nouns", enhance_with_proper_nouns)
        workflow.add_node("proper_noun_search", create_tool_node_with_fallback([proper_noun_tool]))
        
        # Add schema selection model
        model_get_schema = llm.bind_tools([get_schema_tool])
        workflow.add_node(
            "model_get_schema",
            lambda state: {
                "messages": [model_get_schema.invoke(state["messages"])],
            },
        )
        
        workflow.add_node("query_gen", query_gen_node)
        workflow.add_node("correct_query", model_check_query)
        workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))
        
        # Add edges
        workflow.add_edge(START, "greeting")
        workflow.add_edge("greeting", "first_tool_call")
        workflow.add_edge("first_tool_call", "list_tables_tool")
        workflow.add_edge("list_tables_tool", "model_get_schema")
        workflow.add_edge("model_get_schema", "get_schema_tool")
        workflow.add_edge("get_schema_tool", "enhance_with_proper_nouns")
        workflow.add_edge("enhance_with_proper_nouns", "query_gen")
        
        # Define conditional edges
        def should_continue(state: State) -> Literal[END, "correct_query", "query_gen", "proper_noun_search"]:
            messages = state["messages"]
            last_message = messages[-1]
            # If there is a tool call to submit final answer, then we finish
            if getattr(last_message, "tool_calls", None):
                return END
            if hasattr(last_message, "content") and isinstance(last_message.content, str) and last_message.content.startswith("Error:"):
                return "query_gen"
            # If we need more context, try searching for proper nouns
            if "can't find" in last_message.content.lower() or "no results" in last_message.content.lower():
                return "proper_noun_search"
            else:
                return "correct_query"
        
        workflow.add_conditional_edges("query_gen", should_continue)
        workflow.add_edge("proper_noun_search", "query_gen")
        workflow.add_edge("correct_query", "execute_query")
        workflow.add_edge("execute_query", "query_gen")
        
        # Compile the workflow
        return workflow.compile()
    
    # Create agent if not already created
    if st.session_state.agent is None:
        with st.spinner("Initializing SQL Agent with Proper Noun Enhancement..."):
            st.session_state.agent = create_sql_agent()
            st.success("SQL Agent initialized!")

# Chat interface
if st.session_state.agent is not None:
    # Display chat messages
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                if message.tool_calls:
                    st.write("Processing your query...")
                else:
                    st.write(message.content)
        elif isinstance(message, ToolMessage):
            # Tool messages are typically not shown to users
            pass
    
    # Chat input
    if user_query := st.chat_input("Ask a question about your database"):
        # Add user message to chat history
        user_message = HumanMessage(content=user_query)
        st.session_state.messages.append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Process with agent
        with st.spinner("Thinking..."):
            # Initialize state with messages
            state = {"messages": [user_message], "proper_nouns": st.session_state.proper_nouns}
            
            # Get agent response
            final_state = st.session_state.agent.invoke(state)
            
            # Extract AI responses
            for message in final_state["messages"]:
                if isinstance(message, AIMessage) and not message.tool_calls:
                    # Add to message history and display
                    st.session_state.messages.append(message)
                    with st.chat_message("assistant"):
                        st.write(message.content)
                elif isinstance(message, AIMessage) and message.tool_calls:
                    if message.tool_calls[0]["name"] == "SubmitFinalAnswer":
                        final_answer = message.tool_calls[0]["args"]["final_answer"]
                        final_response = AIMessage(content=final_answer)
                        st.session_state.messages.append(final_response)
                        with st.chat_message("assistant"):
                            st.write(final_answer)
else:
    st.info("Please upload a database file and enter your OpenAI API key to start.")

# Add file cleanup on session end
def cleanup():
    if st.session_state.db_path and os.path.exists(st.session_state.db_path):
        try:
            os.unlink(st.session_state.db_path)
        except:
            pass

import atexit
atexit.register(cleanup)