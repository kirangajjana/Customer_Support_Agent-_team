import streamlit as st
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.team import Team
import os
from dotenv import load_dotenv


load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# Define the Data Agent
Data_agent = Agent(
    name="Data Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    description="Finds companies in a specified location and categorizes them into startups, MNCs, and product-based companies.",
    instructions=[
        "Use DuckDuckGo search to find companies in the specified location.",
        "Categorize results into 'Startups', 'MNCs', and 'Product-Based Companies'.",
        "Provide company names, locations, and relevant details.",
        "Pass the list of companies to the Job Search Agent."
    ]
)

# Define the Job Search Agent
Jobsearch_agent = Agent(
    name="Job Search Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    description="Finds job openings at the companies provided by the Data Agent.",
    instructions=[
        "Use DuckDuckGo search to find job openings for each company provided by the Data Agent.",
        "Check company career pages or job listing sites for freshers and experienced roles.",
        "Categorize the openings into 'Fresher' and 'Experienced' roles.",
        "Provide job titles, locations, and application links where available.",
        "If no openings are found for a company, mention 'No current openings found'.",
        "Pass the list of companies with job openings to the Location Agent."
    ]
)

# Define the Location Agent
location_agent = Agent(
    name="Location Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    description="Provides travel guidance to companies provided by data_agent and job_search agent based on major landmarks and transport options.",
    instructions=[
        "Use DuckDuckGo search to gather travel options for each company provided by the Job Search Agent.",
        "Identify the nearest major landmarks (e.g., railway station, bus stand, shopping mall).",
        "Provide distances from these landmarks to the company.",
        "Suggest public transport options (buses, metro) with estimated fares.",
        "Provide cab/taxi options (e.g., Uber, Ola) with estimated fares and booking links.",
        "Ensure information is structured and accurate for easy readability.",
        "Pass the final structured travel details to the Multi-Agent Team."
    ]
)

# Define the Multi-Agent Team
multi_agent_team = Team(
    name="Multi-Agent Team",
    mode="collaborate",
    model=OpenAIChat(id="gpt-4o"),
    members=[Data_agent, Jobsearch_agent, location_agent],
    show_tool_calls=True,
    markdown=True,
    description="A team of agents collaborating to provide company listings, job openings, and travel guidance in a structured manner.",
    instructions=[
        "Ensure agents work sequentially: Data Agent → Job Search Agent → Location Agent.",
        "The Data Agent gathers companies from the specified location.",
        "The Job Search Agent finds openings only for the companies listed by the Data Agent.",
        "The Location Agent provides travel options only for the companies with job openings.",
        "Ensure structured, detailed, and relevant information.",
        "Provide final combined results in a clear format."
    ],
    show_members_responses=True,
)

# Function to run the multi-agent query and display results
def run_agents(query):
    response = multi_agent_team.run(query).content
    return response

# Streamlit Application
st.title("Multi-Agent System Showcase")

# Tabs for displaying results
tabs = st.tabs(["Query Input", "Data Agent", "Job Search Agent", "Location Agent", "Final Output"])

# Query input
with tabs[0]:
    st.header("Enter a Query")
    user_query = st.text_input("Enter your job and travel search query:")
    if user_query:
        st.write(f"Running query: {user_query}")
        result = run_agents(user_query)
        st.write(result)

# Data Agent tab
with tabs[1]:
    st.header("Data Agent")
    st.write("This agent searches for companies in the specified location and categorizes them.")
    st.write("Example input: 'Find companies in Hyderabad'")
    st.write("Output will show company names, locations, and categories.")

# Job Search Agent tab
with tabs[2]:
    st.header("Job Search Agent")
    st.write("This agent finds job openings at the companies listed by Data Agent.")
    st.write("Example input: 'Find job openings in Hyderabad' for the companies found.")
    st.write("Output will categorize job openings into 'Fresher' and 'Experienced' roles.")

# Location Agent tab
with tabs[3]:
    st.header("Location Agent")
    st.write("This agent provides travel guidance for the job openings found by the Job Search Agent.")
    st.write("It will give travel options, distances from major landmarks, and public transport details.")

# Final Output tab
with tabs[4]:
    st.header("Final Output")
    if user_query:
        st.write("Final combined results from all agents will be displayed here.")
        st.write(result)
