import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.team import Team
from agno.models.google import Gemini
from dotenv import load_dotenv


load_dotenv()


os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

# Define the Data Agent
Data_agent = Agent(
    name="Data Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    description="Finds and categorizes 10 startups, 10 MNCs, and 10 product-based companies in the specified location.",
    instructions=[
        "Use DuckDuckGo search to find company lists in the specified location.",
        "Perform three separate searches:",
        "   1. Search for 'Top 10 startup companies in [location]'.",
        "   2. Search for 'Top 10 MNC companies in [location]'.",
        "   3. Search for 'Top 10 product-based companies in [location]'.",
        "Ensure each category contains exactly 10 unique companies.",
        "Provide company names, locations, and relevant details (industry, notable products/services).",
        "Pass the structured list of 30 companies to the Job Search Agent for further processing."
    ]
)

# Define the Job Search Agent
Jobsearch_agent = Agent(
    name="Job Search Agent",
     model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    description="Finds job openings at the companies provided by the Data Agent based on the user's job role preference and experience level.",
    instructions=[
        "Use DuckDuckGo search to find job openings for each company provided by the Data Agent.",
        "Search specifically in the company's official career page and job listing sites like LinkedIn, Naukri, and Indeed.",
        "Focus on job roles specified by the user (e.g., 'Software Engineer', 'Data Analyst').",
        "Check for available roles based on experience level: 'Fresher' and 'Experienced'.",
        "Extract relevant job details including job title, experience required, location, and application link.",
        "If no job openings matching the user’s preference are found, mention 'No relevant openings found'.",
        "Ensure the search is structured and includes role-specific filtering.",
        "Pass the structured list of companies with job openings to the Location Agent for further processing."
    ]
)

# Define the Location Agent
location_agent = Agent(
    name="Location Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    description="Provides detailed travel guidance to companies based on major landmarks and transport options.",
    instructions=[
        "Use DuckDuckGo search to gather travel options for each company provided by the Job Search Agent.",
        "For each company, perform structured searches to find:",
        "   1. Nearest major landmarks (railway stations, metro stations, bus stands, popular locations).",
        "   2. Distance from these landmarks to the company (in km or meters).",
        "   3. Public transport options (bus routes, metro lines) with estimated travel time and fare.",
        "   4. Cab/taxi options (Uber, Ola) with approximate fare estimates and ride duration.",
        "   5. Any additional transport methods (bike rentals, auto-rickshaws) if available.",
        "Ensure that details are structured clearly for easy readability.",
        "Pass the final structured travel details to the Multi-Agent Team for integration with company and job information."
    ]
)

# Define the Multi-Agent Team
multi_agent_team = Team(
    name="Multi-Agent Team",
    mode="collaborate",
    model=OpenAIChat(id="gpt-4o"),
    members=[Data_agent, Jobsearch_agent, location_agent],
    share_member_interactions=True, 
    show_tool_calls=True,
    enable_team_history=True,
    num_of_interactions_from_history=5,
    enable_agentic_context=True,
    description="A structured multi-agent system that provides categorized company listings, job openings, and travel guidance.",
    instructions=[
        "Ensure agents work in a structured pipeline: Data Agent → Job Search Agent → Location Agent.",
        "The Data Agent fetches **10 startups, 10 MNCs, and 10 product-based companies** in the given location.",
        "The Job Search Agent retrieves job openings **based on user preferences (job role & experience level)** from company career pages or job portals.",
        "The Location Agent provides detailed **commute options** for companies **with active job openings**, including nearest landmarks, distances, public transport, and estimated fares.",
        "Each agent should **only process data received from the previous agent** and not generate independent outputs.",
        "Ensure structured, accurate, and readable responses at each step.",
        "Provide the **final aggregated response** in a well-formatted manner for easy user understanding.",
        "if you wanted any specific information just ask him that you want that information"
        "if you dont have the information about that just mention that you were not having that information"
    ],
)

query="i wanted to search for job in hyderabad and traveling options to reach the company from all the main points like busstand,railway station etc, i am a datascientist with 5 year of experience"
response=multi_agent_team.run(query).content
print(response)
