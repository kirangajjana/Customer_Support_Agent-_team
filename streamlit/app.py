import os
import streamlit as st
import time
from typing import Optional
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.team import Team
from dotenv import load_dotenv
from requests_ratelimiter import LimiterSession
from pyrate_limiter import Duration, RequestRate, Limiter

# Load environment variables
load_dotenv()
apikey= os.getenv("GOOGLE_API_KEY")

if not apikey:
    st.error("❌ OpenAI API Key not found. Please set OPENAI_API_KEY in your .env file.")
    st.stop()

# Set up rate limiting: 10 requests per second
rate = RequestRate(10, Duration.SECOND)
limiter = Limiter(rate)
session = LimiterSession(limiter=limiter)

# Streamlit UI
st.set_page_config(page_title="Job Search & Company locator agent", page_icon="🚀", layout="wide")

# Define UI Styling
def main():
    st.markdown(
    """
    <style>
        /* Background Gradient */
        .stApp {
            background: linear-gradient(135deg, #f0f2f6 0%, #e6eaf4 100%);
            background-attachment: fixed;
        }
        
        /* Container Styling */
        .stContainer {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Page Title */
        .title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #4CAF50;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        }

        /* Subtitle */
        .subtitle {
            text-align: center;
            font-size: 20px;
            color: #666;
            margin-bottom: 20px;
        }

        /* Custom Button */
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            border-radius: 10px;
            width: 100%;
            padding: 10px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        }

        .stButton > button:hover {
            background-color: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }

        /* Input Fields */
        .stTextInput > div > div > input {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 8px;
            border: 1px solid #ddd;
            padding: 10px;
        }

        /* Selectbox */
        .stSelectbox > div > div > div {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 8px;
            border: 1px solid #ddd;
            padding: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

    # Rest of the code remains the same as in the original script
    st.markdown("<p class='title'>Job Search & Company Lookup Agent</p>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Find jobs and travel options in your preferred location effortlessly.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        location = st.text_input("📍 Enter job location:", "Hyderabad")
        job_role = st.text_input("💼 Enter job role:", "Data Scientist")
    
    with col2:
        experience = st.selectbox("🎓 Experience Level:", ["Fresher", "Experienced"], index=1)
        search_button = st.button("🔍 Search Jobs")


    # Initialize session state for job response
    if "job_response" not in st.session_state:
        st.session_state.job_response = None

    # Validation checks
    if search_button:
        if not location or not job_role:
            st.warning("Please enter both location and job role.")
            return

        with st.spinner("🚀 Fetching job listings and commute options..."):
            try:
                query = f"I want to search for a {job_role} job in {location} with {experience} experience and travel options from main points like bus stand, railway station, etc."
                
                # Ensure multi_agent_team is defined before use
                if 'multi_agent_team' in globals():
                    response = multi_agent_team.run(query)
                    st.session_state.job_response = response.content if hasattr(response, 'content') else str(response)
                else:
                    st.error("Multi-agent team not properly configured.")
                    return

            except Exception as e:
                st.error(f"❌ Error fetching results: {e}")
                import traceback
                traceback.print_exc()  # This will print the full error traceback

        st.success("✅ Search Completed!")

    if st.session_state.job_response:
        with st.expander("🔍 View Detailed Results"):
            st.text_area("Results:", st.session_state.job_response, height=300)

# Define Agents with more robust error handling and type hints
def create_agent(
    name: str, 
    description: str, 
    instructions: list[str]
) -> Agent:
    """
    Helper function to create agents with consistent configuration
    """
    try:
        return Agent(
            name=name,
            model=Gemini(id="gemini-1.5-flash", api_key=apikey),
            tools=[DuckDuckGoTools()],
            description=description,
            instructions=instructions
        )
    except Exception as e:
        st.error(f"Error creating {name}: {e}")
        return None

# Create agents with the helper function
Data_agent = create_agent(
    name="Data Agent",
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

Jobsearch_agent = create_agent(
    name="Job Search Agent",
    description="Finds job openings at the companies provided by the Data Agent based on the user's job role preference and experience level.",
    instructions=[
        "Use DuckDuckGo search to find job openings for each company provided by the Data Agent.",
        "Search specifically in the company's official career page and job listing sites like LinkedIn, Naukri, and Indeed.",
        "Focus on job roles specified by the user (e.g., 'Software Engineer', 'Data Analyst').",
        "Check for available roles based on experience level: 'Fresher' and 'Experienced'.",
        "Extract relevant job details including job title, experience required, location, and application link.",
        "If no job openings matching the user's preference are found, mention 'No relevant openings found'.",
        "Ensure the search is structured and includes role-specific filtering.",
        "Pass the structured list of companies with job openings to the Location Agent for further processing."
    ]
)

location_agent = create_agent(
    name="Location Agent",
    description="Provides detailed travel guidance to companies based on major landmarks and transport options.",
    instructions=[
        "Use DuckDuckGo search to gather travel options for each company provided by the Job Search Agent.",
        "For each company, perform structured searches to find:",
        "   1. Nearest major landmarks (railway stations, metro stations, bus stands, popular locations).",
        "   2. Distance from these landmarks to the company (in km or meters).",
        "   3. Public transport options (bus routes, metro lines) with estimated travel time and fare.",
    ]
)

# Create multi-agent team with error handling
try:
    multi_agent_team = Team(
        name="Multi-Agent Team",
        mode="collaborate",
        model=Gemini(id="gemini-1.5-flash", api_key=apikey),
        members=[agent for agent in [Data_agent, Jobsearch_agent, location_agent] if agent is not None],
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
            "Provide the **final aggregated response** in a well-formatted manner for easy user understanding."
        ]
    )
except Exception as e:
    st.error(f"Error creating multi-agent team: {e}")
    multi_agent_team = None

def run_app():
    try:
        main()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_app()