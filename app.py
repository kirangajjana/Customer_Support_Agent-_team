import os
from fastapi import FastAPI
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from dotenv import load_dotenv


load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

app = FastAPI()

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    description="You are a helpful assistant.",
    markdown=True,
)

@app.get("/ask")
async def ask(query: str):
    response = agent.run(query)
    return {"response": response.content}