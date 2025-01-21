from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

import logging

from autogen_core import TRACE_LOGGER_NAME

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(TRACE_LOGGER_NAME)
logger.setLevel(logging.DEBUG)

# Define a tool
async def get_weather(city: str) -> str:
    return f"The weather in {city} is 73 degrees and Sunny."

async def get_news(who: str) -> str:
    return f"This {who}'s great news!"

async def main() -> None:
    agent = AssistantAgent(
        name="agent",
        model_client=OpenAIChatCompletionClient(
            model="ollama/llama3.1",
            api_key="NotRequiredSinceWeAreLocal",
            base_url="http://localhost:4000",
            model_capabilities={
                "json_output": False,
                "vision": False,
                "function_calling": True,
            }
        ),
        tools=[get_weather, get_news],
    )

    # Define a team with a single agent and maximum auto-gen turns of 1.
    agent_team = RoundRobinGroupChat([agent], max_turns=1)

    stream = agent_team.run_stream(task="Tell me news about Ye")
    await Console(stream)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
