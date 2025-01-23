import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Create an OpenAI model client.
model_client = OpenAIChatCompletionClient(
    model="ollama/llama3.1",
    api_key="NotRequiredSinceWeAreLocal",
    base_url="http://localhost:4000",
    model_capabilities={
        "json_output": False,
        "vision": False,
        "function_calling": False,
    }
)

# Create the primary agent.
primary_agent = AssistantAgent(
    "primary",
    model_client=model_client,
    system_message="You are a helpful AI assistant.",
)

# Create the critic agent.
critic_agent = AssistantAgent(
    "critic",
    model_client=model_client,
    system_message="Provide constructive feedback. Respond with 'APPROVE' to when your feedbacks are addressed.",
)

# Define a termination condition that stops the task if the critic approves.
text_termination = TextMentionTermination("APPROVE")

# Create a team with the primary and critic agents.
team = RoundRobinGroupChat([primary_agent, critic_agent], termination_condition=text_termination)

# Function to run the team
async def run_team():
#     async for message in team.run_stream(task="Write a short poem about the fall season."):
#         if isinstance(message, TaskResult):
#             print(f"\nStop Reason: {message.stop_reason}")
#         else:
#             print(f"\n{message}")

    await Console(team.run_stream(task="Write a short poem about the fall season."))  # Stream the messages to the console.

if __name__ == "__main__":
    asyncio.run(run_team())