import functools
import operator
from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph

from my_agent.config.config import get_settings
from my_agent.tools.plot_tool import python_repl_tool
from my_agent.tools.random_tool import random_number
from my_agent.tools.supervisor import get_supervisor_node
from my_agent.utils.helper import create_agent, agent_node

# Get the project config
config = get_settings()

# Define the model
llm = ChatOpenAI(model=config.AZURE_OPENAI_MODEL_VERSION, api_key=config.OPENAI_API_KEY)
'''llm = AzureChatOpenAI(
    name="agent_model",
    azure_deployment=config.AZURE_OPENAI_DEPLOYMENT_NAME,
    api_version='2024-08-01-preview',
    model_version=config.AZURE_OPENAI_MODEL_VERSION,
    streaming=config.STREAMING_MODEL == 'true',
)'''

# Tools
tools = [random_number, python_repl_tool]

# Random_Number_Generator as a node
random_agent = create_agent(llm, [random_number], "You get random numbers")
random_node = functools.partial(agent_node, agent=random_agent, name="Random_Number_Generator")

# Coder as a node
code_agent = create_agent(llm, [python_repl_tool], "You generate charts using matplotlib.")
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

# defining the AgentState that holds messages and where to go next
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str

# defining the StateGraph
workflow = StateGraph(AgentState)

# agents as a node, supervisor_chain as a node
workflow.add_node("Random_Number_Generator", random_node)
workflow.add_node("Coder", code_node)
workflow.add_node("Supervisor", get_supervisor_node(llm))

# when agents are done with the task, next one should be supervisor ALWAYS
workflow.add_edge("Random_Number_Generator", "Supervisor")
workflow.add_edge("Coder", "Supervisor")

# Supervisor decides the "next" field in the graph state,
# which routes to a node or finishes. (Remember the special node END above)
workflow.add_conditional_edges(
                    "Supervisor",
                    lambda x: x["next"],
                    {
                       "Random_Number_Generator": "Random_Number_Generator",
                       "Coder": "Coder",
                       "FINISH": END
                    })

# starting point should be supervisor
workflow.set_entry_point("Supervisor")
graph = workflow.compile()
