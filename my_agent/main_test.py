from langchain_core.messages import HumanMessage

from my_agent.graph import graph

if __name__ == '__main__':
    for s in graph.stream(
            {
                "messages": [
                    HumanMessage(content="Get 10 random numbers and generate a histogram")
                ]
            }, config={"recursion_limit": 20}
    ):
        if "__end__" not in s:
            print(s)
            print("----")