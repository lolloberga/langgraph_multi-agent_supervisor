import random

from langchain_core.tools import tool


#for generating random numbers
@tool("random_number", return_direct=False)
def random_number() -> str:
    """Returns a random number between 0-100. input the word 'random'"""
    return str(random.randint(0, 100))
