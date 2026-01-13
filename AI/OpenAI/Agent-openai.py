#Example from https://github.com/openai/openai-agents-python

from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.

# Testing
#beta agent for fast execution
import timeit


b = '''
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.
'''

t = timeit.timeit(b, number=3)
print(round(t, 3), "ms")

#Output
#Function calls itself—  
#mirrors within reflections,  
#end found, it unwinds.
#A function calls self,  
#Infinite steps, yet so clear—  
#Ends when base case nears.
#A function calls self,  
#choing through code's deep woods—  
#End case beckons home.
# Execution time: 7.865 ms
