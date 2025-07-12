import sys
import os
from langchain_core.messages import HumanMessage

sys.path.append(os.path.abspath(os.path.join('..', 'src')))

from graph import build_rag_graph

aivisor_graph = build_rag_graph()

# user_input = input('>> ')
user_input = 'Quero um roteiro de 3 noites em Munique. Chegaremos no dia 14/08/2025 às 09:00 e voltaremos no dia 17.'


aivisor_graph.invoke({'messages': [HumanMessage(content=user_input)]})