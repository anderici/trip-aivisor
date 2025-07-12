import os
import sys
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import pathlib

# Garante que o src/ está no path
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from graph import build_rag_graph

# Inicializa o grafo fora da função para manter performance
aivisor_graph = build_rag_graph()

st.set_page_config(page_title="Trip AIvisor", page_icon="🧳")
st.title("🧳 Trip AIvisor")

# Inicializa a sessão
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Campo de entrada do usuário (input de chat)
user_input = st.chat_input("Descreva o que você quer no seu roteiro de viagem...")

if user_input:
    # Adiciona input do usuário ao histórico
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.spinner("Gerando roteiro..."):
        try:
            # Invoca o grafo com todo o histórico atual
            output = aivisor_graph.invoke({
                "messages": st.session_state.chat_history
            })

            # Pega a resposta final e a nota do roteiro
            final_script = output.get("final_script", "Nenhum roteiro gerado.")
            final_score = output.get("quality", None)

            # Adiciona a resposta final no histórico como mensagem do assistente
            st.session_state.chat_history.append(AIMessage(content=final_script))

            if final_score is not None:
                st.success(f"Nota do roteiro: {final_score}/1000")

        except Exception as e:
            st.error(f"Erro ao gerar roteiro: {e}")

# Exibe o histórico do chat (usuário e assistente)
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)