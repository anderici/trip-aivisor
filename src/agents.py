from openai import OpenAI
import os
import json
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

client = OpenAI(api_key=os.getenv('OPENAI_KEY'))

def chat_completion(messages, model="gpt-4-1106-preview"):
     # Mapeia para o formato aceito pelo OpenAI SDK: dicts com role/content
    role_map = {
        "human": "user",
        "ai": "assistant",
        "system": "system"
    }
    messages_payload = [
        {"role": role_map[msg.type], "content": msg.content} for msg in messages
    ]
    
    response = client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=messages_payload,
        temperature=0,
        # response_format='json'
    )
    return response.choices[0].message.content.strip()

def init_state(state):
    print('---------- Init ----------')
    print('State: ', state)
    
    return {'messages': [], 'quality': 0, 'iterations': 0, 'final_script': ''}

def travel_agent(state):
    print('---------- Travel Agent Start ----------')

    system_prompt = """Com base na descrição do cliente, você deve gerar um roteiro completo, considerando todas as informações recebidas //
        como data e hora de chegada e partida, locais da viagem. Certifique-se de fornecer a melhor experiência para o cliente //
        dada as informações que lhe forem fornecidas. Você deve ser bem detalhista. Por exemplo, se colocar um trajeto que pode ser //
        feito de trem, liste horários desse trem, os valores e onde o cliente pode comprar as passagens. Quando sugerir um lugar que //
        exige a compra de tickets, faça a mesma coisa. 
        Nunca dê uma sugestão genérica. Ao invés de falar para comer em um restaurante ou padaria local, sempre sugira um nome.
        Sempre dê o preço das coisas.
        Ao sugerir uma atração ou um restaurante, indique como chegar com meios de transporte."""
    
    humam_messages = [msg for msg in state['messages'] if isinstance(msg, HumanMessage)]
    ai_messages = [msg for msg in state['messages'] if isinstance(msg, AIMessage)]
    system_message = [SystemMessage(content=system_prompt)]
    messages = system_message + humam_messages + ai_messages

    reply_content = chat_completion(messages=messages)

    # Cria um novo AIMessage com a resposta
    new_ai_message = AIMessage(content=reply_content)

    print('Roteiro: ', new_ai_message.content)

    print('---------- Travel Agent End ----------')

    return {"messages": [new_ai_message], "iterations": state["iterations"] + 1, 'final_script': new_ai_message.content}


def reviewer_agent(state):
    print('---------- Reviewer Agent Start ----------')

    system_prompt = """Com base na descrição do cliente e do roteiro criado pelo agente de viagens, você deve revisar minuciosamente //
        o roteiro que lhe foi passado. Você deve atribuir uma nota de 0 a 1000 a esse roteiro, além de indicar pontos de melhoria. //
        Certifique-se também de revisar as datas da sugestões para não deixar, por exemplo, o agente de viagens recomendar uma atração //
        num dia que essa atração não funciona ou que um restaurante não abre.
        Certifique-se de revisar se as sugestões de atrações, restaurantes, hotéis, etc, realmente existem. Será péssimo colocar algo //
        no roteiro que não exista naquele lugar ou esteja permanentemente fechado.

        Responda SOMENTE com um JSON válido, sem explicações ou mensagens adicionais.
        {
            "score": int,
            "review": str
        }"""
    
    humam_messages = [msg for msg in state['messages'] if isinstance(msg, HumanMessage)]
    ai_messages = [msg for msg in state['messages'] if isinstance(msg, AIMessage)]
    system_message = [SystemMessage(content=system_prompt)]
    messages = system_message + humam_messages + ai_messages
    
    reply_content = chat_completion(messages)

    try:
        parsed = json.loads(reply_content)
        score = int(parsed.get("score", 0))
        review = parsed.get("review", "")
    except json.JSONDecodeError:
        score = 0
        review = "Erro ao processar a resposta do modelo."

    print('Revisão: ', review)
    print('Score: ', score)

    comment = ''

    
    print('---------- Reviewer Agent End ----------')

    if score < int(os.getenv('QUALITY_TRESHOLD')):
        comment = HumanMessage(content='A qualidade do seu roteiro não está boa. Por favor, revise-o de acordo com os meus comentários: ' + review)
        return {"messages": [comment], 'quality': score}
    else:
        return {"messages": [], 'quality': score}

def summary_state(state):
    print('---------- Summary START ----------')
    print(state.get("final_script", "Nenhum roteiro gerado."))
    print('---------- Summary END ----------')