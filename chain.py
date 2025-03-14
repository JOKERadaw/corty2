import streamlit as st
from similaritysearch import similaritysearch  # your custom FAISS-based function
from langchain.llms import LlamaCpp
from langchain.chains import LLMMathChain
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain.utilities.wikipedia import WikipediaAPIWrapper
from langchain.llms import Ollama

# Connect to your local Ollama instance


# 1. Load your locally installed LLaMA 3.2 model (ensure the model path is correct)
llama = Ollama(model="llama3.2")

# 2. Set up the built-in math tool using LLMMathChain
math_tool = LLMMathChain(llm=llama, verbose=True)

# 3. Set up the built-in Wikipedia tool
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# 4. Define an agent function that selects a tool based on the query
def agent(query: str) -> str:
    lower_query = query.lower()
    # If the query is math-related, use the math tool

    # If the query is Wikipedia-related, use the Wikipedia tool
    if any(keyword in lower_query for keyword in ["wikipedia", "who is", "what is", "history of"]):
        return wiki_tool.run(query)
    else:
        # Otherwise, perform a similarity search on your documents
        docs = similaritysearch(query)  # returns top 3 documents
        # Concatenate the results to create context
        context = "\n".join([str(doc) for doc in docs])
        # Build a prompt that includes the retrieved context
        prompt = f"Using the following context, answer the question:\n\nContext:\n{context}\n\nQuestion: {query}"
        # Get an answer from your local LLaMA model
        return llama(prompt)

# 5. Set up the Streamlit webapp interface
translations = {
    "English": {
       "title": "Corty: An Unofficial Cortical Lab Documentation Helper 🧠 ",
        "query_prompt": "Enter your query:",
        "submit_button": "Submit",
        "processing": "Processing..."
    },
    "Spanish": {
        "title": "Corty: Un Ayudante no oficial de Documentación del Laboratorio Cortical 🧠",
        "query_prompt": "Ingrese su consulta:",
        "submit_button": "Enviar",
        "processing": "Procesando..."
    },
    "French": {
        "title": "Corty: Un Assistant de Documentation non officiel pour Laboratoire Cortical 🧠",
        "query_prompt": "Entrez votre requête:",
        "submit_button": "Soumettre",
        "processing": "Traitement en cours..."
    },
    "German": {
        "title": "Corty: Ein inoffizieller Dokumentations-Helfer für das Kortikale Labor 🧠",
        "query_prompt": "Geben Sie Ihre Anfrage ein:",
        "submit_button": "Einreichen",
        "processing": "Verarbeitung..."
    },
    "Italian": {
        "title": "Corty: Un Assistente non ufficiale per la Documentazione del Laboratorio Corticale 🧠",
        "query_prompt": "Inserisci la tua domanda:",
        "submit_button": "Invia",
        "processing": "Elaborazione..."
    },
    "Chinese": {
        "title": "Corty: 非官方皮质实验室文档助手 🧠",
        "query_prompt": "输入您的查询：",
        "submit_button": "提交",
        "processing": "处理中..."
    },
    "Japanese": {
        "title": "Corty: 非公式皮質ラボ文書ヘルパー 🧠",
        "query_prompt": "クエリを入力してください：",
        "submit_button": "送信",
        "processing": "処理中..."
    },
    "Arabic": {
        "title": "كورتي: مساعد توثيق غير رسمي لمختبر القشرة الدماغية 🧠",
        "query_prompt": "أدخل استعلامك:",
        "submit_button": "إرسال",
        "processing": "جاري المعالجة..."
    }
}  
languages = ["English", "Spanish", "French", "German","Italian", "Chinese", "Japanese", "Arabic"]
selected_language = st.selectbox(
    "Select Language",
    options=languages,
    index=0  # Default to first option (English)
)


text = translations[selected_language]
    
# Display the title in the selected language
st.title(text["title"])

# Display the query input in the selected language
user_input = st.text_input(text["query_prompt"])

# Display the submit button in the selected language
if st.button(text["submit_button"]):
    with st.spinner(text["processing"]):
        # Replace with your actual agent call
        answer = agent(user_input)
    st.write(answer)
# Display the selected language

