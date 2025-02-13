import chromadb
from langchain_ollama import OllamaLLM
import streamlit as st

# Подключение к ChromaDB для хранения и поиска контекста
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(
    name="chat_data",
    metadata={"description": "Chat history"}
)

def add_documents_to_collection(documents, ids):
    """Добавление документов в коллекцию и вывод в терминал."""
    if documents and ids:
        collection.add(documents=documents, ids=ids)

def query_chromadb(query_texts, n_results=3):
    """Запрос в ChromaDB для получения релевантного контекста."""
    try:
        results = collection.query(query_texts=query_texts, n_results=n_results)
        return results["documents"]
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return []

# Настройка модели Ollama LLM
llm_model = "llama3.2"

def query_ollama(prompt):
    """Запрос к LLM Ollama для получения ответа."""
    try:
        llm = OllamaLLM(model=llm_model, host="http://localhost:11434")
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return "Error: Unable to process the query."

# Streamlit интерфейс
st.title("AI Assistant for Kazakhstan Constitution")

# Ввод нескольких запросов
queries = st.text_area("Enter your questions about the Constitution (separate with a newline):")

if st.button("Submit"):
    if queries:
        # Разбиваем ввод на отдельные запросы
        query_list = queries.split("\n")
        
        responses = []  # Список для хранения ответов

        for query in query_list:
            if query.strip():  # Проверяем, что запрос не пустой
                # Получение контекста из ChromaDB для каждого запроса
                chroma_results = query_chromadb(query)
                
                # Если есть контекст, формируем запрос для модели с учетом контекста
                if chroma_results:
                    # Преобразуем все элементы контекста в строки, чтобы избежать ошибки TypeError
                    context = "\n\n".join([str(doc) for doc in chroma_results])
                    query_with_context = f"Context: {context}\n\nQuestion: {query}"
                    response = query_ollama(query_with_context)
                else:
                    response = query_ollama(query)
                
                # Сохраняем ответ
                responses.append(f"Answer to '{query}': {response}")
        
        # Показываем все ответы
        for response in responses:
            st.write(response)
