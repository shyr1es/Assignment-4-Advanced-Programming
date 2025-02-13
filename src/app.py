import os
import chromadb
import streamlit as st
from langchain_ollama import OllamaLLM
from constitution_parser import extract_text_from_pdf

# Подключение к ChromaDB для хранения и поиска контекста
chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))
collection = chroma_client.get_or_create_collection(
    name="chat_data",
    metadata={"description": "Chat history"}
)

# Функции для работы с ChromaDB
def add_documents_to_collection(documents, ids):
    """Добавление документов в коллекцию и вывод в терминал."""
    if documents and ids:
        collection.add(documents=documents, ids=ids)
        print(f"Saved {len(documents)} documents with IDs: {ids}")
    else:
        print("No documents to save.")

def query_chromadb(query_text, n_results=3):
    """Запрос в ChromaDB для получения релевантного контекста."""
    try:
        results = collection.query(query_texts=[query_text], n_results=n_results)
        return results
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return None

# Настройка модели Ollama LLM
llm_model = "llama3.2"

def query_ollama(prompt):
    """Запрос к LLM Ollama для получения ответа."""
    try:
        llm = OllamaLLM(model=llm_model, host="http://localhost:11434")  # Указываем хост и порт явно
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return "Error: Unable to process the query."

# Streamlit интерфейс
st.title("AI Assistant for Kazakhstan Constitution")

# Загрузка PDF файлов
uploaded_files = st.file_uploader("Upload your Constitution PDF", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    documents = []
    ids = []

    for file in uploaded_files:
        try:
            # Извлечение текста из загруженных PDF файлов
            file_content = extract_text_from_pdf(file)
            documents.append(file_content)
            ids.append(file.name)
        except Exception as e:
            st.error(f"Error reading file {file.name}: {e}")

    if documents:
        # Добавление документов в ChromaDB
        add_documents_to_collection(documents, ids)
        st.success(f"Uploaded and saved {len(uploaded_files)} documents.")
        st.write("Documents saved:")
        for doc in documents:
            st.write(doc[:500] + ("..." if len(doc) > 500 else ""))  # Показываем часть текста для проверки
    else:
        st.warning("No valid documents uploaded.")

# Ввод пользователя
query = st.text_input("Enter your question about the Constitution:")

if st.button("Submit"):
    if query:
        # Получение контекста из ChromaDB
        chroma_results = query_chromadb(query)

        # Проверяем, есть ли контекст из документов
        if chroma_results and "documents" in chroma_results and chroma_results["documents"]:
            # Формируем контекст из найденных документов
            context_list = [str(doc) if isinstance(doc, list) else doc for doc in chroma_results["documents"]]
            context = "\n\n".join(context_list)
            st.write("Context from Documents:")
            st.write(context[:500] + ("..." if len(context) > 500 else ""))  # Показываем до 500 символов контекста

            # Формируем запрос с учётом контекста
            query_with_context = f"Context: {context}\n\nQuestion: {query}"
            response = query_ollama(query_with_context)
        else:
            st.warning("No relevant context found in uploaded documents. Using question only.")
            response = query_ollama(query)

        # Отображаем ответ от LLM
        if response:
            st.write("Answer:", response)
        else:
            st.warning("No response generated.")

        # Сохранение запроса и ответа в ChromaDB
        add_documents_to_collection([query, response], [f"query_{query}", f"response_{query}"])
    else:
        st.warning("Please enter a question.")

# Кнопка для отображения истории чатов
if st.button("Show Chat History"):
    history = collection.peek()
    if history and "documents" in history:
        st.write("Chat History:")
        for doc_id, doc in zip(history["ids"], history["documents"]):
            st.write(f"ID: {doc_id}, Content: {doc[:200]}...")  # Показываем только первые 200 символов
    else:
        st.warning("No chat history found.")
