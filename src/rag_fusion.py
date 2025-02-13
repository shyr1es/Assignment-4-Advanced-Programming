import chromadb
from langchain_ollama import OllamaLLM
import streamlit as st
from constitution_parser import extract_text_from_pdf

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

def query_chromadb(query_text, n_results=3):
    """Запрос в ChromaDB для получения релевантного контекста."""
    try:
        results = collection.query(query_texts=[query_text], n_results=n_results)
        return results["documents"]
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return []

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
        if chroma_results:
            # Преобразуем все элементы контекста в строки, если они не строки
            context = "\n\n".join([str(doc) for doc in chroma_results])
            st.write("Context from Documents:")
            st.write(context[:500] + ("..." if len(context) > 500 else ""))  # Показываем до 500 символов контекста

            # Формируем запрос с учётом контекста
            query_with_context = f"Context: {context}\n\nQuestion: {query}"
            response = query_ollama(query_with_context)
        else:
            st.warning("No relevant context found in uploaded documents. Using question only.")
            response = query_ollama(query)

        st.write("Answer:", response)
