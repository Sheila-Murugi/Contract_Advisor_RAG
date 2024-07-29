# Contract Advisor Project

Welcome to the Contract Advisor Project! This project aims to build and evaluate a Question-Answer (QA) pipeline using LangChain and OpenAI's GPT models to assist with contract analysis.

## Overview

The Contract Advisor Project leverages state-of-the-art natural language processing (NLP) techniques to answer questions based on contract documents. It retrieves relevant documents, generates accurate answers, and evaluates the accuracy of the generated responses.

## Features

- Document retrieval based on relevance
- Question answering using GPT models
- Evaluation of generated answers against expected answers

## Requirements

- Python 3.8+
- Install the necessary Python packages:
  ```bash
  pip install openai langchain chromadb
  ```

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/contract-advisor.git
   cd contract-advisor
   ```

2. **Set up OpenAI API Key:**
   Ensure you have an OpenAI API key. Set the environment variable:
   ```bash
   export OPENAI_API_KEY='your_openai_api_key'
   ```

## Usage

1. **Load Contract Documents:**
   Ensure your contract documents are in a text format.

2. **Initialize and Run the Pipeline:**

   ```python
   from langchain.embeddings import OpenAIEmbeddings
   from langchain.document_loaders import TextLoader
   from langchain.vectorstores import Chroma
   from langchain.chains import VectorDBQA
   from langchain.chat_models import ChatOpenAI

   # Load and prepare documents
   loader = TextLoader("path_to_contracts")
   documents = loader.load()

   # Set up embeddings and vector store
   embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
   vectorstore = Chroma.from_documents(documents, embeddings)

   # Initialize retriever and QA pipeline
   retriever = vectorstore.as_retriever()
   llm = ChatOpenAI(model_name="gpt-3.5-turbo")
   qa_pipeline = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

   # Evaluate the pipeline
   questions = [
       "What is the termination clause in this contract?",
       "What is the time tracking clause in this contract?"
   ]
   expected_answers = [
       "The termination clause in this contract states that either party, at any given time, may terminate this Agreement, for any reason whatsoever, with or without cause, upon fourteen (14) days' prior written notice. Notwithstanding the above, the Company may terminate this Agreement immediately and without prior notice if Advisor refuses or is unable to perform the Services, or is in breach of any provision of this Agreement.",
       "The time tracking clause in this contract states that the advisor shall provide the Company with a written report, in a format acceptable by the Company, setting forth the number of hours in which he provided the Services, on a daily basis, as well as an aggregated monthly report at the last day of each calendar month..."
   ]

   evaluation_results = evaluate_pipeline(qa_pipeline, questions, expected_answers)
   print(evaluation_results)
   ```

3. **Evaluation:**
   The `evaluate_pipeline` function calculates the accuracy of the QA pipeline by comparing the generated answers with the expected answers.

## Evaluation Function

```python
def evaluate_pipeline(qa_pipeline, questions, expected_answers):
    correct_answers = 0
    total_questions = len(questions)
    results = []

    for question, expected in zip(questions, expected_answers):
        retrieved_docs = retriever.get_relevant_documents(question)
        combined_docs = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = f"Documents: {combined_docs}\n\nQuestion: {question}\n\nAnswer:"

        messages = [
            {"role": "system", "content": "You are a helpful assistant for answering questions based on provided documents."},
            {"role": "user", "content": prompt}
        ]
        answer = chat_completion(messages)

        # Evaluate accuracy and relevance
        correct = expected.lower() in answer.lower()
        correct_answers += int(correct)
        results.append({"question": question, "expected": expected, "answer": answer, "correct": correct})

    accuracy = correct_answers / total_questions
    return {"accuracy": accuracy, "results": results}
```

---

Thank you for using the Contract Advisor Project! We hope it proves useful for your contract analysis needs.
