def get_answer(llm, vector_store, question):
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA(llm=llm, retriever=retriever)
    return qa_chain.run(question)