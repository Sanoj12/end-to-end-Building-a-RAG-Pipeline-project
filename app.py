import wikipedia
from transformers import AutoTokenizer,AutoModelForQuestionAnswering,pipeline
from sentence_transformers import SentenceTransformer #embedding
import streamlit as st
import faiss

import numpy as np



###step-1 get data

def get_wikipedia_data(topic):
    try:
        page = wikipedia.page(topic)
        return page.content

    except wikipedia.exceptions.PageError:

        return None

    except wikipedia.exceptions.DisambiguationError as e:
        #  the topic is multiple
        print(f" topic, Options: {e.options}")
        return None
    

##user input 
topic = input("enter a topic ")
document = get_wikipedia_data(topic)

if not document:
    print("could not retreive information")



####tokenization   --we will split the text into smaller overlapping chunks


#tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
#tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

def split_text(text, chunk_size=256 ,chunk_overlap =20):

    #text converted into token 
    tokens = tokenizer.tokenize(text)
   
    ##initalize empty list

    chunks = []
    
    start = 0 #starting positon


    ##spliting into chunks

    while start < len(tokens): 
        end = min(start + chunk_size , len(tokens)) #if start =240 and end = 256 we don't have the total number of tokens eg: chunk_size =256 ,then end=240(to prevent overflow)
        chunks.append(tokenizer.convert_tokens_to_string(tokens[start:end]))
        if end == len(tokens):
            break

        start = end - chunk_overlap

    return chunks




chunks= split_text(document)
print(len(chunks))




##### convert the text into embeddingd and store in faiss index:
##text into numerical embedding
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embeddings = embedding_model.encode(chunks)

## the number of dimensions in each vector  it is 768
dimension =embeddings.shape[1]   ##shape[1] number of columns (2,768) 

##print(dimension)

##create faiss index - using fast nearest neighbor search -euclidean distance
 
index= faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))  ##embedding faiss index for nearest neighbors search




####querying the rag output




query = input("ask a question about the topic:")



query_embeddings = embedding_model.encode([query]) ##pass into a list



k=3 ##top  3 most similar 

##similarity search
distances ,indices = index.search(query_embeddings,k)


retrieved_chunks =[chunks[i] for i in indices[0]]

print("Retreived_chunks:")

for chunk in retrieved_chunks:
     print("- " + chunk)






###answering with llm

qa_model_name ="deepset/xlm-roberta-large-squad2"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)


qa_model =AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

qa_pipeline = pipeline("question-answering" ,model=qa_model ,tokenizer=qa_tokenizer )


context = " ".join(retrieved_chunks)

answer =qa_pipeline(question=query ,context=context)
print(f"answer: {answer['answer']}")







###############################STREAMLIT App

st.title("Q&A RAG APPLICATION 🚀")
st.write("Hello, this is a Q&A application based on wikipedia content")

topic = st.text_input("enter a topic")


if topic:

    document = get_wikipedia_data(topic)


if not document:

    st.error("could not retreive information")

else:

    st.success("data retreive successfully!")
    chunks = split_text(document)
    embeddings = embedding_model.encode(chunks)
    dimension =embeddings.shape[1] 
     
    index= faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings)) 

query = st.text_input("ask a question about the topic:")

if query:
    query_embeddings = embedding_model.encode([query])
    k=3
    ##similarity search
    distances ,indices = index.search(query_embeddings,k)


    retrieved_chunks =[chunks[i] for i in indices[0]]

    st.subheader("Retrieve Chunks:")

    for chunk in retrieved_chunks:
        st.write(f"-  {chunk}")

    
    context = " ".join(retrieved_chunks)
    answer =qa_pipeline(question=query ,context=context)

    st.subheader("Answer:")
    st.write(answer['answer'])








