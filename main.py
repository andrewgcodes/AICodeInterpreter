#imports
import pandas as pd
import tiktoken
import openai
import os
import streamlit as st
import warnings
warnings.filterwarnings('ignore')



# embedding model parameters
# imports
import ast  # for converting embeddings saved as strings back to arrays
from scipy import spatial  # for calculating vector similarities for search


# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo-0301"
openai.api_key = os.getenv("OPENAI_API_KEY")

@st.cache_data
def load_data():
    df = pd.read_csv("steveJobs1200charchunks_with_embeddings_1k.csv")
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
    return df
st.title("Talk to Steve Jobs")
st.info("Please wait for the embeddings to load. The speed of this site is limited by OpenAI's API speed, which is currently throttled by heavy demand.")
df = load_data()
st.balloons()

def display_result_card(result):
    card_style = """
    <style>
        .card {
            background: #FFFFFF;
            border: 1px solid #E0E0E0;
            border-radius: 12px;
            box-shadow: 0px 1px 3px rgba(0, 0, 0, 0.1), 0px 1px 2px rgba(0, 0, 0, 0.06);
            padding: 15px;
            margin-bottom: 15px;
            transition: transform .2s;
            font-family: 'SF Pro Text', 'SF Pro Icons', 'Helvetica Neue', 'Helvetica', 'Arial', sans-serif;
            color: #333;
        }
        .card:hover {
            box-shadow: 0px 10px 15px -3px rgba(0, 0, 0, 0.1), 0px 4px 6px -2px rgba(0, 0, 0, 0.05);
            transform: scale(1.02);
        }
        .css-15zrgzn {display: none}
        .css-eczf16 {display: none}
        .css-jn99sy {display: none}
    </style>
    """

    card_content = f"""
    <div class="card">
        <p>{result}</p>
    </div>
    """

    st.markdown(card_style, unsafe_allow_html=True)
    st.markdown(card_content, unsafe_allow_html=True)
# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["Text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df, top_n=5)
    introduction = 'Use the below sections from the memoir of Steve Jobs to answer the subsequent question. If the answer cannot be found in the extracts, write "I could not find an answer from the book" and provide information based on your own knowledge of Steve Jobs.'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\Steve Jobs memoir section:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You are a knowledgeable and helpful AI Steve Jobs who answers questions about Steve Jobs based on his memoir as well as your personal knowledge of Steve Jobs, Apple founder. Pretend you are Steve Jobs. Provide short direct quotes from the provided sections of Steve Jobs' memoir in your answers."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message

question = st.text_input("Ask me a question", value = "what are your thoughts about designing", max_chars = 100)
if len(question) > 1:
    with st.spinner(text="Steve is thinking..."):
        try:
            answer = ask(question)
            #st.write(answer)
            display_result_card(answer)
        except:
            st.write("Sorry, something went wrong on OpenAI's side")
