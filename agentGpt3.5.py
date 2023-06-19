import pandas as pd
from dotenv import load_dotenv, find_dotenv
import streamlit as st
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.document_loaders.csv_loader import CSVLoader


# Sidebar contents
with st.sidebar:
    st.title('💬 Transactions Anaylsis')

load_dotenv(find_dotenv())
df=pd.read_csv('./Transactions.csv')

#df.head()

#chat = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.0)
agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df,verbose=True)
st.header("Ask questions below")
if st.button("What are the common categories"):
       with st.spinner(text="In progress..."):
          answer = agent.run("What are the common categories")
          st.write(answer)
if st.button("What are the common dates"):
       with st.spinner(text="In progress..."):
          answer = agent.run("What are the common dates")
          st.write(answer)
query = st.text_input("Ask a question about your CSV: ")
if query is not None and query != "":
            with st.spinner(text="In progress..."):
                st.write(str(agent.run(query)))

st.write("Data Preview:")
st.dataframe(df.head())
#agent.run("Average amount of first 3 column")