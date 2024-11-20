import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
# load_dotenv()

# # Retrieve the API key from the .env file


# if not GROQ_API_KEY:
#     st.error("GROQ API key is not set. Please configure the .env file.")
#     st.stop()
    
# Set up the language model
llm = ChatGroq(
    temperature=0,
    groq_api_key=st.secrets["GROQ"]["API_KEY"],
    model_name="llama-3.2-90b-vision-preview"
)

# Define function to generate metadata
def generate_metadata(df):
    return df.columns.tolist()

# Define function to process user prompts
def ask_and_execute(prompt, df):
    columns = generate_metadata(df)
    generated_code = llm.invoke(
        f"""
        ### COLUMN NAMES:
        {columns}

        ### QUESTION:
        {prompt}

        ### INSTRUCTION:
        Your task is to write python code for the above prompt on a Pandas DataFrame called 'df' and don't use something that would require imports like plt.show only use functions available in pandas. Respond strictly with the code only, no explanations, no formatting, no backticks.

        ### ANSWER:
        """
    )
    generated_code = generated_code.content.strip()
    if not generated_code.startswith("result = "):
        generated_code = f"result = {generated_code}"
    
    try:
        local_vars = {}
        exec(generated_code, globals(), local_vars)
        if "result" in local_vars:
            answer = local_vars['result']
            final_response = llm.invoke(
                f"""
                ### COLUMN NAMES:
                {columns}

                ### QUESTION:
                {prompt}

                ### OUTPUT:
                {answer}

                ### INSTRUCTION:
                You will be given column names, the user question, and the output. Just reply to the user with their answer. If the answer is not given, say "I can't answer."

                ### ANSWER:
                """
            )
            return final_response.content
        else:
            return "Executed without result."
    except Exception as e:
        return str(e)

# Streamlit app
st.title("Excel Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# File uploader
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

# DataFrame placeholder
df = None
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("File uploaded successfully. Here's a preview:")
    st.dataframe(df)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your data:"):
    # Display user message in chat container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    if df is not None:
        with st.chat_message("assistant"):
            # Generate response
            with st.spinner("Processing..."):
                response = ask_and_execute(prompt, df)
            st.markdown(response)
            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        with st.chat_message("assistant"):
            error_message = "Please upload an Excel file first."
            st.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
