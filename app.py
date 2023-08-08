from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
import multiprocessing
from tempfile import NamedTemporaryFile
import pandas as pd
import json
import requests
from pypdf import PdfReader

# 1. Convert PDF file into images via pypdfium2

# Replace the original three functions with this one
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    number_of_pages = len(reader.pages)
    text = ''
    for page_number in range(number_of_pages):
        page = reader.pages[page_number]
        text += page.extract_text()
    return text

# 3. Extract structured info from text via LLM
def extract_structured_data(content: str, data_points):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    template = """
    You are an expert admin who will extract core information from documents

    {content}

    Above is the content; please try to extract all data points from the content above 
    and export in a JSON array format:
    {data_points}

    Now please extract details from the content  and export in a JSON array format, 
    return ONLY the JSON array:
    """

    prompt = PromptTemplate(
        input_variables=["content", "data_points"],
        template=template,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    results = chain.run(content=content, data_points=data_points)

    return results




# 5. Streamlit app
def main():
    default_data_points = """{
        "WIC Number": "what is the 5 digit WIC",
        "Performance From": "what is the Perform From date",
        "Performance To": "what is the Performance To date",
        "Scan amount": "what is the DOLLARS PER",
    }"""

    st.set_page_config(page_title="Doc extraction", page_icon=":bird:")

    st.header("Doc extraction :bird:")

    data_points = st.text_area(
        "Data points", value=default_data_points, height=170)

    uploaded_files = st.file_uploader(
        "upload PDFs", accept_multiple_files=True)

    if uploaded_files is not None and data_points is not None:
        results = []
        for file in uploaded_files:
            with NamedTemporaryFile(dir='.', suffix='.csv') as f:
                f.write(file.getbuffer())
                content = extract_text_from_pdf(f.name)
                print(content)
                data = extract_structured_data(content, data_points)
                json_data = json.loads(data)
                if isinstance(json_data, list):
                    results.extend(json_data)  # Use extend() for lists
                else:
                    results.append(json_data)  # Wrap the dict in a list

        if len(results) > 0:
            try:
                df = pd.DataFrame(results)
                st.subheader("Results")
                st.data_editor(df)
                
            except Exception as e:
                st.error(
                    f"An error occurred while creating the DataFrame: {e}")
                st.write(results)  # Print the data to see its content


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
