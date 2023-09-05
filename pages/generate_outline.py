from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import openai
import streamlit as st

from llama_index import (
    Document,
    VectorStoreIndex,
    ServiceContext,
    set_global_service_context,
)
from llama_index.llms import OpenAI


### Streamlit Multi-Page
st.set_page_config(
    page_title="Generate Outline",
    page_icon="🔖",
)

col1, col2 = st.columns(2, gap="large")

# Session State
if "blog_title" not in st.session_state:
    st.session_state["blog_title"] = "Blog Title"

if "outline" not in st.session_state:
    st.session_state["outline"] = "Here comes generated outline."

if "query_engine" not in st.session_state:
    st.session_state["query_engine"] = None

### Selenium Prepare
# Set the options: run in headless mode
options = Options()
options.add_argument("--headless")

# Initialize the Chrome driver
driver = webdriver.Chrome(
    service=ChromeService(ChromeDriverManager().install()), options=options
)

### Create title by yourself
with col1:
    st.header("Generate Outline")
    blog_title = st.text_area(label="blog title", value=st.session_state["blog_title"])
    st.divider()

    ### Get HTML file by Selenium
    url_1 = st.text_input(label="URL01", value="https://google.com")
    url_2 = st.text_input(label="URL02")
    url_3 = st.text_input(label="URL03")
    url_4 = st.text_input(label="URL04")
    url_5 = st.text_input(label="URL05")
    url_6 = st.text_input(label="URL06")
    url_7 = st.text_input(label="URL07")

    text_list = []
    ## Scrape with Beautifulsoup
    submit_button = st.button(
        label="Generate Outline from title and URLs", type="primary"
    )
    if submit_button:
        if not blog_title:
            st.warning("Please create a title.")
        else:
            st.session_state["blog_title"] = blog_title
            urls = [url_1, url_2, url_3, url_4, url_5, url_6, url_7]
            with st.spinner("Wait for it..."):
                for url in urls:
                    if url:
                        driver.get(url)
                        # Get HTML
                        html = driver.page_source
                        # Scrape with Beautifulsoup
                        soup = BeautifulSoup(html, "lxml")
                        text = soup.get_text()
                        text_list.append(text)
                driver.quit()
                ### Send as a Vector Data using Llama Index
                # Try with GPT-4 by customizing LLM
                # define LLM
                # GPT-4: Max tokens by model is 8192
                llm = OpenAI(model="gpt-4", temperature=0.1, max_tokens=5200)

                # configure service context
                service_context = ServiceContext.from_defaults(llm=llm)
                set_global_service_context(service_context)

                # Read the document
                documents = [Document(text=t) for t in text_list]
                # Build Index
                index = VectorStoreIndex.from_documents(documents)
                # Build Query Engine
                query_engine = index.as_query_engine()
                # Save in the session state
                st.session_state["query_engine"] = query_engine

                # Throw a Query
                outline_response = query_engine.query(
                    f'Create an outline in Japanese for blog post: title "{blog_title}".'
                )
                # Save in the session state
                st.session_state["outline"] = outline_response.response

with col2:
    st.subheader("Outline")
    if st.session_state["outline"]:
        # Show the result
        st.write(st.session_state["outline"])
        st.divider()

        # Save as a file
        st.download_button(
            label="Download the outline as a text file",
            data=st.session_state["outline"],
            file_name=f"{blog_title}_outline.txt",
            mime="text/plain",
        )
