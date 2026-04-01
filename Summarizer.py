import os
import zipfile
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

# Load environment variables
load_dotenv()

# --- Set API key -----
api_key = os.getenv('mistral_key')
if not api_key:
    st.error("Mistral API key not found. Set it  as environment variable MISTRAl_API_KEY.")
    st.stop()
os.environ["MISTRAL_API_KEY"] = api_key

# --- Model selection ----
MODEL_NAME ="mistral-small-latest" 

@st.cache_resource
def load_llm():
    return ChatMistralAI(model=MODEL_NAME, temperature=0.7)

@st.cache_resource
def load_agent():
    llm = load_llm()
    system_message = (
        "You are a Professional Article Writer specializing in writing articles for Medium, LinkedIn, and tech blogs."
    )
    agent = create_agent(
        model=llm,
        tools=[],   # No external tools – summarization only
        system_prompt=system_message,
        middleware=[
            SummarizationMiddleware(
                model=llm,
                trigger=("tokens", 1000),   # Summarize when conversation reaches 1000 tokens
                keep=("tokens", 200),       # Keep last 200 tokens verbatim
            ),
        ],
    )
    return agent

# --- Helper functions  ---
def extract_transcript(link: str) -> str:
    """Extract YouTube transcript using YoutubeLoader."""
    loader = YoutubeLoader.from_youtube_url(link)
    docs = loader.load()
    return docs[0].page_content

def get_text_chunks(text: str, chunk_size=5000, chunk_overlap=200):
    """Split text into manageable chunks for recursive summarization."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_text(text)

def recursive_summarize(text: str, agent):
    """Summarize long transcripts chunk by chunk using the agent."""
    chunks = get_text_chunks(text)
    running_summary = ""

    # System prompt for the summarizer (given in the user message)
    system_prompt = (
        "You are a recursive summarization engine. "
        "Continuously summarize incoming content. "
        "Maintain a concise but complete running summary. "
        "When content grows large, compress older information. "
        "Preserve key technical details, concepts, and relationships. "
        "Output ONLY the updated summary."
    )

    for chunk in chunks:
        response = agent.invoke({
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"""
Current summary:
{running_summary}

New content:
{chunk}

While summarizing the text in the form of an article, strictly follow:
**CRITICAL INSTRUCTIONS**:
- IGNORE introductory notes like "welcome", "in this video"
- IGNORE all channel names, "subscribe", "like", "comment", "follow", "check description"
- IGNORE marketing phrases: "my course", "my discord", "affiliate links", "sponsors"
- FOCUS ONLY on technical content, code, tutorials, actionable insights

**MANDATORY ARTICLE STRUCTURE** (exact Medium/LinkedIn format):
- Write in **first-person professional tone**
- Use **bold subheadings**, **numbered lists**
- Include **code snippets** for technical videos
- Make **Actionable Steps** copy-paste ready
- End with a **short summary of the article**
"""
                }
            ]
        })
        # Extract the last assistant message content
        running_summary = response["messages"][-1].content

    return running_summary

def estimate_transcript_length(link: str) -> bool:
    """Quick length estimator (characters → tokens)."""
    transcript = extract_transcript(link)
    return len(transcript) >= 1000   # True for long transcripts

# --- Define the summarization pipeline ---
llm = load_llm()
agent = load_agent()

# Base summarizer (short transcripts)
summarizer_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a Professional Article Writer specializing in writing articles for Medium, LinkedIn, and tech blogs."
    ),
    HumanMessagePromptTemplate.from_template(
        """
Transform YouTube transcript into **engaging, professional articles** with:

**CRITICAL INSTRUCTIONS**:
- **IGNORE** Introductionary notes like welcome, In this video
- **IGNORE** all channel names, "subscribe", "like", "comment", "follow", "check description" 
- **IGNORE** marketing phrases: "my course", "my discord", "affiliate links", "sponsors"
- **FOCUS ONLY** on technical content, code, tutorials, actionable insights

**MANDATORY ARTICLE STRUCTURE** (exact Medium/LinkedIn format):
- Write in **first-person professional tone** 
- Use **bold subheadings**, **numbered lists**.
- Include **code snippets** for technical videos
- Make **Actionable Steps** copy-paste ready
- End with **short summary of the article**
{transcript}
"""
    )
])

base_summarizer = (
    RunnablePassthrough()
    | RunnableLambda(extract_transcript)
    | summarizer_prompt
    | llm
    | StrOutputParser()
)

# Long summarizer (using recursive summarization)
def long_summarizer_pipeline(link: str) -> str:
    transcript = extract_transcript(link)
    return recursive_summarize(transcript, agent)

long_summarizer = RunnablePassthrough() | RunnableLambda(long_summarizer_pipeline)

# --- Webpage generation prompt and chain ---
web_dev_system = """You are a Senior Frontend Web Developer with 10+ years experience in HTML5, CSS3, and modern JavaScript (ES6+).

Your task: Generate COMPLETE, PRODUCTION-READY frontend code based on user requirements.

**MANDATORY OUTPUT FORMAT** (exact delimiters):
--html--
[html code here]
--html--h

--css--
[css code here]
--css--

--js--
[java script code here]
--js--
"""

web_dev_human = '''
Create a **production-ready article webpages** in the style of **Medium, Dev.to, Hashnode, and Substack**.

**MANDATORY REQUIREMENTS**:
- **Mobile-first responsive design** (perfect on all devices)
- **Clean, modern typography** (system fonts + readability first)
- **Medium-like article layout** with card-based design
- **Dark/light theme toggle**
- **Smooth animations** and **scroll effects**
- **SEO optimized** with proper meta tags
- **Accessibility compliant** (ARIA labels, keyboard navigation)

**CONTENT TO USE**: {article_content}
'''

web_dev_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(web_dev_system),
    HumanMessagePromptTemplate.from_template(web_dev_human)
])

webpage_chain = web_dev_template | llm | StrOutputParser()

# --- Branch summarizer (routes based on transcript length) ---
smart_summarizer = RunnableBranch(
    (RunnableLambda(estimate_transcript_length), long_summarizer),
    base_summarizer
) | webpage_chain

# --- Streamlit UI ---
st.set_page_config(page_title="YouTube to Article Webpage", page_icon="📄", layout="wide")
st.title("🎥 YouTube to Article Webpage")
st.markdown("Enter a YouTube URL and get a polished article webpage generated from its transcript.")

with st.form("input_form"):
    youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    submitted = st.form_submit_button("Generate Article Webpage")

if submitted and youtube_url:
    with st.spinner("Processing... This may take a minute or two."):
        try:
            # Run the full pipeline
            result = smart_summarizer.invoke(youtube_url)

            # Extract the three parts (HTML, CSS, JS) from the delimited output
            html_part = result.split("--html--")[1].split("--html--")[0].strip()
            css_part = result.split("--css--")[1].split("--css--")[0].strip()
            js_part = result.split("--js--")[1].split("--js--")[0].strip()

            # Display the generated webpage (if HTML is present)
            if html_part:
                st.subheader("Generated Webpage Preview")
                # Use an iframe to safely render the HTML
                st.components.v1.html(html_part, height=600, scrolling=True)

                # Provide download buttons
                st.subheader("Download Files")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.download_button(
                        label="📄 Download HTML",
                        data=html_part,
                        file_name="index.html",
                        mime="text/html"
                    )
                with col2:
                    st.download_button(
                        label="🎨 Download CSS",
                        data=css_part,
                        file_name="style.css",
                        mime="text/css"
                    )
                with col3:
                    st.download_button(
                        label="📜 Download JS",
                        data=js_part,
                        file_name="script.js",
                        mime="application/javascript"
                    )

                # Create a ZIP archive of all three files
                with zipfile.ZipFile("website.zip", "w") as zipf:
                    zipf.writestr("index.html", html_part)
                    zipf.writestr("style.css", css_part)
                    zipf.writestr("script.js", js_part)
                with open("website.zip", "rb") as f:
                    zip_bytes = f.read()
                with col4:
                    st.download_button(
                        label="📦 Download ZIP",
                        data=zip_bytes,
                        file_name="website.zip",
                        mime="application/zip"
                    )
            else:
                st.error("No HTML code generated. Please check the model output.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.stop()
elif submitted and not youtube_url:
    st.warning("Please enter a YouTube URL.")

# Optional: Show instructions
with st.expander("ℹ️ How it works"):
    st.markdown("""
    1. The app extracts the transcript from the provided YouTube video.
    2. It uses an LLM to summarise the transcript into a professional article (with special handling for long transcripts).
    3. The summary is then fed to another LLM prompt that generates a complete HTML/CSS/JS webpage, formatted like a Medium or Dev.to article.
    4. The generated code is displayed and can be downloaded as individual files or a ZIP archive.
    """)
