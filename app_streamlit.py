import streamlit as st
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from googletrans import Translator
from transformers import pipeline

# Initialize the Ollama Llama2 model
llama_model = Ollama(model="llama3")
prompt_template = PromptTemplate(input_variables=["keywords"], template="Generate a blog post about {keywords}")
llm_chain = LLMChain(llm=llama_model, prompt=prompt_template)

# Initialize other components
translator = Translator()
summarizer = pipeline("summarization")

# Functions for content generation and processing
def generate_content(keywords):
    response = llm_chain.invoke({"keywords": keywords})
    return response["text"]

def optimize_seo(content, keywords):
    for keyword in keywords.split(','):
        content = content.replace(keyword.strip(), f"**{keyword.strip()}**")
    return content

def customize_tone(content, tone):
    if tone == 'formal':
        return content.replace("!", ".").replace("?", ".")
    return content

def translate_content(content, target_language):
    translation = translator.translate(content, dest=target_language)
    return translation.text

def summarize_content(content):
    summary = summarizer(content, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Streamlit UI
st.title("Automated Content Generation Tool")
st.write("Generate high-quality content for blogs, articles, or social media posts based on specific keywords or topics.")

keywords = st.text_input("Enter keywords or topics:")
tone = st.selectbox("Select tone:", ["neutral", "formal", "casual"])
target_language = st.text_input("Enter target language (e.g., 'es' for Spanish):", "en")

if st.button("Generate Content"):
    if keywords:
        with st.spinner("Generating content..."):
            content = generate_content(keywords)
            optimized_content = optimize_seo(content, keywords)
            customized_content = customize_tone(optimized_content, tone)
            if target_language != 'en':
                translated_content = translate_content(customized_content, target_language)
            else:
                translated_content = customized_content
            summary = summarize_content(translated_content)
            
            st.subheader("Generated Content")
            st.write(translated_content)
            
            st.subheader("Content Summary")
            st.write(summary)
    else:
        st.error("Please enter keywords or topics to generate content.")

# REQUIREMENTS:  pip install streamlit langchain transformers googletrans==4.0.0-rc1 langchain-community
