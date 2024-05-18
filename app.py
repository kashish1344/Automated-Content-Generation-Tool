from flask import Flask, request, render_template
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from googletrans import Translator
from transformers import pipeline
import markdown

app = Flask(__name__)

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

def summarize_content(content, max_chunk_length=500, max_summary_length=100):
    # Split content into manageable chunks
    chunks = [content[i:i + max_chunk_length] for i in range(0, len(content), max_chunk_length)]
    
    summaries = []
    for chunk in chunks:
        try:
            summary = summarizer(chunk, max_length=max_summary_length, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            summaries.append(f"Error summarizing chunk: {str(e)}")
    
    # Join the summaries
    return ' '.join(summaries)

@app.route('/', methods=['GET', 'POST'])
def index():
    generated_content = ""
    content_summary = ""
    if request.method == 'POST':
        keywords = request.form['keywords']
        tone = request.form['tone']
        target_language = request.form['target_language']
        
        if keywords:
            content = generate_content(keywords)
            optimized_content = optimize_seo(content, keywords)
            customized_content = customize_tone(optimized_content, tone)
            if target_language != 'en':
                translated_content = translate_content(customized_content, target_language)
            else:
                translated_content = customized_content
            summary = summarize_content(translated_content)
            
            # Convert markdown content to HTML
            generated_content = markdown.markdown(translated_content)
            content_summary = markdown.markdown(summary)
        else:
            generated_content = "Please enter keywords or topics to generate content."

    return render_template('index.html', generated_content=generated_content, content_summary=content_summary)

if __name__ == '__main__':
    app.run(debug=True)
