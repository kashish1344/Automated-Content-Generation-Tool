
# Automated Content Generation Tool

This project is a Flask-based web application that generates high-quality content for blogs, articles, or social media posts based on specific keywords. The application includes features like SEO optimization, tone customization, multi-language support, and content summarization.

## Features

- **Content Generation:** Generates content based on user-provided keywords using the Ollama Llama2 model.
- **SEO Optimization:** Enhances the content by highlighting the provided keywords.
- **Tone Customization:** Adjusts the tone of the generated content (e.g., formal or informal).
- **Translation:** Translates the content into the desired language using Google Translate.
- **Summarization:** Summarizes the content to provide a concise version.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/automated-content-generation-tool.git
    cd automated-content-generation-tool
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1. **Start the Flask application:**
    ```bash
    python app.py
    ```

2. **Open your browser and navigate to:**
    ```
    http://127.0.0.1:5000/
    ```

## Usage

1. **Enter Keywords:** Provide the keywords or topics for which you want to generate content.
2. **Select Tone:** Choose the desired tone (e.g., formal, informal).
3. **Select Target Language:** Choose the language for translation (default is English).
4. **Generate Content:** Click the button to generate the content.

The application will generate the content, optimize it for SEO, adjust the tone, translate it if necessary, and provide a summary. The generated content and its summary will be displayed in the browser.

## Code Overview

- **`app.py`:** The main Flask application file containing routes and logic for content generation, SEO optimization, tone customization, translation, and summarization.
- **`templates/index.html`:** The HTML template for the web interface.

### Core Functions

- **`generate_content(keywords)`:** Generates content based on the provided keywords.
- **`optimize_seo(content, keywords)`:** Enhances the content by highlighting the provided keywords.
- **`customize_tone(content, tone)`:** Adjusts the tone of the generated content.
- **`translate_content(content, target_language)`:** Translates the content into the desired language.
- **`summarize_content(content, max_chunk_length, max_summary_length)`:** Summarizes the content.

## Dependencies

- Flask
- langchain-community
- googletrans
- transformers
- markdown
