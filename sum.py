import streamlit as st
import whisperx
import tempfile
from transformers import pipeline

# Initialize the summarization pipeline
summarizer = pipeline("summarization")
# Initialize the sentiment analysis pipeline with the specified model
sentiment_analyzer = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", return_all_scores=True)

# Function to read text from a local file
def read_text_from_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text

# Function to split the text into chunks
def split_text_into_chunks(text, max_chunk_size=512):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_chunk_size):
        chunk = ' '.join(words[i:i + max_chunk_size])
        chunks.append(chunk)
    return chunks

# Define the Streamlit app layout
st.title("Whisper Diarization")

# Add file uploader widget
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "mp4", "wav"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Save the uploaded audio file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Load the temporary audio file
    audio = whisperx.load_audio(tmp_file_path)
    model = whisperx.load_model("large-v2", device="cuda", compute_type="float16", language="en")

    result = model.transcribe(audio, batch_size=4)

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cuda")
    result = whisperx.align(result["segments"], model_a, metadata, audio, "cuda", return_char_alignments=False)
    diarize_model = whisperx.DiarizationPipeline(use_auth_token="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", device="cuda")
    diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Display diarization results in the app interface
    st.write("Diarization Results:")
    for segment in result['segments']:
        speaker = segment['speaker']
        text = segment['text']
        if speaker == "SPEAKER_01":
            st.markdown(f"<span style='color: Cyan;'>Speaker {speaker}:</span> {text}", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color: red;'>Speaker {speaker}:</span> {text}", unsafe_allow_html=True)

    # Save diarization results to a file
    file_path = "diarization_results.txt"
    with open(file_path, "w", encoding="utf-8") as file:
        for segment in result['segments']:
            speaker = segment['speaker']
            text = segment['text']
            encoded_text = text.encode('utf-8', 'ignore').decode('utf-8')
            file.write(f"Speaker {speaker}: {encoded_text}\n")

    # Add a button to summarize the transcript
    if st.button("Summarize Transcript"):
        # Read the text from the diarization results file
        text = read_text_from_file(file_path)

        # Split the text into chunks
        chunks = split_text_into_chunks(text)

        # Summarize each chunk and combine the summaries
        summaries = []
        for chunk in chunks:
            summary = summarizer(chunk, max_length=80, min_length=20, do_sample=False)
            summaries.append(summary[0]['summary_text'])

        # Combine the summaries
        combined_summary = ' '.join(summaries)

        # Display the combined summary
        st.write("Combined Summary:")
        st.write(combined_summary)

        # Store the combined summary in session state for sentiment analysis
        st.session_state['combined_summary'] = combined_summary

# Add a button for sentiment analysis
if 'combined_summary' in st.session_state and st.button("Analyze Sentiment"):
    # Perform sentiment analysis on the combined summary
    combined_summary = st.session_state['combined_summary']
    sentiment_results = sentiment_analyzer(combined_summary)

    # Display the sentiment analysis results
    st.write("Sentiment Analysis Results:")
    for result in sentiment_results:
        for sentiment in result:
            st.write(f"Label: {sentiment['label']}, Score: {sentiment['score']:.4f}")
