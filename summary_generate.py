from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader ,WebBaseLoader
from typing import List, Union
from gtts import gTTS
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


def load_text_file(file_path: str) -> List[Document]:
    loader = TextLoader(file_path,encoding='utf-8')
    print('loading data from text file')
    return loader.load()

def load_pdf_file(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    print('loading data from pdf file')
    return loader.load()

def load_word_file(file_path: str) -> List[Document]:
    loader = Docx2txtLoader(file_path)
    print('loading data from word file')
    return loader.load()

def load_url(file_path: str) -> List[Document]:
    loader = WebBaseLoader(file_path)
    print('loading data from url')
    return loader.load()

def load_document(file_path: str) -> List[Document]:
    """
    Load a document based on its file extension
    
    Args:
        file_path (str): Path to the document
        
    Returns:
        List[Document]: List of LangChain Document objects
    """
    if file_path.endswith('.txt'):
        return load_text_file(file_path)
    elif file_path.endswith('.pdf'):
        return load_pdf_file(file_path)
    elif file_path.endswith('.docx'):
        return load_word_file(file_path)
    elif file_path.startswith('https:') or file_path.startswith('http:') or file_path.startswith('www.'):
        return load_url(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
def text_to_speech(text, output_file, language='en'):
    """Convert text to speech and save as MP3"""
    print('Converting summary to audio and saving audio file')
    tts = gTTS(text=text, lang=language)
    tts.save(output_file)


def generate_summary(text):
    response=llm.invoke(f"""here is {text}. 
                        Can you generate summary from this data and focus on every small detail. 
                        And generate summary properly in formal way so it is easily understandable. 
                        I have to generate audio from this summary. So do not * or any other thing in the heading.
                        And also do not place any thing in the backquote (``) instead put it in the normal quote.""")
    print('generating summary')
    with open('summary.txt','w') as f:
        f.write(response.content)

    return response.content
    
def main():
    url=input("Enter the URL or file path: ")
    documents = load_document(url)
    text_with_next_lines=(documents[0].page_content)
    text=(','.join(text_with_next_lines.split('\n\n')))
    summary=generate_summary(text)
    output_path='summary_audio.wav'
    text_to_speech(summary,output_path)

if __name__ == "__main__":
    main()