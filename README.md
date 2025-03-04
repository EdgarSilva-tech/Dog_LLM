Dog LLM is an AI-powered application that combines image recognition and natural language processing to answer questions about dog breeds. The app uses a pre-trained EfficientNet model for breed classification and leverages OpenAI's GPT model for answering questions. Features - Image upload for dog breed recognition - Question answering about the detected dog breed - RAG (Retrieval Augmented Generation) for enhanced responses - Filtered RAG for more accurate and relevant answers - Evaluation metrics for answer quality.

Setup - Clone the repository as so git clone https://github.com/EdgarSilva-tech/Dog_LLM.git, Install the required dependencies: pip install -r requirements.txt. Set up your OpenAI API key in the `.env` file: OPENAI_API_KEY=your_api_key_here.

Running the Application: To run the Streamlit app: streamlit run src/app.py. 
Project Structure: 
- `app.py`: Main Streamlit application 
- `data_ingestion.py`: Scripts for downloading and preprocessing the dataset 
- `data_loader.py`: Data loading utilities for the machine learning model 
- `eval.py`: Evaluation metrics and feedback functions 
- `LLM.py`: Language model setup and chain configuration 
- `model.py`: Image classification model definition and utilities

Docker Support - The project includes Docker support for easy deployment. Use the following commands: docker build -t dog-llm followed by docker run -p 8000:8000 dog-llm to create and run the Docker container.
