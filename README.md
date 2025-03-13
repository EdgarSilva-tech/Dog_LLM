<h1>Dog Breed Identification and Information System</h1>
<p>This project is a comprehensive Dog Breed Identification and Information System that combines machine learning for image classification with natural language processing to provide detailed information about dog breeds.</p>
<h2>Features</h2>
<ul>
<li>Image-based dog breed identification</li>
<li>Natural language queries about dog breeds</li>
<li>Interactive web interface using Streamlit</li>
<li>Retrieval-augmented generation (RAG) for accurate information retrieval</li>
<li>Evaluation and feedback system for model responses</li>
</ul>
<h2>Components</h2>
<ol>
<li><strong>Web Application</strong> (<code>app.py</code>):</li>
<li>Built with Streamlit</li>
<li>
<p>Allows users to upload dog images and ask questions about breeds</p>
</li>
<li>
<p><strong>Machine Learning Model</strong> (<code>model.py</code>):</p>
</li>
<li>Uses EfficientNet-B0 for dog breed classification</li>
<li>
<p>Pre-trained on ImageNet and fine-tuned for dog breed identification</p>
</li>
<li>
<p><strong>Language Model</strong> (<code>LLM.py</code>):</p>
</li>
<li>Utilizes OpenAI's GPT model for natural language processing</li>
<li>
<p>Implements a retrieval system for accurate information lookup</p>
</li>
<li>
<p><strong>Data Processing</strong> (<code>data_ingestion.py</code>, <code>data_loader.py</code>):</p>
</li>
<li>Scripts for downloading, preparing, and loading the dataset</li>
<li>
<p>Includes data augmentation and transformation pipelines</p>
</li>
<li>
<p><strong>Evaluation System</strong> (<code>eval.py</code>):</p>
</li>
<li>Implements feedback mechanisms to evaluate model responses</li>
<li>
<p>Uses TruLens for tracking and improving model performance</p>
</li>
<li>
<p><strong>Docker Support</strong> (<code>compose.yaml</code>, <code>Dockerfile</code>):</p>
</li>
<li>Containerization for easy deployment and scaling</li>
</ol>
<h2>Setup and Installation</h2>
<ol>
<li>Clone the repository</li>
<li>Install dependencies: <code>pip install -r requirements.txt</code></li>
<li>Set up environment variables:</li>
<li>Create a <code>.env</code> file and add your OpenAI API key: <code>OPENAI_API_KEY=your_api_key_here</code></li>
<li>Run the data ingestion script: <code>python data_ingestion.py</code></li>
<li>Start the Streamlit app: <code>streamlit run src/app.py</code></li>
</ol>
<h2>Usage</h2>
<ol>
<li>Open the Streamlit app in your browser</li>
<li>Upload an image of a dog</li>
<li>Ask questions about the identified breed</li>
<li>Choose between standard RAG or filtered RAG for responses</li>
</ol>
<h2>Docker Deployment</h2>
<p>To run the application using Docker:</p>
<ol>
<li>Build the Docker image: <code>docker build -t dog-breed-app .</code></li>
<li>Run the container: <code>docker run -p 8000:8000 -e OPENAI_API_KEY=your_api_key_here dog-breed-app</code></li>
</ol>
<h2>Contributing</h2>
<p>Contributions to improve the project are welcome. Please follow these steps:</p>
<ol>
<li>Fork the repository</li>
<li>Create a new branch</li>
<li>Make your changes and commit them</li>
<li>Push to your fork and submit a pull request</li>
</ol>
<h2>License</h2>
<p>This project is licensed under the MIT License - see the LICENSE file for details.</p>
