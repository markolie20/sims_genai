# Sim World: LLM-Powered 2D Simulation

This project is a 2D simulation where autonomous agents (Sims) navigate a grid-based world, make decisions using Google's Gemini LLM, and interact with their environment and each other. It features a Retrieval Augmented Generation (RAG) system to provide Sims with contextual world knowledge (themed around medieval times), enhancing their decision-making capabilities.

## Features

*   **LLM-Driven Sim Behavior:** Each Sim uses Google's Gemini model to decide its actions based on its state, environment, messages, and world knowledge.
*   **Dynamic World:** Sims have attributes like hunger and mood that influence their decisions.
*   **Object Interaction:** Sims can interact with objects like apples (to eat), trumpets (to improve mood), and farms (to work, improve mood, and earn money).
*   **Inter-Sim Communication:** Sims can send simple messages to each other.
*   **User Interaction:** Users can click on Sims to give them direct instructions.
*   **Retrieval Augmented Generation (RAG):**
    *   The simulation loads knowledge about medieval life from specified URLs.
    *   This text is chunked, embedded (using `HuggingFaceEmbeddings` with `all-MiniLM-L6-v2`), and stored in a FAISS vector store.
    *   Sims query this vector store to get relevant context for their decisions.
*   **Fun Fact Tool:** A LangChain tool is integrated to periodically fetch and inject random fun facts into the Sims' "awareness."
*   **GUI:** Built with Tkinter, providing a visual representation of the grid, Sims, and objects.
*   **Asynchronous LLM Calls:** Sim decision-making (LLM calls) happens in separate threads to prevent the GUI from freezing.
*   **Medieval Theme:** The Sim's system prompt and the RAG knowledge base are geared towards a medieval setting.

## Technologies Used

*   **Python 3.9+**
*   **Tkinter:** For the graphical user interface.
*   **LangChain:** Framework for developing applications powered by language models.
    *   `ChatGoogleGenerativeAI`: For interacting with Google's Gemini models.
    *   `GoogleGenerativeAIEmbeddings` / `HuggingFaceEmbeddings`: For creating text embeddings.
    *   `ChatPromptTemplate`, `SystemMessage`, `HumanMessage`: For structuring prompts to the LLM.
    *   `FAISS`: For efficient similarity search in the vector store.
    *   `RetrievalQA`: Chain for question-answering over a knowledge base (though the project uses a more direct retriever approach).
    *   `Tool`: For integrating external functionalities (like the Fun Fact API).
*   **Google Generative AI:** Access to Gemini models via API.
*   **Sentence-Transformers:** (`all-MiniLM-L6-v2`) Used by `HuggingFaceEmbeddings` for creating text embeddings locally for RAG.
*   **FAISS:** For creating a local vector store for RAG.
*   **Requests:** For making HTTP requests (e.g., to the Fun Fact API and for `rag.py`).
*   **BeautifulSoup4:** (Used by `rag.py`) For parsing HTML content from URLs.
*   **python-dotenv:** For managing environment variables (like API keys).
*   **Threading:** For running LLM calls in the background.

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    tkinter
    langchain
    langchain-google-genai
    langchain-community
    langchain-huggingface
    google-generativeai
    python-dotenv
    requests
    sentence-transformers
    faiss-cpu # or faiss-gpu if you have a compatible GPU and CUDA setup
    beautifulsoup4
    numpy
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `tkinter` is usually part of the standard Python installation, but it's good to list it.)*

4.  **Set up Environment Variables:**
    Create a file named `.env` in the root directory of the project and add your Google API Key:
    ```env
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
    ```
    You can obtain a Google API key from the [Google AI Studio](https://aistudio.google.com/app/apikey).

5.  **Project Files:**
    Ensure you have the following Python files in your project directory:
    *   `sim_world_main.py` (The main script you provided)
    *   `rag.py` (The script for processing URLs)

## Running the Application

Once the setup is complete, you can run the simulation:

```bash
python sim_world_main.py
