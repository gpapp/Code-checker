# Code Analyzer Service

This project provides a service for analyzing Python code (scripts or Jupyter Notebooks) using the Gemini large language model (LLM) from Google. It can analyze code from various sources, including direct code input, GitHub URLs, and Google Cloud Platform (GCP) Bucket URLs.

## Features

-   **Code Analysis:** Provides a detailed analysis of the code's structure, logic, and clarity.
-   **Performance Evaluation:** Assesses the code's potential performance bottlenecks, efficiency, and scalability.
-   **Coding Standards Evaluation:** Evaluates the code's adherence to Python coding standards (PEP 8) and best practices.
-   **Error Handling Evaluation:** Assesses the code's error handling capabilities.
-   **Security Evaluation:** Evaluates the code's security aspects.
-   **No-Go Flag:** Determines if the code is acceptable or if it has too many errors to be considered acceptable.
-   **Multiple Input Sources:**
    -   Direct code input.
    -   GitHub URL (supports `.py` and `.ipynb` files).
    -   GCP Bucket URL (supports `.py` and `.ipynb` files).
-   **Gemini LLM Integration:** Leverages the power of Google's Gemini model for code analysis.
-   **FastAPI Framework:** Built using the FastAPI framework for high performance and ease of use.
- **Comprehensive testing**: Unit and integration tests are included.

## Prerequisites

-   Python 3.8+
-   Google Cloud Project (for GCP Bucket support)
-   GitHub Account (for GitHub URL support)
-   API keys:
    -   Google Gemini API key
    -   GitHub Personal Access Token (PAT)

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd Code-checker
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**

    -   Create a `.env` file in the root directory of the project.
    -   Add the following variables to the `.env` file, replacing the placeholders with your actual keys:

    ```properties
    GOOGLE_API_KEY=YOUR_GOOGLE_GEMINI_API_KEY
    GITHUB_TOKEN=YOUR_GITHUB_PERSONAL_ACCESS_TOKEN
    ```

## Usage

1.  **Run the FastAPI application:**

    ```bash
    uvicorn server:app --reload
    ```

    This will start the server on `http://0.0.0.0:8000`.

2.  **Interact with the API:**

    You can interact with the API using tools like `curl`, `Postman`, or any HTTP client.

    **Endpoint:** `/analyze/`

    **Method:** `POST`

    **Request Body (JSON):**

    You can provide one of the following in the request body:

    -   **Direct Code Input:**

        ```json
        {
          "code": "print('Hello, world!')",
          "file_type": "python"
        }
        ```
    - **Direct Code Input for jupyter notebook:**

        ```json
        {
          "code": "print('Hello, world!')",
          "file_type": "jupyter notebook"
        }
        ```

    -   **GitHub URL:**

        ```json
        {
          "github_url": "https://github.com/username/repo/blob/main/file.py"
        }
        ```
        or
        ```json
        {
          "github_url": "https://github.com/username/repo/blob/main/file.ipynb"
        }
        ```

    -   **GCP Bucket URL:**

        ```json
        {
          "gcp_bucket_url": "gs://your-bucket-name/file.py"
        }
        ```
        or
        ```json
        {
          "gcp_bucket_url": "gs://your-bucket-name/file.ipynb"
        }
        ```

    **Response Body (JSON):**

    ```json
    {
      "analysis": "...",
      "performance_evaluation": "...",
      "coding_standards_evaluation": "...",
      "no_go": true/false,
      "raw_gemini_response": "..."
    }
    ```

    **Example using curl:**

    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"code": "print(\"Hello, world!\")", "file_type": "python"}' http://0.0.0.0:8000/analyze/
    ```
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"github_url": "https://github.com/gpapp/notebooks/blob/main/Clean%20up%20transcription.ipynb"}' http://0.0.0.0:8000/analyze/
    ```

## Testing

To run the tests:

1.  Ensure you have installed the test dependencies from `requirements.txt`.
2.  Run pytest from the project root:

    ```bash
    pytest
    ```

## Project Structure

