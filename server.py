from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import os
import re
import urllib
from typing import Dict, Any, Optional
import google.generativeai as genai
from google.cloud import storage  # For GCP Bucket Support
from dotenv import load_dotenv
from github import Github, Auth

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API key (now loaded from .env)

# Configure the Gemini API key (replace with your actual key)
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Initialize the Gemini model
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    safety_settings=safety_settings,
)


app = FastAPI(
    title="Code Analyzer Service",
    description="Analyzes Python code (scripts or Jupyter Notebooks) using Gemini LLM.",
    version="0.1.0",
)


class CodeAnalysisRequest(BaseModel):
    """
    Request model for code analysis.
    """
    code: Optional[str] = None  # For direct code input (optional)
    file_type: Optional[str] = None # Required when using code
    github_url: Optional[HttpUrl] = None  # For GitHub URL input
    gcp_bucket_url: Optional[str] = None  # For GCP Bucket URL input (gs://bucket-name/file-name)


class CodeAnalysisResponse(BaseModel):
    """
    Response model for code analysis.
    """

    analysis: str
    performance_evaluation: str
    coding_standards_evaluation: str
    no_go: bool
    raw_gemini_response: str


def analyze_code_with_gemini(code: str, file_type: str) -> Dict[str, Any]:
    """
    Analyzes the code using Gemini LLM.

    Args:
        code: The code to analyze.
        file_type: The type of the file (e.g., "python", "jupyter").

    Returns:
        A dictionary containing the analysis, performance evaluation, coding standards evaluation, no-go flag, and raw Gemini response.
    """
    prompt = f"""
    You are a senior software engineer reviewing code.
    Analyze the following {file_type} code:

    ```
    {code}
    ```

    Provide a detailed analysis of the code, including:
    1.  **Overall Analysis**: A general evaluation of the code's structure, logic, and clarity.
    2.  **Performance Evaluation**: Assess the code's potential performance bottlenecks, efficiency, and scalability.
    3.  **Coding Standards Evaluation**: Evaluate the code's adherence to Python coding standards (PEP 8) and best practices.
    4. **Error handling**: Evaluate the code's error handling capabilities.
    5. **Security**: Evaluate the code's security aspects.

    Based on your analysis, determine if the code is acceptable or if it has too many errors to be considered acceptable. 
    Only consider errors that are potentially insecure or highly inefficient to set the "no_go" flag to true. Otherwise, set it to false.

    Return the result in the following JSON format:
    {{
        "analysis": "...",
        "performance_evaluation": "...",
        "coding_standards_evaluation": "...",
        "no_go": true/false
    }}
    """

    try:
        response = model.generate_content(prompt)
        raw_response = response.text
        # Extract JSON from Gemini's response
        json_match = re.search(r"\{[\s\S]*\}", raw_response)
        if json_match:
            json_str = json_match.group(0)
            try:
                import json
                analysis_data = json.loads(json_str)
                return {
                    "analysis": analysis_data.get("analysis", "No analysis provided."),
                    "performance_evaluation": analysis_data.get(
                        "performance_evaluation", "No performance evaluation provided."
                    ),
                    "coding_standards_evaluation": analysis_data.get(
                        "coding_standards_evaluation",
                        "No coding standards evaluation provided.",
                    ),
                    "no_go": analysis_data.get("no_go", False),
                    "raw_gemini_response": raw_response,
                }
            except json.JSONDecodeError:
                return {
                    "analysis": "Error decoding JSON from Gemini response.",
                    "performance_evaluation": "Error decoding JSON from Gemini response.",
                    "coding_standards_evaluation": "Error decoding JSON from Gemini response.",
                    "no_go": True,
                    "raw_gemini_response": raw_response,
                }
        else:
            return {
                "analysis": "No JSON found in Gemini response.",
                "performance_evaluation": "No JSON found in Gemini response.",
                "coding_standards_evaluation": "No JSON found in Gemini response.",
                "no_go": True,
                "raw_gemini_response": raw_response,
            }
    except Exception as e:
        return {
            "analysis": f"Error during Gemini API call: {e}",
            "performance_evaluation": f"Error during Gemini API call: {e}",
            "coding_standards_evaluation": f"Error during Gemini API call: {e}",
            "no_go": True,
            "raw_gemini_response": f"Error during Gemini API call: {e}",
        }


def download_file_from_github(url: HttpUrl) -> str:
    """Downloads a file from a GitHub URL using PyGithub."""
    try:
        # Parse the GitHub URL to extract repository information
        match = re.match(r"https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)", str(url))
        if not match:
            raise ValueError("Invalid GitHub URL format.")

        owner, repo, branch, path = match.groups()

        # Authenticate with GitHub (optional, but recommended for rate limiting)
        # Replace with your GitHub personal access token if needed
        
        auth = Auth.Token(os.environ.get("GITHUB_TOKEN", ""))
        g = Github(auth=auth) 
        g.get_user().login

        repo = g.get_repo(f"{owner}/{repo}")
        contents = repo.get_contents(urllib.parse.unquote(path))
        return contents.decoded_content.decode("utf-8")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid GitHub URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error downloading from GitHub: {e}")

def download_file_from_gcp_bucket(gcp_bucket_url: str) -> str:
    """Downloads a file from a GCP bucket URL and returns the file's content."""
    try:
        # Extract bucket name and file name from URL
        if not gcp_bucket_url.startswith("gs://"):
            raise ValueError("Invalid GCP Bucket URL format. It should start with 'gs://'")

        bucket_name, file_name = gcp_bucket_url[5:].split("/", 1)

        # Initialize the GCP Storage Client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        # Download the file
        file_content = blob.download_as_text()
        return file_content

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error downloading file from GCP Bucket: {e}"
        )


@app.post("/analyze/", response_model=CodeAnalysisResponse)
async def analyze_code( 
    request_data: Optional[CodeAnalysisRequest] = None
):
    """
    Analyzes code from various sources: uploaded file, GitHub URL, or GCP Bucket URL.

    Args:
        file: The uploaded file (optional).
        request_data: JSON request containing code, github_url, or gcp_bucket_url.

    Returns:
        CodeAnalysisResponse: The analysis results.
    """
    if not request_data:
        raise HTTPException(status_code=400, detail="No input provided")
    
    if request_data and not (request_data.code or request_data.github_url or request_data.gcp_bucket_url):
        raise HTTPException(status_code=400, detail="Please provide code, github_url or gcp_bucket_url in request data.")


    code = None
    file_type = None
    if request_data.code:
        if not request_data.file_type:
            raise HTTPException(status_code=400, detail="file_type is required when using code.")
        code = request_data.code
        file_type = request_data.file_type
    elif request_data.github_url:
        code = download_file_from_github(request_data.github_url)
        file_type = "python" if request_data.github_url.path.endswith(".py") else "jupyter notebook" if request_data.github_url.path.endswith(".ipynb") else None
    elif request_data.gcp_bucket_url:
        code = download_file_from_gcp_bucket(request_data.gcp_bucket_url)
        file_type = "python" if request_data.gcp_bucket_url.endswith(".py") else "jupyter notebook" if request_data.gcp_bucket_url.endswith(".ipynb") else None
    else:
        raise HTTPException(status_code=400, detail="Invalid request. Need code, github_url or gcp_bucket_url.")
    if not file_type:
        raise HTTPException(status_code=400, detail="Could not determine file type from provided url.")


    if not code:
      raise HTTPException(status_code=400, detail="No code found to analyze.")
    
    gemini_analysis = analyze_code_with_gemini(code, file_type)

    return CodeAnalysisResponse(
        analysis=gemini_analysis["analysis"],
        performance_evaluation=gemini_analysis["performance_evaluation"],
        coding_standards_evaluation=gemini_analysis["coding_standards_evaluation"],
        no_go=gemini_analysis["no_go"],
        raw_gemini_response=gemini_analysis["raw_gemini_response"],
    )

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
    