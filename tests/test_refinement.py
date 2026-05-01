import pytest
from unittest.mock import MagicMock, patch
from gemma_asr_service import refine_transcription_timed

def test_refine_transcription_mocked():
    # Mock words
    words = [
        {"word": "Ez", "start": 0.0, "end": 0.5},
        {"word": "egy", "start": 0.5, "end": 1.0},
        {"word": "teszt.", "start": 1.0, "end": 1.5}
    ]
    
    # Mock response from Ollama/OpenAI
    mock_response = MagicMock()
    # The service expects a JSON string with batch_summary and ID mapping
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"batch_summary": "Test summary", "0": "Ez egy sikeres teszt."}'))
    ]
    
    with patch('openai.OpenAI') as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        refined_text, refined_words = refine_transcription_timed(words)
        
        assert "sikeres" in refined_text
        assert len(refined_words) > 0
        # Check if timestamps were preserved/interpolated
        assert refined_words[0]["start"] == 0.0
        assert abs(refined_words[-1]["end"] - 1.5) < 0.1
