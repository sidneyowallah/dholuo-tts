import sys
from unittest.mock import MagicMock, patch
import pytest

# Mock TTS and other heavy dependencies BEFORE importing api.main
# This prevents ModuleNotFoundError if coqui-tts is not installed in test env
sys.modules["TTS"] = MagicMock()
sys.modules["TTS.tts"] = MagicMock()
sys.modules["TTS.tts.configs"] = MagicMock()
sys.modules["TTS.tts.configs.shared_configs"] = MagicMock()
sys.modules["TTS.tts.configs.vits_config"] = MagicMock()
sys.modules["TTS.tts.models"] = MagicMock()
sys.modules["TTS.tts.models.vits"] = MagicMock()
sys.modules["TTS.tts.utils"] = MagicMock()
sys.modules["TTS.tts.utils.text"] = MagicMock()
sys.modules["TTS.tts.utils.text.tokenizer"] = MagicMock()
sys.modules["TTS.utils"] = MagicMock()
sys.modules["TTS.utils.audio"] = MagicMock()
sys.modules["tagger"] = MagicMock()  # Mock tagger as well if needed

# Mock other external dependencies
sys.modules["redis"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.io"] = MagicMock()
sys.modules["scipy.io.wavfile"] = MagicMock()

# Now import app
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

@pytest.fixture
def mock_inference():
    with patch("api.routes.inference_engine") as mock:
        # Mock synthesize
        mock.synthesize.return_value = {
            "audio": "UklGRi4AAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=", # Empty wav base64
            "duration": 1.0,
            "sample_rate": 22050
        }
        # Mock tagging
        mock.get_tagging.return_value = [("word", "tag")]
        # Mock phonemize
        mock.text_to_ipa.return_value = "w 3 r d"
        mock._initialized = True
        yield mock

@pytest.fixture
def mock_cache():
    with patch("api.routes.cache") as mock:
        mock.get.return_value = None
        mock.is_connected.return_value = False
        yield mock

@pytest.fixture
def mock_cache():
    with patch("api.routes.cache") as mock:
        mock.get.return_value = None
        mock.is_connected.return_value = False
        yield mock

def test_health(mock_inference, mock_cache):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_synthesize(mock_inference, mock_cache):
    payload = {
        "text": "Nyithindo",
        "voice": "female"
    }
    response = client.post("/api/v1/synthesize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "audio" in data
    assert "duration" in data

def test_tag(mock_inference, mock_cache):
    payload = {"text": "Nyithindo"}
    response = client.post("/api/v1/tag", json=payload)
    assert response.status_code == 200
    assert "tagged_pairs" in response.json()

def test_phonemize(mock_inference, mock_cache):
    payload = {"text": "Nyithindo"}
    response = client.post("/api/v1/phonemize", json=payload)
    assert response.status_code == 200
    assert "ipa_text" in response.json()

def test_batch(mock_inference, mock_cache):
    # Mock BackgroundTasks is handled by TestClient usually forsync execution?
    # Actually TestClient runs background tasks synchronously.
    
    # We need to mock cache set/get for job management to work in the route logic
    mock_cache.get.return_value = None # Initial check might be None?
    
    # But batch route sets cache first.
    # We can mock the side effects or just let it pass.
    # The route logic calls `cache.set`.
    
    payload = {"texts": ["one", "two"]}
    response = client.post("/api/v1/batch", json=payload)
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert job_id is not None
