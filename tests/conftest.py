import pytest
from dotenv import load_dotenv
import os
from evalbench.runtime_setup.config import EvalConfig
from evalbench.runtime_setup.runtime import set_config

@pytest.fixture(scope='session', autouse=True)
def configure_environment():
    load_dotenv()
    cfg = EvalConfig(groq_api_key=os.getenv('GROQ_API_KEY'))
    set_config(cfg)
