import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

PACKAGE_DIR: Final[Path] = Path(__file__).parents[2]
load_dotenv(PACKAGE_DIR / '.env')

OPENAI_API_KEY: Final[str] = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY: Final[str] = os.getenv('ANTHROPIC_API_KEY')
GOOGLE_API_KEY: Final[str] = os.getenv('GOOGLE_API_KEY')

VERSION = '0.1.0'
