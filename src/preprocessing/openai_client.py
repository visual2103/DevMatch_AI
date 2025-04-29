import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
import re
import logging
from typing import Tuple, Dict
import time
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    logger.error("OPENAI_API_KEY was not found in the environment variables.")
    logger.error("Please create a .env file in the root directory of the project and add the API key in the format: OPENAI_API_KEY=your_key_here")
    raise ValueError(
        "OPENAI_API_KEY was not found. "
        "Please create a .env file in the root directory of the project"
        "and add the API key in the format: OPENAI_API_KEY=your_key_here"
    )

client = OpenAI(api_key=api_key)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def analyze_text(text: str, prompt: str) -> Tuple[Dict[str, int], Dict[str, str]]:
    try:
        full_prompt = prompt.format(text=text)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=900,
            temperature=0
        )
        content = response.choices[0].message.content

        industry_scores = {}
        explanations = {}

        for line in content.splitlines():
            line = line.replace("–", "-").replace("—", "-").strip()
            line = re.sub(r"\s+", " ", line)
            match_score = re.match(r"(.+?)\s*-\s*([0-9]{1,3})\s*%", line)
            match_expl = re.match(r"(.+?):\s*(.+)", line)

            if match_score:
                industry = match_score.group(1).strip()
                score = int(match_score.group(2))
                if not 0 <= score <= 100:
                    raise ValueError(f"Invalid score {score} for industry {industry}")
                industry_scores[industry] = score
            elif match_expl:
                industry = match_expl.group(1).strip()
                explanation = match_expl.group(2).strip()
                explanations[industry] = explanation

        if not industry_scores:
            raise ValueError("No industry scores found in API response")

        return industry_scores, explanations

    except openai.RateLimitError as e:
        logger.error(f"Rate limit exceeded: {str(e)}")
        raise
    except openai.APIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise