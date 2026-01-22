from openai import OpenAI
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).parent

load_dotenv()


class FeedbackAnalyzer:
    def __init__(self):
        self.client: OpenAI = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def _detect_file_type(self, df) -> str:
        headers = [col.lower().strip() for col in df.columns]
        
        if 'question' in headers and 'answer' in headers:
            return 'survey'
        
        elif 'review' in headers and 'rate' in headers:
            return 'reviews'
        
        else:
            raise ValueError("Zły format danych!")
        
    def _csv_to_json(self, file_path) -> tuple[list, str]:
        df = pd.read_csv(file_path, encoding='utf-8')
        
        df.columns = df.columns.str.strip()
        file_type = self._detect_file_type(df)
        data = df.to_dict('records')
        return data, file_type
    
    def _get_prompt(self, file_type:str, data:dict) -> str:
        prompts = {
            "survey": f"""Przeanalizuj poniższe odpowiedzi z ankiety.

                        Dane w formacie JSON:
                        {data}

                        Wykonaj następującą analizę:
                        1. Zidentyfikuj główne tematy i wzorce w odpowiedziach
                        2. Pogrupuj podobne odpowiedzi
                        3. Wyciągnij kluczowe wnioski dla każdego pytania
                        4. Przedstaw rekomendacje na podstawie odpowiedzi

                        Odpowiedź przedstaw w języku polskim w uporządkowanej formie.""",
            "reviews": f"""Przeanalizuj poniższe opinie użytkowników.

                        Dane w formacie JSON:
                        {data}

                        Wykonaj następującą analizę:
                        1. Analiza sentymentu (pozytywne/negatywne/neutralne)
                        2. Najczęściej występujące problemy
                        3. Najczęściej występujące pochwały
                        4. Statystyki ocen (rozkład ocen 1-5)
                        5. Kluczowe wnioski i rekomendacje

                        Odpowiedź przedstaw w języku polskim w uporządkowanej formie."""
        }
        return prompts[file_type]
    
    def analyse(self, file_name:str):
        print(f"Wczytywanie pliku: {file_name}")
        file_path:Path = BASE_DIR / file_name
        data, file_type = self._csv_to_json(file_path.resolve())
        prompt = self._get_prompt(file_type, data)
        print("Analiza...")
        response = self.client.responses.create(
            model="gpt-4.1",
            input=[
                {
                    "role":"system",
                    "content": (
                        "Jestes analitykiem biznesowym, przeanalizuj opinie"
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        print(response.output)