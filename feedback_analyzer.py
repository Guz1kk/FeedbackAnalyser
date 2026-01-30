from openai import OpenAI
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os
import random
import json

BASE_DIR = Path(__file__).parent
load_dotenv()


class FeedbackAnalyzer:
    def __init__(self, model: str = "gpt-4.1"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Brak OPENAI_API_KEY w zmiennych środowiskowych.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _detect_file_type(self, df: pd.DataFrame) -> str:
        headers = [col.lower().strip() for col in df.columns]
        if "question" in headers and "answer" in headers:
            return "survey"
        if "review" in headers and "rate" in headers:
            return "reviews"
        raise ValueError("Zły format danych! Oczekuję: (question, answer) albo (review, rate).")

    def _read_csv(self, file_path: Path) -> pd.DataFrame:
        df = pd.read_csv(file_path, encoding="utf-8", sep=";")
        df.columns = df.columns.str.strip()
        return df

    def _safe_sample(self, items: list, k: int) -> list:
        items = [x for x in items if isinstance(x, str) and x.strip()]
        if len(items) <= k:
            return items
        return random.sample(items, k)

    def _prep_reviews_payload(self, df: pd.DataFrame, sample_size: int = 80) -> dict:
        df = df.copy()
        df["review"] = df["review"].astype(str).fillna("").str.strip()

        # rate bywa "5", "4.0", "5/5", itd. -> próbujemy wyciągnąć liczbę 1-5
        df["rate_raw"] = df["rate"]
        df["rate"] = (
            df["rate"].astype(str)
            .str.replace(",", ".", regex=False)
            .str.extract(r"(\d+(\.\d+)?)")[0]
        )
        df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
        df = df.dropna(subset=["review"])

        rating_dist = (
            df["rate"]
            .dropna()
            .round()
            .clip(1, 5)
            .astype(int)
            .value_counts()
            .reindex([1,2,3,4,5], fill_value=0)
            .to_dict()
        )

        payload = {
            "meta": {
                "n_rows": int(len(df)),
                "n_with_rating": int(df["rate"].notna().sum()),
                "avg_rating": float(df["rate"].dropna().mean()) if df["rate"].notna().any() else None,
            },
            "rating_distribution_1_5": rating_dist,
            "sample_reviews": self._safe_sample(df["review"].tolist(), sample_size),
        }
        return payload

    def _prep_survey_payload(self, df: pd.DataFrame, per_question_sample: int = 50) -> dict:
        df = df.copy()
        df["question"] = df["question"].astype(str).fillna("").str.strip()
        df["answer"] = df["answer"].astype(str).fillna("").str.strip()
        df = df[(df["question"] != "") & (df["answer"] != "")]

        questions = []
        for q, grp in df.groupby("question", dropna=False):
            answers = grp["answer"].tolist()
            questions.append({
                "question": q,
                "n_answers": int(len(answers)),
                "sample_answers": self._safe_sample(answers, per_question_sample)
            })

        payload = {
            "meta": {"n_rows": int(len(df)), "n_questions": int(df["question"].nunique())},
            "questions": questions
        }
        return payload

    def _prompt_reviews(self, payload: dict) -> str:
        return f"""
Przeanalizuj opinie użytkowników na podstawie PODSUMOWANIA oraz PRÓBKI opinii (nie zakładaj, że próbka zawiera wszystkie możliwe problemy).

Dane (JSON):
{json.dumps(payload, ensure_ascii=False, indent=2)}

Wykonaj analizę i zwróć wynik po polsku w MARKDOWN z nagłówkami:
## Sentyment
- podaj szacowany udział pozytywne/neutralne/negatywne na podstawie próbki + uzasadnij

## Najczęstsze problemy
- lista 5–10 punktów, każdy: opis + przykładowe cytaty (krótkie) z próbki

## Najczęstsze pochwały
- lista 5–10 punktów, każdy: opis + przykładowe cytaty (krótkie) z próbki

## Statystyki ocen
- wykorzystaj rating_distribution_1_5 oraz avg_rating jeśli dostępne
- krótki komentarz do rozkładu

## Rekomendacje
- 5–10 konkretnych działań (priorytet: wysoki/średni/niski) + spodziewany efekt

Zasady:
- Nie wymyślaj faktów, cytaty tylko z sample_reviews.
- Jeśli brakuje ocen, zaznacz to wyraźnie.
""".strip()

    def _prompt_survey(self, payload: dict) -> str:
        return f"""
Przeanalizuj odpowiedzi ankietowe. Masz listę pytań i próbki odpowiedzi (próbka może nie zawierać wszystkich wątków).

Dane (JSON):
{json.dumps(payload, ensure_ascii=False, indent=2)}

Zwróć wynik po polsku w MARKDOWN:

## Podsumowanie ogólne
- 3–7 najważniejszych obserwacji

## Analiza per pytanie
Dla każdego pytania:
### [treść pytania]
- Najczęstsze wzorce (tematy)
- Grupowanie odpowiedzi (3–7 klastrów: nazwa + opis)
- Kluczowe wnioski
- Rekomendacje

## Rekomendacje przekrojowe
- 5–10 działań (priorytet + dlaczego)

Zasady:
- Nie wymyślaj danych, opieraj się na sample_answers.
""".strip()

    def analyse(self, file_name: str, sample_size: int = 80, per_question_sample: int = 50) -> str:
        file_path = (BASE_DIR / file_name).resolve()
        df = self._read_csv(file_path)
        file_type = self._detect_file_type(df)

        if file_type == "reviews":
            payload = self._prep_reviews_payload(df, sample_size=sample_size)
            prompt = self._prompt_reviews(payload)
        else:
            payload = self._prep_survey_payload(df, per_question_sample=per_question_sample)
            prompt = self._prompt_survey(payload)

        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": "Jesteś analitykiem biznesowym. Pisz konkretnie i strukturalnie."},
                {"role": "user", "content": prompt},
            ],
        )

        return response.output[0].content[0].text
