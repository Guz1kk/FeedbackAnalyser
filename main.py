from feedback_analyzer import FeedbackAnalyzer


def main() -> None:

    file_name = "question_answer.csv" #"review_test.csv"

    analyzer = FeedbackAnalyzer(model="gpt-4.1")

    try:
        result = analyzer.analyse(
            file_name=file_name,
            sample_size=80,          # używane tylko dla reviews
            per_question_sample=50   # używane tylko dla survey
        )

        with open("results_q&a.txt", "w", encoding="utf-8") as f:
            f.write(result)

        print("Analiza zakończona. Wynik zapisany w results.txt")

    except Exception as e:
        print("Błąd podczas analizy:")
        print(e)


if __name__ == "__main__":
    main()
