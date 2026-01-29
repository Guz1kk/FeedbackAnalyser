from matplotlib import text
from feedback_analyzer import FeedbackAnalyzer

def main()-> None:
    file_name:str = 'survey.csv'
    analyzer = FeedbackAnalyzer()
    result = analyzer.analyse(file_name)
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write(result)

if __name__ == '__main__':
    main()  