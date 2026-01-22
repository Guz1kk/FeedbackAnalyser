from feedback_analyzer import FeedbackAnalyzer

def main()-> None:
    file_name:str = 'b.csv'
    analyzer = FeedbackAnalyzer()
    result = analyzer.analyse(file_name)

if __name__ == '__main__':
    main()
    