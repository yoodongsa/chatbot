import pandas as pd                      # CSV 파일을 읽기 위해 pandas 모듈 사용
import Levenshtein                      # 레벤슈타인 거리 계산을 위한 외부 라이브러리 (pip install python-Levenshtein)

class SimpleChatBot:
    def __init__(self, filepath):
        # 생성자에서 데이터 파일을 로드하여 질문과 답변 리스트로 저장
        self.questions, self.answers = self.load_data(filepath)

    def load_data(self, filepath):
        """
        CSV 파일에서 질문(Q)과 답변(A) 데이터를 읽어와 각각 리스트로 반환하는 함수
        """
        data = pd.read_csv(filepath)         # CSV 파일을 읽어 DataFrame으로 저장
        questions = data['Q'].tolist()       # 'Q' 열을 리스트로 변환해 질문 목록 생성
        answers = data['A'].tolist()         # 'A' 열을 리스트로 변환해 답변 목록 생성
        return questions, answers            # 질문과 답변 리스트 반환

    def find_best_answer(self, input_sentence):
        """
        사용자 입력 문장과 레벤슈타인 거리가 가장 짧은 질문을 찾아
        해당 질문에 대응하는 답변을 반환하는 함수
        """
        min_distance = float('inf')          # 가장 작은 거리 값을 저장 (초기값은 무한대)
        best_match_index = -1                # 가장 유사한 질문의 인덱스 저장용

        # 모든 질문에 대해 레벤슈타인 거리 계산
        for idx, question in enumerate(self.questions):
            distance = Levenshtein.distance(input_sentence, question)  # 입력과 질문 간의 레벤슈타인 거리 계산
            if distance < min_distance:
                min_distance = distance        # 더 작은 거리를 발견하면 갱신
                best_match_index = idx         # 해당 인덱스를 기억

        return self.answers[best_match_index]  # 가장 유사한 질문에 대응하는 답변 반환

# 학습용 CSV 파일 경로
filepath = 'ChatbotData.csv'

# 챗봇 객체 생성
chatbot = SimpleChatBot(filepath)

# 사용자 입력을 반복적으로 받아서 응답하는 루프
while True:
    input_sentence = input('You: ')           # 사용자로부터 질문 입력 받기
    if input_sentence.lower() == '종료':       # '종료' 입력 시 대화 종료
        break
    response = chatbot.find_best_answer(input_sentence)  # 최적의 응답 찾기
    print('Chatbot:', response)               # 챗봇의 응답 출력
