from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TfidfVectorizer 객체 생성
vectorizer = TfidfVectorizer()

# 한국어 문장들
sentence1 = "나는 학교에 갔다"
sentence2 = "나는 영화관에 갔다"
# sentence2 = "너는 그제 밥을 지었습니다."

# 문장들을 벡터화
tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])
print(tfidf_matrix)

# 문장1과 문장2의 코사인 유사도 계산
cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

print(f"문장 1: {sentence1}")
print(f"문장 2: {sentence2}")
print(f"두 문장의 코사인 유사도: {cosine_sim[0][0]}")