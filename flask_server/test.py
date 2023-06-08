import torch
import torch.nn as nn
import random
import pandas as pd

from model.classifier import KoBERTforSequenceClassfication
from kobert_transformers import get_tokenizer
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer



# -------------------------------------------------- #
#                                                    #
# [ 학습된 모델을 불러와 감정키워드, 수치값 도출을 위한 파일  #
#                                                    #
# -------------------------------------------------- #



# def load_wellness_answer():
#     root_path = "."
#     category_path = f"{root_path}/data/sentiment_emotion.txt"
#     answer_path = f"{root_path}/data/sentiment_system.txt"
#
#     c_f = open(category_path, 'r', encoding="utf-8")
#     a_f = open(answer_path, 'r', encoding="utf-8")
#
#     category_lines = c_f.readlines()
#     answer_lines = a_f.readlines()
#
#     category = {}
#     answer = {}
#     for line_num, line_data in enumerate(category_lines):
#         data = line_data.split('\t')
#         category[data[1][:-1]] = data[0]
#
#     for line_num, line_data in enumerate(answer_lines):
#         data = line_data.split('\t')
#         keys = answer.keys()
#         if (data[0] in keys):
#             answer[data[0]] += [data[1][:-1]]
#         else:
#             answer[data[0]] = [data[1][:-1]]
#
#     return category, answer

def kobert_input(tokenizer, str, device=None, max_seq_len=512):
    index_of_words = tokenizer.encode(str)
    token_type_ids = [0] * len(index_of_words)
    attention_mask = [1] * len(index_of_words)

    # Padding Length
    padding_length = max_seq_len - len(index_of_words)

    # Zero Padding
    index_of_words += [0] * padding_length
    token_type_ids += [0] * padding_length
    attention_mask += [0] * padding_length

    data = {
        'input_ids': torch.tensor([index_of_words]).to(device),
        'token_type_ids': torch.tensor([token_type_ids]).to(device),
        'attention_mask': torch.tensor([attention_mask]).to(device),
    }
    return data

def wordEmbedding(eb_data, question):
    # 두 개의 벡터로부터 코사인 유사도를 구하는 함수 cos_sim를 정의합니다.
    def cos_sim(A, B):
        return dot(A, B) / (norm(A) * norm(B))

    # 모든 질문 샘플들의 문장 임베딩 값들을 전부 비교하여 코사인 유사도 값이 가장 높은 질문 샘플을 찾아냅니다.
    # 그리고 해당 질문 샘플과 짝이 되는 답변 샘플을 리턴합니다.

    embedding = model2.encode(question)
    eb_data['score'] = eb_data.apply(lambda x: cos_sim(x['embedding'], embedding), axis=1)
    print("--------- 코사인 끝 --------")


    return eb_data.loc[eb_data['score'].idxmax()]['system']


if __name__ == "__main__":
    root_path = "."
    checkpoint_path = f"{root_path}/checkpoint"
    save_ckpt_path = f"{checkpoint_path}/kobert-sentiment-text-classification.pth"

    # 답변과 카테고리 불러오기
    # category, answer = load_wellness_answer()

    ctx = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(ctx)
    print('사용할 디바이스:', device)
    print('cuda 버전확인', torch.version.cuda)
    print("파이토치 버전 : ",torch.__version__)
    print(torch.cuda.get_device_name(0))
    print("쿠다 가능 :{}".format(torch.cuda.is_available()))

    # 저장한 Checkpoint 불러오기
    checkpoint = torch.load(save_ckpt_path, map_location=device)

    # 감정분석
    model = KoBERTforSequenceClassfication()
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    model.to(ctx)
    model.eval()

    # 워드 임베딩
    model2 = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

    tokenizer = get_tokenizer()

    eb_data = pd.read_csv('data/sentiment_human_test2.csv', encoding='utf-8')

    print("---------워드 임베딩 시작하는 부분-------")


    # 워드 임베딩 시작(human, system)
    # 데이터에서 모든 질문열. 즉, train_data['Q']에 대해서 문장 임베딩 값을 구한 후 embedding이라는 새로운 열에 저장합니다.
    eb_data['embedding'] = eb_data.apply(lambda row: model2.encode(str(row.human)), axis=1)
    print("--------- 임베딩 끝 --------")

    while 1:
        sent = input('\nQuestion: ')  # '요즘 기분이 우울한 느낌이에요'
        data = kobert_input(tokenizer, sent, device, 512)

        if '종료' in sent:
            break
        output = model(**data)

        logit = output[0]
        softmax_logit = torch.softmax(logit, dim=-1)
        softmax_logit = softmax_logit.squeeze()

        # 감정분석 결과 Index 반환 ( ek 반환 )
        max_index = torch.argmax(softmax_logit).item()
        print("--------- 감정값 획득  --------")
        max_index_value = softmax_logit[torch.argmax(softmax_logit)].item()



        # eb_data = pd.read_csv('./data/ek0.csv', encoding='utf-8')

        # # 워드 임베딩 시작(human, system)
        # # 데이터에서 모든 질문열. 즉, train_data['Q']에 대해서 문장 임베딩 값을 구한 후 embedding이라는 새로운 열에 저장합니다.
        # eb_data['embedding'] = eb_data.apply(lambda row: model2.encode(str(row.human)), axis = 1)
        #
        # # 두 개의 벡터로부터 코사인 유사도를 구하는 함수 cos_sim를 정의합니다.
        # def cos_sim(A, B):
        #     return dot(A, B) / (norm(A) * norm(B))
        #
        # # 모든 질문 샘플들의 문장 임베딩 값들을 전부 비교하여 코사인 유사도 값이 가장 높은 질문 샘플을 찾아냅니다.
        # # 그리고 해당 질문 샘플과 짝이 되는 답변 샘플을 리턴합니다.
        #
        # def return_answer(question):
        #     embedding = model2.encode(question)
        #     eb_data['score'] = eb_data.apply(lambda x: cos_sim(x['embedding'], embedding), axis=1)
        #     return eb_data.loc[eb_data['score'].idxmax()]['system']




        # 감정키워드 변수
        emotion = ""

        # system data 변수
        result_system = ""


        # 감정키값(ek) -> 감정 키워드
        if max_index >= 0 and max_index < 10:
            if max_index == 0:
                emotion = "기쁨/감사하는"
                # eb_data = pd.read_csv('data/ek0_9/ek0.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 1:
                emotion = "기쁨/기쁨"
                # eb_data = pd.read_csv('data/ek0_9/ek1.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 2:
                emotion = "기쁨/느긋"
                # eb_data = pd.read_csv('data/ek0_9/ek2.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 3:
                emotion = "기쁨/만족스러운"
                # eb_data = pd.read_csv('data/ek0_9/ek3.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 4:
                emotion = "기쁨/신뢰하는"
                # eb_data = pd.read_csv('data/ek0_9/ek4.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 5:
                emotion = "기쁨/신이 난"
                # eb_data = pd.read_csv('data/ek0_9/ek5.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 6:
                emotion = "기쁨/안도"
                # eb_data = pd.read_csv('data/ek0_9/ek6.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 7:
                emotion = "기쁨/자신하는"
                # eb_data = pd.read_csv('data/ek0_9/ek7.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 8:
                emotion = "기쁨/편안한"
                # eb_data = pd.read_csv('data/ek0_9/ek8.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 9:
                emotion = "기쁨/흥분"
                # eb_data = pd.read_csv('data/ek0_9/ek9.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
        elif max_index >= 10 and max_index < 20:
            if max_index == 10:
                emotion = "당황/고립된"
                # eb_data = pd.read_csv('data/ek10_19/ek10.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 11:
                emotion = "당황/남의 시선을 의식하는"
                # eb_data = pd.read_csv('data/ek10_19/ek11.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 12:
                emotion = "당황/당황"
                # eb_data = pd.read_csv('data/ek10_19/ek12.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 13:
                emotion = "당황/부끄러운"
                # eb_data = pd.read_csv('data/ek10_19/ek13.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 14:
                emotion = "당황/열등감"
                # eb_data = pd.read_csv('data/ek10_19/ek14.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 15:
                emotion = "당황/외로운"
                # eb_data = pd.read_csv('data/ek10_19/ek15.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 16:
                emotion = "당황/죄책감의"
                # eb_data = pd.read_csv('data/ek10_19/ek16.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 17:
                emotion = "당황/한심한"
                # eb_data = pd.read_csv('data/ek10_19/ek17.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 18:
                emotion = "당황/혐오스러운"
                # eb_data = pd.read_csv('data/ek10_19/ek18.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 19:
                emotion = "당황/혼란스러운"
                # eb_data = pd.read_csv('data/ek10_19/ek19.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
        elif max_index >= 20 and max_index < 30:
            if max_index == 20:
                emotion = "분노/구역질 나는"
                # eb_data = pd.read_csv('data/ek20_29/ek20.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 21:
                emotion = "분노/노여워하는"
                # eb_data = pd.read_csv('data/ek20_29/ek21.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 22:
                emotion = "분노/방어적인"
                # eb_data = pd.read_csv('data/ek20_29/ek22.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 23:
                emotion = "분노/분노"
                # eb_data = pd.read_csv('data/ek20_29/ek23.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 24:
                emotion = "분노/성가신"
                # eb_data = pd.read_csv('data/ek20_29/ek24.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 25:
                emotion = "분노/악의적인"
                # eb_data = pd.read_csv('data/ek20_29/ek25.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 26:
                emotion = "분노/안달하는"
                # eb_data = pd.read_csv('data/ek20_29/ek26.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 27:
                emotion = "분노/좌절한"
                # eb_data = pd.read_csv('data/ek20_29/ek27.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 28:
                emotion = "분노/짜증내는"
                # eb_data = pd.read_csv('data/ek20_29/ek28.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 29:
                emotion = "분노/툴툴대는"
                # eb_data = pd.read_csv('data/ek20_29/ek29.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
        elif max_index >= 30 and max_index < 40:
            if max_index == 30:
                emotion = "불안/걱정스러운"
                # eb_data = pd.read_csv('data/ek30_39/ek30.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 31:
                emotion = "불안/당혹스러운"
                # eb_data = pd.read_csv('data/ek30_39/ek31.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 32:
                emotion = "불안/두려운"
                # eb_data = pd.read_csv('data/ek30_39/ek32.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 33:
                emotion = "불안/불안"
                # eb_data = pd.read_csv('data/ek30_39/ek33.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 34:
                emotion = "불안/스트레스 받는"
                # eb_data = pd.read_csv('data/ek30_39/ek34.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 35:
                emotion = "불안/조심스러운"
                # eb_data = pd.read_csv('data/ek30_39/ek35.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 36:
                emotion = "불안/초조한"
                # eb_data = pd.read_csv('data/ek30_39/ek36.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 37:
                emotion = "불안/취약한"
                # eb_data = pd.read_csv('data/ek30_39/ek37.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 38:
                emotion = "불안/혼란스러운"
                # eb_data = pd.read_csv('data/ek30_39/ek38.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 39:
                emotion = "불안/회의적인"
                # eb_data = pd.read_csv('data/ek30_39/ek39.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
        elif max_index >= 40 and max_index < 50:
            if max_index == 40:
                emotion = "상처/가난한, 불우한"
                # eb_data = pd.read_csv('data/ek40_49/ek40.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 41:
                emotion = "상처/고립된"
                # eb_data = pd.read_csv('data/ek40_49/ek41.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 42:
                emotion = "상처/괴로워하는"
                # eb_data = pd.read_csv('data/ek40_49/ek42.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 43:
                emotion = "상처/배신당한"
                # eb_data = pd.read_csv('data/ek40_49/ek43.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 44:
                emotion = "상처/버려진"
                # eb_data = pd.read_csv('data/ek40_49/ek44.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 45:
                emotion = "상처/상처"
                # eb_data = pd.read_csv('data/ek40_49/ek45.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 46:
                emotion = "상처/억울한"
                # eb_data = pd.read_csv('data/ek40_49/ek46.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 47:
                emotion = "상처/질투하는"
                # eb_data = pd.read_csv('data/ek40_49/ek47.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 48:
                emotion = "상처/충격 받은"
                # eb_data = pd.read_csv('data/ek40_49/ek48.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 49:
                emotion = "상처/희생된"
                # eb_data = pd.read_csv('data/ek40_49/ek49.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
        elif max_index >= 50 and max_index < 60:
            if max_index == 50:
                emotion = "슬픔/낙담한"
                # eb_data = pd.read_csv('data/ek50_59/ek50.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 51:
                emotion = "슬픔/눈물이 나는"
                # eb_data = pd.read_csv('data/ek50_59/ek51.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 52:
                emotion = "슬픔/마비된"
                # eb_data = pd.read_csv('data/ek50_59/ek52.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 53:
                emotion = "슬픔/비통한"
                # eb_data = pd.read_csv('data/ek50_59/ek53.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 54:
                emotion = "슬픔/슬픔"
                # eb_data = pd.read_csv('data/ek50_59/ek54.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 55:
                emotion = "슬픔/실망한"
                # eb_data = pd.read_csv('data/ek50_59/ek55.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 56:
                emotion = "슬픔/염세적인"
                # eb_data = pd.read_csv('data/ek50_59/ek56.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 57:
                emotion = "슬픔/우울한"
                # eb_data = pd.read_csv('data/ek50_59/ek57.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 58:
                emotion = "슬픔/환멸을 느끼는"
                # eb_data = pd.read_csv('data/ek50_59/ek58.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
            elif max_index == 59:
                emotion = "슬픔/후회되는"
                # eb_data = pd.read_csv('data/ek50_59/ek59.csv', encoding='utf-8')
                result_system = wordEmbedding(eb_data, sent)
        elif max_index == 60:
            emotion = "평온/평온"
            # eb_data = pd.read_csv('data/ek60.csv', encoding='utf-8')
            result_system = wordEmbedding(eb_data, sent)
        else:
            print("------ [에러]max_index 값이 0보다 작거나 61보다 큽니다. ------")


        print(f'Answer: {result_system}, emotion: {emotion} index: {max_index}, softmax_value: {max_index_value}')
        print('-' * 50)