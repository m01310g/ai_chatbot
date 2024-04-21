from konlpy.tag import Komoran
import pickle

class Preprocess:
    def __init__(self, word2index_dic='E:/ai_chatbot/train_tools/dict/chatbot_dict.bin', userdic='E:/ai_chatbot/util/user_dic.tsv'):
        # 단어 인덱스 사전 불러오기
        if(word2index_dic != ''):
            f = open(word2index_dic, 'rb')
            self.word_index = pickle.load(f)
            f.close()
            print("success load word_dic")

        else:
            self.word_index = None
            print("fail load word_dic")

        # 형태소 분석기 초기화
        self.komoran = Komoran(userdic=userdic)

        # 제외할 품사
        self.exclusion_tags = [
            # 주격조사, 보격조사, 관형격조사, 목적격조사, 부사격조사, 호격조사, 인용격조사
            'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ',
            # 보조사, 접속조사
            'JX', 'JC',
            # 마침표, 물음표, 느낌표, 쉼표, 가운데점, 콜론, 빗금, 따옴표, 괄호표, 줄표, 줄임표, 붙임표
            'SF', 'SP', 'SS', 'SE', 'SO',
            # 선어말어미, 종결어미, 연결어미, 명사형전성어미, 관형형전성어미
            'EP', 'EF', 'EC', 'ETN', 'ETM',
            # 명사파생접미사, 동사파생접미사, 형용사파생접미사
            'XSN', 'XSV', 'XSA'
        ]

    # Komoran POS tagging
    def pos(self, sentence):
        return self.komoran.pos(sentence)
    
    # 불용어 제거
    # 필요한 품사 정보만 가져오기
    def get_keywords(self, pos, without_tag=False):
        f = lambda x: x in self.exclusion_tags
        word_list = []
        for p in pos:
            if f(p[1]) is False:
                word_list.append(p if without_tag is False else p[0])
        return word_list
    
    # 키워드를 단어 인덱스 시퀀스로 변환
    def get_wordidx_sequence(self, keywords):
        if self.word_index is None:
            return []
        w2i = []
        for word in keywords:
            try:
                w2i.append(self.word_index[word])
            except KeyError:
                # 해당 단어가 사전에 없는 경우 OOV 처리
                w2i.append(self.word_index['OOV'])
        return w2i