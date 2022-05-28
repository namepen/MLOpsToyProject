import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

from collections import Counter
import json

@st.cache
def load_raw_data(json_path):
    #FIXME: 어떻게 모듈로서 실행하는지 모름. 루트로부터 찾아들어가야하는데 어떻게 하는지 아직 모름.
    try:
        with open(json_path, 'r') as f:
            raw_data_dict = json.load(f)
    except:
        with open(f'../.{json_path}') as f:
            raw_data_dict = json.load(f)
    return raw_data_dict['data']

@st.cache
def get_sample_data(raw_data_dict, idx):
    header_dict = raw_data_dict[idx]['header']
    body_df = pd.DataFrame(raw_data_dict[idx]['body'])
    return header_dict, body_df

@st.cache
def get_header_df(raw_data_dict:dict):
    """N 개의 채팅 데이터(body)와 메타정보(head)를 담고 있는 전체 데이터를 입력받아,
    메타정보를 요약한 1개의 N행 데이터프레임을 반환함.

    Args:
        raw_data_dict (dict): N 개의 채팅 데이터(body)와 메타정보(head)를 담고 있는 전체 데이터

    Returns:
        DataFrame: 메타정보를 요약한 1개의 N행 데이터프레임
    """
    def _participants_info(header)->dict:
        # 대화 참가자들의 평균 나이를 컬럼으로 추가함.
        average_age = []
        for participant in header['participantsInfo']:
            average_age.append(int(participant['age'][:2])) # e.g. '20대' = int('20')
        v = sum(average_age)/len(average_age)
        return {'averageAge':v}
    _li = []
    for e in raw_data_dict:
        header = e['header']
        _dialogue_info = header['dialogueInfo'].copy()
        _dialogue_info.update(_participants_info(header))
        _li.append(pd.Series(_dialogue_info))
    header_df = pd.DataFrame(_li)
    for col in ['numberOfParticipants', 'numberOfParticipants', 'numberOfTurns']:
        header_df[col] = header_df[col].astype('int32', copy=False)
        #NOTE: st.write() 는 int64 type 이 포함된 DataFrame 을 시각화할 수 없음.
    return header_df

@st.cache
def get_countplot(header_df:pd.DataFrame, feature:str):
    fig = plt.figure()
    sns.countplot(x=feature, data=header_df, color='salmon')
    return fig

def main():
    st.title('Chatting Data EDA')
    # Start
    state_data_load = st.text('데이터를 로딩 중입니다...')
    raw_data_dict   = load_raw_data('./data/sample/sample.json')
    state_data_load.text('')
    # End
    if st.checkbox('샘플 데이터 보기'):
        sample_idx = st.slider('', 1, 100, 1)
        # Start
        state_sample_data_load = st.text(f'{sample_idx}번째 대화에 대한 정보를 여는 중입니다...')
        header_dict, body_df   = get_sample_data(raw_data_dict, sample_idx-1)
        state_sample_data_load.text('')
        # End
        st.subheader(f'💬 {sample_idx}번째 대화')
        st.write(header_dict)
        st.write(body_df)
    st.subheader('📐 전체 데이터 분포')
    header_df = get_header_df(raw_data_dict)
    st.write(header_df)
    selected_feature = st.selectbox('특징 등장 빈도 보기', ['대화 참여자 수', '대화를 주고받은 횟수', '채팅 수', '주제'])
    _translate = {
        '대화 참여자 수':'numberOfParticipants',
        '대화를 주고받은 횟수':'numberOfTurns',
        '채팅 수':'numberOfUtterances',
        '주제':'topic',
        '평균 나이':'averageAge',
    }
    # Start
    fig = get_countplot(header_df, _translate[selected_feature])
    # End
    st.pyplot(fig)

if __name__ == '__main__':
    main()