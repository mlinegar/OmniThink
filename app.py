import time
import json
import base64

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from http import HTTPStatus

import sys

import os
import subprocess

import os
import sys
from argparse import ArgumentParser
from src.tools.lm import OpenAIModel_dashscope
from src.tools.rm import GoogleSearchAli, GoogleSearchAli_readpage
from src.tools.mindmap import MindMap
from src.actions.outline_generation import OutlineGenerationModule
from src.dataclass.Article import Article
from src.actions.article_generation import ArticleGenerationModule
from src.actions.article_polish import ArticlePolishingModule
from src.actions.html_generation import HtmlGenerationModule

bash_command = "pip install --upgrade pip"
process = subprocess.Popen(bash_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Load environment variables and API keys
# load_dotenv()

openai_kwargs = {
    'api_key': os.getenv("OPENAI_API_KEY"),
    'api_provider': os.getenv('OPENAI_API_TYPE'),
    'temperature': 1.0,
    'top_p': 0.9,
    'api_base': os.getenv('AZURE_API_BASE'),
    'api_version': os.getenv('AZURE_API_VERSION'),
}


lm = OpenAIModel_dashscope(model='gpt-4o', max_tokens=1000, **openai_kwargs)
lm4outline = OpenAIModel_dashscope(model='gpt-4o', max_tokens=1000, **openai_kwargs)
lm4gensection = OpenAIModel_dashscope(model='gpt-4o', max_tokens=2000, **openai_kwargs)
lm4polish = OpenAIModel_dashscope(model='claude-3-7-sonnet-20250219', max_tokens=4000, **openai_kwargs)
lm4html = OpenAIModel_dashscope(model='claude-3-7-sonnet-20250219', max_tokens=4000, **openai_kwargs)

# rm = GoogleSearchAli(k=5)
rm = GoogleSearchAli_readpage(k=5)


st.set_page_config(page_title='DeepResearch', layout="wide")


st.title('🤖 DeepResearch')
# st.markdown('_OmniThink is a tool that helps you think deeply about a topic, generate an outline, and write an article._')

# Sidebar for configuration and examples
with st.sidebar:
    st.header('⚙️ 配置')
    MAX_ROUNDS = st.number_input('🔍 检索深度', min_value=0, max_value=10, value=2, step=1)
    models = ['gpt-4o', '即将推出']
    selected_example = st.selectbox('🤖 语言模型:', models)
    searchers = ['Google搜索', '即将推出']
    selected_example = st.selectbox('🔎 搜索引擎', searchers)

    n_max_doc = st.number_input('📄 单次检索网页数量', min_value=1, max_value=50, value=10, step=5)
    st.header('📚 示例')
    examples = ['通义实验室', '2024花莲地震', '泰勒·斯威夫特', '尹锡悦']
    selected_example = st.selectbox('选择示例', examples)
    status_placeholder = st.empty()

mind_map = MindMap(
    retriever=rm,
    gen_concept_lm=lm,
    depth = MAX_ROUNDS
)

def Think(input_topic):

    generator = mind_map.build_map(input_topic)   

    st.markdown(f'### 🔍 正在对 {input_topic} 相关内容进行深度搜索...')

    for idx, layer in enumerate(generator):
        print(layer)
        print('layer!!!')
        st.markdown(f'### 第 {idx + 1} 层深度思考检索...')
        status_placeholder.text(f"正在进行第 {idx + 1} 层深度思考检索，预计需要 {(idx+1)*3} 分钟。")
        for node in layer:
            category = node.category

            print(f'category: {category}')
            with st.expander(f'📌 {category}'):
                st.markdown(f'### {node.category} 的概念')
                print(node.concept)
                for concept in node.concept:
                    st.markdown(f'* {concept}')
                st.markdown(f'### {node.category} 的网络信息')
                for idx, info in enumerate(node.info):
                    st.markdown(f'{idx + 1}. {info["title"]} \n {info["snippets"]}')

    st.markdown(f'正在为 {mind_map.get_web_number()} 个检索到的网页构建索引表...')
    mind_map.prepare_table_for_retrieval()
    return '__finish__', '__finish__'

def GenOutline(input_topic):
    status_placeholder.text("📝 正在生成大纲，预计需要1分钟。")
    ogm = OutlineGenerationModule(lm)
    outline = ogm.generate_outline(topic= input_topic, mindmap = mind_map)

    return outline

def GenArticle(input_topic, outline):
    status_placeholder.text("✍️ 正在撰写文章，预计需要3分钟。")

    article_with_outline = Article.from_outline_str(topic=input_topic, outline_str=outline)
    ag = ArticleGenerationModule(retriever = rm, article_gen_lm = lm, retrieve_top_k = 3, max_thread_num = 10)
    article = ag.generate_article(topic = topic, mindmap = mind_map, article_with_outline = article_with_outline)
    # ap = ArticlePolishingModule(article_gen_lm = lm, article_polish_lm = lm)
    # article = ap.polish_article(topic = topic, draft_article = article)
    return article.to_string()


with st.form('my_form'):
    topic = st.text_input('🔍 请输入您感兴趣的主题', value=selected_example, placeholder='请输入您感兴趣的主题')
    submit_button = st.form_submit_button('🚀 生成！')

    if submit_button:
        if topic:
            st.markdown('### 🤔 思考过程')
            summary, news_timeline = Think(topic)
            st.session_state.summary = summary
            st.session_state.news_timeline = news_timeline

            st.markdown('### 📝 大纲生成')
            with st.expander("大纲生成", expanded=True):
                outline = GenOutline(topic)
                st.text(outline)

            st.markdown('### ✍️ 文章生成')
            with st.expander("文章生成", expanded=True):
                article = GenArticle(topic, outline)
                st.markdown(article)

        else:
            st.error('❌ 请输入主题。')


