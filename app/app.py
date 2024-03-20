import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, Request
from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler

from slack_sdk.web.async_client import AsyncWebClient
from google.cloud import logging
import re
import pickle

import vertexai
from vertexai.language_models import TextGenerationModel

load_dotenv()

try:
    project_id = os.environ['PROJECT_ID']
    region = os.environ['REGION']
    slack_bot_token = os.environ['SLACK_BOT_TOKEN']
    slack_signing_secret = os.environ['SLACK_SIGNING_SECRET']
except KeyError as e:
    sys.exit(f"Environment variable not set: {e}")

app = App(token=slack_bot_token, signing_secret=slack_signing_secret)
app_handler = SlackRequestHandler(app)
api = FastAPI(
    docs_url="/"
)
router = APIRouter()


@api.get('/health')
async def health():
    response = {
        'status': 'up'
    }
    return response


@api.post('/slack/events')
async def events(req: Request):
    return await app_handler.handle(req)

# VertexAIを初期化
vertexai.init(project=project_id, location=region)

text_model = TextGenerationModel.from_pretrained("text-bison")
PARAMETERS = {
    "max_output_tokens": 1024,
    "temperature": 0.20,
    "top_p": 0.95,
    "top_k": 40,
}

RESPONSE_STYLE = """"""

# cloud logging
logging_client = logging.Client()

# cloud logging: 書き込むログの名前
logger_name = "palm2_slack_chatbot"

# cloud logging: ロガーを選択する
logger = logging_client.logger(logger_name)

# 本動作はここから


def generate_response(
    client: AsyncWebClient,
    ts: str,
    conversation_thread: str,
    user_id: str,
    channel_id: str,
    prompt: str,
) -> None:
    """
    ユーザーIDがボットのIDまたはNoneでなく、かつチャンネルIDが存在する場合、Slackチャンネルにメッセージを投稿する。

    Parameters
    ----------
    ts : str
        メッセージのタイムスタンプ
    user_id : str
        ユーザーID
    channel_id : str
        チャンネルID
    prompt : str
        プロンプト
    """
    response = text_model.predict(prompt, **PARAMETERS)

    # ブロックされたか確認する
    is_blocked = response.is_blocked
    is_empty_response = len(response.text.strip(" \n")) < 1

    if is_blocked or is_empty_response:
        payload = "入力または出力が Google のポリシーに違反している可能性があるため、出力がブロックされています。プロンプトの言い換えや、パラメータ設定の調整を試してください。"
    else:
        # slackで**などのmarkdownを正しく表示できないので削除し、簡潔にする
        payload = remove_markdown(response.text)

    # レスポンスをslackへ返す
    client.chat_postMessage(channel=channel_id, thread_ts=ts, text=payload)

    keyword = get_keyword(text_model, prompt, PARAMETERS)
    send_log(logger, user_id, prompt, payload, keyword)


def send_log(logger, user_id: str, prompt: str, payload: str, keyword: str) -> None:
    """
    ログを送信する

    Parameters
    ----------
    user_id : str
        ユーザーID
    prompt : str
        プロンプト
    payload : str
        ペイロード
    """

    # ログに書き込むデータを持つ辞書を作成する
    data = {
        "slack_user_id": user_id,
        "prompt": prompt,
        "response": payload,
        "keyword": keyword,
    }

    # 辞書をJSON文字列に変換する
    logger.log_struct(data)


def get_keyword(text_model, prompt: str, parameters) -> str:
    """
    キーワードを取得する

    Parameters
    ----------
    prompt : str
        プロンプト

    Returns
    ----------
    str
        キーワード
    """
    get_keyword_prompt = f"""勉強会の計画

    以下の文章からテーマまたコンテキストとなるキーワードを1つ生成してください: サボテンの育て方
    コンテンツの消費者は、サボテンの育て方に興味がある人々
    サボテンは、初心者でも育てやすい植物です。しかし、サボテンを健康に育てるためには、いくつかの注意点があります。
    サボテンの育て方について、以下の点を教えてください。
    - サボテンの種類
    - サボテンの土
    - サボテンの水やり
    - サボテンの肥料
    - サボテンの剪定
    - サボテンの病気と害虫
    テーマまたコンテキストとなるキーワード: サボテンの育て方

    以下の文章からテーマまたコンテキストとなるキーワードを1つ生成してください: 鍼灸学において、よだれつわりの治療法を教えてください。また、理由と考え方も教えてください。読み手は鍼灸師だとします。鍼を打つべきなツボとその理由を詳しく教えてください
    テーマまたコンテキストとなるキーワード: 鍼灸学医療

    以下の文章からテーマまたコンテキストとなるキーワードを1つ生成してください: 金融業界のビジネスプラン
    金融業界のビジネスプランを作成してください。
    ビジネスプランには、以下の項目を含めてください。
    - 会社の概要
    - 製品やサービスの概要
    - 市場分析
    - 競合分析
    - マーケティング戦略
    - 財務計画
    ビジネスプランは、3年間で利益を出すことを目標としてください。
    テーマまたコンテキストとなるキーワード: ビジネスプランの作成

    以下の文章からテーマまたコンテキストとなるキーワードを1つ生成してください: 30代男性向けのマーケティングのアイディアを教えてください
    テーマまたコンテキストとなるキーワード: マーケティング戦略

    以下の文章からテーマまたコンテキストとなるキーワードを1つ生成してください: {prompt}
    テーマまたコンテキストとなるキーワード:
    """
    response = text_model.predict(get_keyword_prompt, **parameters)
    is_blocked = response.is_blocked
    if is_blocked:
        keyword = "入力または出力が Google のポリシーに違反している可能性があるため、出力がブロックされています。プロンプトの言い換えや、パラメータ設定の調整を試してください。現在、英語にのみ対応しています。"
    else:
        keyword = remove_markdown(response.text)
    return keyword


def remove_markdown(text):
    # インラインコードブロックを削除する
    text = re.sub(r"`(.+?)`", r"\1", text)
    # マルチラインコードブロックを削除する
    text = re.sub(r"```(.+?)```", r"\1", text)
    # ボールドマークダウンを削除する
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    # イタリックマークダウンを削除する
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    # ヘッダーマークダウンを---で置き換える
    text = re.sub(r"^#{1,6}\s*(.+)", r"---\1---", text, flags=re.MULTILINE)
    return text.replace("[Example]:", "")


@app.event("message")
def handle_incoming_message(client: AsyncWebClient, payload: dict) -> None:
    """
    受信メッセージを処理する

    Parameters
    ----------
    payload : dict
        ペイロード
    """
    channel_id = payload.get("channel")
    user_id = payload.get("user")
    prompt = payload.get("text")
    ts = payload.get("ts")
    thread_ts = payload.get("thread_ts")
    conversation_thread = ts if thread_ts is None else thread_ts
    generate_response(client, ts, conversation_thread,
                      user_id, channel_id, prompt)
