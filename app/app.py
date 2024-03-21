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
from google.protobuf.json_format import MessageToDict

import vertexai
from vertexai.language_models import TextGenerationModel
from typing import List

from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
from google.protobuf import json_format
import json

load_dotenv()

try:
    project_id = os.environ['PROJECT_ID']
    region = os.environ['REGION']
    slack_bot_token = os.environ['SLACK_BOT_TOKEN']
    slack_signing_secret = os.environ['SLACK_SIGNING_SECRET']
    data_store_id = os.environ['DATA_STORE_ID']
    engine_id = os.environ['ENGINE_ID']
    vertex_ai_search_location = os.environ['VERTEX_AI_SEARCH_LOCATION']

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
    if "x-slack-retry-num" in req.headers:
        return
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


def generate_response_by_vertex_ai_search(
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
    response = search_sample(project_id=project_id,
                             location=vertex_ai_search_location,
                             engine_id=engine_id,
                             search_query=prompt)

    # レスポンスをslackへ返す
    client.chat_postMessage(channel=channel_id, thread_ts=ts,
                            text=response)


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


# @app.event("message")
@app.event("app_mention")
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
    generate_response_by_vertex_ai_search(client, ts, conversation_thread,
                                          user_id, channel_id, prompt)


def search_sample(
    project_id: str,
    location: str,
    engine_id: str,
    search_query: str,
) -> List[discoveryengine.SearchResponse]:

    client_options = (
        ClientOptions(
            api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )

    # Create a client
    client = discoveryengine.SearchServiceClient(client_options=client_options)

    # The full resource name of the search app serving config
    serving_config = f"projects/{project_id}/locations/{location}/collections/default_collection/engines/{engine_id}/servingConfigs/default_config"

    # Optional: Configuration options for search
    # Refer to the `ContentSearchSpec` reference for all supported fields:
    # https://cloud.google.com/python/docs/reference/discoveryengine/latest/google.cloud.discoveryengine_v1.types.SearchRequest.ContentSearchSpec
    content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
        # For information about snippets, refer to:
        # https://cloud.google.com/generative-ai-app-builder/docs/snippets
        snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
            return_snippet=False
        ),
        # For information about search summaries, refer to:
        # https://cloud.google.com/generative-ai-app-builder/docs/get-search-summaries
        summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
            summary_result_count=5,
            include_citations=False,
            ignore_adversarial_query=True,
            ignore_non_summary_seeking_query=True,
            model_prompt_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelPromptSpec(
                preamble="YOUR_CUSTOM_PROMPT"
            ),
            model_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelSpec(
                version="gemini-1.0-pro-001/answer_gen/v1",
            ),
        ),
    )

    # Refer to the `SearchRequest` reference for all supported fields:
    # https://cloud.google.com/python/docs/reference/discoveryengine/latest/google.cloud.discoveryengine_v1.types.SearchRequest
    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=search_query,
        page_size=5,
        content_search_spec=content_search_spec,
        query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
            condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
        ),
        spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
            mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
        ),
    )

    response = client.search(request)

    # 要約文取得
    summary = response.summary.summary_text.replace(
        "<b>", "").replace("</b>", "")

    # 関連ファイル取得
    references = []
    for r in response.results:
        r_dct = MessageToDict(r._pb)
        link = r_dct['document']['derivedStructData']['link']
        references.append(link)

    result = {
        "summary": summary,
        "references": references
    }

    logger.log_struct(result)
    return result


def multi_turn_search_sample(
    project_id: str,
    location: str,
    data_store_id: str,
    search_queries: List[str],
) -> List[discoveryengine.ConverseConversationResponse]:

    #  For more information, refer to:
    # https://cloud.google.com/generative-ai-app-builder/docs/locations#specify_a_multi-region_for_your_data_store
    client_options = (
        ClientOptions(
            api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )

    # Create a client
    client = discoveryengine.ConversationalSearchServiceClient(
        client_options=client_options
    )

    # Initialize Multi-Turn Session
    conversation = client.create_conversation(
        # The full resource name of the data store
        # e.g. projects/{project_id}/locations/{location}/dataStores/{data_store_id}
        parent=client.data_store_path(
            project=project_id, location=location, data_store=data_store_id
        ),
        conversation=discoveryengine.Conversation(),
    )

    for search_query in search_queries:
        # Add new message to session
        request = discoveryengine.ConverseConversationRequest(
            name=conversation.name,
            query=discoveryengine.TextInput(input=search_query),
            serving_config=client.serving_config_path(
                project=project_id,
                location=location,
                data_store=data_store_id,
                serving_config="default_config",
            ),
            # Options for the returned summary
            summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
                # Number of results to include in summary
                summary_result_count=3,
                include_citations=True,
            ),
        )
        response = client.converse_conversation(request)

        print(f"Reply: {response.reply.summary.summary_text}\n")

        for i, result in enumerate(response.search_results, 1):
            result_data = result.document.derived_struct_data
            print(f"[{i}]")
            print(f"Link: {result_data['link']}")
            print(f"First Snippet: {result_data['snippets'][0]['snippet']}")
            print(
                "First Extractive Answer: \n"
                f"\tPage: {result_data['extractive_answers'][0]['pageNumber']}\n"
                f"\tContent: {result_data['extractive_answers'][0]['content']}\n\n"
            )
            print("\n\n")

        return response
