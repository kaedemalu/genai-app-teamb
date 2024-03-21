import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, Request
from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler

from slack_sdk.web.async_client import AsyncWebClient
from google.cloud import logging
from google.protobuf.json_format import MessageToDict

import vertexai
from typing import List

from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
import pprint
import pickle
from typing import Optional
from vertexai.preview.generative_models import GenerativeModel
from typing import Optional
from google.cloud import bigquery, storage
from slack_sdk import WebClient

import vertexai
from vertexai.preview.generative_models import (
    GenerationResponse,
    GenerativeModel,
    grounding,
    Tool,
)


load_dotenv()

try:
    project_id = os.environ['PROJECT_ID']
    vertex_ai_location = os.environ['REGION']
    slack_bot_token = os.environ['SLACK_BOT_TOKEN']
    slack_signing_secret = os.environ['SLACK_SIGNING_SECRET']
    data_store_id = os.environ['DATA_STORE_ID']
    data_store_location = os.environ['DATA_STORE_LOCATION']
    engine_id = os.environ['ENGINE_ID']
    vertex_ai_search_location = os.environ['VERTEX_AI_SEARCH_LOCATION']
    chat_history_bucket_name = os.environ['CHAT_HISTORY_BUCKET_NAME']

except KeyError as e:
    sys.exit(f"Environment variable not set: {e}")

app = App(token=slack_bot_token, signing_secret=slack_signing_secret)
app_handler = SlackRequestHandler(app)
api = FastAPI(
    docs_url="/"
)
router = APIRouter()
slack_client = WebClient(slack_bot_token)


@api.get('/health')
async def health():
    response = {
        'status': 'up'
    }
    return response

# bucket_name = "vertex-ai-conversation-sample-kamiya-history"
base_blob_name = "chat-history"


@api.post('/slack/events')
async def events(req: Request):
    if "x-slack-retry-num" in req.headers:
        return
    return await app_handler.handle(req)

generation_config = {
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 40,
    "candidate_count": 1,
    "max_output_tokens": 8192,
}

model = GenerativeModel("gemini-1.0-pro-001",
                        generation_config=generation_config)

RESPONSE_STYLE = """"""

# cloud logging
logging_client = logging.Client()

# cloud logging: 書き込むログの名前
logger_name = "palm2_slack_chatbot"

# cloud logging: ロガーを選択する
logger = logging_client.logger(logger_name)


def download_blob(bucket_name, source_blob_name):
    """Downloads a blob from the bucket if it exists, otherwise returns None."""
    # Google Cloud Storageクライアントを作成
    storage_client = storage.Client()

    # 指定されたバケットを取得
    bucket = storage_client.bucket(bucket_name)

    # 指定されたblobを取得
    blob = bucket.blob(source_blob_name)

    # blobが存在するかどうかをチェック
    if blob.exists():
        # 存在する場合、blobの内容をバイトとしてダウンロード
        serialized_object = blob.download_as_bytes()
        if serialized_object:
            # シリアル化された会話のステートを逆シリアル化し、使える形にする
            return pickle.loads(serialized_object)
    else:
        # 存在しない場合、Noneを返す
        return []


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_string(source_file_name)


def serialize_to_pickle(python_object):
    # Serialize the Python object to a pickle
    serialized_object = pickle.dumps(python_object)
    return serialized_object


def process_and_upload_chat_history(chat, bucket_name, chat_history_blob_name):
    """チャットの履歴を処理してGCSにアップロードする関数"""
    # チャットの履歴を取得
    my_historical_chat = chat.history

    # 会話のステートをシリアル化
    serialized_chat_state = serialize_to_pickle(my_historical_chat)

    # GCSへアップロード
    upload_blob(bucket_name, serialized_chat_state, chat_history_blob_name)


def process_user_message_and_get_response(
    bucket_name, chat_history_blob_name, user_message, model
):
    # GCS から会話のステートをダウンロード
    chat_history = download_blob(bucket_name, chat_history_blob_name)

    # チャットモデルの初期化
    chat = model.start_chat(history=chat_history)

    # ユーザーメッセージをモデルに送信し、レスポンスを取得
    response = chat.send_message(user_message)

    # 会話の履歴をアップロード
    process_and_upload_chat_history(chat, bucket_name, chat_history_blob_name)

    return response.text


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

    chat_history_blob_name = f"{base_blob_name}_{user_id}_{conversation_thread}.pkl"
    result = generate_text_with_grounding(project_id=project_id,
                                          location=vertex_ai_location,
                                          data_store_location=data_store_location,
                                          data_store_id=data_store_id,
                                          query=prompt)

    response_text = None

    if result == "検索結果なし":
        response_text = process_user_message_and_get_response(
            chat_history_bucket_name, chat_history_blob_name, prompt, model
        )
    else:
        # response_text = search_sample(project_id=project_id,
        #                               location=vertex_ai_search_location,
        #                               engine_id=engine_id,
        #                               search_query=prompt,
        #                               user_id=user_id)
        messages = get_thread_messages(channel_id, conversation_thread)
        response_text = multi_turn_search_sample(
            project_id=project_id,
            location=vertex_ai_search_location,
            data_store_location=data_store_location,
            data_store_id=data_store_id,
            search_queries=messages
        )

    # レスポンスをslackへ返す
    client.chat_postMessage(channel=channel_id, thread_ts=ts,
                            text=response_text)


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


def get_thread_messages(channel_id, thread_ts):

    print(f"channel_id: {channel_id}, thread_ts: {thread_ts}")

    messages_text_list = []  # メッセージのテキストを格納するリスト

    try:
        # スレッド内のメッセージを取得
        result = slack_client.conversations_replies(
            channel=channel_id, ts=thread_ts)
        
        messages = result['messages']

        print(f"messages: {messages}")

        for msg in messages:
            print(f"msg: {msg}")
            # ボットからのメッセージを除外してユーザーからのメッセージのみを追加
            if 'bot_id' not in msg:
                messages_text_list.append(msg['text'])  # メッセージのテキストをリストに追加
    except Exception as e:
        print(f"エラーが発生しました: {e}")

    return messages_text_list  # メッセージのテキストリストを返却

# typeがtextのテキストを抽出する関数
def extract_text_elements(data):
    texts = []  # 抽出したテキストを保存するリスト
    if isinstance(data, dict):  # 入力が辞書の場合
        for value in data.values():
            texts.extend(extract_text_elements(value))  # 再帰的に探索
    elif isinstance(data, list):  # 入力がリストの場合
        for item in data:
            texts.extend(extract_text_elements(item))  # 再帰的に探索
    elif isinstance(data, dict) and data.get('type') == 'text':  # textタイプの要素を見つけた場合
        texts.append(data['text'])  # テキストをリストに追加
    return texts

def search_sample(
    project_id: str,
    location: str,
    engine_id: str,
    search_query: str,
    user_id: str = None

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
            return_snippet=True
        ),
        # For information about search summaries, refer to:
        # https://cloud.google.com/generative-ai-app-builder/docs/get-search-summaries
        summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
            summary_result_count=5,
            include_citations=True,
            ignore_adversarial_query=True,
            ignore_non_summary_seeking_query=True,
            model_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelSpec(
                version="gemini-1.0-pro-001/answer_gen/v1"
            ),
        )
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
        user_pseudo_id=user_id
    )

    response = client.search(request)
    response_ = MessageToDict(response._pb)
    pprint.pprint(response_)

    # 要約文取得
    summary = response.summary.summary_text.replace(
        "<b>", "").replace("</b>", "")

    # 関連ファイル取得
    references = []
    for r in response.results:
        r_dct = MessageToDict(r._pb)
        title = r_dct['document']['derivedStructData']['title']
        link = r_dct['document']['derivedStructData']['link']
        reference = {"title": title, "url": link}
        references.append(reference)

    result = {
        "summary": summary,
        "references": references
    }

    logger.log_struct(result)

    slack_message = format_slack_message(result)
    print(slack_message)

    return slack_message


def format_slack_message(result):
    # 要約を指定された形式にする
    formatted_summary = f"*要約*: {result['summary']}"

    # 参照をSlackの番号付きリスト形式にする
    formatted_references = []
    for idx, ref in enumerate(result['references'], start=1):
        formatted_references.append(f"{idx}. <{ref['url']}|{ref['title']}>")

    # Slackメッセージを組み立てる
    slack_message = f"{formatted_summary}\n\n*参照*:\n" + \
        "\n".join(formatted_references)

    return slack_message


def multi_turn_search_sample(
    project_id: str,
    location: str,
    data_store_location: str,
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
            project=project_id, location=data_store_location, data_store=data_store_id
        ),
        conversation=discoveryengine.Conversation(),
    )

    results_list = []  # 最終結果を格納するリスト
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
        response_ = MessageToDict(response._pb)

        # pprint.pprint(response_.keys())
        # pprint.pprint(response_["reply"]["summary"])

        # 要約文取得（HTMLタグを削除）
        summary = response_["reply"]["summary"]["summaryText"].replace(
            "<b>", "").replace("</b>", "")

        # 関連ファイル取得
        references = []
        for data in response_["reply"]["summary"]["summaryWithMetadata"]["references"][:5]:  # searchResultsの最初の5件を処理
            # data = result["document"]["derivedStructData"]
            title = data.get('title', 'タイトルが見つかりません')
            link = data.get('uri', 'リンクが見つかりません')
            references.append({"title": title, "url": link})

        # 一つの結果を辞書として構築
        result_dict = {
            "query": search_query,
            "summary": summary,
            "references": references
        }

        # 結果をリストに追加
        results_list.append(result_dict)
        logger.log_struct(result_dict)

    slack_message = format_slack_message(results_list[-1])
    print(slack_message)

    return slack_message


def generate_text_with_grounding(
    project_id: str,
    location: str,
    data_store_location: Optional[str],
    data_store_id: Optional[str],
    query: str,
) -> GenerationResponse:

    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)

    # Load the model
    model = GenerativeModel(model_name="gemini-1.0-pro")
    data_store_path = f"projects/{project_id}/locations/{data_store_location}/collections/default_collection/dataStores/{data_store_id}"

    # Create Tool for grounding
    if data_store_path:
        # Use Vertex AI Search data store
        # Format: projects/{project_id}/locations/{location}/collections/default_collection/dataStores/{data_store_id}
        tool = Tool.from_retrieval(
            grounding.Retrieval(grounding.VertexAISearch(
                datastore=data_store_path))
        )
    else:
        # Use Google Search for grounding (Private Preview)
        tool = Tool.from_google_search_retrieval(
            grounding.GoogleSearchRetrieval())

    prompt = f"以下の質問に答えてください。わからなければ、「検索結果なし」と答えてください。質問：{query}"
    response = model.generate_content(prompt, tools=[tool])
    data = response.to_dict()
    pprint.pprint(data)
    text_value = data['candidates'][0]['content']['parts'][0]['text']

    return text_value
