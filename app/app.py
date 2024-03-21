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
from vertexai.language_models import TextGenerationModel
from typing import List

from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
import pprint

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
                             search_query=prompt,
                             user_id=user_id)

    # レスポンスをslackへ返す
    client.chat_postMessage(channel=channel_id, thread_ts=ts,
                            text=response)


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
                version="gemini-1.0-pro-001/answer_gen/v1",
                # version="preview",
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
