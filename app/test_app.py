# test_app.py
import unittest
from unittest.mock import patch
from app import (search_sample, grounding,
                 generate_text_with_grounding, multi_turn_search_sample, get_thread_messages, generate_response_by_vertex_ai_search)
from dotenv import load_dotenv
import os
from slack_sdk.web.async_client import AsyncWebClient

load_dotenv()


class TestSearchSample(unittest.TestCase):

    # @patch('app.discoveryengine.SearchResponse')
    # def test_search_sample(self, mock_search):

    #     # モックの設定
    #     mock_response = mock_search.return_value
    #     mock_response.some_method.return_value = 'expected result'

    #     project_id = os.getenv('PROJECT_ID', 'default_project_id')
    #     # location = os.getenv('LOCATION', 'default_location')
    #     location = "global"
    #     engine_id = "jagu-e-r-qa-bot_1710151254495"
    #     search_query = '小売分科会について教えてください'  # ここではテスト用のクエリを直接指定

    #     # テスト実行
    #     result = search_sample(project_id, location,
    #                            engine_id, search_query)

    #     # 検証
    #     self.assertEqual(result.some_method(), 'expected result')

    # @patch('app.discoveryengine.SearchResponse')
    # def test_generate_text_with_grounding(self, mock_search):

    #     # モックの設定
    #     mock_response = mock_search.return_value
    #     mock_response.some_method.return_value = 'expected result'

    #     project_id = os.getenv('PROJECT_ID', 'default_project_id')
    #     data_store_location = os.getenv('DATA_STORE_LOCATION', 'default_project_id')
    #     data_store_id = os.getenv('DATA_STORE_ID', 'default_project_id')
    #     location = "asia-northeast1"

    #     # search_query = '小売分科会について教えてください'  # ここではテスト用のクエリを直接指定
    #     search_query = 'こんにちは。あなたは誰ですか？'  # ここではテスト用のクエリを直接指定
    #     # search_query = '今日の天気について教えてください'  # ここではテスト用のクエリを直接指定

    #     # テスト実行
    #     result = generate_text_with_grounding(project_id=project_id,
    #                        location=location,
    #                        data_store_location=data_store_location,
    #                        data_store_id=data_store_id,
    #                        query=search_query)

    #     # 検証
    #     self.assertEqual(result.some_method(), 'expected result')

    # @patch('app.discoveryengine.SearchResponse')
    # def test_multi_turn_search_sample(self, mock_search):

    #     # モックの設定
    #     mock_response = mock_search.return_value
    #     mock_response.some_method.return_value = 'expected result'

    #     project_id = os.environ['PROJECT_ID']
    #     # vertex_ai_location = os.environ['REGION']
    #     # slack_bot_token = os.environ['SLACK_BOT_TOKEN']
    #     # slack_signing_secret = os.environ['SLACK_SIGNING_SECRET']
    #     data_store_id = os.environ['DATA_STORE_ID']
    #     data_store_location = os.environ['DATA_STORE_LOCATION']
    #     # engine_id = os.environ['ENGINE_ID']
    #     vertex_ai_search_location = os.environ['VERTEX_AI_SEARCH_LOCATION']
    #     # chat_history_bucket_name = os.environ['CHAT_HISTORY_BUCKET_NAME']
    #     user_id="U05C253622C"

    #     # project_id=project_id,
    #     # location=vertex_ai_location,
    #     # data_store_location=data_store_location,
    #     # data_store_id=data_store_id,
    #     # search_queries=messages

    #     # messages = ["小売分科会について教えてください。", "ミートアップはどうなってますか？", "登壇者の一覧を教えてください"]
    #     messages = ["あなたは誰ですか？"]

    #     print(project_id, vertex_ai_search_location,
    #           data_store_location, data_store_id, messages)

    #     # テスト実行
    #     result = multi_turn_search_sample(project_id, vertex_ai_search_location,
    #                                       data_store_location, data_store_id, messages, user_id)

    #     # 検証
    #     self.assertEqual(result.some_method(), 'expected result')

    # @patch('app.discoveryengine.SearchResponse')
    # def test_get_thread_messages(self, mock_search):

    #     # モックの設定
    #     mock_response = mock_search.return_value
    #     mock_response.some_method.return_value = 'expected result'
    #     channel_id="C06MX2A6NU8"
    #     # thread_ts="1711080476.081309"
    #     thread_ts="1711079981.710919"
    #     get_thread_messages(channel_id, thread_ts)

    @patch('app.discoveryengine.SearchResponse')
    def test_generate_response_by_vertex_ai_search(self, mock_search):

        client = AsyncWebClient()
        channel_id="C06MX2A6NU8"
        user_id = "U05C253622C"
        prompt = "小売分科会について教えてください"
        ts = None
        # thread_ts="1711079981.710919"
        thread_ts="1711063295.156109"
        conversation_thread = ts if thread_ts is None else thread_ts
        user_id="U05C253622C"
        generate_response_by_vertex_ai_search(client, ts, conversation_thread,
                                            user_id, channel_id, prompt)


if __name__ == '__main__':
    unittest.main()
