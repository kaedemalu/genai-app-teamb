# test_app.py
import unittest
from unittest.mock import patch
from app import search_sample
from dotenv import load_dotenv
import os

load_dotenv()


class TestSearchSample(unittest.TestCase):

    @patch('app.discoveryengine.SearchResponse')
    def test_search_sample(self, mock_search):

        # モックの設定
        mock_response = mock_search.return_value
        mock_response.some_method.return_value = 'expected result'

        project_id = os.getenv('PROJECT_ID', 'default_project_id')
        # location = os.getenv('LOCATION', 'default_location')
        location = "global"
        engine_id = "jagu-e-r-qa-bot_1710151254495"
        search_query = '小売分科会について教えてください'  # ここではテスト用のクエリを直接指定

        # テスト実行
        result = search_sample(project_id, location,
                               engine_id, search_query)

        # 検証
        self.assertEqual(result.some_method(), 'expected result')


if __name__ == '__main__':
    unittest.main()
