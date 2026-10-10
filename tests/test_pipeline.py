import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from bs4 import BeautifulSoup
import paper_daily as daily
import create_index
from paper_store import load_papers, save_papers

ROOT = Path(__file__).resolve().parents[1]
LINK = 'https://arxiv.org/html/1234.56789'
LIST = BeautifulSoup('''<dl id="articles"><dt><a title="View HTML" href="/html/1234.56789">html</a></dt>
<dd><div class="meta"><div class="list-title">Title: Paper</div></div></dd></dl>''', 'html.parser')
DETAIL = BeautifulSoup('<div class="ltx_abstract"><p>Abstract</p></div>', 'html.parser')


class SummaryTests(unittest.TestCase):
    def connection(self, status=200, content='总结。分数：4分'):
        connection = Mock()
        response = connection.getresponse.return_value
        response.status = status
        response.read.return_value = json.dumps({'choices': [{'message': {'content': content}}]}).encode()
        return connection

    def test_success_parses_score_and_closes_connection(self):
        connection = self.connection()
        with patch.object(daily.http.client, 'HTTPSConnection', return_value=connection):
            result = daily.call_llm_for_summary('title', 'abstract', 'intro', 'related')
        self.assertEqual(result['score'], 4)
        self.assertEqual(result['error'], '')
        connection.close.assert_called_once()

    def test_http_errors_fail_without_reading_or_logging_provider_body(self):
        for status in (401, 403, 429, 500):
            with self.subTest(status=status):
                connection = self.connection(status)
                connection.getresponse.return_value.read.return_value = b'sensitive-provider-body'
                with patch.object(daily.http.client, 'HTTPSConnection', return_value=connection):
                    with self.assertRaisesRegex(RuntimeError, f'HTTP {status}') as raised:
                        daily.call_llm_for_summary('t', 'a', 'i', 'r')
                self.assertNotIn('sensitive-provider-body', str(raised.exception))
                connection.getresponse.return_value.read.assert_not_called()
                connection.close.assert_called_once()

    def test_network_exception_does_not_expose_exception_text(self):
        connection = self.connection()
        connection.request.side_effect = OSError('sensitive-request-value')
        with patch.object(daily.http.client, 'HTTPSConnection', return_value=connection):
            with self.assertRaises(RuntimeError) as raised:
                daily.call_llm_for_summary('t', 'a', 'i', 'r')
        self.assertNotIn('sensitive-request-value', str(raised.exception))
        connection.close.assert_called_once()

    def test_invalid_summary_is_not_accepted(self):
        connection = self.connection(content='No score')
        with patch.object(daily.http.client, 'HTTPSConnection', return_value=connection):
            with self.assertRaises(RuntimeError):
                daily.call_llm_for_summary('t', 'a', 'i', 'r')

    def test_cli_rejects_missing_key_and_prompt_without_printing_key(self):
        with tempfile.TemporaryDirectory() as temp:
            for key, prompt in (('', 'prompt'), ('fake-test-only-marker', '')):
                env = {**os.environ, 'LLM_API_KEY': key, 'LLM_PROMPT': prompt}
                result = subprocess.run([sys.executable, str(ROOT / 'paper_daily.py')],
                                        cwd=temp, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 1)
                self.assertNotIn('fake-test-only-marker', result.stdout + result.stderr)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.archive = self.root / 'papers'
        self.original = {'2025-01-01': [{'arxiv_html_link': LINK,
            'title': '历史论文', 'introduction': 'Keep original',
            'llm_summary': '大模型总结失败', 'llm_score': 0, 'llm_error': 'old error'}]}
        save_papers(self.original, self.archive)
        patcher = patch.object(daily, 'JSON_SAVE_PATH', str(self.archive))
        patcher.start()
        self.addCleanup(patcher.stop)
        daily.all_papers_global = {}

    def test_successful_retry_updates_in_place_and_keeps_old_metadata(self):
        with patch.object(daily, 'get_arxiv_soup', side_effect=[LIST, DETAIL]), \
             patch.object(daily, 'call_llm_for_summary', return_value={'summary': '分数：4分', 'score': 4, 'error': ''}):
            daily.crawl_and_process_papers('mock-url', 1)
        result = load_papers(self.archive)
        self.assertEqual(sum(map(len, result.values())), 1)
        self.assertEqual(result['2025-01-01'][0]['introduction'], 'Keep original')
        self.assertEqual(result['2025-01-01'][0]['llm_score'], 4)
        with patch.object(daily, 'get_arxiv_soup', return_value=LIST), \
             patch.object(daily, 'call_llm_for_summary') as summarize:
            daily.crawl_and_process_papers('mock-url', 1)
        summarize.assert_not_called()
        self.assertEqual(sum(map(len, load_papers(self.archive).values())), 1)

    def test_failed_summary_aborts_without_adding_failed_rows(self):
        with patch.object(daily, 'get_arxiv_soup', side_effect=[LIST, DETAIL]), \
             patch.object(daily, 'call_llm_for_summary', side_effect=RuntimeError('LLM HTTP 403')):
            with self.assertRaisesRegex(RuntimeError, '403'):
                daily.crawl_and_process_papers('mock-url', 1)
        self.assertEqual(load_papers(self.archive), self.original)

    def test_save_failure_is_not_swallowed(self):
        with patch.object(daily, 'get_arxiv_soup', side_effect=[LIST, DETAIL]), \
             patch.object(daily, 'call_llm_for_summary', return_value={'summary': '分数：4分', 'score': 4, 'error': ''}), \
             patch.object(daily, 'save_papers', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                daily.crawl_and_process_papers('mock-url', 1)
        self.assertEqual(load_papers(self.archive), self.original)

    def test_cli_403_exits_nonzero_without_publishing_or_exposing_body(self):
        # Run the actual entry point in an isolated directory with mocked I/O.
        save_papers(self.original, self.root / 'data' / 'papers')
        script = """
import runpy
from unittest.mock import Mock, patch
connection = Mock()
connection.getresponse.return_value.status = 403
connection.getresponse.return_value.read.return_value = b'sensitive-provider-body'
page = Mock()
page.text = '<dl id="articles"><dt><a title="View HTML" href="/html/1234.56789">html</a></dt><dd><div class="meta">Paper</div></dd></dl>'
with patch('http.client.HTTPSConnection', return_value=connection), patch('requests.get', return_value=page), patch('time.sleep'):
    runpy.run_path(SCRIPT_PATH, run_name='__main__')
""".replace('SCRIPT_PATH', repr(str(ROOT / 'paper_daily.py')))
        env = {**os.environ, 'LLM_API_KEY': 'fake-test-only-marker',
               'LLM_PROMPT': 'prompt', 'PYTHONPATH': str(ROOT)}
        result = subprocess.run([sys.executable, '-c', script], cwd=self.root,
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('LLM HTTP 403', result.stderr)
        self.assertNotIn('sensitive-provider-body', result.stdout + result.stderr)
        self.assertNotIn('fake-test-only-marker', result.stdout + result.stderr)
        self.assertFalse((self.root / 'README.md').exists())
        records = load_papers(self.root / 'data' / 'papers')
        self.assertEqual(records['2025-01-01'], self.original['2025-01-01'])
        self.assertEqual(sum(map(len, records.values())), 1)

    def test_missing_list_page_fails(self):
        with patch.object(daily, 'get_arxiv_soup', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'could not be fetched'):
                daily.crawl_and_process_papers('mock-url', 1)

    def test_corrupt_archive_aborts_before_network(self):
        (self.archive / '2025-01-01.json').write_text('broken')
        with patch.object(daily, 'get_arxiv_soup') as fetch:
            with self.assertRaises(ValueError):
                daily.crawl_and_process_papers('mock-url', 1)
        fetch.assert_not_called()

    def test_offline_render_uses_latest_available_dates_after_long_outage(self):
        md = self.root / 'README.md'
        html = self.root / 'index.html'
        daily.json_to_markdown(str(self.archive), str(md))
        self.assertTrue(create_index.json_to_html(str(self.archive), str(html), str(ROOT / 'template.html')))
        self.assertIn('2025-01-01', md.read_text())
        self.assertIn('历史论文', html.read_text())
        self.assertNotIn('{date_sections}', html.read_text())

    def test_empty_archive_markdown_has_clear_error_not_index_error(self):
        empty = self.root / 'empty'
        save_papers({}, empty)
        with self.assertRaisesRegex(ValueError, 'No paper data'):
            daily.json_to_markdown(str(empty), str(self.root / 'empty.md'))


if __name__ == '__main__':
    unittest.main()
