import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import paper_store as store


class PaperStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'archive'
        self.data = {
            '2025-01-01': [{'arxiv_html_link': 'https://arxiv.org/html/1',
                            'title': '历史中文', 'introduction': 'All original text',
                            'extra': {'keep': True}}],
            '2025-01-02': [],
        }

    def test_migration_preserves_all_fields_and_duplicates(self):
        self.data['2025-01-01'] *= 2
        legacy = Path(self.temp.name) / 'legacy.json'
        legacy.write_text(json.dumps(self.data), encoding='utf-8')
        store.save_papers(store.load_papers(legacy), self.path)
        self.assertEqual(store.load_papers(self.path), self.data)
        self.assertTrue(legacy.exists())

    def test_only_changed_day_and_manifest_are_written(self):
        store.save_papers(self.data, self.path)
        before = (self.path / '2025-01-01.json').stat().st_mtime_ns
        updated = copy.deepcopy(self.data)
        updated['2025-01-03'] = [{'arxiv_html_link': 'new', 'title': 'new'}]
        with patch.object(store, 'atomic_write', wraps=store.atomic_write) as writer:
            store.save_papers(updated, self.path)
        self.assertEqual([call.args[0].name for call in writer.call_args_list],
                         ['2025-01-03.json', 'manifest.json'])
        self.assertEqual((self.path / '2025-01-01.json').stat().st_mtime_ns, before)
        self.assertEqual(store.load_papers(self.path), updated)

    def test_missing_archive_is_not_treated_as_empty(self):
        with self.assertRaises(FileNotFoundError):
            store.load_papers(self.path)

    def test_corrupt_or_missing_shard_is_rejected_without_writes(self):
        store.save_papers(self.data, self.path)
        shard = self.path / '2025-01-01.json'
        shard.write_text('[]', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            store.load_papers(self.path)
        with self.assertRaises(ValueError):
            store.save_papers(self.data, self.path)
        self.assertEqual(shard.read_text(), '[]')
        shard.unlink()
        with self.assertRaisesRegex(ValueError, 'file set'):
            store.load_papers(self.path)

    def test_removing_historical_rows_is_rejected(self):
        store.save_papers(self.data, self.path)
        self.data['2025-01-01'] = []
        with self.assertRaisesRegex(ValueError, 'remove historical'):
            store.save_papers(self.data, self.path)
        self.assertEqual(len(store.load_papers(self.path)['2025-01-01']), 1)

    def test_oversized_day_is_rejected_before_any_writes(self):
        with patch.object(store, 'MAX_FILE_BYTES', 10):
            with self.assertRaisesRegex(ValueError, 'size budget'):
                store.save_papers(self.data, self.path)
        self.assertFalse(self.path.exists())

    def test_unsafe_dates_are_rejected(self):
        with self.assertRaises(ValueError):
            store.save_papers({'../bad': []}, self.path)
        self.assertFalse(self.path.exists())

    def test_failed_atomic_write_preserves_original(self):
        target = Path(self.temp.name) / 'original'
        target.write_text('original')
        with patch.object(Path, 'replace', side_effect=OSError('disk error')):
            with self.assertRaises(OSError):
                store.atomic_write(target, b'new')
        self.assertEqual(target.read_text(), 'original')
        self.assertEqual(list(Path(self.temp.name).iterdir()), [target])


if __name__ == '__main__':
    unittest.main()
