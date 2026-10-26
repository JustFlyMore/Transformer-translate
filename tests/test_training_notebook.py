"""Run with the project's Python: python -m unittest discover -s tests -v."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "src/model/Transformer_translate.ipynb"


def load_definitions():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {"__name__": "notebook_test", "project_dir": ROOT}
    for index in (0, 2, 4, 6, 8, 10, 13):
        exec(compile("".join(notebook["cells"][index]["source"]),
                     f"{NOTEBOOK}:cell{index}", "exec"), namespace)
    return namespace


class TrainingNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.ns = load_definitions()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.zh = Path(self.temp.name) / "zh.txt"
        self.en = Path(self.temp.name) / "en.txt"
        self.zh_lines = ["你好。", "这是一个用于测试的句子。", "", "数据训练。", "很长的句子" * 20]
        self.en_lines = ["Hello.", "This is a sentence for testing.", "", "Training data.", "A long sentence " * 20]
        self.zh.write_text("\n".join(self.zh_lines) + "\n", encoding="utf-8")
        self.en.write_text("\n".join(self.en_lines) + "\n", encoding="utf-8")

    def dataset(self):
        return self.ns["TranslationDataset"](
            self.zh, self.en, self.ns["zh_sp"], self.ns["en_sp"], max_len=16)

    def loader(self):
        return torch.utils.data.DataLoader(self.dataset(), batch_size=2,
                                          collate_fn=self.ns["collate_fn"])

    def model(self, device):
        torch.manual_seed(42)
        return self.ns["TranslationTransformer"](
            self.ns["src_vocab_size"], self.ns["tgt_vocab_size"], d_model=16,
            nhead=2, num_encoder_layers=1, num_decoder_layers=1,
            dim_feedforward=32, dropout=0, max_len=16,
            src_pad_id=self.ns["zh_sp"].pad_id(),
            tgt_pad_id=self.ns["en_sp"].pad_id()).to(device)

    def test_tokenization_and_collation_match_original(self):
        dataset = self.dataset()
        for i, (zh, en) in enumerate(zip(self.zh_lines, self.en_lines)):
            for actual, text, sp in zip(dataset[i], (zh, en),
                                        (self.ns["zh_sp"], self.ns["en_sp"])):
                expected = [sp.bos_id()] + sp.encode(text.strip(), out_type=int)[:14] + [sp.eos_id()]
                self.assertEqual(actual.tolist(), expected)
        samples = [dataset[i] for i in range(len(dataset))]
        batch = self.ns["collate_fn"](samples)
        padded = []
        for side, sp in enumerate((self.ns["zh_sp"], self.ns["en_sp"])):
            sequences = [pair[side].tolist() for pair in samples]
            result = torch.full((len(samples), max(map(len, sequences))), sp.pad_id(), dtype=torch.long)
            for i, seq in enumerate(sequences):
                result[i, :len(seq)] = torch.tensor(seq)
            padded.append(result)
        labels = padded[1][:, 1:].clone()
        labels[labels == self.ns["en_sp"].pad_id()] = -100
        torch.testing.assert_close(batch["src"], padded[0])
        torch.testing.assert_close(batch["decoder_input"], padded[1][:, :-1])
        torch.testing.assert_close(batch["labels"], labels)

    def test_empty_and_misaligned_data(self):
        self.zh.write_text("", encoding="utf-8")
        self.en.write_text("", encoding="utf-8")
        self.assertEqual(len(self.dataset()), 0)
        self.en.write_text("extra\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "行数"):
            self.dataset()
        self.zh.write_text("one\ntwo\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "行数"):
            self.dataset()

    def test_sampler_covers_tail_and_changes_epoch(self):
        sampler = self.ns["LengthBucketBatchSampler"](
            [i % 19 + 2 for i in range(103)], batch_size=8, pool_batches=3)
        first = list(sampler)
        self.assertEqual(sorted(i for batch in first for i in batch), list(range(103)))
        self.assertEqual(len(first), len(sampler))
        self.assertTrue(all(1 <= len(batch) <= 8 for batch in first))
        self.assertEqual(first, list(sampler))
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))
        for count in (0, 1, 8, 9):
            for shuffle in (False, True):
                small = self.ns["LengthBucketBatchSampler"]([2] * count, 8, shuffle)
                batches = list(small)
                self.assertEqual(len(batches), len(small))
                self.assertEqual(sorted(i for batch in batches for i in batch), list(range(count)))

    def reference_epoch(self, model, loader, device, optimizer=None):
        model.train(optimizer is not None)
        loss_total, token_total = 0.0, 0
        with torch.set_grad_enabled(optimizer is not None):
            for batch in loader:
                src, dec, labels = (batch[k].to(device) for k in ("src", "decoder_input", "labels"))
                mask = torch.ones(dec.size(1), dec.size(1), dtype=torch.bool, device=device).triu(1)
                logits = model(src, dec, src.eq(self.ns["zh_sp"].pad_id()),
                               dec.eq(self.ns["en_sp"].pad_id()), mask)
                loss_sum = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                    ignore_index=-100, reduction="sum")
                count = labels.ne(-100).sum()
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    (loss_sum / count.clamp_min(1)).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                loss_total += loss_sum.item()
                token_total += count.item()
        return loss_total / token_total, token_total

    def compare_epochs(self, device):
        for training in (False, True):
            with self.subTest(device=device, training=training):
                original = self.model(device)
                optimized = copy.deepcopy(original)
                old_optimizer = torch.optim.Adam(original.parameters(), lr=1e-4) if training else None
                new_optimizer = torch.optim.Adam(optimized.parameters(), lr=1e-4) if training else None
                expected_loss, tokens = self.reference_epoch(original, self.loader(), device, old_optimizer)
                actual = self.ns["run_epoch"](
                    optimized, self.loader(), device, self.ns["zh_sp"].pad_id(),
                    self.ns["en_sp"].pad_id(), optimizer=new_optimizer, log_interval=0)
                self.assertAlmostEqual(expected_loss, actual["loss"], places=5)
                self.assertEqual(tokens, actual["tokens"])
                self.assertGreater(actual["tokens_per_second"], 0)
                for before, after in zip(original.parameters(), optimized.parameters()):
                    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
                self.assertTrue(all(p.grad is None for p in optimized.parameters()))

    def test_cpu_loss_and_parameter_updates_match(self):
        self.compare_epochs(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_loss_and_parameter_updates_match(self):
        self.compare_epochs(torch.device("cuda"))

    def test_all_cells_compile(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"cell{i}", "exec")


if __name__ == "__main__":
    unittest.main()
