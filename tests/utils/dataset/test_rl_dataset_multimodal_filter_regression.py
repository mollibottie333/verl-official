from io import BytesIO
import sys
import types

import datasets
from PIL import Image

from verl.utils.dataset.rl_dataset import RLHFDataset


class _DummyProcessor:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        assert add_generation_prompt is True
        assert tokenize is False
        return "dummy-prompt"

    def __call__(self, text, images=None, videos=None, videos_kwargs=None, **kwargs):
        length = 4
        if images:
            length += 2 * len(images)
        if videos:
            length += 3 * len(videos)
        return {"input_ids": [list(range(length))]}


def _build_test_dataset(max_prompt_length: int) -> RLHFDataset:
    dataset = RLHFDataset.__new__(RLHFDataset)
    dataset.tokenizer = None
    dataset.processor = _DummyProcessor()
    dataset.prompt_key = "prompt"
    dataset.image_key = "images"
    dataset.video_key = "videos"
    dataset.image_patch_size = 14
    dataset.max_prompt_length = max_prompt_length
    dataset.filter_overlong_prompts = True
    dataset.apply_chat_template_kwargs = {}
    dataset.tool_schemas = None
    dataset.num_workers = 1
    return dataset


def _install_fake_qwen_vl_utils(monkeypatch):
    def _fake_process_vision_info(messages, image_patch_size=14, return_video_metadata=True):
        images = []
        videos = []
        for message in messages:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for chunk in content:
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("type") == "image":
                    images.append(chunk.get("image"))
                elif chunk.get("type") == "video":
                    videos.append((chunk.get("video"), {"dummy": True}))
        return images, videos

    fake_module = types.SimpleNamespace(process_vision_info=_fake_process_vision_info)
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", fake_module)


def test_multimodal_filter_uses_vision_length(monkeypatch):
    _install_fake_qwen_vl_utils(monkeypatch)

    dataset = _build_test_dataset(max_prompt_length=5)
    dataframe = datasets.Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": "<image> describe this image"}],
                "images": [{"image": "unused"}],
                "videos": [],
            },
            {
                "prompt": [{"role": "user", "content": "plain text"}],
                "images": [],
                "videos": [],
            },
        ]
    )

    filtered = dataset.maybe_filter_out_long_prompts(dataframe)

    assert len(filtered) == 1
    assert filtered[0]["prompt"][0]["content"] == "plain text"


def test_multimodal_filter_does_not_reuse_mutated_image_dict(monkeypatch):
    _install_fake_qwen_vl_utils(monkeypatch)

    image_bytes = BytesIO()
    Image.new("RGB", (8, 8), color="red").save(image_bytes, format="PNG")

    dataset = _build_test_dataset(max_prompt_length=10)
    dataframe = datasets.Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": "<image> describe this image"}],
                "images": [{"bytes": image_bytes.getvalue()}],
                "videos": [],
            }
        ]
    )

    filtered = dataset.maybe_filter_out_long_prompts(dataframe)

    assert len(filtered) == 1
