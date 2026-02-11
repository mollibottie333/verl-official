import datasets

from verl.utils.dataset.rl_dataset import RLHFDataset


class _DummyProcessor:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        assert add_generation_prompt is True
        assert tokenize is False
        return "dummy-prompt"

    def __call__(self, text, images=None, videos=None, videos_kwargs=None):
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


def test_multimodal_filter_uses_vision_length(monkeypatch):
    import verl.utils.dataset.vision_utils as vision_utils

    monkeypatch.setattr(vision_utils, "process_image", lambda image, image_patch_size=14: image)
    monkeypatch.setattr(
        vision_utils,
        "process_video",
        lambda video, image_patch_size=14, return_video_metadata=True: (video, {"dummy": True}),
    )

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
