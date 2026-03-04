# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Variant of mathv2_like: same extraction (Final Answer + boxed/loose), outcome
# scoring uses math_dpsk_scoring (choice / math-verify / LLM judge) instead of grade_answer.

import json
import os
import re
import datetime
from typing import Optional
from math_verify import parse, StringExtractionConfig, LatexExtractionConfig, ExprExtractionConfig

from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed
from verl.utils.reward_score import genRM
from verl.utils.reward_score.math_dpsk_scoring import score_answer_text
import sympy

_DEBUG_LOG_PATH = os.environ.get(
    "MATHV2_DEBUG_LOG_PATH",
    f"debug_mathv2_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
)


def _log_answer_gt_score(record: dict) -> None:
    try:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Failed to write debug log to {_DEBUG_LOG_PATH}: {e}")

choices = ["a", "b", "c", "d","e", "A", "B", "C", "D", 'E']

verifier_prompt = r"""
    Your task is to evaluate the quality of a thinking process to a problem. The problem asks for an answer to the given multimodal question. Please evaluate the thinking process and score it according to the following criteria:

    - If the thinking process is completely correct with all steps executed properly and clearly demonstrated, including faithful visual cues, accurate handling of spatial transformations and topological relations, and rigorous error-free arithmetic while ensuring the reasoning chain is consistent and without internal contradictions, then the score is 1.
    - If the thinking process is generally correct but with some details omitted or minor errors, such as providing correct visual descriptions but failing to state all relevant information, showing slight logical leaps or minor self-corrections, or failing to fully state assumptions in ambiguous cases without breaking the core logic, then the score is 0.5.
    - If the thinking process does not actually address the required problem, contains fatal errors such as hallucinating visual details or violating physical laws, or has severe omissions including being self-contradictory, shifting logical standards mid-process, or relying on memory recall and guessing instead of active analysis, then the score is 0
    - Additionally, if the thinking process involves improper use of images, such as failing to fully utilize image information when required, misperceiving visual details, or basing extensive reasoning on image information that is only weakly relevant to the problem, then the score is 0.

    Please carefully reason out and analyze the quality of the thinking process below , and in your final response present a detailed evaluation of the thinking process' s quality followed by your score. Therefore, your final response should be in the following format:

    [Your evaluation here. You are required to present in detail the key steps of the  thinking process or the steps for which you had doubts regarding their correctness, and explicitly analyze whether each step is accurate: for correct steps, explain why you initially doubted their correctness and why they are indeed correct; for erroneous steps, explain the reason for the error and the impact of that error on the  thinking process. You should analyze your  thinking process faithfully. E.g., if there are issues in your final  thinking process, you should point it out.] 

    Based on my evaluation, the final overall score should be: \\boxed{{[the final overall score (0, 0.5, or 1, and nothing else) based on the above criteria.
    ]}}
    ---
    Here is your task input:
    
    ## Problem
    {problem}
    
    ## Thinking Process
    {solution_str}
"""

meta_verifier_prompt = r"""
    You are given a "problem","thinking process", and "thinking process evaluation", and you need to assess whether this "thinking process evaluation" is reasonable .

    First, "thinking process evaluation" is generated to evaluate the quality of the "thinking process", by prompting a verifier with the rules below (these are not your rules):
    Please evaluate the thinking process and score it according to the following criteria:

    - If the thinking process is completely correct with all steps executed properly and clearly demonstrated, including precise visual grounding, accurate handling of spatial transformations and topological relations, and rigorous error-free arithmetic while ensuring the reasoning chain is consistent and without internal contradictions, then the score is 1.
    - If the thinking process is generally correct but with some details omitted or minor errors, such as providing correct visual descriptions but failing to state all relevant information, showing slight logical leaps or minor self-corrections, or failing to fully state assumptions in ambiguous cases without breaking the core logic, then the score is 0.5.
    - If the thinking process does not actually address the required problem, contains fatal errors such as hallucinating visual details or violating physical laws, or has severe omissions including being self-contradictory, shifting logical standards mid-process, or relying on memory recall and guessing instead of active analysis, then the score is 0
    - Additionally, if the thinking process involves improper use of images, such as failing to fully utilize image information when required, misperceiving visual details, or basing extensive reasoning on image information that is only weakly relevant to the problem, then the score is 0.


    Next, I will introduce the rules for you to analyze the quality of the "thinking process evaluation".


    1. Your task is to analyze the "thinking process evaluation". You do not need to solve the "problem", nor do you need to strictly assess whether the "thinking process" is accurate. Your only task is to strictly follow the rules below to evaluate whether the "thinking process evaluation" is reasonable.

    2. You need to analyze the content of the "thinking process evaluation" from four aspects:
        - Restatement Analysis: In the "thinking process evaluation", certain behaviors of the visual information and resoning steps may be restated. You need to return to the original image and text of "thinking process" and check whether the "thinking process" actually has these behaviors mentioned in the "thinking process evaluation"
        - Defect Analysis: "reasoning evalution" may point out errors or defacts in the "thinking process". You need to carefully analyze whether the mentioned errors and defects are indeed valid.
        - Logical Analysis: Whether the "thinking process evaluation"'s logic is consistent.
        - Score Analysis: Whether the final score given by the "thinking process evaluation" matches the defects it found. You need to analyze according to the scoring rules given above.


    3. The most important part is **defect analysis**: In this part, your core task is to check whether the errors or defects of the "thinking process" pointed out in the "thinking process evaluation" are reasonable. In other words, any positive components about the "thinking process" in the "thinking process evaluation", regardless of whether they are reasonable, are not within your evaluation scope.
    For example: If the "thinking process evaluation" says that a certain conclusion in the "thinking process" is correct, but actually this conclusion is incorrect, then you do not need to care about this point. All parts that the "thinking process evaluation" considers correct do not belong to your evaluation scope.
    Specifically: If the "thinking process evaluation" believes that the "thinking process" is completely accurate and has not found any errors or defects, then regardless of whether the "thinking process" itself is actually accurate, even if there are obvious errors, you should still consider its analysis of errors to be reasonable.

    Importantly, for defects found by the "thinking process evaluation", you need to analyze two points simultaneously:
    - whether this defect actually exists
    - whether the "thinking process evaluation"'s analysis of this defect is accurate.
    These two aspects constitute the analysis of defects.

    4. About the **logical analysis**, if there are some logical errors in the "thinking process evaluation", you need to identify them. However, please note that identifying incorrect steps in the "thinking process" as correct steps does not constitute a logical error. In practice, logical errors include but are not limited to:
    If the "thinking process evaluation" identifies some reasoning step(s) in the "thinking process" as incorrect, then it cannot further indicate that subsequent conclusion(s) depending on those reasoning step(s) are wrong, but can only indicate that subsequent conclusion(s) are "not rigorously demonstrated."
    Self-contradictory reasoning steps.
    Inaccurate restatement of content from "thinking process"

    5. Finally, you need to present your analysis of the "thinking process evaluation" in your output and also rate its quality based on the rules below: First, if there is at least one unreasonable defect among the defects found by the "thinking process evaluation", then you only need to do **defect analysis**: If all defects found by the "thinking process evaluation" are unreasonable, then you should rate it with \(0\). If some defects found by the "thinking process evaluation" are reasonable and some are unreasonable, then your rating should be \(0.5\).


    Next, if the "thinking process evaluation" points out no errors or defects, or all defects found by the evaluation are reasonable, then you should do the following things:
    - Analyze whether "logical errors" exist in the "thinking process evaluation"(**logical analysis**) or whether "thinking process evaluation" gives a wrong score according to the rules for "thinking process evaluation"(**score analysis**). If yes, you should rate the "thinking process evaluation" with \(0.5\) ; if no, your rating should be \(1\)


    Your output should follow the format below:
    Here is my analysis of the "thinking process evaluation":
    [Your analysis here.]
    Based on my analysis, I will rate the "thinking process evaluation" as:
    \\boxed{{[the numerical rating of the "thinking process evaluation" (0, 0.5, or 1, and nothing else) based on the criteria above.]}}

    ---
    Here is your task input:
    
    ## Problem
    {problem}
    
    ## Thinking Process
    {solution_str}

    ## Thinking Process Evaluation
    {reasoning_evaluation_str}
"""

def extract_content_from_tag(text: str,header_name: str) -> Optional[str]:
    """Extract content from Markdown headers.    
    Args:
        text: The text containing the tags or headers
        tag: The tag name (e.g., 'answer', 'self-evaluation', 'redacted_reasoning')
              Maps to: 'answer' -> 'Answer', 'self-evaluation' -> 'Self Evaluation', 
                      'redacted_reasoning' -> 'Thinking Process'
    
    Returns:
        The content inside the section, or None if not found
    """
    markdown_pattern = rf"##\s+{re.escape(header_name)}\s*\n(.*?)(?=\n##\s+|\Z)"
    markdown_match = re.search(markdown_pattern, text, re.DOTALL | re.IGNORECASE)
    if markdown_match:
        return markdown_match.group(1).strip()
       
    return None


def res_format_reward(predict_str: str) -> float:
    """Compute R_o format reward.
    
    Checks for format: ## Raw Thinking  ... 
                       ## Solution ... 
                       ## Final Answer ...\box{} ...
                       ## Self Evaluation ...\box{} ...
    
    Args:
        predict_str: The prediction string to check
    
    Returns:
        1.0 if format matches, 0.0 otherwise
    """
    patterns = [
        r"##\s+Raw\s+Thinking",
        r"##\s+Solution",
        r"##\s+Final\s+Answer",
        r"##\s+Self\s+Evaluation",
    ]
    flags = re.IGNORECASE
    positions = []
    for pattern in patterns:
        match = re.search(pattern, predict_str, flags)
        if not match:
            return 0.0
        positions.append(match.start())
    
    for i in range(len(positions) - 1):
        if positions[i] >= positions[i + 1]:
            return 0.0
    return 1.0


def match_last_box_content(s):
    box_matches = [m.start() for m in re.finditer(r'\\boxed\{', s)]
    if not box_matches:
        return None
    last_box_start = box_matches[-1]
    stack = ['{']
    result = None
    i = last_box_start + len(r'\boxed{')
    while i < len(s):
        if s[i] == '{':
            stack.append(i)
        elif s[i] == '}':
            stack.pop()
            if not stack:
                result = s[last_box_start + len(r'\boxed{'):i]
                break
        i += 1
    return result

async def acc_reward_ro(
    predict_str: str,
    ground_truth: str,
    use_boxed: bool = True,
    extra_info: Optional[dict] = None,
) -> tuple:
    """Compute R_o accuracy reward (geo-like).

    Extraction logic is from mathv2_like (Final Answer section, boxed or loose parse).
    Scoring logic is from math_dpsk_scoring (choice / math-verify / LLM judge).

    Args:
        predict_str: The prediction string
        ground_truth: The ground truth answer
        use_boxed: Whether to extract boxed content
        extra_info: Optional dict with 'question' and 'images' for LLM judge

    Returns:
        (r_o_format, r_o_acc)
    """
    answer_str = extract_content_from_tag(predict_str, "Final Answer")
    if answer_str is None:
        print('answer str is None')
        return 0.0, 0.0

    answer = match_last_box_content(answer_str)
    if answer is None:
        r_o_format = 0.0
    else:
        r_o_format = 1.0
    if answer is not None:
        r_o_acc, method = await score_answer_text(answer, ground_truth, extra_info)
        record = {'ground_truth': ground_truth, 'r_o_acc': r_o_acc, 'reward_mechanism': method}
    else:
        r_o_acc = 0.0
        method = 'no_answer'
        record = {method}
    _log_answer_gt_score(record)
    return r_o_format, r_o_acc


async def compute_ro(
    predict_str: str,
    ground_truth: str,
    use_boxed: bool = True,
    extra_info: Optional[dict] = None,
) -> tuple:
    """Compute R_o reward (geo3k-like).

    Args:
        predict_str: The prediction string
        ground_truth: The ground truth answer
        use_boxed: Whether to extract boxed content
        extra_info: Optional dict for LLM judge (question, images)

    Returns:
        (thinking_format, acc_reward * format_reward, acc_reward, format_reward)
    """
    thinking_format = res_format_reward(predict_str)
    format_reward, acc_reward = await acc_reward_ro(
        predict_str, ground_truth, use_boxed, extra_info
    )
    return thinking_format, acc_reward * format_reward, acc_reward, format_reward


def format_reward_rp(predict_str: str) -> float:
    """Compute R_p format reward.
    
    Checks for \\box{} in ## Self Evaluation section.
    
    Args:
        predict_str: The prediction string to check
    
    Returns:
        Tuple of (1.0 if \\box{} found in self-evaluation, boxed content), or (0.0, None) otherwise
    """
    self_eval_content = extract_content_from_tag(predict_str, "Self Evaluation")
    if self_eval_content is None:
        return 0.0, None
    
    boxed = last_boxed_only_string(self_eval_content)
    if boxed is None:
        return 0.0, None
    boxed_content = remove_boxed(boxed)
    return 1.0, boxed_content



async def compute_rp(
    predict_str: str,
    problem: str,
    images: Optional[list[str]] = None,
    alpha: float = 0.76,
    beta: float = 0.24) -> float:
    """Compute R_p reward (mathv2-like).
    
    R_p = R_format(Y, Z) * (alpha * R_Y + beta * R_Z)
    where:
    - R_format: checks for \\box{} in self-evaluation
    - R_Y = |1 - (s' - s)|
    - R_Z = |1 - (s' - s)| * m_s
    - s: genRM(content in ## Answer section)
    - m_s: genRM(content in ## Answer and ## Self Evaluation sections)
    - s': box content from ## Self Evaluation ...\\box{} ...
    
    Args:
        predict_str: The prediction string
        problem: The problem statement
        alpha: Weight for R_Y (default 0.76)
        beta: Weight for R_Z (default 0.24)
    
    Returns:
        R_p score
    """
    r_format,s_prime_str = format_reward_rp(predict_str)
    if r_format == 0.0:
        return 0.0, 0.0, 0.0, -1, -1
    
    if s_prime_str is None:
        print('no score in actor\'s response')
        return 0.0, 0.0, 0.0, -1, -1
    
    try:
        s_prime = float(s_prime_str)
        if s_prime < 0.0 or s_prime > 1.0:
            return 0.0, 0.0, 0.0, -1, -1
    except (ValueError, TypeError):
        print('error')
        return 0.0, 0.0, 0.0, -1, -1
    
    think_content = extract_content_from_tag(predict_str, "Solution")
    if think_content is None:
        print('no answer content found')
        return 0.0, 0.0, 0.0, -1, -1

    s = await genRM.compute_genrm_score(problem, images, think_content, reasoning_evaluation_str=None, prompt_template=verifier_prompt)

    self_eval_content = extract_content_from_tag(predict_str, "Self Evaluation")
    
    m_s = await genRM.compute_genrm_score(problem, images, think_content, reasoning_evaluation_str=self_eval_content, prompt_template=meta_verifier_prompt)

    diff = abs(s_prime - s)
    r_score = 1.0 - diff
    print(f"r_score: {r_score}, s_prime(predict): {s_prime}, s(gt): {s}")
    r_z = r_score * m_s
    r_y = s

    r_p = r_format * (alpha * r_y + beta * r_z)
    return r_p,r_y,r_z, s,m_s


async def compute_score(
    predict_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    use_boxed: bool = True,
    format_score: float = 0.1,
    alpha: float = 0.5,
    beta: float = 0.5,
    prompt_template=None,
) -> float:
    """Compute the combined reward score.
    
    Reward = alpha * R_o + beta * (R_o * R_p)
    
    Args:
        predict_str: The prediction string
        ground_truth: The ground truth answer
        extra_info: Additional information dict, should contain 'question' key for the problem statement
        use_boxed: Whether to extract boxed content for R_o
        format_score: Weight for format score in R_o (default 0.1)
        alpha: Weight for R_o term (default 0.5)
        beta: Weight for (R_o * R_p) term (default 0.5)
        prompt_template: Optional prompt template for genRM in R_p computation.
    
    Returns:
        Combined reward score
    """
    r_t_f, r_o_out, r_o_acc, r_o_format = await compute_ro(
        predict_str, ground_truth, use_boxed, extra_info
    )
    
    if r_o_acc > 0.0:
        problem = extra_info["question"]
        if "images" in extra_info:
            images = extra_info["images"]
        else:
            images = None
        r_p, r_y, r_z, s, m_s = await compute_rp(predict_str, problem, images)
        
        reward = (alpha * r_p + beta * (r_o_out * r_p)) * r_t_f
    else:
        reward = r_o_acc 
        r_p,r_y,r_z,s,m_s = 0.0,0.0,0.0,0.0,0.0
    return {
            "score": reward, 
            "process_reward": r_p,
            "outcome_reward": r_o_out,
            "raw_thinking_exist_reward": r_t_f,
            "outcome_accuracy_reward": r_o_acc,
            "verifier_reward_R_y": r_y,
            "meta_verifier_reward_R_z": r_z,
            "verifier_score_s": s,
            "meta_verifier_score_m_s": m_s
        }


async def compute_eva_score(
    predict_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    use_boxed: bool = True,
    format_score: float = 0.1,
    alpha: float = 1.0,
    beta: float = 1.0,
    alpha_rp: float = 0.5,
    beta_rp: float = 0.5,
    prompt_template=None,
) -> float:
    """Compute the combined reward score.
    
    Reward = alpha * R_o + beta * (R_o * R_p)
    
    Args:
        predict_str: The prediction string
        ground_truth: The ground truth answer
        extra_info: Additional information dict, should contain 'question' key for the problem statement
        use_boxed: Whether to extract boxed content for R_o
        format_score: Weight for format score in R_o (default 0.1)
        alpha: Weight for R_o term (default 1.0)
        beta: Weight for (R_o * R_p) term (default 1.0)
        alpha_rp: Weight for R_Y in R_p (default 0.5)
        beta_rp: Weight for R_Z in R_p (default 0.5)
        prompt_template: Optional prompt template for genRM in R_p computation.
    
    Returns:
        Combined reward score
    """
    r_t_f, r_o_out, r_o_acc, r_o_format = await compute_ro(
        predict_str, ground_truth, use_boxed, extra_info
    )
    
    if r_o_acc > 0.0:
        problem = extra_info["question"]
        if "images" in extra_info:
            images = extra_info["images"]
        else:
            images = None
        r_p, r_y, r_z, s, m_s = await compute_rp(predict_str, problem, images)
        
        reward = alpha * r_p  + beta *(r_o_out * r_p)
    else:
        reward = r_o_acc 
        r_p,r_y,r_z,s,m_s = 0.0,0.0,0.0,0.0,0.0
    return {
            "score": reward,
        }
