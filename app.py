"""StoryLab KR v0.1 — Korean UI adaptation of Interactive Story Generator.
Original project: https://github.com/ChanMeng666/interactive-story-generator
This file is modified from the original concept and preserves its interactive text-story workflow.
License: Apache-2.0 (retain the upstream LICENSE file when applying to the fork).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STYLES = ["판타지", "SF", "미스터리", "모험", "로맨스", "공포", "현실 드라마"]
THEMES = ["성장", "추리", "사랑", "가족", "일상", "우정", "상실과 회복", "갈등"]
CHARACTERS = {
    "모험가": "새로운 세계를 탐험하며 위험을 감수하는 용감한 인물",
    "탐정": "예리한 관찰력과 논리적인 추론 능력을 가진 인물",
    "예술가": "일상에서 의미를 발견하고 독창적으로 표현하는 인물",
    "과학자": "미지의 현상을 탐구하고 질문을 멈추지 않는 인물",
    "평범한 인물": "현실적인 고민과 복잡한 내면을 가진 인물",
}
SYSTEM_PROMPT = """당신은 한국어로 글을 쓰는 전문 스토리 작가입니다.
사용자가 제시한 설정과 이전 장면을 존중하며 하나의 연속된 이야기를 작성하세요.
인물의 성격, 관계, 시간, 장소, 원인과 결과가 이전 내용과 모순되지 않도록 주의하세요.
새로운 사건을 도입할 때는 충분한 복선을 고려하고, 문학적인 과장보다 생생하고 구체적인 행동과 대사를 우선하세요.
응답은 자연스러운 한국어로 작성하세요. 사용자가 명시적으로 요청하지 않으면 이야기를 처음부터 다시 시작하지 마세요.
"""
MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")


def korean_story(user_message, history, style, theme, character, temperature, max_tokens, top_p):
    if not user_message or not user_message.strip():
        raise gr.Error("장면 설명을 입력해 주세요.")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise gr.Error("HF_TOKEN이 설정되지 않았습니다. 배포 환경의 비밀 변수에 등록해 주세요.")
    if history is None:
        history = []

    # Maintain previous input and AI output in the context sent to the model.
    previous = []
    for user_text, ai_text in history:
        if user_text:
            previous.append(f"사용자 지시: {user_text}")
        if ai_text:
            previous.append(f"이전 이야기: {ai_text}")
    context = "\n\n".join(previous)
    prompt = (
        f"장르: {style}\n주제: {theme}\n등장인물: {character}\n"
        f"기존 이야기:\n{context or '(첫 장면)'}\n\n"
        f"현재 사용자의 지시: {user_message}\n\n"
        "기존 이야기가 있다면 이어서 작성하고, 없으면 첫 장면을 시작하세요."
    )
    updated = history + [[user_message, ""]]
    client = InferenceClient(token=token)
    answer = ""
    try:
        for event in client.chat_completion(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            stream=True,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            top_p=float(top_p),
        ):
            delta = event.choices[0].delta.content if event.choices else None
            if delta:
                answer += delta
                updated[-1][1] = answer
                yield updated
    except Exception:
        logger.exception("Story generation failed")
        updated[-1][1] = "스토리 생성에 실패했습니다. HF_TOKEN, 모델 접근 권한, API 상태를 확인해 주세요."
        yield updated
    else:
        if not answer:
            updated[-1][1] = "모델이 빈 응답을 반환했습니다. 모델 설정을 확인해 주세요."
            yield updated


def save_story(history, style, theme, character):
    if not history:
        raise gr.Error("저장할 이야기가 없습니다.")
    folder = Path("stories")
    folder.mkdir(exist_ok=True)
    filename = folder / f"story_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.txt"
    with filename.open("w", encoding="utf-8") as file:
        file.write(f"StoryLab KR\n장르: {style}\n주제: {theme}\n인물: {character}\n\n")
        for idx, (user_text, ai_text) in enumerate(history, 1):
            file.write(f"[장면 {idx}]\n사용자: {user_text or ''}\nAI: {ai_text or ''}\n\n")
    return str(filename.resolve())


def create_demo():
    with gr.Blocks(theme=gr.themes.Soft(), title="StoryLab KR") as demo:
        gr.Markdown("# 🎬 StoryLab KR\n장르와 등장인물을 정하고 장면을 입력하면 AI가 한국어로 이야기를 이어갑니다.")
        with gr.Tab("✍️ 스토리 만들기"):
            with gr.Row():
                with gr.Column(scale=1):
                    style = gr.Dropdown(STYLES, value="현실 드라마", label="장르")
                    theme = gr.Dropdown(THEMES, value="일상", label="주제")
                    template = gr.Dropdown(list(CHARACTERS), value="평범한 인물", label="인물 템플릿")
                    character = gr.Textbox(label="인물 설정", lines=3, value=CHARACTERS["평범한 인물"])
                    scene = gr.Textbox(label="장면 설명", lines=4, placeholder="언제, 어디에서, 누가, 어떤 행동을 하나요?")
                    with gr.Row():
                        generate_btn = gr.Button("✨ 스토리 생성", variant="primary")
                        clear_btn = gr.Button("🗑️ 초기화")
                    save_btn = gr.Button("💾 TXT 다운로드 파일 만들기")
                    story_file = gr.File(label="저장된 스토리")
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="스토리", height=610, type="tuples")
        with gr.Tab("⚙️ 생성 설정"):
            temperature = gr.Slider(0.1, 1.8, value=0.7, step=0.1, label="창의성 (Temperature)")
            max_tokens = gr.Slider(64, 2048, value=512, step=64, label="최대 생성 길이 (토큰)")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="단어 선택 다양성 (Top-p)")
        with gr.Accordion("📖 사용 방법", open=False):
            gr.Markdown("1. 장르·주제와 인물을 설정합니다.\n2. 장면을 한국어로 적습니다.\n3. 생성 버튼을 누릅니다.\n4. 이어질 장면을 입력하거나 TXT 파일로 저장합니다.\n\n**주의:** 아직 장기 기억·스토리보드·영상 생성 기능은 없습니다.")

        template.change(lambda key: CHARACTERS[key], inputs=template, outputs=character)

        def run(message, history, s, t, c, temp, length, p):
            yield from korean_story(message, history, s, t, c, temp, length, p)
        inputs = [scene, chatbot, style, theme, character, temperature, max_tokens, top_p]
        generate_btn.click(run, inputs=inputs, outputs=chatbot).then(lambda: "", outputs=scene)
        scene.submit(run, inputs=inputs, outputs=chatbot).then(lambda: "", outputs=scene)
        clear_btn.click(lambda: ([], None), outputs=[chatbot, story_file])
        save_btn.click(save_story, inputs=[chatbot, style, theme, character], outputs=story_file)
    return demo


if __name__ == "__main__":
    create_demo().queue().launch(server_port=int(os.getenv("PORT", "7860")), share=False)
