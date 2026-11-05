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
MODEL = os.getenv("HF_MODEL", "openai/gpt-oss-120b")
PROVIDER = os.getenv("HF_PROVIDER", "auto")


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
    for msg in history:
        if msg.get("role") == "user":
            previous.append(f"사용자 지시: {msg.get('content', '')}")
        elif msg.get("role") == "assistant":
            previous.append(f"이전 이야기: {msg.get('content', '')}")
    context = "\n\n".join(previous)
    prompt = (
        f"장르: {style}\n주제: {theme}\n등장인물: {character}\n"
        f"기존 이야기:\n{context or '(첫 장면)'}\n\n"
        f"현재 사용자의 지시: {user_message}\n\n"
        "기존 이야기가 있다면 이어서 작성하고, 없으면 첫 장면을 시작하세요."
    )
    updated = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": ""}]
    client = InferenceClient(token=token, provider=PROVIDER, timeout=60)
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
                updated[-1]["content"] = answer
                yield updated
    except Exception:
        logger.exception("Story generation failed")
        updated[-1]["content"] = "스토리 생성에 실패했습니다. HF_TOKEN, 모델 접근 권한, API 상태를 확인해 주세요."
        yield updated
    else:
        if not answer:
            updated[-1]["content"] = "모델이 빈 응답을 반환했습니다. 모델 설정을 확인해 주세요."
            yield updated


def save_story(history, style, theme, character):
    if not history:
        raise gr.Error("저장할 이야기가 없습니다.")
    folder = Path("stories")
    folder.mkdir(exist_ok=True)
    filename = folder / f"story_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.txt"
    with filename.open("w", encoding="utf-8") as file:
        file.write(f"StoryLab KR\n장르: {style}\n주제: {theme}\n인물: {character}\n\n")
        for idx, msg in enumerate(history, 1):
            role = "사용자" if msg.get("role") == "user" else "AI"
            file.write(f"[{idx}] {role}: {msg.get('content', '')}\n\n")
    return str(filename.resolve())



# --- Video production planning (v0.2) ---
import json
import re

SHOT_PRESETS = [
    ("와이드 / Establishing", "24-35mm", "공간과 인물의 관계를 먼저 보여준다"),
    ("미디엄 / Action", "50mm", "핵심 행동과 인물의 반응을 따라간다"),
    ("클로즈업 / Detail", "85mm", "감정 또는 이야기의 결정적 디테일에 집중한다"),
]


def build_storyboard(title, genre, runtime, ratio, synopsis, characters, scene_text, shots_per_scene):
    """Create an editable planning draft, not an AI-generated script."""
    if not (title or "").strip():
        raise gr.Error("프로젝트명을 입력해 주세요.")
    if not (scene_text or "").strip():
        raise gr.Error("씬 구성을 입력해 주세요. 빈 줄로 씬을 구분합니다.")
    raw_scenes = [part.strip() for part in re.split(r"\\n\\s*\\n", scene_text) if part.strip()]
    if len(raw_scenes) > 50:
        raise gr.Error("한 번에 최대 50개 씬까지 처리할 수 있습니다.")
    count = int(shots_per_scene)
    scenes = []
    for i, description in enumerate(raw_scenes, 1):
        shots = []
        for j in range(count):
            size, lens, direction = SHOT_PRESETS[j % len(SHOT_PRESETS)]
            shots.append({
                "shot_id": f"S{i:02d}-C{j + 1:02d}",
                "shot_size": size,
                "lens": lens,
                "action": description,
                "direction": direction,
                "dialogue": "",
                "duration_seconds": None,
                "image_prompt": (
                    f"Photorealistic cinematic film still, {ratio} composition, "
                    f"{size}, {lens} lens look. Scene: {description}. "
                    f"Character continuity: {characters or 'use approved character reference'}. "
                    "Natural skin texture, grounded lighting, coherent location and wardrobe. "
                    "No subtitles, no watermark, no added text."
                ),
                "video_prompt": (
                    f"Live-action cinematic shot, {ratio}. {description}. "
                    f"Camera framing: {size}, {lens} look. {direction}. "
                    "Maintain exact character, costume and environment continuity from reference. "
                    "Natural restrained movement, no scene jump, no added text."
                ),
            })
        scenes.append({"scene_id": f"S{i:02d}", "description": description, "shots": shots})
    return {
        "schema_version": "0.2",
        "project": {"title": title.strip(), "genre": genre, "runtime_minutes": float(runtime), "ratio": ratio,
                    "synopsis": (synopsis or "").strip(), "characters": (characters or "").strip()},
        "scenes": scenes,
    }


def storyboard_markdown(project):
    info = project["project"]
    lines = [f"# {info['title']} — 스토리보드 초안", "",
             f"장르: {info['genre']} · 러닝타임 목표: {info['runtime_minutes']:g}분 · 화면비: {info['ratio']}", "",
             "## 시놉시스", info["synopsis"] or "(미작성)", "", "## 캐릭터 / 연속성 기준",
             info["characters"] or "(미작성)", "",
             "> 아래 컷 구성과 프롬프트는 규칙 기반 초안입니다. 연출·타임코드·인물 고정은 검토 후 확정하세요.", ""]
    for scene in project["scenes"]:
        lines += [f"## {scene['scene_id']} — {scene['description']}", ""]
        for shot in scene["shots"]:
            lines += [f"### {shot['shot_id']} · {shot['shot_size']}",
                      f"- 렌즈 기준: {shot['lens']}",
                      f"- 연출: {shot['direction']}",
                      f"- 행동: {shot['action']}",
                      f"- 이미지 프롬프트: {shot['image_prompt']}",
                      f"- 영상 프롬프트: {shot['video_prompt']}", ""]
    return "\\n".join(lines)


def export_project(project):
    if not project or not project.get("scenes"):
        raise gr.Error("먼저 스토리보드 초안을 생성해 주세요.")
    folder = Path("exports")
    folder.mkdir(exist_ok=True)
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = folder / f"storylab_{name}.json"
    destination.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(destination.resolve())


def create_demo():
    with gr.Blocks(title="StoryLab KR") as demo:
        gr.Markdown("# 🎬 StoryLab KR\n한국어 스토리 개발과 영상 제작용 스토리보드 초안을 한곳에서 만듭니다.")
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
                    chatbot = gr.Chatbot(label="스토리", height=610)
        with gr.Tab("🎥 영상 스토리보드"):
            gr.Markdown("## 영상 프로젝트 설계\\n씬을 빈 줄로 구분해 입력하세요. 생성되는 컷과 프롬프트는 **규칙 기반 작업 초안**이며 AI가 장면을 해석한 결과가 아닙니다.")
            with gr.Row():
                with gr.Column(scale=1):
                    project_title = gr.Textbox(label="프로젝트명", value="새 영상 프로젝트")
                    project_genre = gr.Dropdown(STYLES, value="현실 드라마", label="장르")
                    project_runtime = gr.Number(value=3, minimum=0.1, maximum=240, label="목표 러닝타임(분)")
                    project_ratio = gr.Dropdown(["16:9", "9:16", "1:1", "4:5"], value="16:9", label="화면비")
                    project_synopsis = gr.Textbox(label="시놉시스 / 로그라인", lines=4, placeholder="작품이 어떤 이야기인지 적어주세요.")
                    project_characters = gr.Textbox(label="등장인물 / 고정할 외형·의상", lines=4, placeholder="캐릭터 시트 또는 승인한 인물 레퍼런스의 핵심 기준")
                    project_scenes = gr.Textbox(
                        label="씬 구성 (빈 줄로 씬 구분)", lines=9,
                        value="새벽의 패스트푸드점. 엄마가 혼자 메뉴를 고르다 망설인다.\\n\\n엄마가 햄버거를 받아 들고 조용히 미소 짓는다.",
                    )
                    project_shots = gr.Slider(1, 3, step=1, value=3, label="씬당 기본 컷 수")
                    storyboard_btn = gr.Button("🎬 컷별 스토리보드 초안 생성", variant="primary")
                    export_btn = gr.Button("📦 프로젝트 JSON 다운로드")
                    project_file = gr.File(label="프로젝트 파일")
                with gr.Column(scale=2):
                    storyboard_view = gr.Markdown(label="스토리보드 미리보기", value="프로젝트 정보를 입력하고 생성 버튼을 눌러주세요.")
                    storyboard_state = gr.State({})
            storyboard_btn.click(
                lambda a,b,c,d,e,f,g,h: (lambda p: (storyboard_markdown(p), p))(build_storyboard(a,b,c,d,e,f,g,h)),
                inputs=[project_title, project_genre, project_runtime, project_ratio, project_synopsis,
                        project_characters, project_scenes, project_shots],
                outputs=[storyboard_view, storyboard_state],
            )
            export_btn.click(export_project, inputs=storyboard_state, outputs=project_file)
        with gr.Tab("⚙️ 생성 설정"):
            temperature = gr.Slider(0.1, 1.8, value=0.7, step=0.1, label="창의성 (Temperature)")
            max_tokens = gr.Slider(64, 2048, value=512, step=64, label="최대 생성 길이 (토큰)")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="단어 선택 다양성 (Top-p)")
        with gr.Accordion("📖 사용 방법", open=False):
            gr.Markdown("1. 장르·주제와 인물을 설정합니다.\n2. 장면을 한국어로 적습니다.\n3. 생성 버튼을 누릅니다.\n4. 이어질 장면을 입력하거나 TXT 파일로 저장합니다.\n\n**주의:** 스토리보드 탭은 규칙 기반 초안만 만듭니다. 자동 이미지·영상 생성과 장기 기억은 아직 없습니다.")

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
