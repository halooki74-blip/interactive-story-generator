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
    raw_scenes = [part.strip() for part in re.split(r"\n\s*\n", scene_text) if part.strip()]
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
    return "\n".join(lines)


def export_project(project):
    if not project or not project.get("scenes"):
        raise gr.Error("먼저 스토리보드 초안을 생성해 주세요.")
    folder = Path("exports")
    folder.mkdir(exist_ok=True)
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = folder / f"storylab_{name}.json"
    destination.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(destination.resolve())



# --- StoryLab KR interview-led dramaturgy engine (v0.3) ---
INTERVIEW_STEPS = [
    ("씨앗", "어떤 장면, 경험, 질문에서 이 작품이 시작됐나요? 한두 문장으로 들려주세요."),
    ("주인공", "주인공은 어떤 사람이고, 지금 가장 간절하게 원하는 것은 무엇인가요?"),
    ("장애물", "그걸 얻지 못하게 가로막는 사람·상황·내적 두려움은 무엇인가요?"),
    ("대가", "그 장애물을 넘지 못하면 주인공은 무엇을 잃나요?"),
    ("정보 차이", "관객이 먼저 알지만 주인공은 아직 모르는 사실이 있을까요? 없으면 '제안해줘'라고 답해도 돼요."),
    ("복선", "초반에 사소하게 등장했다가 후반에 의미가 바뀔 물건·대사·행동은 무엇인가요?"),
    ("충돌", "주인공의 믿음을 가장 세게 흔들 사건은 무엇인가요?"),
    ("변화", "결말에서 주인공이 얻는 새로운 깨달음이나 감정은 무엇인가요?"),
    ("형식", "영상의 장르, 길이, 톤을 알려주세요. 예: 생활 드라마, 10분, 절제된 감동."),
]


def interview_question(index, answers):
    """Context-aware wording while preserving a stable step order."""
    if index >= len(INTERVIEW_STEPS):
        return "인터뷰가 끝났어요. 아래 '스토리 설계안 생성'을 눌러주세요."
    stage, question = INTERVIEW_STEPS[index]
    if index == 2 and len(answers) > 1 and answers[1].strip():
        question = f"앞서 말한 주인공({answers[1][:100]})이 원하는 것을 가장 어렵게 만드는 장애물은 무엇인가요?"
    if index == 3 and len(answers) > 2 and answers[2].strip():
        question = f"그 장애물({answers[2][:100]})을 넘지 못하면 어떤 구체적인 대가를 치르나요?"
    return f"**{index+1}/{len(INTERVIEW_STEPS)} · {stage}**\n\n{question}"


def interview_next(answer, answers):
    history = list(answers or [])
    if len(history) >= len(INTERVIEW_STEPS):
        return interview_question(len(history), history), history, "", "인터뷰 완료. 스토리 설계안을 생성해 주세요."
    if not (answer or "").strip():
        raise gr.Error("답변을 입력해 주세요. 모르면 '제안해줘'라고 적어도 됩니다.")
    history.append(answer.strip())
    return interview_question(len(history), history), history, "", f"답변 {len(history)}/{len(INTERVIEW_STEPS)}개 수집 완료"


def interview_back(answers):
    history = list(answers or [])
    if history:
        last = history.pop()
        return interview_question(len(history), history), history, last, f"{len(history)+1}번째 질문을 수정할 수 있어요."
    return interview_question(0, []), [], "", "첫 질문입니다."


INTERVIEW_WRITER_PROMPT = """당신은 한국어 영화 시나리오 작가 겸 스토리 에디터입니다.
인터뷰 답변을 이야기의 원천으로 삼아 영화적 드라마 구조를 설계하세요.
아래의 원칙을 반드시 충족하되, 템플릿을 억지로 끼워 넣어 개연성을 해치지 마세요.

1. 욕망: 주인공에게 구체적인 외적 목표와 내적 결핍을 주고, 처음부터 끝까지 능동적으로 선택하게 합니다.
2. 장애물: 모든 주요 막에 주인공을 적극적으로 방해하는 사람/상황/내적 갈등을 배치합니다. 장애물의 강도를 점진적으로 올리고 실패의 대가를 명확히 합니다.
3. 씨뿌리기/씨거두기: 2~3개의 구체적 물건·대사·행동을 초반에 심고 마지막에 인과적으로 회수합니다. 복선과 회수 위치를 표로 표시하세요.
4. 히치콕식 서스펜스: 관객은 위험 또는 중요한 사실을 먼저 알지만 등장인물은 모르는 '정보의 비대칭'을 가능한 한 초반에 구성하고, 기다림·선택·지연을 통해 긴장을 만드세요. 단순 반전이나 점프 스케어와 구분하세요.
5. 정-반-합: 정(주인공의 신념·일상), 반(신념을 깨는 갈등·반례), 합(선택과 대가를 거친 새로운 태도)로 구성하되, 합은 무조건 해피엔딩이 아닙니다.
6. 감동: 설교나 감정 설명 대신 작은 행동·선택·이미지로 공감이 생기도록 설계하세요.
7. 이야기의 사건은 동기-행동-결과의 인과로 연결하고, 인물의 성격이 장면마다 갑자기 달라지지 않아야 합니다.
8. 사용자가 답하지 않거나 '제안해줘'라고 한 설정은 [AI 제안·미확정]으로 표시하고, 제공한 사실과 구분합니다.
9. 참고로 '히치콕식'은 특정 영화의 장면을 복제하라는 뜻이 아니라 정보 비대칭과 긴장의 원리를 사용하라는 뜻입니다.

다음 제목 순서로 작성하세요:
# 핵심 로그라인
# 사용자 답변에서 확정된 사실
# AI 제안·미확정 설정
# 주인공의 욕망과 치명적인 장애물
# 정-반-합 3막 구조
# 관객이 먼저 아는 정보와 서스펜스 설계
# 씨뿌리기 → 씨거두기 설계표
# 주요 장면 6~8개 (각 장면의 목표/장애물/선택/결과 포함)
# 마지막 감정 이미지
# 검증 체크리스트 (복선 회수, 인과성, 인물 일관성, 갈등의 상승)
응답은 자연스러운 한국어로 작성하세요."""


def generate_interview_story(answers):
    collected = list(answers or [])
    if len(collected) < 4:
        raise gr.Error("적어도 4개 질문에 답한 뒤 생성해 주세요.")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise gr.Error("실제 AI 설계안 생성에는 HF_TOKEN이 필요합니다. 배포 환경의 Secrets에 등록해 주세요.")
    detail = "\n".join(
        f"{i + 1}. {INTERVIEW_STEPS[i][0]} — {value}" for i, value in enumerate(collected)
    )
    client = InferenceClient(token=token, provider=PROVIDER, timeout=120)
    result = ""
    try:
        events = client.chat_completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": INTERVIEW_WRITER_PROMPT},
                {"role": "user", "content": "다음 인터뷰를 바탕으로 스토리를 개발하세요.\n\n" + detail},
            ],
            stream=True, max_tokens=3500, temperature=0.75, top_p=0.9,
        )
        for event in events:
            part = event.choices[0].delta.content if event.choices else None
            if part:
                result += part
                yield result
    except Exception as exc:
        logger.exception("Interview story generation failed")
        raise gr.Error("AI 호출 실패. HF_TOKEN, 모델 가용성, 사용량 제한을 확인해 주세요.") from exc
    if not result:
        raise gr.Error("AI가 빈 응답을 반환했습니다. 모델 설정을 확인해 주세요.")


def export_interview(answers, generated):
    if not answers and not generated:
        raise gr.Error("저장할 인터뷰가 없습니다.")
    folder = Path("exports")
    folder.mkdir(exist_ok=True)
    filename = folder / f"interview_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    data = {
        "schema_version": "0.3",
        "questions": [{"stage": INTERVIEW_STEPS[i][0], "question": INTERVIEW_STEPS[i][1], "answer": a}
                      for i, a in enumerate(answers or [])],
        "story_design": generated or "",
    }
    filename.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(filename.resolve())



def create_demo():
    with gr.Blocks(title="StoryLab KR") as demo:
        gr.Markdown("# 🎬 StoryLab KR\n한국어 스토리 개발과 영상 제작용 스토리보드 초안을 한곳에서 만듭니다.")
        with gr.Tab("🎤 작가 인터뷰"):
            gr.Markdown(
                "## 이야기의 씨앗부터 함께 찾기\n"
                "질문을 하나씩 답하면 주인공의 욕망·장애물·복선·서스펜스를 정리합니다. "
                "모르는 부분은 **제안해줘**라고 적으세요. AI의 추정은 확정 사실과 구분합니다."
            )
            interview_answers = gr.State([])
            with gr.Row():
                with gr.Column(scale=1):
                    question_view = gr.Markdown(interview_question(0, []))
                    interview_answer = gr.Textbox(
                        label="내 답변", lines=4, placeholder="짧게 적어도 됩니다. 모르면 '제안해줘'."
                    )
                    with gr.Row():
                        back_button = gr.Button("↩ 이전 질문")
                        next_button = gr.Button("답변하고 다음 질문", variant="primary")
                    interview_status = gr.Markdown("0/9개 답변 수집")
                    develop_button = gr.Button("✨ 인터뷰로 스토리 설계안 생성", variant="primary")
                    export_interview_btn = gr.Button("💾 인터뷰·설계안 JSON 저장")
                    interview_file = gr.File(label="인터뷰 프로젝트 파일")
                with gr.Column(scale=2):
                    interview_result = gr.Markdown("답변을 모으면 이곳에 로그라인, 정반합, 장애물, 복선 회수표, 서스펜스 설계안이 생성됩니다.")
            next_button.click(
                interview_next,
                inputs=[interview_answer, interview_answers],
                outputs=[question_view, interview_answers, interview_answer, interview_status],
            )
            back_button.click(
                interview_back,
                inputs=interview_answers,
                outputs=[question_view, interview_answers, interview_answer, interview_status],
            )
            develop_button.click(
                generate_interview_story, inputs=interview_answers, outputs=interview_result
            )
            export_interview_btn.click(
                export_interview, inputs=[interview_answers, interview_result], outputs=interview_file
            )
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
            gr.Markdown("## 영상 프로젝트 설계\n씬을 빈 줄로 구분해 입력하세요. 생성되는 컷과 프롬프트는 **규칙 기반 작업 초안**이며 AI가 장면을 해석한 결과가 아닙니다.")
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
                        value="새벽의 패스트푸드점. 엄마가 혼자 메뉴를 고르다 망설인다.\n\n엄마가 햄버거를 받아 들고 조용히 미소 짓는다.",
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
