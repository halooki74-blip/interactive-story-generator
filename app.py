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



# --- StoryLab KR interview -> character report -> drama doctor (v0.4) ---
INTERVIEW_STEPS = [
    ("기억의 씨앗", "어떤 기억이나 장면에서 이야기가 시작됐나요?"),
    ("감정과 감각", "그 기억에서 가장 선명한 감정이나 소리·냄새·빛은 무엇인가요?"),
    ("공간", "그 장면은 어디였나요? 기억나는 공간을 구체적으로 알려주세요."),
    ("함께한 사람", "그때 혼자였나요, 누군가 곁에 있었나요?"),
    ("행동", "그 순간 무엇을 하고 있었나요?"),
    ("행동의 이유", "그 행동을 하면서 어떤 생각을 했나요? 기억나지 않아도 괜찮아요."),
    ("인물 탐색", "그 기억에 등장하는 사람 중 한 명은 어떤 사람이었나요?"),
    ("관계의 단서", "그 사람의 특징이 드러나는 구체적인 행동을 기억하나요?"),
    ("나의 반응", "그 사람과 함께 있을 때 당신은 보통 어떻게 행동했나요?"),
    ("현재의 시선", "지금 그 시절의 자신이나 인물에게 해주고 싶은 말은 무엇인가요?"),
]
INTERVIEW_COUNT = len(INTERVIEW_STEPS)


def interview_question(index, answers):
    """A bounded, answer-aware interview; no invented biographical facts."""
    if index >= INTERVIEW_COUNT:
        return "**10/10 완료** — 캐릭터 분석 보고서를 생성하세요."
    label, base = INTERVIEW_STEPS[index]
    previous = (answers[-1] if answers else "").strip()
    if index == 3 and any(word in previous for word in ("방", "창문", "집", "학교", "공원")):
        base = "그 공간에 혼자 있었나요, 아니면 누군가 함께 있었나요?"
    if index == 4 and previous in ("혼자", "혼자였어", "혼자 있었어"):
        base = "혼자 그곳에 있을 때 구체적으로 무엇을 하고 있었나요?"
    if index == 6 and any(word in " ".join(answers) for word in ("엄마", "어머니")):
        base = "기억 속 엄마는 어떤 사람이었나요? 말투나 행동 하나를 떠올려주세요."
    if index == 7 and previous:
        base = "그 인물의 성격이 드러나는 실제 행동이나 말 한마디를 기억하나요?"
    if index == 8 and previous:
        base = "그런 행동을 보았을 때 어린 당신은 어떻게 반응했나요?"
    return f"**{index + 1}/{INTERVIEW_COUNT} · {label}**\n\n{base}"


def interview_next(answer, answers):
    history = list(answers or [])
    if len(history) >= INTERVIEW_COUNT:
        return interview_question(len(history), history), history, "", "10문항 인터뷰 완료"
    if not (answer or "").strip():
        raise gr.Error("답변을 입력해 주세요. 기억나지 않으면 '기억나지 않아'라고 입력할 수 있습니다.")
    history.append(answer.strip())
    return (interview_question(len(history), history), history, "",
            f"{len(history)}/{INTERVIEW_COUNT}문항 수집 완료")


def interview_back(answers):
    history = list(answers or [])
    if history:
        previous = history.pop()
        return (interview_question(len(history), history), history, previous,
                f"{len(history)}/{INTERVIEW_COUNT}문항 완료 · 답변 수정 중")
    return interview_question(0, []), [], "", "첫 번째 질문"


def require_complete(answers):
    items = list(answers or [])
    if len(items) != INTERVIEW_COUNT:
        raise gr.Error("캐릭터 보고서는 10개 질문을 모두 완료한 후 생성할 수 있습니다.")
    return items


def interview_transcript(answers):
    return "\n".join(
        f"{i+1}. [{INTERVIEW_STEPS[i][0]}] 질문: {INTERVIEW_STEPS[i][1]}\n답변: {value}"
        for i, value in enumerate(answers)
    )


CHARACTER_REPORT_PROMPT = """당신은 한국어 영화 캐릭터 리서처입니다.
10문항 인터뷰에서 개별 등장인물의 정체성을 구별하고 '분석 보고서'만 작성하세요.
사용자에게 추가 확인 질문이나 피드백 요청을 하지 마세요.
사용자가 직접 답한 내용을 [확인된 진술], 기억의 불명확함을 [불확실한 기억],
당신의 성격 해석과 설정을 [AI 해석·미확정]으로 구분합니다.
실제 답변에 없는 인물, 성별, 나이, 가족관계, 사건, 대사, 행동을 사실로 만들지 마세요.
1인칭 화자/현재의 화자/기억 속 인물이 같은지 불확실하면 표시하세요.
각 인물별로 행동, 말투, 감정 반응, 관계, 고유한 특징, 미확인 항목을 정리하고
서로 다른 캐릭터의 아이덴티티와 관계의 특징을 설명하세요.
근거가 약한 성격 단정은 하지 말고, 이후 창작에서의 일관성 기준을 제안하세요.
출력 순서:
# A · 캐릭터 분석 리포트
## 확인된 사실과 불확실한 기억
## 등장인물 식별
## 인물별 캐릭터 바이블 (확인/해석 명시)
## 관계와 개별성
## 인물 일관성 및 미확인 항목
마지막에 질문이나 승인 요청을 넣지 마세요."""


DRAMA_DIAGNOSIS_PROMPT = """당신은 한국어 영화 드라마 닥터입니다.
인터뷰와 A 캐릭터 리포트의 사실/가설 구분을 보존하세요.
작품의 정서와 인물의 개별성을 해치지 않고 극적 동력을 평가하세요.
목표, 동기, 장애물, 실패의 대가, 주인공의 중요한 선택,
전환점, 인과 관계, 씨뿌리기와 회수, 관객-인물 정보 비대칭, 정반합을 점검하세요.
이 장치들을 무조건 강요하지 말고 장르와 정서에 맞을 때만 추천하세요.
점수를 꾸며내지 말고 근거와 남은 약점을 설명하세요.
동력이 충분하다면 무리한 사건 추가 없이 그 사실을 보고하세요.
동력이 부족하다면 서로 다른 보강 장치 3개를 제시하세요.
각 장치는 [AI 창작 제안·미확정] 표시, 구체적인 사건, 작동 원리,
캐릭터에 미치는 영향, 감정 훼손 위험을 포함합니다.
마지막에 사용자에게 세 가지 중 하나를 부드럽게 선택하도록 안내하세요.
출력 순서:
# B · 극적 구조 진단
## 현재 이야기의 정서와 핵심 가능성
## 극적 동력 진단 및 보완 우선순위
## 추천 장치 A / B / C
추천 장치는 반드시 각각 '### 추천안 A', '### 추천안 B', '### 추천안 C' 한 줄 제목으로 나누어 구체적인 내용을 작성하세요.
## 가장 적합한 추천과 그 이유
## 선택 전에 유지할 캐릭터/기억의 기준"""


DRAMA_REWRITE_PROMPT = """당신은 한국어 영화 작가이며 드라마 닥터입니다.
사용자가 선택한 장치만 핵심 장치로 채택하고 과도한 사건을 추가하지 마세요.
인터뷰의 확인된 사실, 불확실한 기억, AI 설정을 명백히 구분하세요.
엄마의 메모 등 새로 고안한 사물과 대사는 역사적 사실처럼 표시하지 마세요.
인물별 성격, 행동 방식, 관계의 일관성을 유지하세요.
선택된 장치를 인과적으로 적용하여 목표와 현재 행동 동기, 장애물,
실패의 대가, 주인공이 내리는 중요한 선택, 전환과 결말을 개선하세요.
씨뿌리기와 씨거두기는 의미가 달라지는 소수의 구체적 요소로 구성하세요.
히치콕 정보 비대칭은 반드시 넣지 말고 적절하면 정보 비대칭인지
전형적인 서스펜스인지 감정적 아이러니인지 정확히 구별하세요.
정반합을 활용할 수 있지만 작품을 억지로 도식화하지 마세요.
출력 순서:
# B · 선택한 장치 적용 결과
## 보존된 실제 기억 / 불확실한 기억 / AI 창작
## 주인공 동기·목표·장애물·대가·선택
## 인물의 관계와 전환
## 씨뿌리기 → 씨거두기
## 장면별 재설계 (6~8장면)
## 개선 전후 평가와 남은 약점
## 다음 제작 단계로 전달할 설정"""


def call_storylab_ai(system_prompt, prompt, max_tokens=3500):
    token = os.getenv("HF_TOKEN")
    if not token:
        raise gr.Error("HF_TOKEN이 없습니다. 배포 환경의 Secrets에 설정해 주세요.")
    client = InferenceClient(token=token, provider=PROVIDER, timeout=120)
    chunks = []
    try:
        events = client.chat_completion(
            model=MODEL,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": prompt}],
            stream=True, max_tokens=max_tokens, temperature=0.65, top_p=0.9,
        )
        for event in events:
            delta = event.choices[0].delta.content if event.choices else None
            if delta:
                chunks.append(delta)
                yield "".join(chunks)
    except Exception as exc:
        logger.exception("StoryLab AI stage failed")
        raise gr.Error("AI 생성에 실패했습니다. 모델 접근 권한, HF_TOKEN과 사용량 제한을 확인하세요.") from exc
    if not chunks:
        raise gr.Error("AI 모델이 빈 응답을 반환했습니다.")


def generate_character_report(answers):
    items = require_complete(answers)
    yield from call_storylab_ai(
        CHARACTER_REPORT_PROMPT,
        "다음은 창작자의 인터뷰 답변입니다. 분석 보고서만 작성하세요.\n\n" + interview_transcript(items),
    )


def diagnose_drama(answers, character_report):
    items = require_complete(answers)
    if "# A · 캐릭터 분석 리포트" not in (character_report or ""):
        raise gr.Error("완료된 A 캐릭터 분석 보고서가 필요합니다.")
    yield from call_storylab_ai(
        DRAMA_DIAGNOSIS_PROMPT,
        "원본 인터뷰:\n" + interview_transcript(items)
        + "\n\nA 캐릭터 보고서:\n" + character_report,
    )


DRAMA_DEVICE_OPTIONS = ["추천안 A", "추천안 B", "추천안 C", "직접 입력"]


def rewrite_drama(answers, character_report, diagnosis, selection, custom_device):
    items = require_complete(answers)
    if "# A · 캐릭터 분석 리포트" not in (character_report or "") or "# B · 극적 구조 진단" not in (diagnosis or ""):
        raise gr.Error("완료된 A 보고서와 B 진단이 필요합니다.")
    if selection == "직접 입력":
        device = (custom_device or "").strip()
        if not device:
            raise gr.Error("직접 적용할 장치를 입력해 주세요.")
    else:
        choice = (selection or "추천안 B").split()[-1]
        # Apply the actual recommendation text, not a hard-coded device name.
        match = re.search(
            rf"(?ms)^### 추천안 {choice}[ \t]*\n(.*?)(?=^### 추천안 [ABC]|^## |\Z)",
            diagnosis,
        )
        if not match or not match.group(1).strip():
            raise gr.Error("B 진단 결과에서 선택한 추천안을 찾지 못했습니다. 진단을 다시 실행하거나 장치를 직접 입력하세요.")
        device = match.group(1).strip()
    yield from call_storylab_ai(
        DRAMA_REWRITE_PROMPT,
        "원본 인터뷰:\n" + interview_transcript(items)
        + "\n\nA 캐릭터 보고서:\n" + character_report
        + "\n\nB 극적 진단과 추천:\n" + diagnosis
        + "\n\n사용자 선택 장치:\n" + device
        + "\n선택한 장치는 AI 창작 설정이며 기억의 사실로 간주하면 안 됩니다.",
        max_tokens=4500,
    )


def export_interview(answers, character_report, diagnosis, rewrite):
    items = require_complete(answers)
    folder = Path("exports")
    folder.mkdir(exist_ok=True)
    filename = folder / f"interview_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    data = {
        "schema_version": "0.4",
        "questions": [{"stage": INTERVIEW_STEPS[i][0], "question": INTERVIEW_STEPS[i][1],
                       "answer": answer} for i, answer in enumerate(items)],
        "character_report_A": character_report or "",
        "drama_diagnosis_B": diagnosis or "",
        "drama_rewrite_B": rewrite or "",
        "notice": "AI 해석/창작 설정은 인터뷰의 확정된 사실과 별개입니다.",
    }
    filename.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(filename.resolve())



# --- C: Scene-based screenplay drafting and targeted revision (v0.5) ---
SCREENPLAY_PROMPT = """당신은 한국어 실사 단편영화 시나리오 작가입니다.
원본 인터뷰(확인된 진술/불확실한 기억), A 캐릭터 분석, B 극적 설계의
우선순위를 지킵니다. AI가 추가한 사건/대사는 실제 경험으로 단정하지 마세요.
등장인물의 고유한 행동, 말투, 관계, 감정 표현을 장면마다 일관되게 유지하세요.
대사는 설명적이지 않고, 행동과 시각적 디테일로 표현하세요.
각 씬에는 현재 목표, 행동, 상대 반응, 결과와 다음 장면으로 이어지는 인과가 필요합니다.
선택된 복선과 회수의 의미 변화가 장면으로 드러나도록 하세요.
원래 정서가 고요하다면 무리한 갈등, 반전, 위기를 추가하지 마세요.
추가 설정은 서두의 'AI 창작 설정'에 별도로 기재하세요.
출력 형식을 정확히 지키세요:
# C · 시나리오 초안
## 제작 기준
(장르, 길이, 인물 일관성, 사실/창작 경계)
## AI 창작 설정
(미확정 장면/대사/사건 명시)
## 씬별 시나리오
### SCENE 01
INT./EXT. 장소 - 시간
행동과 관찰 가능한 표정
인물명: 대사
[씬의 목표 / 장애물 / 선택 / 결과]
### SCENE 02
(동일 형식 반복)
... 요청된 마지막 씬까지 씬 번호를 연속으로 부여합니다.
## 연속성·복선 점검
(인물 일관성, 설정 충돌, 복선-회수 위치, 미해결 문제)
씬 내용 중에는 ### SCENE NN 외의 ### 제목을 사용하지 마세요.
"""

REVISION_PROMPT = """당신은 한국어 시나리오 스크립트 닥터입니다.
오직 제공된 한 개의 씬만 수정하고, 다른 씬을 상상하거나 변경하지 마세요.
캐릭터 바이블과 B 구조를 준수하세요. 앞뒤 씬의 기존 사건과 모순되지 않게 하세요.
인물 정체성, 시간, 공간, 소품, 복선, 설정의 원인-결과를 존중하세요.
사용자의 수정 지시가 실제 기억과 충돌한다면 실제 기억을 변조하지 않고
영화적 각색이라는 사실이 명확하도록 작성하세요.
반드시 수정된 씬 본문만 반환하세요. 번호/제목/다른 씬/해설/코드블록 금지.
"""

SCENE_HEADING_RE = re.compile(r"(?m)^### SCENE (\d{2})[ \t]*$")


def screenplay_scene_spans(screenplay):
    """Return stable scene spans without including the following scene."""
    matches = list(SCENE_HEADING_RE.finditer(screenplay or ""))
    return [
        (int(m.group(1)), m.start(), matches[i + 1].start() if i + 1 < len(matches)
         else _script_scene_end(screenplay, m.end()))
        for i, m in enumerate(matches)
    ]


def _script_scene_end(screenplay, start):
    tail = re.search(r"(?m)^## 연속성·복선 점검[ \t]*$", screenplay[start:])
    return start + tail.start() if tail else len(screenplay)


def generate_screenplay(answers, character_report, drama_design, genre, runtime, scene_count):
    items = require_complete(answers)
    if "# A · 캐릭터 분석 리포트" not in (character_report or ""):
        raise gr.Error("C단계에는 완료된 A 캐릭터 분석 보고서가 필요합니다.")
    if "# B · 선택한 장치 적용 결과" not in (drama_design or ""):
        raise gr.Error("C단계에는 B에서 선택한 장치로 재설계한 결과가 필요합니다.")
    if runtime is None or not 1 <= float(runtime) <= 120:
        raise gr.Error("러닝타임은 1~120분이어야 합니다.")
    count = int(scene_count)
    if not 3 <= count <= 12:
        raise gr.Error("씬 수는 3~12개여야 합니다.")
    prompt = (
        f"장르: {genre}\n목표 러닝타임: {float(runtime):g}분\n씬 수: 정확히 {count}개\n\n"
        "확인된 인터뷰(답변 내용 자체를 우선):\n" + interview_transcript(items)
        + "\n\nA 캐릭터 보고서(가설은 사실로 승격 금지):\n" + character_report
        + "\n\nB 사용자 선택 후 재설계:\n" + drama_design
        + "\n\n인물·사건·결과를 연결해 실제 촬영 가능한 한국어 시나리오를 작성하세요."
    )
    for draft in call_storylab_ai(SCREENPLAY_PROMPT, prompt, max_tokens=6500):
        yield draft


def revise_screenplay_scene(screenplay, scene_number, instruction, character_report, drama_design):
    script = (screenplay or "").strip()
    if not script:
        raise gr.Error("먼저 C 시나리오 초안을 생성해 주세요.")
    if not (instruction or "").strip():
        raise gr.Error("해당 씬에서 수정할 내용을 입력해 주세요.")
    spans = screenplay_scene_spans(script)
    if not spans:
        raise gr.Error("시나리오에서 '### SCENE 01' 형식의 씬 제목을 찾지 못했습니다.")
    number = int(scene_number)
    selected = [entry for entry in spans if entry[0] == number]
    if len(selected) != 1:
        raise gr.Error("해당 씬이 없거나 중복되어 있습니다. 씬 번호를 확인하세요.")
    _, start, end = selected[0]
    heading_match = SCENE_HEADING_RE.match(script, start)
    original = script[heading_match.end():end].strip()
    prompt = (
        f"A 캐릭터 보고서:\n{character_report}\n\n"
        f"B 승인된 극적 구조:\n{drama_design}\n\n"
        f"수정할 SCENE {number:02d} 원문:\n{original}\n\n"
        f"사용자 수정 지시:\n{instruction}\n\n"
        "씬 본문만 출력. 다른 씬은 프로그램이 원문 그대로 유지합니다."
    )
    final = ""
    for response in call_storylab_ai(REVISION_PROMPT, prompt, max_tokens=2500):
        final = response
    replacement = final.strip()
    if not replacement or SCENE_HEADING_RE.search(replacement) or re.search(
        r"(?m)^## ", replacement
    ):
        raise gr.Error("AI 응답에 다른 씬 제목이 섞였거나 비어 있습니다. 다시 시도해 주세요.")
    # Preserve every byte outside the selected scene to avoid unintended edits.
    return script[:heading_match.end()] + "\n" + replacement + "\n\n" + script[end:]


def export_screenplay(answers, character_report, drama_design, screenplay):
    items = require_complete(answers)
    if not (screenplay or "").strip():
        raise gr.Error("저장할 시나리오가 없습니다.")
    folder = Path("exports")
    folder.mkdir(exist_ok=True)
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = folder / f"screenplay_{suffix}.json"
    payload = {
        "schema_version": "0.5",
        "interview": [
            {"stage": INTERVIEW_STEPS[i][0], "answer": a} for i, a in enumerate(items)
        ],
        "character_report_A": character_report or "",
        "drama_design_B": drama_design or "",
        "screenplay_C": screenplay,
        "scene_count_detected": len(screenplay_scene_spans(screenplay)),
        "provenance_notice": "원본 기억과 AI 창작 설정을 별도로 확인해야 합니다.",
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(destination.resolve())


def create_demo():
    with gr.Blocks(title="StoryLab KR") as demo:
        gr.Markdown("# 🎬 StoryLab KR\n한국어 스토리 개발과 영상 제작용 스토리보드 초안을 한곳에서 만듭니다.")
        with gr.Tab("🎤 작가 인터뷰"):
            gr.Markdown("## 10문항 인물 탐색 → A 캐릭터 보고서 → B 드라마 닥터\n"
                        "한 번에 한 질문씩 진행합니다. 실제 기억이 불확실하면 그대로 적어 주세요. "
                        "A단계는 보고만 하고 승인을 요구하지 않습니다. "
                        "B단계는 극적 동력이 부족할 때 장치를 추천하고, 선택된 장치만 적용합니다.")
            interview_answers = gr.State([])
            with gr.Row():
                with gr.Column(scale=1):
                    question_view = gr.Markdown(interview_question(0, []))
                    interview_answer = gr.Textbox(label="내 답변", lines=4,
                        placeholder="실제 기억을 편하게 적어주세요. '기억나지 않아'도 가능.")
                    with gr.Row():
                        back_button = gr.Button("↩ 이전 질문")
                        next_button = gr.Button("답변하고 다음 질문", variant="primary")
                    interview_status = gr.Markdown("0/10문항 완료")
                    report_button = gr.Button("A · 캐릭터 분석 보고서 생성", variant="primary")
                    diagnose_button = gr.Button("B · 극적 구조 진단 및 장치 추천")
                    gr.Markdown("### B · 추천 장치 선택 후 재설계")
                    device_selection = gr.Dropdown(DRAMA_DEVICE_OPTIONS, value="추천안 B",
                                                   label="B 진단에 표시된 추천안 선택")
                    custom_device = gr.Textbox(label="직접 입력 장치 (선택 사항)", lines=2)
                    rewrite_button = gr.Button("B · 선택 장치로 재설계", variant="primary")
                    export_interview_btn = gr.Button("💾 A/B 보고서 JSON 저장")
                    interview_file = gr.File(label="인터뷰 프로젝트 파일")
                with gr.Column(scale=2):
                    gr.Markdown("### A · 캐릭터 분석 보고서")
                    character_view = gr.Markdown("")
                    gr.Markdown("### B · 극적 구조 진단")
                    diagnosis_view = gr.Markdown("")
                    gr.Markdown("### B · 선택한 장치 적용 결과")
                    rewrite_view = gr.Markdown("")
            next_button.click(interview_next,
                inputs=[interview_answer, interview_answers],
                outputs=[question_view, interview_answers, interview_answer, interview_status]).then(
                lambda: ("", "", "", ""), outputs=[character_view, diagnosis_view, rewrite_view, screenplay_view], queue=False)
            back_button.click(interview_back,
                inputs=interview_answers,
                outputs=[question_view, interview_answers, interview_answer, interview_status]).then(
                lambda: ("", "", "", ""), outputs=[character_view, diagnosis_view, rewrite_view, screenplay_view], queue=False)
            report_button.click(generate_character_report,
                inputs=interview_answers, outputs=character_view).then(
                lambda: ("", "", ""), outputs=[diagnosis_view, rewrite_view, screenplay_view], queue=False)
            diagnose_button.click(diagnose_drama,
                inputs=[interview_answers, character_view], outputs=diagnosis_view).then(
                lambda: ("", ""), outputs=[rewrite_view, screenplay_view], queue=False)
            rewrite_button.click(rewrite_drama,
                inputs=[interview_answers, character_view, diagnosis_view,
                        device_selection, custom_device], outputs=rewrite_view).then(
                lambda: "", outputs=screenplay_view, queue=False)
            export_interview_btn.click(export_interview,
                inputs=[interview_answers, character_view, diagnosis_view, rewrite_view],
                outputs=interview_file)
            gr.Markdown("## C · 시나리오 엔진\nB 재설계까지 완료한 뒤 씬별 시나리오를 생성합니다. "
                        "특정 씬을 지정해서 수정하면 다른 씬은 원문 그대로 유지합니다.")
            with gr.Row():
                screenplay_genre = gr.Dropdown(STYLES, value="현실 드라마", label="시나리오 장르")
                screenplay_runtime = gr.Number(value=10, minimum=1, maximum=120,
                                                label="목표 러닝타임(분)")
                screenplay_count = gr.Slider(3, 12, step=1, value=6, label="씬 수")
            screenplay_generate_btn = gr.Button("C · 씬별 시나리오 생성", variant="primary")
            screenplay_view = gr.Textbox(label="C · 시나리오 편집 가능 초안", lines=22,
                                         placeholder="B 설계를 완료한 뒤 생성하세요. ### SCENE 01 형식 유지.")
            with gr.Accordion("✍️ 지정 씬만 수정", open=False):
                screenplay_scene_no = gr.Number(value=1, minimum=1, maximum=12,
                                                 precision=0, label="수정할 씬 번호")
                screenplay_instruction = gr.Textbox(label="수정 방향", lines=3,
                    placeholder="예: 3번 씬의 대사를 줄이고 엄마의 행동으로 감정을 표현해줘")
                screenplay_revise_btn = gr.Button("선택 씬만 수정", variant="primary")
            screenplay_save_btn = gr.Button("💾 인터뷰·A·B·C 통합 JSON 저장")
            screenplay_file = gr.File(label="통합 시나리오 프로젝트")
            screenplay_generate_btn.click(generate_screenplay,
                inputs=[interview_answers, character_view, rewrite_view,
                        screenplay_genre, screenplay_runtime, screenplay_count],
                outputs=screenplay_view)
            screenplay_revise_btn.click(revise_screenplay_scene,
                inputs=[screenplay_view, screenplay_scene_no, screenplay_instruction,
                        character_view, rewrite_view], outputs=screenplay_view)
            screenplay_save_btn.click(export_screenplay,
                inputs=[interview_answers, character_view, rewrite_view, screenplay_view],
                outputs=screenplay_file)

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
