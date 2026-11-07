"""Offline smoke tests: no Hugging Face token or network calls required."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app


ANSWERS = [
    "어린 시절 비 오는 날",
    "라디오 음악이 주는 행복",
    "방 창문 옆",
    "혼자",
    "일기를 쓰고 있었어",
    "하루의 감상을 적었어",
    "엄마는 자주 혼냈어",
    "혼내고 감을 챙겨줬던 것 같아",
    "맨날 나한테만 뭐라고 해! 토라졌다가 풀렸어",
    "그 시절을 소중히 생각해",
]
CHAR = "# A · 캐릭터 분석 리포트\n인물 A, 엄마"
DIAG = (
    "# B · 극적 구조 진단\n"
    "### 추천안 A\n시간 제한을 추가\n"
    "### 추천안 B\n일기장 속 엄마의 메모 발견\n"
    "### 추천안 C\n현재의 관계에서 선택\n"
    "## 가장 적합한 추천과 그 이유\nB"
)
REWRITE = "# B · 선택한 장치 적용 결과\n메모를 발견한다"
SCRIPT = (
    "# C · 시나리오 초안\n"
    "## 씬별 시나리오\n"
    "### SCENE 01\nINT. 방 - 낮\n아이가 일기를 쓴다.\n\n"
    "### SCENE 02\nINT. 방 - 밤\n엄마가 방문을 닫는다.\n\n"
    "## 연속성·복선 점검\n메모를 회수한다.\n"
)


class StoryLabTests(unittest.TestCase):
    def test_interview_progress_back_and_limit(self):
        history = []
        for idx, value in enumerate(ANSWERS):
            question, history, cleared, progress = app.interview_next(value, history)
            self.assertEqual(len(history), idx + 1)
            self.assertEqual(cleared, "")
        self.assertIn("10/10", question)
        self.assertEqual(len(app.require_complete(history)), 10)
        question, history, previous, status = app.interview_back(history)
        self.assertEqual(len(history), 9)
        self.assertEqual(previous, ANSWERS[-1])
        self.assertIn("10/10", question)

    def test_incomplete_interview_rejected(self):
        with self.assertRaises(Exception):
            list(app.generate_character_report(ANSWERS[:9]))

    def test_a_report_uses_answers(self):
        requests = []
        def fake_ai(system, prompt, max_tokens=3500):
            requests.append(prompt)
            yield CHAR
        with patch.object(app, "call_storylab_ai", fake_ai):
            self.assertEqual(list(app.generate_character_report(ANSWERS))[-1], CHAR)
        self.assertIn("방 창문 옆", requests[0])

    def test_b_uses_exact_recommendation_not_static_option(self):
        requests = []
        def fake_ai(system, prompt, max_tokens=3500):
            requests.append(prompt)
            yield REWRITE
        with patch.object(app, "call_storylab_ai", fake_ai):
            self.assertEqual(list(app.rewrite_drama(
                ANSWERS, CHAR, DIAG, "추천안 B", ""
            ))[-1], REWRITE)
        self.assertIn("일기장 속 엄마의 메모 발견", requests[0])
        self.assertNotIn("선택한 장치는 AI 창작 설정이며 기억의 사실로 간주하면 안 됩니다.\n\n현재의 관계에서 선택", requests[0])

    def test_b_missing_recommendation_rejected(self):
        with self.assertRaises(Exception):
            list(app.rewrite_drama(ANSWERS, CHAR, "# B · 극적 구조 진단", "추천안 B", ""))

    def test_c_preconditions(self):
        with self.assertRaises(Exception):
            list(app.generate_screenplay(ANSWERS, "", REWRITE, "현실 드라마", 10, 5))
        with self.assertRaises(Exception):
            list(app.generate_screenplay(ANSWERS, CHAR, "", "현실 드라마", 10, 5))

    def test_c_scene_revision_preserves_other_scenes(self):
        def fake_ai(system, prompt, max_tokens=3500):
            yield "INT. 방 - 밤\n엄마가 조용히 감을 놓고 나간다."
        with patch.object(app, "call_storylab_ai", fake_ai):
            result = app.revise_screenplay_scene(SCRIPT, 2, "대사를 행동으로", CHAR, REWRITE)
        self.assertIn("엄마가 조용히 감을 놓고 나간다", result)
        self.assertIn("아이가 일기를 쓴다.", result)
        self.assertIn("## 연속성·복선 점검\n메모를 회수한다.", result)
        self.assertNotIn("엄마가 방문을 닫는다", result)
        self.assertEqual(len(app.screenplay_scene_spans(result)), 2)

    def test_scene_heading_variants(self):
        text = ("### SCENE 1 — 도입\n한 장면\n\n"
                "### SCENE 02: 발견\n다음 장면\n"
                "## 연속성·복선 점검\n검토")
        spans = app.screenplay_scene_spans(text)
        self.assertEqual([number for number, _, _ in spans], [1, 2])

    def test_rainy_frequency_end_to_end_mock(self):
        """Ten interview answers -> A -> B diagnosis -> selected reveal -> C draft."""
        answers = [
            "어릴 때 아련했던 행복한 기억",
            "라디오에서 흘러나오던 음악이 행복감을 줬어",
            "비가 한 방울씩 떨어지는 소리가 들리던 내 방 창문 옆",
            "혼자였어",
            "일기를 쓰고 있었어",
            "정확한 글은 기억나지 않지만 충실했던 하루와 엄마에 대한 감상",
            "맨날 혼났지만 어딘가 따뜻한 기억",
            "혼낸 뒤 감 같은 것을 챙겨줬던 것 같아 (불확실)",
            "맨날 나한테만 뭐라고 해! 토라졌다가 이내 수그러들었어",
            "그 시절을 소중히 생각해",
        ]
        calls = []
        report = "# A · 캐릭터 분석 리포트\\n어린 나, 엄마, 현재의 나"
        diagnosis = (
            "# B · 극적 구조 진단\\n### 추천안 A\\n시간 제한\\n"
            "### 추천안 B\\n일기장 속 엄마의 메모 발견\\n"
            "### 추천안 C\\n현재 관계\\n"
            "## 가장 적합한 추천과 그 이유\\nB"
        )
        rewrite = "# B · 선택한 장치 적용 결과\\n메모로 감정 재해석"
        script = (
            "# C · 시나리오 초안\\n## 씬별 시나리오\\n"
            "### SCENE 01\\n일기를 펼친다\\n"
            "### SCENE 02\\n엄마의 메모를 발견한다\\n"
            "### SCENE 03\\n새 일기를 쓴다\\n"
            "## 연속성·복선 점검\\n감과 메모의 회수"
        )
        responses = [report, diagnosis, rewrite, script]

        def fake_ai(system, prompt, max_tokens=3500):
            calls.append((system, prompt))
            yield responses[len(calls) - 1]

        with patch.object(app, "call_storylab_ai", fake_ai):
            a = list(app.generate_character_report(answers))[-1]
            b = list(app.diagnose_drama(answers, a))[-1]
            selected = list(app.rewrite_drama(
                answers, a, b, "추천안 B", ""
            ))[-1]
            c = list(app.generate_screenplay(
                answers, a, selected, "현실 드라마", 10, 3
            ))[-1]
        self.assertEqual(len(calls), 4)
        self.assertIn("비가 한 방울씩", calls[0][1])
        self.assertIn(report, calls[1][1])
        self.assertIn("일기장 속 엄마의 메모 발견", calls[2][1])
        self.assertIn(rewrite, calls[3][1])
        self.assertIn("그 시절을 소중히 생각해", calls[3][1])
        self.assertEqual(len(app.screenplay_scene_spans(c)), 3)

    def test_gradio_ui_constructs_without_ai(self):
        demo = app.create_demo()
        self.assertIsNotNone(demo)
        self.assertGreater(len(demo.config.get("dependencies", [])), 10)

    def test_export_schema_and_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(app, "Path", wraps=Path):
                # Avoid writing data into user project during the test.
                with patch("app.datetime") as mocked_date:
                    mocked_date.now.return_value.strftime.return_value = "20261107_000000_000000"
                    current = Path.cwd()
                    import os
                    try:
                        os.chdir(tmp)
                        filename = app.export_screenplay(ANSWERS, CHAR, REWRITE, SCRIPT)
                        self.assertTrue(Path(filename).exists())
                        import json
                        data = json.loads(Path(filename).read_text(encoding="utf-8"))
                        self.assertEqual(len(data["interview"]), 10)
                        self.assertEqual(data["scene_count_detected"], 2)
                    finally:
                        os.chdir(current)


if __name__ == "__main__":
    unittest.main()
