# -*- coding: utf-8 -*-
"""Build the furigana-annotated speaking script + Q&A page (2026-10-15 progress report).
Run with review-ai/.venv/bin/python (needs fugashi + unidic-lite)."""
import html
import re
import sys
from pathlib import Path

import fugashi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from script_content import PHRASES, QA, QA_TRANSLATION, SCRIPT, TRANSLATION  # noqa: E402

TAGGER = fugashi.Tagger()
KANJI = re.compile(r"[㐀-鿿々〆ヶ]")
# surface -> hiragana reading, for words UniDic reads wrongly in this context
OVERRIDES = {
    "日本": "にほん", "何": "なに", "方": "ほう", "1つ": "ひとつ", "2つ": "ふたつ", "3つ": "みっつ",
    "6つ": "むっつ", "1つ目": "ひとつめ", "2つ目": "ふたつめ", "3つ目": "みっつめ",
    "4つ目": "よっつめ", "5つ目": "いつつめ", "上": "うえ", "下": "した", "私": "わたし",
    # phrases UniDic splits badly; matched before tokenizing
    "主構成": "しゅこうせい", "主表": "しゅひょう", "主実験": "しゅじっけん", "主結果": "しゅけっか",
    "除けて": "のぞけて", "数分": "すうふん", "提出日": "ていしゅつび",
    "観点別": "かんてんべつ", "人手": "ひとで", "頭打ち": "あたまうち", "上乗せ": "うわのせ", "内挿": "ないそう",
    "締めて": "しめて", "前面": "ぜんめん", "総合評価点": "そうごうひょうかてん", "評価点": "ひょうかてん",
    "選び方": "えらびかた", "規約上": "きやくじょう",
}
SEEN = {}


def hira(s):
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def ruby_token(surface, reading):
    if not KANJI.search(surface) or not reading or reading == "*":
        return html.escape(surface)
    reading = hira(reading)
    SEEN.setdefault(surface, reading)
    pre, suf = "", ""
    s, r = surface, reading
    while s and r and s[-1] == r[-1] and not KANJI.match(s[-1]):
        suf, s, r = s[-1] + suf, s[:-1], r[:-1]
    while s and r and s[0] == r[0] and not KANJI.match(s[0]):
        pre, s, r = pre + s[0], s[1:], r[1:]
    if not s or not r:
        return html.escape(surface)
    return f"{html.escape(pre)}<ruby>{html.escape(s)}<rt>{html.escape(r)}</rt></ruby>{html.escape(suf)}"


def ruby(text):
    # keep numbered-counter overrides ("1つ目", "2つ") intact before tokenizing
    out, i = [], 0
    pattern = re.compile("|".join(sorted((re.escape(k) for k in OVERRIDES if len(k) > 1), key=len, reverse=True)))
    for m in pattern.finditer(text):
        out.append(_ruby_plain(text[i:m.start()]))
        out.append(ruby_token(m.group(0), OVERRIDES[m.group(0)]))
        i = m.end()
    out.append(_ruby_plain(text[i:]))
    return "".join(out)


def _ruby_plain(text):
    parts = []
    for w in TAGGER(text):
        surf = w.surface
        reading = OVERRIDES.get(surf) or (w.feature.kana if w.feature.kana else None)
        # whitespace UniDic drops between tokens
        parts.append(html.escape(w.white_space) + ruby_token(surf, reading))
    return "".join(parts)


def mmss(sec):
    return f"{sec // 60}:{sec % 60:02d}"


def build():
    total = sum(s[2] for s in SCRIPT)
    script_html, cum = [], 0
    for page, title, sec, paras, zh in SCRIPT:
        cum += sec
        body = []
        for p, tr in zip(paras, TRANSLATION[page]):
            m = re.match(r"【(.+?)】(.*)", p)
            if m:
                body.append(f'<p class="ja"><span class="cue">{html.escape(m.group(1))}</span>{ruby(m.group(2))}</p>')
            else:
                body.append(f'<p class="ja">{ruby(p)}</p>')
            body.append(f'<p class="tr">{html.escape(tr)}</p>')
        timing = (f'<span class="t">{mmss(sec)}</span><span class="cum">{ruby("累計")} {mmss(cum)}</span>'
                  if sec else f'<span class="t">{ruby("対話")}</span><span class="cum">{ruby("時間は先生次第")}</span>')
        script_html.append(f"""
<section class="slide" id="p{page}">
  <header class="slide-head">
    <span class="pg">p.{page}</span>
    <h3>{ruby(title)}</h3>
    <span class="timing">{timing}</span>
  </header>
  {''.join(body)}
  <p class="zh">{html.escape(zh)}</p>
</section>""")

    cats, qa_html = [], []
    for n, (cat, lik, q, qzh, ans, azh) in enumerate(QA, 1):
        if cat not in cats:
            cats.append(cat)
        qa_html.append(f"""
<article class="qa" data-cat="{html.escape(cat)}" data-lik="{lik}" id="q{n}">
  <header class="qa-head">
    <span class="qn">Q{n}</span>
    <span class="lik lik-{ {'高': 'hi', '中': 'mid', '低': 'lo'}[lik] }">{ruby("可能性")} {LIK[lik]}</span>
    <span class="qcat">{ruby(cat)}</span>
  </header>
  <p class="q ja">{ruby(q)}</p>
  <p class="zh qzh">{html.escape(qzh)}</p>
  <div class="ans">
    <span class="ans-label">{ruby("回答例")}</span>
    {''.join(f'<p class="ja">{ruby(a)}</p><p class="tr">{html.escape(t)}</p>' for a, t in zip(ans, QA_TRANSLATION[n - 1]))}
    <p class="zh">{html.escape(azh)}</p>
  </div>
</article>""")

    chips = "".join(f'<button type="button" class="chip" data-filter="{html.escape(c)}" id="f-{i}">{ruby(c)}</button>'
                    for i, c in enumerate(cats))
    phrases = "".join(f"""
<div class="phrase">
  <span class="when">{ruby(w)}</span>
  <p class="ja">{ruby(p)}</p>
  <p class="zh">{html.escape(zh)}</p>
</div>""" for w, p, zh in PHRASES)

    n_hi = sum(1 for q in QA if q[1] == "高")
    page = TEMPLATE
    for k, v in dict(total=mmss(total), n_slides=str(len(SCRIPT)), n_q=str(len(QA)), n_hi=str(n_hi),
                     script="".join(script_html), qa="".join(qa_html), chips=chips, phrases=phrases).items():
        page = page.replace("@@" + k + "@@", v)
    (HERE / "meeting_script.html").write_text(page, encoding="utf-8")
    (HERE / "furigana_check.tsv").write_text(
        "".join(f"{k}\t{v}\n" for k, v in sorted(SEEN.items())), encoding="utf-8")
    print("ok", len(SEEN), "annotated words")


LIK = {k: ruby_token(k, r) for k, r in (("高", "こう"), ("中", "ちゅう"), ("低", "てい"))}  # likelihood labels


def furigana_template(t):
    """Furigana for the fixed Japanese text of the page (header, navigation, section intros)."""
    for s in ("研究進捗報告 ・ 2026年10月15日", "報告の台本と想定問答", "台本", "想定問答", "困ったときの一言",
              "回答を隠す", "発表の台本"):
        t = t.replace(f">{s}<", f">{ruby(s)}<")
    for a, b in (
        ("<span>台本 <b>", f"<span>{ruby('台本')} <b>"),
        ("<span>話す時間の目安 <b>", f"<span>{ruby('話す時間の目安')} <b>"),
        ("</b>（p.12 の相談は別）</span>", f"</b>{ruby('（p.12 の相談は別）')}</span>"),
        ("<span>想定問答 <b>@@n_q@@ 問</b>（可能性 高 @@n_hi@@ 問）</span>",
         f"<span>{ruby('想定問答')} <b>@@n_q@@ {ruby('問')}</b>（{ruby('可能性')} {LIK['高']} @@n_hi@@ {ruby('問')}）</span>"),
        (">可能性 高のみ</button>", f">{ruby('可能性')} {LIK['高']}{ruby('のみ')}</button>"),
    ):
        assert t.count(a) == 1, a
        t = t.replace(a, b)
    for body in [x.split("</p>")[0] for x in t.split('<p class="lede">')[1:]]:  # section intros
        t = t.replace(f'<p class="lede">{body}</p>', f'<p class="lede">{ruby(body)}</p>')
    return t


TEMPLATE = furigana_template((HERE / "template.html").read_text(encoding="utf-8"))

if __name__ == "__main__":
    build()


def build_md():
    """Markdown twin of the page. Furigana kept as inline <ruby> HTML."""
    L = [f"# {ruby('進捗報告の台本と想定問答')}（2026年10月15日）", "",
         f"> {ruby('ふりがなは')} `<ruby>` {ruby('タグで入っています。VSCode のプレビューや GitHub では漢字の上に表示されます。')}", "",
         f"## {ruby('発表の台本')}", ""]
    cum = 0
    for page, title, sec, paras, zh in SCRIPT:
        cum += sec
        t = f"{mmss(sec)}（{ruby('累計')} {mmss(cum)}）" if sec else ruby("対話（時間は先生次第）")
        L += [f"### p.{page} {ruby(title)}", "", f"*{t}*", ""]
        for p, tr in zip(paras, TRANSLATION[page]):
            m = re.match(r"【(.+?)】(.*)", p)
            L += [f"**{m.group(1)}** {ruby(m.group(2))}" if m else ruby(p), "", f"*中文：{tr}*", ""]
        L += [f"> 要点：{zh}", ""]
    L += ["---", "", f"## {ruby('想定問答')}", ""]
    cat = None
    for n, (c, lik, q, qzh, ans, azh) in enumerate(QA, 1):
        if c != cat:
            L += [f"### {ruby(c)}", ""]
            cat = c
        L += [f"**Q{n}**（{ruby('可能性')} {LIK[lik]}）{ruby(q)}", "", f"> {qzh}", "", f"{ruby('回答例')}："]
        L += [""] + [f"{ruby(a)}\n\n*中文：{t}*\n" for a, t in zip(ans, QA_TRANSLATION[n - 1])] + [f"> 要点：{azh}", ""]
    L += ["---", "", f"## {ruby('困ったときの一言')}", ""]
    for w, p, zh in PHRASES:
        L += [f"- **{ruby(w)}**：{ruby(p)}（{zh}）"]
    (HERE / "progress_script_2026-10-15.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("md ok")


if __name__ == "__main__":
    build_md()
