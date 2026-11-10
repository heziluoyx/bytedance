"""中文Markdown与HTML报告的通用输出工具。"""
import html
import re

import numpy as np
import pandas as pd


def save_report(path, lines):
    """输出同源Markdown和本地可直接打开的HTML。"""
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    blocks, in_table = [], False
    for line in lines:
        if line.startswith("|"):
            if re.fullmatch(r"\|[-|: ]+\|", line):
                continue
            cells = [html.escape(cell.strip()) for cell in line.strip("|").split("|")]
            tag = "td" if in_table else "th"
            if not in_table:
                blocks.append("<table>")
                in_table = True
            blocks.append(
                "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"
            )
            continue
        if in_table:
            blocks.append("</table>")
            in_table = False
        if line.startswith("!["):
            match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
            blocks.append(
                f'<figure><img src="{html.escape(match[2])}"><figcaption>{html.escape(match[1])}</figcaption></figure>'
            )
        elif line.startswith("## "):
            blocks.append("<h2>" + html.escape(line[3:]) + "</h2>")
        elif line.startswith("# "):
            blocks.append("<h1>" + html.escape(line[2:]) + "</h1>")
        elif line:
            content = re.sub(
                r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html.escape(line)
            )
            content = re.sub(
                r"\[(.*?)\]\((https://[^ )]+)\)", r'<a href="\2">\1</a>', content
            )
            blocks.append("<p>" + content + "</p>")
    if in_table:
        blocks.append("</table>")
    path.with_suffix(".html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'
        + html.escape(path.name)
        + '</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.9 "Microsoft YaHei",sans-serif;color:#1d2730}h1,h2{color:#172f43}table{border-collapse:collapse;width:100%;font-size:14px;margin:24px 0}td,th{border:1px solid #ddd;padding:8px}th{background:#244b63;color:white}tr:nth-child(even){background:#f1f5f7}img{max-width:100%;height:auto}figure{margin:32px 0}figcaption{color:#51626d}a{color:#146095}</style><main>'
        + "\n".join(blocks)
        + "</main></html>",
        encoding="utf-8",
    )


def table(frame, columns):
    """列名到展示名的简单映射。"""
    result = ["| " + " | ".join(columns.values()) + " |", "|" + "---|" * len(columns)]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(
                ("是" if value else "否")
                if isinstance(value, (bool, np.bool_))
                else "不可估计"
                if pd.isna(value)
                else f"{value:.4f}"
                if isinstance(value, (float, np.floating))
                else str(value)
            )
        result.append("| " + " | ".join(values) + " |")
    return result + [""]
