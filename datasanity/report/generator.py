from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape


@dataclass
class HtmlReport:
    html: str

    def save(self, filepath: str | Path = "datasanity_report.html") -> Path:
        path = Path(filepath)
        path.write_text(self.html, encoding="utf-8")
        return path


def generate_html_report(results: dict) -> HtmlReport:
    templates_dir = Path(__file__).parent / "templates"
    project_root = Path(__file__).resolve().parents[2]
    css_path = project_root / "assets" / "styles" / "report.css"

    css = ""
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")

    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html")

    html = template.render(results=results)

    # 🔥 Ubaci CSS ručno (nema Jinja u style tagu)
    html = html.replace("/*__EMBEDDED_CSS__*/", css)

    return HtmlReport(html=html)
