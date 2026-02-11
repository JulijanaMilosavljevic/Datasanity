from .checks.imbalance import check_class_imbalance
from .checks.missing import check_missing_values
from .checks.constants import check_constant_columns
from .checks.id_columns import check_id_like_columns
from .checks.duplicates import check_duplicates
from .checks.leakage import check_target_leakage
from .checks.advice import generate_modeling_advice
from .report.generator import generate_html_report
from .checks.severity import compute_dataset_severity
from .checks.model_suggest import suggest_models
from .report.codegen import generate_training_code


class DataSanityReport:
    def __init__(self, results: dict):
        self.results = results

    def to_dict(self) -> dict:
        return self.results

    def to_html(self) -> str:
        return generate_html_report(self.results).html


def check_dataset(df, target: str) -> DataSanityReport:
    results = {
        "shape": df.shape,
        "imbalance": check_class_imbalance(df, target),
        "missing": check_missing_values(df),
        "constants": check_constant_columns(df),
        "id_columns": check_id_like_columns(df),
        "duplicates": check_duplicates(df),
        "leakage": check_target_leakage(df, target),
    }

    results["advice"] = generate_modeling_advice(results)
    results["severity"] = compute_dataset_severity(results)
    results["model_suggestion"] = suggest_models(df, target, results)
    results["code_snippet"] = generate_training_code(results["model_suggestion"])
    return DataSanityReport(results)
