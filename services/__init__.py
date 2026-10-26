"""
Services package for SyllabusLens.
Exposes document extraction, analysis pipeline, coverage analytics, evaluation, and exports.
"""

from services.document_service import DocumentService, ExtractedDocument
from services.analysis_service import AnalysisService, CompleteAnalysisResult
from services.coverage_service import CoverageService, AggregateCoverageMetrics, SectionCoverageStat
from services.evaluation_service import EvaluationService, MethodEvaluationReport, ConfusionMatrixMetrics
from services.export_service import ExportService

__all__ = [
    "DocumentService",
    "ExtractedDocument",
    "AnalysisService",
    "CompleteAnalysisResult",
    "CoverageService",
    "AggregateCoverageMetrics",
    "SectionCoverageStat",
    "EvaluationService",
    "MethodEvaluationReport",
    "ConfusionMatrixMetrics",
    "ExportService"
]
