"""Domain errors. Messages are written for end users and say how to recover."""


class SyllabusLensError(Exception):
    """Base class for expected, user-reportable errors."""


class NotFoundError(SyllabusLensError):
    pass


class InvalidSubjectError(SyllabusLensError):
    """No subject, or an unknown subject, was supplied. Never falls back to 'all documents'."""


class CrossSubjectError(SyllabusLensError):
    """A document outside the selected subject was requested without explicit cross-subject mode."""


class IsolationViolation(SyllabusLensError):
    """Internal invariant failure: a section from a disallowed subject reached the corpus."""


class DocumentValidationError(SyllabusLensError):
    pass


class DuplicateDocumentError(DocumentValidationError):
    def __init__(self, message: str, existing_id: str):
        super().__init__(message)
        self.existing_id = existing_id


class ConfirmationError(SyllabusLensError):
    """A destructive action was attempted without matching confirmation."""
