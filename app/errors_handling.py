from fastapi import HTTPException
from typing import Any, Dict, Optional

class BaseAPIException(HTTPException):
    """الفئة الأساسية اللي هتورث منها كل الأخطاء في المشروع"""
    def __init__(
        self, 
        status_code: int, 
        detail: str, 
        error_code: str, 
        headers: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code

class DataBaseConnectionError(BaseAPIException):
    """خطأ في الاتصال بقاعدة بيانات PostgreSQL"""
    def __init__(self, detail: str = "تعذر الاتصال بقاعدة البيانات"):
        super().__init__(
            status_code=503, # Service Unavailable
            detail=detail,
            error_code="DB_CONNECTION_ERROR"
        )

class AIProcessingError(BaseAPIException):
    """خطأ ناتج من كود زميلتك (Gemini أو Embeddings)"""
    def __init__(self, detail: str = "حدث خطأ أثناء تحليل البيانات بواسطة الذكاء الاصطناعي"):
        super().__init__(
            status_code=500, # Internal Server Error
            detail=detail,
            error_code="AI_PROCESSING_ERROR"
        )

class InsufficientTokensError(BaseAPIException):
    """خطأ في حالة نفاد رصيد التوكينز (الـ 10 دولار)"""
    def __init__(self, detail: str = "لقد وصلت للحد الأقصى المسموح به من التوكينز"):
        super().__init__(
            status_code=402, # Payment Required
            detail=detail,
            error_code="INSUFFICIENT_TOKENS"
        )

class InvalidInputDataError(BaseAPIException):
    """خطأ إذا كان النص المرسل من فلاتر فارغ أو غير صالح"""
    def __init__(self, detail: str = "البيانات المرسلة غير صالحة أو فارغة"):
        super().__init__(
            status_code=400, # Bad Request
            detail=detail,
            error_code="INVALID_INPUT_DATA"
        )