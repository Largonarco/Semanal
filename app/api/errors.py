from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class APIError(Exception):
    def __init__(
        self,
        status_code: int,
        title: str,
        detail: str | None = None,
        type_: str = "about:blank",
    ) -> None:
        self.status_code = status_code
        self.title = title
        self.detail = detail
        self.type = type_
        super().__init__(detail or title)


def _problem(
    status: int, title: str, detail: str | None = None, type_: str = "about:blank"
) -> JSONResponse:
    body = {"type": type_, "title": title, "status": status}
    if detail is not None:
        body["detail"] = detail
    return JSONResponse(
        status_code=status, content=body, media_type="application/problem+json"
    )


async def _api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    return _problem(exc.status_code, exc.title, exc.detail, exc.type)


async def _http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    title = exc.detail if isinstance(exc.detail, str) else "HTTP error"
    return _problem(exc.status_code, title)


async def _validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return _problem(
        422,
        title="Request validation failed",
        detail=str(exc.errors()),
    )


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return _problem(500, title="Internal server error", detail=str(exc))


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(APIError, _api_error_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)
