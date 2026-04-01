from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.v1.router import router as v1_router
from app.api.error_handlers import register_error_handlers


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    # e.g. connect to DB, load ML models, etc.
    yield
    # --- Shutdown ---
    # e.g. close DB connections, release resources


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global error handlers
    register_error_handlers(app)

    # Mount versioned API
    app.include_router(v1_router, prefix=settings.API_V1_PREFIX)

    from starlette.middleware.base import BaseHTTPMiddleware
    class DebugMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            print(f"DEBUG [Middleware]: Received {request.method} {request.url}")
            print(f"DEBUG [Middleware]: Headers: Origin={request.headers.get('origin', 'N/A')}, "
                  f"Access-Control-Request-Method={request.headers.get('access-control-request-method', 'N/A')}")
            response = await call_next(request)
            print(f"DEBUG [Middleware]: Response Status: {response.status_code}")
            return response

    app.add_middleware(DebugMiddleware)

    return app


app = create_app()
