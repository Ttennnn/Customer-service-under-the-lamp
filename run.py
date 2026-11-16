"""Local development entry point.

PyCharm can run this file directly (green triangle) to start the FastAPI app
with auto-reload. It loads ``app.main:app`` through uvicorn, so the working
directory must be the repository root (where ``.env`` and ``knowledge/`` live).
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
