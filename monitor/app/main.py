import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from monitor.app.router import router

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
# Make sure monitor router logs show up
logging.getLogger("monitor.app.router").setLevel(logging.DEBUG)

app = FastAPI(title="Vibration Monitor")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)
app.mount("/static", StaticFiles(directory="monitor/app/static"), name="static")

@app.get("/")
def root():
    return FileResponse("monitor/app/static/index.html")