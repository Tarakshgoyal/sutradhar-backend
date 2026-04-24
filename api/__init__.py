from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api.routes import (
    graph, visualization, chat, live_feed, reports, weather, news, signals,
    timeline, language, briefs, disinfo, scenarios, predictions, admin,
)
from api.routes import auth


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    graph.driver.close()
    chat.close_agent()
    timeline.close_agent()


app = FastAPI(title="Intelligence Graph", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(graph.router)
app.include_router(visualization.router)
app.include_router(chat.router)
app.include_router(live_feed.router)
app.include_router(reports.router)
app.include_router(weather.router)
app.include_router(news.router)
app.include_router(signals.router)
app.include_router(timeline.router)
app.include_router(language.router)
app.include_router(briefs.router)
app.include_router(disinfo.router)
app.include_router(scenarios.router)
app.include_router(predictions.router)
app.include_router(admin.router)
app.include_router(auth.router)
