"""Weather API routes — current conditions, trends, anomalies, monsoon analysis."""

import logging
from datetime import datetime, timedelta, date
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select, func, and_, desc
import pandas as pd

from models.database import SessionLocal
from models.weather_observation import WeatherObservation
from models.weather_anomaly import WeatherAnomalyRecord
from models.climate_normal import ClimateNormal
from scrapers.weather import INDIA_CITIES, CITY_BY_NAME, WeatherScraper
from agents.weather_anomaly import WeatherTrendAnalyzer
from config import OPENWEATHERMAP_API_KEY

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# ── Helper ───────────────────────────────────────────────────────────


def _parse_period(period: str) -> timedelta:
    """Parse period string like '7d', '30d', '1y', '5y' into timedelta."""
    period = period.strip().lower()
    if period.endswith("d"):
        return timedelta(days=int(period[:-1]))
    if period.endswith("y"):
        return timedelta(days=int(period[:-1]) * 365)
    if period.endswith("m"):
        return timedelta(days=int(period[:-1]) * 30)
    return timedelta(days=30)


def _load_observations(city: str, start: date, end: date) -> pd.DataFrame:
    """Load observations from Postgres into a DataFrame."""
    db = SessionLocal()
    try:
        rows = (
            db.query(WeatherObservation)
            .filter(
                WeatherObservation.city == city,
                WeatherObservation.date >= start,
                WeatherObservation.date <= end,
            )
            .order_by(WeatherObservation.date)
            .all()
        )
        if not rows:
            return pd.DataFrame()

        records = []
        for r in rows:
            records.append({
                "date": r.date,
                "temperature_max": r.temperature_max,
                "temperature_min": r.temperature_min,
                "temperature_mean": r.temperature_mean,
                "precipitation_sum": r.precipitation_sum,
                "wind_speed_max": r.wind_speed_max,
                "humidity_mean": r.humidity_mean,
                "soil_moisture_mean": r.soil_moisture_mean,
                "temp_max_zscore": r.temp_max_zscore,
                "temp_min_zscore": r.temp_min_zscore,
                "precip_zscore": r.precip_zscore,
            })
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df
    finally:
        db.close()


def _load_normals(city: str) -> pd.DataFrame:
    """Load climate normals from Postgres."""
    db = SessionLocal()
    try:
        rows = (
            db.query(ClimateNormal)
            .filter(ClimateNormal.city == city)
            .all()
        )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "month": r.month,
                "variable": r.variable,
                "mean": r.mean,
                "std": r.std,
                "p5": r.p5,
                "p25": r.p25,
                "p75": r.p75,
                "p95": r.p95,
            }
            for r in rows
        ])
    finally:
        db.close()


# ── Endpoints ────────────────────────────────────────────────────────


@router.get("/weather/live")
def get_live_weather(
    city: Optional[str] = Query(None, description="Filter to a single city (default: all 25)"),
):
    """Live current weather for all 25 monitored Indian cities via OpenWeatherMap.

    Data is fetched directly from OWM in real-time (updated every ~10 minutes).
    Unlike /weather/current which reads from the database, this endpoint always
    returns the freshest available conditions.
    """
    if not OPENWEATHERMAP_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="OpenWeatherMap API key not configured (set OPENWEATHERMAP_API_KEY in .env)"
        )

    scraper = WeatherScraper()

    if city:
        city_obj = CITY_BY_NAME.get(city)
        if not city_obj:
            raise HTTPException(status_code=404, detail=f"Unknown city: {city}")
        data = scraper.fetch_live_current(city_obj, OPENWEATHERMAP_API_KEY)
        if not data:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Failed to fetch live weather for {city}. "
                    "If your OWM key was just created, it can take up to 2 hours to activate. "
                    "Check https://openweathermap.org/faq#error401"
                )
            )
        return {"observations": [data], "count": 1, "source": "openweathermap"}

    observations = scraper.fetch_all_cities_live(OPENWEATHERMAP_API_KEY)
    if not observations:
        raise HTTPException(
            status_code=503,
            detail=(
                "OpenWeatherMap returned no data. "
                "New API keys take up to 2 hours to activate after creation. "
                "Check https://openweathermap.org/faq#error401"
            )
        )
    return {
        "observations": observations,
        "count": len(observations),
        "source": "openweathermap",
    }


@router.get("/weather/cities")
def list_cities():
    """List all monitored Indian cities."""
    return {
        "cities": [
            {"name": c.name, "lat": c.lat, "lon": c.lon, "zone": c.zone}
            for c in INDIA_CITIES
        ]
    }


@router.get("/weather/current")
def get_current_weather():
    """Latest observation + anomaly flags for all 25 cities."""
    db = SessionLocal()
    try:
        results = []
        for city_obj in INDIA_CITIES:
            city = city_obj.name
            latest = (
                db.query(WeatherObservation)
                .filter(WeatherObservation.city == city)
                .order_by(desc(WeatherObservation.date))
                .first()
            )
            if not latest:
                continue

            # Check for active anomalies
            active_anomalies = (
                db.query(WeatherAnomalyRecord)
                .filter(
                    WeatherAnomalyRecord.city == city,
                    WeatherAnomalyRecord.start_date <= latest.date,
                    (
                        (WeatherAnomalyRecord.end_date >= latest.date)
                        | (WeatherAnomalyRecord.end_date.is_(None))
                    ),
                )
                .all()
            )

            results.append({
                "city": city,
                "lat": city_obj.lat,
                "lon": city_obj.lon,
                "date": str(latest.date),
                "temperature_max": latest.temperature_max,
                "temperature_min": latest.temperature_min,
                "temperature_mean": latest.temperature_mean,
                "precipitation_sum": latest.precipitation_sum,
                "humidity_mean": latest.humidity_mean,
                "wind_speed_max": latest.wind_speed_max,
                "weather_code": latest.weather_code,
                "temp_max_zscore": latest.temp_max_zscore,
                "precip_zscore": latest.precip_zscore,
                "active_anomalies": [
                    {
                        "type": a.anomaly_type,
                        "severity": a.severity,
                        "start_date": str(a.start_date),
                        "peak_value": a.peak_value,
                    }
                    for a in active_anomalies
                ],
            })

        return {"observations": results, "count": len(results)}
    finally:
        db.close()


@router.get("/weather/trends")
def get_weather_trends(
    city: str = Query(..., description="City name"),
    variable: str = Query("temperature_max", description="Variable to analyze"),
    period: str = Query("1y", description="Time period: 7d, 30d, 1y, 5y"),
):
    """Time-series data for a city and variable with z-scores and normals."""
    if city not in CITY_BY_NAME:
        raise HTTPException(status_code=404, detail=f"Unknown city: {city}")

    delta = _parse_period(period)
    end = date.today()
    start = end - delta

    df = _load_observations(city, start, end)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {city} in the given period")

    normals = _load_normals(city)

    # Build normal reference line (monthly means)
    normal_lookup: dict[int, float] = {}
    if not normals.empty:
        var_normals = normals[normals["variable"] == variable]
        for _, row in var_normals.iterrows():
            normal_lookup[int(row["month"])] = float(row["mean"])

    z_col = {
        "temperature_max": "temp_max_zscore",
        "temperature_min": "temp_min_zscore",
        "precipitation_sum": "precip_zscore",
    }.get(variable)

    records = []
    for idx, row in df.iterrows():
        d = idx.date() if hasattr(idx, "date") else idx
        month = idx.month if hasattr(idx, "month") else 1
        records.append({
            "date": str(d),
            "value": row.get(variable),
            "normal": normal_lookup.get(month),
            "z_score": row.get(z_col) if z_col else None,
        })

    # Compute linear trend if enough data
    trend_info = None
    analyzer = WeatherTrendAnalyzer()
    if variable in df.columns:
        trend = analyzer.compute_annual_trend(df, variable)
        if trend:
            trend_info = {
                "slope_per_decade": trend.slope_per_decade,
                "r_squared": trend.r_squared,
                "p_value": trend.p_value,
                "direction": trend.direction,
            }

    return {
        "city": city,
        "variable": variable,
        "period": period,
        "data": records,
        "trend": trend_info,
    }


@router.get("/weather/anomalies")
def get_anomalies(
    anomaly_type: Optional[str] = Query(None, description="Filter by type"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    city: Optional[str] = Query(None, description="Filter by city"),
    days: int = Query(30, description="Look back N days"),
):
    """Query detected weather anomalies."""
    db = SessionLocal()
    try:
        query = db.query(WeatherAnomalyRecord)

        cutoff = date.today() - timedelta(days=days)
        query = query.filter(WeatherAnomalyRecord.start_date >= cutoff)

        if anomaly_type:
            query = query.filter(WeatherAnomalyRecord.anomaly_type == anomaly_type)
        if severity:
            query = query.filter(WeatherAnomalyRecord.severity == severity)
        if city:
            query = query.filter(WeatherAnomalyRecord.city == city)

        query = query.order_by(desc(WeatherAnomalyRecord.start_date))
        rows = query.limit(200).all()

        return {
            "anomalies": [
                {
                    "id": r.id,
                    "city": r.city,
                    "type": r.anomaly_type,
                    "severity": r.severity,
                    "start_date": str(r.start_date),
                    "end_date": str(r.end_date) if r.end_date else None,
                    "peak_value": r.peak_value,
                    "z_score": r.z_score,
                    "description": r.description,
                    "detected_at": r.detected_at.isoformat() if r.detected_at else None,
                }
                for r in rows
            ],
            "count": len(rows),
        }
    finally:
        db.close()


@router.get("/weather/monsoon")
def get_monsoon_analysis(
    year: int = Query(default=None, description="Year to analyze (default: current)"),
    city: Optional[str] = Query(None, description="City (default: all cities)"),
):
    """Monsoon season (Jun-Sep) analysis."""
    if year is None:
        year = date.today().year

    cities_to_check = [city] if city else [c.name for c in INDIA_CITIES]
    analyzer = WeatherTrendAnalyzer()
    results = []

    for c in cities_to_check:
        if c not in CITY_BY_NAME:
            continue

        df = _load_observations(c, date(year, 1, 1), date(year, 12, 31))
        if df.empty:
            continue

        normals = _load_normals(c)
        if normals.empty:
            continue

        analysis = analyzer.analyze_monsoon(df, normals, c, year)
        if analysis:
            results.append({
                "city": analysis.city,
                "year": analysis.year,
                "total_rainfall_mm": analysis.total_rainfall_mm,
                "normal_rainfall_mm": analysis.normal_rainfall_mm,
                "deficit_pct": analysis.deficit_pct,
                "rain_days": analysis.rain_days,
                "heavy_rain_days": analysis.heavy_rain_days,
                "max_single_day_mm": analysis.max_single_day_mm,
            })

    return {"year": year, "analyses": results, "count": len(results)}


@router.get("/weather/climate-trends")
def get_climate_trends(
    city: str = Query(..., description="City name"),
    variable: str = Query("temperature_max", description="Variable to analyze"),
):
    """Long-term climate trend with extreme event frequency."""
    if city not in CITY_BY_NAME:
        raise HTTPException(status_code=404, detail=f"Unknown city: {city}")

    # Load all available data
    db = SessionLocal()
    try:
        oldest = (
            db.query(func.min(WeatherObservation.date))
            .filter(WeatherObservation.city == city)
            .scalar()
        )
        if not oldest:
            raise HTTPException(status_code=404, detail=f"No data for {city}")
    finally:
        db.close()

    df = _load_observations(city, oldest, date.today())
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {city}")

    analyzer = WeatherTrendAnalyzer()

    # Annual trend
    trend = analyzer.compute_annual_trend(df, variable)
    trend_info = None
    if trend:
        trend_info = {
            "slope_per_decade": trend.slope_per_decade,
            "r_squared": trend.r_squared,
            "p_value": trend.p_value,
            "direction": trend.direction,
            "start_year": trend.start_year,
            "end_year": trend.end_year,
        }

    # Annual means for charting
    if variable in df.columns:
        df_var = df[[variable]].dropna().copy()
        df_var["year"] = df_var.index.year
        annual = df_var.groupby("year")[variable].mean()
        annual_data = [
            {"year": int(y), "mean": round(float(v), 2)}
            for y, v in annual.items()
        ]
    else:
        annual_data = []

    # Extreme day counts (e.g., days above 45°C for temperature_max)
    extreme_threshold = {
        "temperature_max": 45.0,
        "temperature_min": 4.0,
        "precipitation_sum": 64.5,
    }.get(variable)

    extreme_data = []
    if extreme_threshold and variable in df.columns:
        above = variable != "temperature_min"
        freq = analyzer.compute_extreme_frequency(df, variable, extreme_threshold, above=above)
        extreme_data = [
            {"year": int(row["year"]), "count": int(row["count"])}
            for _, row in freq.iterrows()
        ]

    return {
        "city": city,
        "variable": variable,
        "trend": trend_info,
        "annual_means": annual_data,
        "extreme_days": extreme_data,
    }
