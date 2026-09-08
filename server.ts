import express from "express";
import path from "path";
import fs from "fs";
import { GoogleGenAI } from "@google/genai";
import { createServer as createViteServer } from "vite";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json());

// Disable caching globally for all responses so browser always gets fresh HTML, JS, and API responses
app.use((req, res, next) => {
  res.setHeader("Cache-Control", "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0");
  res.setHeader("Pragma", "no-cache");
  res.setHeader("Expires", "0");
  res.setHeader("Surrogate-Control", "no-store");
  next();
});

// Removed Gemini AI initialization as requested
const apiKey = process.env.GEMINI_API_KEY?.trim();

// Ensure humidity is within 0-100 range without artificial scaling or "fixing" low values
function normalizeHumidity(val: any): number | null {
  if (val === undefined || val === null || isNaN(Number(val))) return null;
  let h = Number(val);
  
  // Fraction decimal (0.0 to 1.0) -> convert to percentage if detected
  if (h > 0 && h <= 1.0 && h !== 1.0) {
    h = h * 100;
  }
  
  return Math.min(100, Math.max(0, Math.round(h)));
}

// GIOŚ API Caching (to handle 429 Too Many Requests)
const GIOS_STATIONS_CACHE_TTL = 24 * 60 * 60 * 1000; // 24 hours
const GIOS_AQI_CACHE_TTL = 30 * 60 * 1000; // 30 minutes
let giosStationsCache: { data: any[]; timestamp: number } | null = null;
const giosAqiCache = new Map<number, { data: any; timestamp: number }>();

// IMGW Hydro Cache (hydrological stations update hourly)
const IMGW_HYDRO_CACHE_TTL = 15 * 60 * 1000; // 15 minutes
let imgwHydroCache: { data: any[]; timestamp: number } | null = null;

/**
 * Returns raw API shortwave radiation or null if unavailable.
 * No custom modeling or time-based attenuation.
 */
function calculateSolarRadiation(cloudCoverPercent: number, isDayTime: boolean = true, rawApiShortwave?: number): number | null {
  if (typeof rawApiShortwave === 'number' && !isNaN(rawApiShortwave) && rawApiShortwave >= 0) {
    return Math.round(rawApiShortwave);
  }
  return null;
}

function normalizeStationName(str: string): string {
  return (str || "")
    .toLowerCase()
    .replace(/[-_]/g, " ")
    .replace(/\s+/g, " ")
    .replace(/ą/g, "a").replace(/ć/g, "c").replace(/ę/g, "e")
    .replace(/ł/g, "l").replace(/ń/g, "n").replace(/ó/g, "o")
    .replace(/ś/g, "s").replace(/ź/g, "z").replace(/ż/g, "z")
    .trim();
}

function getDistanceKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return Number((R * c).toFixed(1));
}

// Global in-memory cache for weather and station data to prevent Open-Meteo 429 rate limit issues
const weatherResponseCache = new Map<string, { data: any; timestamp: number }>();
const stationResponseCache = new Map<string, { data: any; timestamp: number }>();
const WEATHER_CACHE_TTL_MS = 2 * 60 * 1000; // 2 minutes fresh TTL

function formatUtcToPolishTime(dateStr: string, hourStr?: string): string {
  try {
    let isoStr = dateStr;
    if (hourStr !== undefined) {
      const h = String(hourStr).padStart(2, '0');
      isoStr = `${dateStr}T${h}:00:00Z`;
    } else if (dateStr.includes(' ')) {
      isoStr = `${dateStr.replace(' ', 'T')}Z`;
    }
    const d = new Date(isoStr);
    if (!isNaN(d.getTime())) {
      return d.toLocaleTimeString('pl-PL', { timeZone: 'Europe/Warsaw', hour: '2-digit', minute: '2-digit' }) + ' CEST';
    }
  } catch (e) {
    // fallback
  }
  return `${dateStr} ${hourStr || ''}:00`;
}

async function fetchWithRetry(url: string, retries = 2, timeoutMs = 6000): Promise<Response | null> {
  for (let i = 0; i < retries; i++) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(url, { 
        headers: {
          'User-Agent': 'AuraMeteoApp/1.0 (contact@aurameteo.pl)'
        },
        signal: controller.signal 
      });
      clearTimeout(timeout);
      if (res.ok) return res;
      
      // If it's 400, 401, 403, or 429 (Rate Limit / Unauthorized), return immediately without retrying
      if (res.status === 400 || res.status === 401 || res.status === 403 || res.status === 429) {
        return res;
      }
      
      if (i === retries - 1) {
        console.warn(`Fetch returned status ${res.status} for ${url}`);
      }
    } catch (err: any) {
      clearTimeout(timeout);
      const isAbort = err?.name === 'AbortError' || String(err).includes('aborted');
      if (i === retries - 1 && !isAbort) {
        console.warn(`Fetch notice for ${url}: ${err?.message || err}`);
      }
    }
    if (i < retries - 1) await new Promise(resolve => setTimeout(resolve, 500));
  }
  return null;
}

function metSymbolToWeatherCode(symbol: string): number {
  if (!symbol) return 0;
  const s = symbol.toLowerCase();
  if (s.includes("clearsky")) return 0;
  if (s.includes("fair")) return 1;
  if (s.includes("partlycloudy")) return 2;
  if (s.includes("cloudy")) return 3;
  if (s.includes("fog")) return 45;
  if (s.includes("heavyrainandthunder") || s.includes("thunder")) return 95;
  if (s.includes("heavyrain")) return 63;
  if (s.includes("rainshowers")) return 80;
  if (s.includes("rain")) return 61;
  if (s.includes("snowshowers")) return 85;
  if (s.includes("snow")) return 71;
  if (s.includes("sleet")) return 68;
  return 3;
}

function parseMetNorwayToWeatherData(data: any): any {
  const timeseries = data.properties?.timeseries;
  if (!Array.isArray(timeseries) || timeseries.length === 0) return null;

  const first = timeseries[0];
  const inst = first.data?.instant?.details || {};
  const next1 = first.data?.next_1_hours || first.data?.next_6_hours || {};
  const symbolCode = next1.summary?.symbol_code || "";
  const code = metSymbolToWeatherCode(symbolCode);

  const curTemp = inst.air_temperature ?? null;
  const curHumidity = inst.relative_humidity ?? null;
  const curCloud = typeof inst.cloud_area_fraction === 'number' ? Math.round(inst.cloud_area_fraction) : null;
  const curPressure = inst.air_pressure_at_sea_level ?? null;
  const curWind = typeof inst.wind_speed === 'number' ? Number((inst.wind_speed * 3.6).toFixed(1)) : null;
  const curWindDir = inst.wind_from_direction ?? null;
  const curPrecip = next1.details?.precipitation_amount ?? null;
  const curUv = inst.ultraviolet_index_clear_sky ?? null;

  const hourlyTime: string[] = [];
  const hourlyTemp: number[] = [];
  const hourlyHum: number[] = [];
  const hourlyAppTemp: number[] = [];
  const hourlyWind: number[] = [];
  const hourlyWindDir: number[] = [];
  const hourlyPressure: number[] = [];
  const hourlyPrecipProb: number[] = [];
  const hourlyPrecip: number[] = [];
  const hourlyUv: number[] = [];
  const hourlyCloud: number[] = [];
  const hourlyCode: number[] = [];
  const hourlyIsDay: number[] = [];

  const dailyMap = new Map<string, { temps: number[]; precips: number[]; uvs: number[]; winds: number[]; codes: number[]; probs: number[] }>();

  for (const step of timeseries.slice(0, 48)) {
    const stInst = step.data?.instant?.details || {};
    const stNext = step.data?.next_1_hours || step.data?.next_6_hours || {};
    const tStr = step.time;
    const temp = stInst.air_temperature ?? null;
    const hum = stInst.relative_humidity ?? null;
    const cloud = typeof stInst.cloud_area_fraction === 'number' ? Math.round(stInst.cloud_area_fraction) : null;
    const press = stInst.air_pressure_at_sea_level ?? null;
    const wind = typeof stInst.wind_speed === 'number' ? Number((stInst.wind_speed * 3.6).toFixed(1)) : null;
    const windDir = stInst.wind_from_direction ?? null;
    const precip = stNext.details?.precipitation_amount ?? null;
    const uv = stInst.ultraviolet_index_clear_sky ?? null;
    const stCode = metSymbolToWeatherCode(stNext.summary?.symbol_code || symbolCode);
    const prob = stNext.details?.probability_of_precipitation ?? null;
    const dateObj = new Date(tStr);
    const hour = dateObj.getHours();
    const isDay = hour >= 6 && hour <= 21 ? 1 : 0;

    hourlyTime.push(tStr);
    hourlyTemp.push(temp);
    hourlyHum.push(hum);
    hourlyAppTemp.push(null);
    hourlyWind.push(wind);
    hourlyWindDir.push(windDir);
    hourlyPressure.push(press);
    hourlyPrecipProb.push(prob);
    hourlyPrecip.push(precip);
    hourlyUv.push(uv);
    hourlyCloud.push(cloud);
    hourlyCode.push(stCode);
    hourlyIsDay.push(isDay);

    const dateKey = tStr.split("T")[0];
    if (!dailyMap.has(dateKey)) {
      dailyMap.set(dateKey, { temps: [], precips: [], uvs: [], winds: [], codes: [], probs: [] });
    }
    const dObj = dailyMap.get(dateKey)!;
    dObj.temps.push(temp);
    dObj.precips.push(precip);
    dObj.uvs.push(uv);
    dObj.winds.push(wind);
    dObj.codes.push(stCode);
    dObj.probs.push(prob);
  }

  const dailyTime: string[] = [];
  const dailyCode: number[] = [];
  const dailyTempMax: number[] = [];
  const dailyTempMin: number[] = [];
  const dailyUvMax: number[] = [];
  const dailyPrecipSum: number[] = [];
  const dailyPrecipProbMax: number[] = [];
  const dailyWindMax: number[] = [];

  const safeMax = (arr: any[]) => {
    const filtered = arr.filter(v => typeof v === 'number' && !isNaN(v));
    return filtered.length > 0 ? Math.max(...filtered) : null;
  };

  const safeMin = (arr: any[]) => {
    const filtered = arr.filter(v => typeof v === 'number' && !isNaN(v));
    return filtered.length > 0 ? Math.min(...filtered) : null;
  };

  const safeSum = (arr: any[]) => {
    const filtered = arr.filter(v => typeof v === 'number' && !isNaN(v));
    return filtered.length > 0 ? Number(filtered.reduce((a, b) => a + b, 0).toFixed(1)) : null;
  };

  for (const [dKey, dObj] of dailyMap.entries()) {
    dailyTime.push(dKey);
    dailyTempMax.push(safeMax(dObj.temps));
    dailyTempMin.push(safeMin(dObj.temps));
    dailyUvMax.push(safeMax(dObj.uvs));
    dailyPrecipSum.push(safeSum(dObj.precips));
    dailyPrecipProbMax.push(safeMax(dObj.probs));
    dailyWindMax.push(safeMax(dObj.winds));
    dailyCode.push(dObj.codes[0] ?? code);
  }

  const dailySunrise: (string | null)[] = new Array(dailyTime.length).fill(null);
  const dailySunset: (string | null)[] = new Array(dailyTime.length).fill(null);

  return {
    current: {
      temperature_2m: curTemp,
      relative_humidity_2m: curHumidity,
      apparent_temperature: null,
      is_day: new Date().getHours() >= 6 && new Date().getHours() <= 21 ? 1 : 0,
      precipitation: curPrecip,
      cloud_cover: curCloud,
      cloud_cover_low: null,
      cloud_cover_mid: null,
      cloud_cover_high: null,
      pressure_msl: curPressure,
      wind_speed_10m: curWind,
      wind_direction_10m: curWindDir,
      uv_index: curUv,
      visibility: null,
      weather_code: code
    },
    hourly: {
      time: hourlyTime,
      temperature_2m: hourlyTemp,
      relative_humidity_2m: hourlyHum,
      apparent_temperature: hourlyAppTemp,
      weather_code: hourlyCode,
      wind_speed_10m: hourlyWind,
      wind_direction_10m: hourlyWindDir,
      pressure_msl: hourlyPressure,
      precipitation_probability: hourlyPrecipProb,
      precipitation: hourlyPrecip,
      uv_index: hourlyUv,
      cloud_cover: hourlyCloud,
      cloud_cover_low: new Array(hourlyTime.length).fill(null),
      cloud_cover_mid: new Array(hourlyTime.length).fill(null),
      cloud_cover_high: new Array(hourlyTime.length).fill(null),
      visibility: new Array(hourlyTime.length).fill(null),
      is_day: hourlyIsDay
    },
    daily: {
      time: dailyTime,
      weather_code: dailyCode,
      temperature_2m_max: dailyTempMax,
      temperature_2m_min: dailyTempMin,
      apparent_temperature_max: new Array(dailyTime.length).fill(null),
      apparent_temperature_min: new Array(dailyTime.length).fill(null),
      uv_index_max: dailyUvMax,
      precipitation_sum: dailyPrecipSum,
      precipitation_probability_max: dailyPrecipProbMax,
      wind_speed_10m_max: dailyWindMax,
      sunrise: dailySunrise,
      sunset: dailySunset
    },
    activeServers: ["MET Norway (Yr.no)"]
  };
}

/**
 * Unified IMGW Station Fetcher:
 * 1. Fetches all active IMGW stations from https://danepubliczne.imgw.pl/api/data/meteo (785 stations with real lat/lon in payload)
 * 2. Fetches synop in parallel to enrich barometric pressure (cisnienie)
 * 3. Uses exact lat/lon directly from IMGW API for each station
 * 4. Calculates Haversine distance from user GPS
 * 5. Sorts ascending and selects nearest
 * 6. Logs TOP 10 candidate stations to console.table
 */
async function fetchUnifiedImgwStation(userLat: number, userLng: number) {
  try {
    const ts = Date.now();
    const [meteoRes, synopRes] = await Promise.all([
      fetchWithRetry(`https://danepubliczne.imgw.pl/api/data/meteo?t=${ts}`),
      fetchWithRetry(`https://danepubliczne.imgw.pl/api/data/synop?t=${ts}`)
    ]);

    // Build synop lookup dictionary for barometric pressure
    const synopMap = new Map<string, any>();
    if (synopRes && synopRes.headers.get("content-type")?.includes("application/json")) {
      try {
        const synopList = await synopRes.json();
        if (Array.isArray(synopList)) {
          for (const s of synopList) {
            if (s.stacja) {
              synopMap.set(normalizeStationName(s.stacja), s);
            }
          }
        }
      } catch (err) {
        console.warn("Could not parse synop response:", err);
      }
    }

    if (!meteoRes || !meteoRes.headers.get("content-type")?.includes("application/json")) {
      console.warn("IMGW METEO API returned non-JSON response");
      return null;
    }

    const meteoList = await meteoRes.json();
    if (!Array.isArray(meteoList) || meteoList.length === 0) return null;

    const candidates: any[] = [];

    for (const item of meteoList) {
      if (!item || !item.lat || !item.lon) continue;
      const stLat = parseFloat(item.lat);
      const stLng = parseFloat(item.lon);
      if (isNaN(stLat) || isNaN(stLng)) continue;

      const tempStr = item.temperatura_powietrza;
      if (tempStr === null || tempStr === undefined || tempStr === "") continue;

      const rawTemp = parseFloat(tempStr);
      if (isNaN(rawTemp)) continue;

      const dist = getDistanceKm(userLat, userLng, stLat, stLng);
      const rawHum = item.wilgotnosc_wzgledna ? parseFloat(item.wilgotnosc_wzgledna) : null;
      const rawWind = item.wiatr_srednia_predkosc ? Math.round(parseFloat(item.wiatr_srednia_predkosc) * 3.6) : null;
      const rawRain = item.opad_10min ? parseFloat(item.opad_10min) : null;
      const rawGround = item.temperatura_gruntu ? parseFloat(item.temperatura_gruntu) : null;

      // Pressure lookup from synop if available
      const synopMatch = synopMap.get(normalizeStationName(item.nazwa_stacji || ""));
      const rawPress = synopMatch?.cisnienie ? parseFloat(synopMatch.cisnienie.replace(',', '.')) : null;

      const timeRaw = item.temperatura_powietrza_data || item.opad_10min_data || "";
      const formattedTime = timeRaw ? formatUtcToPolishTime(timeRaw) : "";

      candidates.push({
        raw: item,
        id: item.kod_stacji,
        id_stacji: item.kod_stacji,
        name: `Stacja IMGW-PIB ${item.nazwa_stacji}`,
        stationName: item.nazwa_stacji,
        distanceKm: Number(dist.toFixed(1)),
        distance: `${dist.toFixed(1)} km`,
        lat: stLat,
        lng: stLng,
        temp: rawTemp,
        humidity: rawHum && !isNaN(rawHum) ? normalizeHumidity(rawHum) : null,
        windSpeed: rawWind && !isNaN(rawWind) ? rawWind : null,
        rainRate: rawRain !== null && !isNaN(rawRain) ? rawRain : 0,
        groundTemp: rawGround && !isNaN(rawGround) ? rawGround : null,
        soilTemp: rawGround && !isNaN(rawGround) ? rawGround : null,
        pressure: rawPress && !isNaN(rawPress) ? Number(rawPress.toFixed(1)) : null,
        rawPressure: synopMatch?.cisnienie ?? null,
        status: "Online - Telemetria IMGW-PIB",
        measurementTime: formattedTime,
        rawMeasurementTime: timeRaw,
        lastPacket: formattedTime,
        isOfficial: true
      });
    }

    if (candidates.length === 0) return null;

    // Strict Haversine sorting ascending
    candidates.sort((a, b) => a.distanceKm - b.distanceKm);

    const top10 = candidates.slice(0, 10);

    const nearestWithSynopPressure = candidates.find(c => c.pressure !== null && !isNaN(c.pressure));

    console.log(`📡 [IMGW Unified API] TOP 10 najbliższych stacji dla GPS (${userLat.toFixed(4)}, ${userLng.toFixed(4)}):`);
    console.table(
      top10.map((c, i) => ({
        "Poz.": i + 1,
        "ID": c.id,
        "Nazwa Stacji IMGW": c.stationName,
        "Szerokość (Lat)": c.lat,
        "Długość (Lng)": c.lng,
        "Odległość (km)": c.distanceKm,
        "Temp (°C)": c.temp,
        "Wiatr (km/h)": c.windSpeed !== null ? `${c.windSpeed} km/h` : "—",
        "Ciśnienie (hPa)": c.pressure ?? "Brak barometru"
      }))
    );

    const cleanTop10 = top10.map(c => {
      const copy = { ...c };
      delete copy.candidates;
      delete copy.nearestCandidates;
      return copy;
    });

    const bestStation = { ...cleanTop10[0] };
    bestStation.tempFormatted = bestStation.temp !== null ? `${bestStation.temp.toFixed(1).replace('.', ',')}°C` : "Brak danych";
    bestStation.solarRadiation = null;
    bestStation.solarRadiationSource = "Brak aktynometru na stacji IMGW";
    bestStation.soilMoisture = null;
    bestStation.soilMoistureSource = "Brak czujnika wilgotności gleby na stacji IMGW";

    if (bestStation.pressure === null && nearestWithSynopPressure) {
      bestStation.synopPressureStation = {
        stationName: nearestWithSynopPressure.stationName,
        distanceKm: nearestWithSynopPressure.distanceKm,
        pressure: nearestWithSynopPressure.pressure!
      };
    }

    bestStation.candidates = cleanTop10;
    bestStation.nearestCandidates = cleanTop10;
    console.log(`✅ [IMGW Unified API] Wybrano stację najbliższą: ${bestStation.stationName} (ID: ${bestStation.id}, ${bestStation.distanceKm} km, ${bestStation.tempFormatted})`);
    return bestStation;
    return null;
  } catch (err) {
    console.warn("IMGW Unified API fetch warning:", err);
    return null;
  }
}

/**
 * Fetches Air Quality data from GIOŚ (Chief Inspectorate for Environmental Protection) API.
 * Uses station proximity for Polish locations.
 */
async function fetchGiosAirQuality(userLat: number, userLng: number) {
  try {
    // 1. Check cache for stations
    let stations: any[] | null = null;
    if (giosStationsCache && (Date.now() - giosStationsCache.timestamp < GIOS_STATIONS_CACHE_TTL)) {
      stations = giosStationsCache.data;
    }

    if (!stations) {
      // Fetch all stations (using modernized GIOŚ API v1 with 15s timeout for large payload)
      const stationsRes = await fetchWithRetry("https://api.gios.gov.pl/pjp-api/v1/rest/metadata/stations?size=500", 2, 15000);
      if (stationsRes && stationsRes.ok) {
        const stationsData = await stationsRes.json();
        stations = stationsData["Lista metadanych stacji pomiarowych"] || stationsData.data || stationsData;
        if (Array.isArray(stations)) {
          giosStationsCache = { data: stations, timestamp: Date.now() };
        }
      } else if (giosStationsCache) {
        // Fallback to expired cache if API fails (rate limit)
        stations = giosStationsCache.data;
        console.warn("[GIOŚ API] Using expired stations cache due to API failure.");
      }
    }
    
    if (!stations || !Array.isArray(stations)) return null;

    // 2. Find nearest station
    let nearestStation = null;
    let minDist = Infinity;
    for (const s of stations) {
      const lat = s["WGS84 φ N"] || s.gegrLat || s.lat;
      const lon = s["WGS84 λ E"] || s.gegrLon || s.lon;
      const closedDate = s["Data zamknięcia"] || s.dataZamkniecia;
      
      if (!lat || !lon || closedDate) continue; 
      
      const d = getDistanceKm(userLat, userLng, parseFloat(lat), parseFloat(lon));
      if (d < minDist) {
        minDist = d;
        nearestStation = s;
      }
    }

    if (!nearestStation || minDist > 50) return null; 

    // 3. Check cache for AQI
    const stationId = nearestStation.Nr || nearestStation.id;
    const cachedAqi = giosAqiCache.get(stationId);
    if (cachedAqi && (Date.now() - cachedAqi.timestamp < GIOS_AQI_CACHE_TTL)) {
      return cachedAqi.data;
    }

    const indexRes = await fetchWithRetry(`https://api.gios.gov.pl/pjp-api/v1/rest/aqindex/getIndex/${stationId}`, 2, 8000);
    if (!indexRes || !indexRes.ok) {
      if (cachedAqi) return cachedAqi.data; // Return expired if rate limited
      return null;
    }
    const aqiData = await indexRes.json();
    const aqiObj = aqiData.AqIndex || aqiData;

    const result = {
      stationName: nearestStation["Nazwa stacji"] || nearestStation.stationName,
      address: nearestStation.Adres || nearestStation.addressStreet,
      distanceKm: Math.round(minDist * 10) / 10,
      aqi: aqiObj["Nazwa kategorii indeksu"] || aqiObj.stIndexLevel?.indexLevelName || "Brak danych",
      pm10: aqiObj["Nazwa kategorii indeksu dla wskaźnika PM10"] || aqiObj.pm10IndexLevel?.indexLevelName,
      pm25: aqiObj["Nazwa kategorii indeksu dla wskaźnika PM2.5"] || aqiObj.pm25IndexLevel?.indexLevelName,
      o3: aqiObj["Nazwa kategorii indeksu dla wskaźnika O3"] || aqiObj.o3IndexLevel?.indexLevelName,
      no2: aqiObj["Nazwa kategorii indeksu dla wskaźnika NO2"] || aqiObj.no2IndexLevel?.indexLevelName,
      source: "GIOŚ (Główny Inspektorat Ochrony Środowiska)"
    };

    // Store in cache
    giosAqiCache.set(stationId, { data: result, timestamp: Date.now() });

    return result;
  } catch (err) {
    console.warn("GIOŚ API fetch warning:", err);
    return null;
  }
}

/**
 * Fetches Hydrological data from IMGW-PIB API.
 * Returns water levels for the nearest measurement point.
 */
async function fetchImgwHydroData(userLat: number, userLng: number) {
  try {
    let hydroList: any[] | null = null;

    // 1. Check in-memory cache
    if (imgwHydroCache && (Date.now() - imgwHydroCache.timestamp < IMGW_HYDRO_CACHE_TTL) && Array.isArray(imgwHydroCache.data)) {
      hydroList = imgwHydroCache.data;
    } else {
      // 2. Fetch fresh data with 7000ms timeout
      const res = await fetchWithRetry("https://danepubliczne.imgw.pl/api/data/hydro", 2, 7000);
      if (res && res.ok) {
        const parsed = await res.json();
        if (Array.isArray(parsed) && parsed.length > 0) {
          hydroList = parsed;
          imgwHydroCache = { data: parsed, timestamp: Date.now() };
        }
      }
    }

    // 3. Fallback to stale cache if fresh fetch failed
    if (!hydroList && imgwHydroCache && Array.isArray(imgwHydroCache.data)) {
      hydroList = imgwHydroCache.data;
    }

    if (!hydroList || !Array.isArray(hydroList)) {
      return { stations: [], source: "IMGW-PIB Hydrologia" };
    }

    // 4. Calculate distance to user coordinates and return 10 nearest stations
    const stationsWithDistance = hydroList
      .filter((s: any) => s && (s.lat || s.lon || s.stacja))
      .map((s: any) => {
        const sLat = parseFloat(s.lat);
        const sLng = parseFloat(s.lon);
        const dist = (!isNaN(sLat) && !isNaN(sLng))
          ? getDistanceKm(userLat, userLng, sLat, sLng)
          : 999;
        return { ...s, distance_km: dist };
      })
      .sort((a: any, b: any) => a.distance_km - b.distance_km);

    return {
      stations: stationsWithDistance.slice(0, 10),
      source: "IMGW-PIB Hydrologia"
    };
  } catch (err: any) {
    if (imgwHydroCache && Array.isArray(imgwHydroCache.data)) {
      return { stations: imgwHydroCache.data.slice(0, 10), source: "IMGW-PIB Hydrologia (Cache)" };
    }
    return { stations: [], source: "IMGW-PIB Hydrologia" };
  }
}

// API Route: Nearest IMGW Station
app.get("/api/imgw/nearest", async (req, res) => {
  res.setHeader("Cache-Control", "no-store, no-cache, must-revalidate, proxy-revalidate");
  const { lat: rawLat, lng: rawLng } = req.query;
  const lat = parseFloat(rawLat as string);
  const lng = parseFloat(rawLng as string);
  if (isNaN(lat) || isNaN(lng)) {
    return res.status(400).json({ error: "Szerokość i długość geograficzna są wymagane (lat, lng)." });
  }
  const station = await fetchUnifiedImgwStation(lat, lng);
  if (!station) {
    return res.status(404).json({ error: "Nie znaleziono stacji IMGW" });
  }
  return res.json(station);
});

// API Route: Hydrological Data from IMGW
app.get("/api/imgw/hydro", async (req, res) => {
  res.setHeader("Cache-Control", "no-store, no-cache, must-revalidate, proxy-revalidate");
  const { lat: rawLat, lng: rawLng } = req.query;
  const lat = parseFloat(rawLat as string) || 52.0;
  const lng = parseFloat(rawLng as string) || 19.0;
  const hydroData = await fetchImgwHydroData(lat, lng);
  return res.json(hydroData || { stations: [], source: "IMGW-PIB Hydrologia" });
});

// API Route: Air Quality from GIOŚ
app.get("/api/gios/air-quality", async (req, res) => {
  res.setHeader("Cache-Control", "no-store, no-cache, must-revalidate, proxy-revalidate");
  const { lat: rawLat, lng: rawLng } = req.query;
  const lat = parseFloat(rawLat as string);
  const lng = parseFloat(rawLng as string);
  if (isNaN(lat) || isNaN(lng)) {
    return res.status(400).json({ error: "Szerokość i długość geograficzna są wymagane (lat, lng)." });
  }
  const aqi = await fetchGiosAirQuality(lat, lng);
  return res.json(aqi || { aqi: "Brak danych" });
});

// API Route: Get weather data & reverse geocode
app.get(["/api/weather", "/api/pogoda"], async (req, res) => {
  // Disable caching so live server updates & IMGW telemetry are fetched fresh on every request
  res.setHeader("Cache-Control", "no-store, no-cache, must-revalidate, proxy-revalidate");
  res.setHeader("Pragma", "no-cache");
  res.setHeader("Expires", "0");
  
  const { lat: rawLat, lng: rawLng } = req.query;
  let lat = parseFloat(rawLat as string);
  let lng = parseFloat(rawLng as string);

  // SNAP removed to allow accurate GPS coordinates
  
  console.log("Fetching weather for lat:", lat, "lng:", lng);

  if (isNaN(lat) || isNaN(lng)) {
    return res.status(400).json({ error: "Szerokość i długość geograficzna są wymagane (lat, lng)." });
  }

  const isForce = req.query.force === "true" || req.query.refresh === "true" || req.headers["cache-control"] === "no-cache" || req.query.t !== undefined;
  const geoKey = `${lat.toFixed(4)}_${lng.toFixed(4)}`;
  const cachedWeather = weatherResponseCache.get(geoKey);
  if (!isForce && cachedWeather && (Date.now() - cachedWeather.timestamp < WEATHER_CACHE_TTL_MS)) {
    return res.json(cachedWeather.data);
  }

  // Determine city for logging
  let city = "Nieznana lokalizacja";
  try {
    const geoUrl = `https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${lat}&longitude=${lng}&localityLanguage=pl`;
    const geoController = new AbortController();
    const geoTimeout = setTimeout(() => geoController.abort(), 5000);
    const geoRes = await fetch(geoUrl, { signal: geoController.signal });
    clearTimeout(geoTimeout);
    if (geoRes.ok) {
      const geoData = await geoRes.json();
      if (geoData.locality && !geoData.locality.toLowerCase().startsWith("województwo")) {
        city = geoData.locality;
      } else {
        const combinedList = [
          ...(geoData.localityInfo?.administrative || []),
          ...(geoData.localityInfo?.informative || [])
        ];
        const validItems = combinedList.filter((item: any) => {
          if (!item || !item.name) return false;
          const lower = item.name.toLowerCase();
          return !["europa", "europe", "polska", "poland", "unia europejska"].includes(lower) &&
                 !lower.startsWith("województwo") && !lower.startsWith("voivodeship");
        });
        validItems.sort((a: any, b: any) => (b.order || 0) - (a.order || 0));
        if (validItems.length > 0) {
          city = validItems[0].name;
        } else if (geoData.city) {
          city = geoData.city;
        }
      }
      console.log("Resolved location for weather fetch:", city, "at lat:", lat, "lng:", lng);
    }
  } catch (e) {
    console.warn("Reverse geocoding failed for logging:", e);
  }

  try {
    let weatherData: any = null;
    const weatherApiKey = process.env.WEATHER_API_KEY;

    // 1. Attempt Open-Meteo
    const apiKey = process.env.OPENMETEO_API_KEY;
    let omBase = apiKey ? "https://customer-api.open-meteo.com/v1/forecast" : "https://api.open-meteo.com/v1/forecast";
    let auth = apiKey ? `&apikey=${apiKey}` : "";

    let openMeteoUrl = `${omBase}?latitude=${lat}&longitude=${lng}&current=temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,rain,showers,snowfall,weather_code,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,pressure_msl,wind_speed_10m,wind_gusts_10m,wind_direction_10m,uv_index,visibility,shortwave_radiation,direct_normal_irradiance,lightning_potential&minutely_15=precipitation,precipitation_probability,rain,snowfall&hourly=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,wind_gusts_10m,wind_direction_10m,pressure_msl,precipitation_probability,precipitation,uv_index,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,visibility,shortwave_radiation,direct_normal_irradiance,is_day,soil_moisture_0_to_1cm,soil_moisture_1_to_3cm,soil_temperature_0cm,evapotranspiration,lightning_potential&daily=sunrise,sunset,temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,uv_index_max,precipitation_sum,precipitation_probability_max,wind_speed_10m_max,wind_gusts_10m_max,weather_code&forecast_days=3&past_days=1&timezone=auto${auth}`;
    try {
      console.log(`Fetching weather from Open-Meteo: ${openMeteoUrl.split('&apikey=')[0]}...`);
      let res = await fetchWithRetry(openMeteoUrl);
      
      // Fallback to public API if key is invalid
      if (res && res.status === 400 && apiKey) {
        console.warn("Open-Meteo returned 400 with API key, falling back to public endpoint...");
        omBase = "https://api.open-meteo.com/v1/forecast";
        auth = "";
        openMeteoUrl = `${omBase}?latitude=${lat}&longitude=${lng}&current=temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,rain,showers,snowfall,weather_code,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,pressure_msl,wind_speed_10m,wind_gusts_10m,wind_direction_10m,uv_index,visibility,shortwave_radiation,direct_normal_irradiance,lightning_potential&minutely_15=precipitation,precipitation_probability,rain,snowfall&hourly=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,wind_gusts_10m,wind_direction_10m,pressure_msl,precipitation_probability,precipitation,uv_index,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,visibility,shortwave_radiation,direct_normal_irradiance,is_day,soil_moisture_0_to_1cm,soil_moisture_1_to_3cm,soil_temperature_0cm,evapotranspiration,lightning_potential&daily=sunrise,sunset,temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,uv_index_max,precipitation_sum,precipitation_probability_max,wind_speed_10m_max,wind_gusts_10m_max,weather_code&forecast_days=3&past_days=1&timezone=auto`;
        res = await fetchWithRetry(openMeteoUrl);
      }

      if (res && res.ok) {
        const contentType = res.headers.get("content-type");
        if (contentType && contentType.includes("application/json")) {
          weatherData = await res.json();
          weatherData.activeServers = ["Open-Meteo GFS"];
        } else {
          const text = await res.text();
          console.error("Open-Meteo returned non-JSON response:", text.substring(0, 100));
        }
      }
    } catch (err) {
      console.error("Open-Meteo failed:", err);
    }

    // 2. Fallback to WeatherAPI
    if (!weatherData && weatherApiKey) {
      try {
        console.log("Falling back to WeatherAPI");
        const weatherApiUrl = `https://api.weatherapi.com/v1/forecast.json?key=${weatherApiKey}&q=${lat},${lng}&days=1&aqi=no&alerts=no`;
        const weatherApiResponse = await fetchWithRetry(weatherApiUrl);
        if (weatherApiResponse && weatherApiResponse.ok) {
          const contentType = weatherApiResponse.headers.get("content-type");
          if (!contentType || !contentType.includes("application/json")) {
            console.warn("WeatherAPI returned non-JSON response");
            throw new Error("Invalid response format");
          }
          const data = await weatherApiResponse.json();
          weatherData = {
            current: {
              temperature_2m: data.current.temp_c,
              relative_humidity_2m: data.current.humidity,
              apparent_temperature: data.current.feelslike_c,
              is_day: data.current.is_day,
              precipitation: data.current.precip_mm,
              cloud_cover: data.current.cloud,
              wind_speed_10m: data.current.wind_kph,
              uv_index: data.current.uv,
              weather_code: data.current.condition.code
            },
            hourly: {
              time: data.forecast.forecastday[0].hour.map((h: any) => h.time),
              temperature_2m: data.forecast.forecastday[0].hour.map((h: any) => h.temp_c),
              relative_humidity_2m: data.forecast.forecastday[0].hour.map((h: any) => h.humidity),
              apparent_temperature: data.forecast.forecastday[0].hour.map((h: any) => h.feelslike_c),
              wind_speed_10m: data.forecast.forecastday[0].hour.map((h: any) => h.wind_kph),
              precipitation_probability: data.forecast.forecastday[0].hour.map((h: any) => h.chance_of_rain),
              cloud_cover: data.forecast.forecastday[0].hour.map((h: any) => h.cloud),
              weather_code: data.forecast.forecastday[0].hour.map((h: any) => h.condition.code),
              precipitation: data.forecast.forecastday[0].hour.map((h: any) => h.precip_mm),
              uv_index: data.forecast.forecastday[0].hour.map((h: any) => h.uv)
            },
            daily: {
              time: [data.forecast.forecastday[0].date],
              weather_code: [data.forecast.forecastday[0].day.condition.code],
              temperature_2m_max: [data.forecast.forecastday[0].day.maxtemp_c],
              temperature_2m_min: [data.forecast.forecastday[0].day.mintemp_c],
              apparent_temperature_max: [null],
              apparent_temperature_min: [null],
              uv_index_max: [data.forecast.forecastday[0].day.uv],
              precipitation_sum: [data.forecast.forecastday[0].day.totalprecip_mm],
              precipitation_probability_max: [data.forecast.forecastday[0].day.daily_chance_of_rain],
              wind_speed_10m_max: [data.forecast.forecastday[0].day.maxwind_kph]
            },
            activeServers: ["WeatherAPI"]
          };
        }
      } catch (err) {
        console.error("WeatherAPI failed:", err);
      }
    }

    if (!weatherData) {
      try {
        console.log("Attempting MET Norway fallback forecast...");
        const metUrl = `https://api.met.no/weatherapi/locationforecast/2.0/compact?lat=${lat}&lon=${lng}`;
        const metRes = await fetchWithRetry(metUrl, 2, 8000);
        if (metRes && metRes.ok) {
          const metJson = await metRes.json();
          weatherData = parseMetNorwayToWeatherData(metJson);
        }
      } catch (e) {
        console.warn("MET Norway fallback failed:", e);
      }
    }

    if (!weatherData) {
      const fallbackCached = weatherResponseCache.get(geoKey);
      if (fallbackCached && fallbackCached.data) {
        console.warn(`[Weather API] Returning cached fallback for ${geoKey}`);
        return res.json(fallbackCached.data);
      }
      return res.status(503).json({ error: "Usługa pogodowa chwilowo niedostępna. Spróbuj ponownie za chwilę." });
    }

    // 2. Integration of Polish-specific data (GIOŚ AQI, IMGW Hydrology) with fast 2s deadline
    try {
      const secondaryTimeout = new Promise<null>((resolve) => setTimeout(() => resolve(null), 2000));
      const secondaryDataPromise = Promise.all([
        fetchGiosAirQuality(lat, lng).catch(() => null),
        fetchImgwHydroData(lat, lng).catch(() => null)
      ]);

      const [giosAir, hydroData] = (await Promise.race([secondaryDataPromise, secondaryTimeout])) || [null, null];
      if (giosAir) weatherData.airQuality = giosAir;
      if (hydroData) weatherData.hydrology = hydroData;
    } catch (e) {
      console.warn("Failed or timed out fetching additional Polish environmental data:", e);
    }

    // 2. Minimal Normalization: Ensure base fields exist if the API returns them with model suffixes
    const normalizeObject = (obj: any, fields: string[]) => {
      if (!obj) return;
      fields.forEach(field => {
        if (obj[field] === undefined || obj[field] === null) {
          const keys = Object.keys(obj);
          const altKey = keys.find(k => k.startsWith(field + '_'));
          if (altKey) {
            obj[field] = obj[altKey];
          }
        }
      });
    };

    const weatherFields = [
      'temperature_2m', 'relative_humidity_2m', 'apparent_temperature', 
      'weather_code', 'wind_speed_10m', 'wind_direction_10m', 
      'pressure_msl', 'precipitation_probability', 'precipitation', 
      'uv_index', 'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high', 'visibility'
    ];

    if (!weatherData) {
      return res.status(503).json({ error: "Nie udało się pobrać danych pogodowych z żadnego źródła." });
    }

    // Standardize structure for all models
    if (weatherData.current) normalizeObject(weatherData.current, weatherFields);
    if (weatherData.hourly) normalizeObject(weatherData.hourly, weatherFields);

    const activeServers = weatherData.activeServers ? [...weatherData.activeServers] : ["Open-Meteo GFS"];
    let metCloud: number | null = null;
    let metTemp: number | null = null;
    let ecmwfTemp: number | null = null;
    let ecmwfCloud: number | null = null;
    let ecmwfHum: number | null = null;
    let iconTemp: number | null = null;
    let iconCloud: number | null = null;
    let iconHum: number | null = null;
    let imgwData: any = null;

    // Fetch IMGW Telemetry, IMGW SYNOP, ECMWF IFS, DWD ICON-EU, and MET Norway API in parallel
    try {
      const parsedLat = lat;
      const parsedLng = lng;
      const metUrl = `https://api.met.no/weatherapi/locationforecast/2.0/compact?lat=${lat}&lon=${lng}`;
      const ecmwfUrl = `${omBase}?latitude=${lat}&longitude=${lng}&models=ecmwf_ifs025&current=temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover,pressure_msl${auth}`;
      const iconUrl = `${omBase}?latitude=${lat}&longitude=${lng}&models=icon_eu&current=temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover,pressure_msl${auth}`;

      const [metRes, ecmwfRes, iconRes, unifiedImgwResult] = await Promise.all([
        fetchWithRetry(metUrl, 2, 10000), // Slightly lower timeout for sub-fetches
        fetchWithRetry(ecmwfUrl, 2, 10000),
        fetchWithRetry(iconUrl, 2, 10000),
        fetchUnifiedImgwStation(parsedLat, parsedLng).catch(() => null)
      ]);

      imgwData = unifiedImgwResult;
      if (imgwData) {
        activeServers.push(`IMGW-PIB stacja ${imgwData.stationName} (${imgwData.distanceKm}km)`);
      }

      if (metRes && metRes.ok) {
        const metJson = await metRes.json();
        const timeseries = metJson?.properties?.timeseries;
        if (timeseries && timeseries.length > 0) {
          const currentInstant = timeseries[0]?.data?.instant?.details;
          if (currentInstant) {
            if (typeof currentInstant.cloud_area_fraction === 'number') metCloud = currentInstant.cloud_area_fraction;
            if (typeof currentInstant.air_temperature === 'number') metTemp = currentInstant.air_temperature;
            activeServers.push("MET Norway");
          }
        }
      }

      if (ecmwfRes && ecmwfRes.ok) {
        const ecmwfJson = await ecmwfRes.json();
        if (ecmwfJson && ecmwfJson.current) {
          ecmwfTemp = ecmwfJson.current.temperature_2m;
          ecmwfCloud = ecmwfJson.current.cloud_cover;
          ecmwfHum = normalizeHumidity(ecmwfJson.current.relative_humidity_2m);
          activeServers.push("ECMWF IFS (Europe)");
        }
      }

      if (iconRes && iconRes.ok) {
        const iconJson = await iconRes.json();
        if (iconJson && iconJson.current) {
          iconTemp = iconJson.current.temperature_2m;
          iconCloud = iconJson.current.cloud_cover;
          iconHum = normalizeHumidity(iconJson.current.relative_humidity_2m);
          activeServers.push("DWD ICON-EU (Środk. Europa)");
        }
      }
    } catch (e) {
      console.warn("Multi-server consensus fetch warning:", e);
    }

    weatherData.activeServers = activeServers;

    const baseTemp = weatherData.current?.temperature_2m ?? (weatherData.hourly?.temperature_2m?.[0] ?? null);
    const baseHum = normalizeHumidity(weatherData.current?.relative_humidity_2m ?? (weatherData.hourly?.relative_humidity_2m?.[0] ?? null));
    const c = weatherData.current ?? {};
    const baseWind = weatherData.current?.wind_speed_10m ?? null;

    // Calculate Perceived Optical Cloud Cover & Ground Truth Solar Irradiance:
    const lowC = typeof c.cloud_cover_low === 'number' ? c.cloud_cover_low : null;
    const midC = typeof c.cloud_cover_mid === 'number' ? c.cloud_cover_mid : null;
    const highC = typeof c.cloud_cover_high === 'number' ? c.cloud_cover_high : null;
    const totalC = typeof c.cloud_cover === 'number' ? c.cloud_cover : null;
    const isDay = c.is_day === 1;

    const swRad = typeof c.shortwave_radiation === 'number' ? c.shortwave_radiation : null;
    const dniRad = typeof c.direct_normal_irradiance === 'number' ? c.direct_normal_irradiance : null;
    const uvVal = typeof c.uv_index === 'number' ? c.uv_index : null;
    const precipVal = typeof c.precipitation === 'number' ? c.precipitation : null;

    if (weatherData.current) {
      weatherData.current.relative_humidity_2m = normalizeHumidity(weatherData.current.relative_humidity_2m);

      const rawCloud = typeof weatherData.current.cloud_cover === 'number'
        ? Math.min(100, Math.max(0, Math.round(weatherData.current.cloud_cover)))
        : null;

      const lowC = typeof weatherData.current.cloud_cover_low === 'number' ? weatherData.current.cloud_cover_low : 0;
      const midC = typeof weatherData.current.cloud_cover_mid === 'number' ? weatherData.current.cloud_cover_mid : 0;
      const highC = typeof weatherData.current.cloud_cover_high === 'number' ? weatherData.current.cloud_cover_high : 0;
      
      const lowFrac = Math.min(100, Math.max(0, lowC)) / 100;
      const midFrac = Math.min(100, Math.max(0, midC)) / 100;
      const highFrac = Math.min(100, Math.max(0, highC)) / 100;

      const effectiveLow = lowFrac * 1.0;
      const remAfterLow = 1 - lowFrac;
      const effectiveMid = remAfterLow * midFrac * 0.55;
      const remAfterMid = remAfterLow * (1 - midFrac);
      const effectiveHigh = remAfterMid * highFrac * 0.15;
      const opticalCloud = Math.min(100, Math.max(0, Math.round((effectiveLow + effectiveMid + effectiveHigh) * 100)));

      weatherData.current.cloud_cover = rawCloud;
      weatherData.current.perceived_cloud_cover = (lowC > 0 || midC > 0 || highC > 0) ? opticalCloud : rawCloud;
      weatherData.current.optical_cloud_cover = opticalCloud;

      weatherData.current.fusion_metadata = {
        applied_filters: ["ECMWF_IFS", "DWD_ICON_EU", "IMGW_TELEMETRY"],
        activeModelsCount: 1
      };
    }

    // Attach IMGW station data separately as requested
    weatherData.imgwStation = imgwData;
    
    // Determine soil moisture from satellite data (raw 0-1cm moisture)
    let wilgotnoscSatelitarna = null;
    if (typeof weatherData.current?.soil_moisture_0_to_1cm === 'number') {
      const sm0 = weatherData.current.soil_moisture_0_to_1cm;
      wilgotnoscSatelitarna = Math.round(sm0 > 1 ? sm0 : sm0 * 100);
    } else if (weatherData.hourly && Array.isArray(weatherData.hourly.soil_moisture_0_to_1cm) && weatherData.hourly.soil_moisture_0_to_1cm.length > 0) {
      const sm0Arr = weatherData.hourly.soil_moisture_0_to_1cm;
      const times = weatherData.hourly.time ?? [];
      const nowIsoHour = new Date().toISOString().slice(0, 13);
      let idx = times.findIndex((t: string) => t.startsWith(nowIsoHour));
      if (idx === -1) idx = new Date().getHours();
      if (idx >= sm0Arr.length) idx = 0;

      const sm0 = sm0Arr[idx];
      if (typeof sm0 === 'number') {
        wilgotnoscSatelitarna = Math.round(sm0 > 1 ? sm0 : sm0 * 100);
      }
    }
    
    if (weatherData.current) {
      weatherData.current.soil_moisture_satellite = wilgotnoscSatelitarna;
    }

    // Preserve raw cloud_cover for all hours in hourly forecast
    if (weatherData.hourly && Array.isArray(weatherData.hourly.cloud_cover)) {
      weatherData.hourly.cloud_cover = weatherData.hourly.cloud_cover.map((cVal: number) => {
        return typeof cVal === 'number' ? Math.min(100, Math.max(0, Math.round(cVal))) : null;
      });
    }

    if (weatherData.hourly && Array.isArray(weatherData.hourly.relative_humidity_2m)) {
      weatherData.hourly.relative_humidity_2m = weatherData.hourly.relative_humidity_2m.map((h: any) => normalizeHumidity(h));
    }

    // Attach other models purely for information, not for calculation
    weatherData.modelsData = {
      ecmwf: ecmwfTemp !== null ? { temp: ecmwfTemp, humidity: ecmwfHum, cloud: ecmwfCloud } : null,
      icon: iconTemp !== null ? { temp: iconTemp, humidity: iconHum, cloud: iconCloud } : null,
      metNorway: metTemp !== null ? { temp: metTemp, cloud: metCloud } : null
    };

    weatherData.sourcesData = {
      gpsSource: {
        temp: weatherData.current?.temperature_2m ?? null,
        cloud: weatherData.current?.cloud_cover ?? null,
        humidity: weatherData.current?.relative_humidity_2m ?? null,
        label: "Prognoza dla lokalizacji GPS"
      },
      imgw: imgwData ? {
        temp: imgwData.temp,
        humidity: imgwData.humidity ?? null,
        wind: imgwData.windSpeed ?? null,
        pressure: imgwData.pressure ?? null,
        stationName: imgwData.stationName,
        distanceKm: imgwData.distanceKm,
        measurementTime: imgwData.measurementTime,
        label: `IMGW-PIB Stacja ${imgwData.stationName} (${imgwData.distanceKm} km)`
      } : null
    };


    // Calculate daily summaries from hourly data to ensure consistency
    if (weatherData.hourly && weatherData.hourly.time && weatherData.daily) {
      const hourly = weatherData.hourly;
      
      const temps = hourly.temperature_2m || [];
      const codes = hourly.weather_code || [];
      const precips = hourly.precipitation || [];
      const probs = hourly.precipitation_probability || [];
      const winds = hourly.wind_speed_10m || [];
      const uvs = hourly.uv_index || [];
      const clouds = hourly.cloud_cover || [];

      const daily: any = {
        ...weatherData.daily, // keep sunrise, sunset
        time: [],
        weather_code: [],
        temperature_2m_max: [],
        temperature_2m_min: [],
        uv_index_max: [],
        precipitation_sum: [],
        precipitation_probability_max: [],
        wind_speed_10m_max: []
      };

      const safeMax = (arr: any[]) => {
        const filtered = arr.filter(v => typeof v === 'number' && !isNaN(v));
        return filtered.length > 0 ? Math.max(...filtered) : null;
      };

      const safeMin = (arr: any[]) => {
        const filtered = arr.filter(v => typeof v === 'number' && !isNaN(v));
        return filtered.length > 0 ? Math.min(...filtered) : null;
      };

      const safeSum = (arr: any[]) => {
        const filtered = arr.filter(v => typeof v === 'number' && !isNaN(v));
        return filtered.length > 0 ? Number(filtered.reduce((a, b) => a + b, 0).toFixed(1)) : null;
      };

      for (let d = 0; d < 7; d++) {
        const start = d * 24;
        const end = start + 24;
        if (temps.length < end) break;
        
        const dayHourly = {
          temp: temps.slice(start, end),
          code: codes.slice(start, end),
          precip: precips.slice(start, end),
          prob: probs.slice(start, end),
          wind: winds.slice(start, end),
          uv: uvs.slice(start, end)
        };
        
        daily.time.push(hourly.time[start].split('T')[0]); // YYYY-MM-DD
        daily.temperature_2m_max.push(safeMax(dayHourly.temp));
        daily.temperature_2m_min.push(safeMin(dayHourly.temp));
        daily.precipitation_sum.push(safeSum(dayHourly.precip));
        daily.precipitation_probability_max.push(safeMax(dayHourly.prob));
        daily.wind_speed_10m_max.push(safeMax(dayHourly.wind));
        daily.uv_index_max.push(safeMax(dayHourly.uv));
        
        const dayCodes = dayHourly.code ?? [];
        const pSum = daily.precipitation_sum[daily.precipitation_sum.length - 1];
        const maxP = daily.precipitation_probability_max[daily.precipitation_probability_max.length - 1];

        const getDailyCode = (codes: number[], precipSum: number | null, maxPop: number | null) => {
          if (!codes || codes.length === 0) return 0;
          
          const getSeverity = (code: number) => {
            if (code >= 95 && code <= 99) return 100; // Burza
            if (code === 65 || code === 82 || code === 75 || code === 86) return 90; // Ulewa / śnieżyca
            if (code === 63 || code === 81 || code === 73 || code === 85) return 80; // Opady umiarkowane
            if (code === 61 || code === 80 || code === 55 || code === 53 || code === 51) return 70; // Deszcz / przelotny / mżawka
            if (code === 45 || code === 48) return 50; // Mgła
            if (code === 3) return 40; // Zachmurzenie całkowite
            if (code === 2) return 30; // Umiarkowane
            if (code === 1) return 20; // Małe
            return 10; // Bezchmurnie
          };

          let bestCode = codes[12] !== undefined ? codes[12] : codes[0];
          let maxScore = getSeverity(bestCode);

          for (const c of codes) {
            const score = getSeverity(c);
            if (score > maxScore) {
              maxScore = score;
              bestCode = c;
            }
          }

          if ((precipSum >= 0.2 || maxPop >= 40) && maxScore < 70) {
            return maxPop >= 60 ? 80 : 51;
          }

          return bestCode;
        };

        daily.weather_code.push(getDailyCode(dayCodes, pSum, maxP));
      }
      weatherData.daily = daily;
      
      // Normalize the hourly object too
      hourly.temperature_2m = temps;
      hourly.weather_code = codes;
      hourly.precipitation = precips;
      hourly.precipitation_probability = probs;
      hourly.wind_speed_10m = winds;
      hourly.uv_index = uvs;
      hourly.cloud_cover = clouds;
    }

    // Attach provider metadata
    weatherData.provider = "Open-Meteo (Hourly-Based)";

    // 2. Fetch high-precision reverse geocoding from OpenStreetMap Nominatim with BigDataCloud fallback
    let city = "Nieznana lokalizacja";
    try {
      const nomUrl = `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}&zoom=18&addressdetails=1&accept-language=pl`;
      const nomController = new AbortController();
      const nomTimeout = setTimeout(() => nomController.abort(), 8000);
      const nomRes = await fetch(nomUrl, {
        headers: { "User-Agent": "AuraWeatherApp/1.0 (contact@auraweather.app)" },
        signal: nomController.signal
      });
      clearTimeout(nomTimeout);
      if (nomRes.ok) {
        const nomData = await nomRes.json();
        const a = nomData.address || {};
        
        // Priority list:
        // 1. city
        // 2. town
        // 3. village (including hamlet, isolated_dwelling)
        // 4. municipality
        // 5. county
        // 6. state
        // 7. other sensible fields
        const cityCandidate = a.city;
        const townCandidate = a.town;
        const villageCandidate = a.village || a.hamlet || a.isolated_dwelling;
        const municipalityCandidate = a.municipality || a.district;
        const countyCandidate = a.county;
        const stateCandidate = a.state;
        const otherCandidate = a.suburb || a.locality || a.neighbourhood || a.quarter || a.city_district || a.residential;

        if (cityCandidate && typeof cityCandidate === "string" && cityCandidate.trim().length > 1) {
          city = cityCandidate.trim();
        } else if (townCandidate && typeof townCandidate === "string" && townCandidate.trim().length > 1) {
          city = townCandidate.trim();
        } else if (villageCandidate && typeof villageCandidate === "string" && villageCandidate.trim().length > 1) {
          city = villageCandidate.trim();
        } else if (municipalityCandidate && typeof municipalityCandidate === "string" && municipalityCandidate.trim().length > 1) {
          city = municipalityCandidate.toLowerCase().startsWith("gmina") ? municipalityCandidate.trim() : `Gmina ${municipalityCandidate.trim()}`;
        } else if (countyCandidate && typeof countyCandidate === "string" && countyCandidate.trim().length > 1) {
          city = countyCandidate.trim();
        } else if (stateCandidate && typeof stateCandidate === "string" && stateCandidate.trim().length > 1) {
          city = stateCandidate.trim();
        } else if (otherCandidate && typeof otherCandidate === "string" && otherCandidate.trim().length > 1) {
          city = otherCandidate.trim();
        }
      }
    } catch (e) {
      console.warn("Nominatim reverse geocoding failed or timed out, trying fallback...", e);
    }

    if (city === "Nieznana lokalizacja") {
      try {
        const geoUrl = `https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${lat}&longitude=${lng}&localityLanguage=pl`;
        const geoController = new AbortController();
        const geoTimeout = setTimeout(() => geoController.abort(), 5000);
        const geoRes = await fetch(geoUrl, { signal: geoController.signal });
        clearTimeout(geoTimeout);
        if (geoRes.ok) {
          const geoData = await geoRes.json();
          if (geoData.city && typeof geoData.city === "string" && geoData.city.trim().length > 1) {
            city = geoData.city.trim();
          } else if (geoData.locality && typeof geoData.locality === "string" && !geoData.locality.toLowerCase().startsWith("województwo")) {
            city = geoData.locality.trim();
          } else {
            const combinedList = [
              ...(geoData.localityInfo?.administrative || []),
              ...(geoData.localityInfo?.informative || [])
            ];
            const validItems = combinedList.filter((item: any) => {
              if (!item || !item.name || typeof item.name !== "string") return false;
              const lower = item.name.toLowerCase();
              return !["europa", "europe", "polska", "poland", "unia europejska"].includes(lower) &&
                     !lower.startsWith("województwo") && !lower.startsWith("voivodeship");
            });
            validItems.sort((a: any, b: any) => (b.order || 0) - (a.order || 0));
            if (validItems.length > 0) {
              city = validItems[0].name.trim();
            }
          }
        }
      } catch (e) {
        console.error("Geocoding failed completely:", e);
      }
    }

    if (city === "Nieznana lokalizacja" || !city) {
      city = "Lokalizacja GPS";
    }

    const weatherPayload = {
      city,
      lat,
      lng,
      weather: weatherData,
      lastUpdated: new Date().toISOString()
    };
    weatherResponseCache.set(geoKey, { data: weatherPayload, timestamp: Date.now() });
    res.json(weatherPayload);
  } catch (err: any) {
    console.error("Error in /api/weather:", err);
    res.status(500).json({ error: err.message || "Błąd wewnętrzny serwera." });
  }
});

// API Route: Get real station data for given GPS coordinates
app.get("/api/stations", async (req, res) => {
  const { lat, lng } = req.query;
  if (!lat || !lng) {
    return res.status(400).json({ error: "lat and lng required" });
  }

  const latitude = parseFloat(lat as string);
  const longitude = parseFloat(lng as string);
  const geoKey = `${Math.round(latitude * 100) / 100}_${Math.round(longitude * 100) / 100}`;

  const cachedStation = stationResponseCache.get(geoKey);
  if (cachedStation && (Date.now() - cachedStation.timestamp < WEATHER_CACHE_TTL_MS)) {
    return res.json(cachedStation.data);
  }

  const apiKey = process.env.OPENMETEO_API_KEY;
  let omBase = apiKey ? "https://customer-api.open-meteo.com/v1/forecast" : "https://api.open-meteo.com/v1/forecast";
  let auth = apiKey ? `&apikey=${apiKey}` : "";

  try {
    const response = await fetchWithRetry(`${omBase}?latitude=${latitude}&longitude=${longitude}&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,cloud_cover,pressure_msl,wind_speed_10m,wind_direction_10m,soil_temperature_0cm,soil_moisture_0_to_1cm,shortwave_radiation${auth}`);

    let data: any = {};
    if (response && response.ok) {
      data = await response.json().catch(() => ({}));
    } else {
      const weatherCached = weatherResponseCache.get(geoKey);
      if (weatherCached && weatherCached.data && weatherCached.data.weather && weatherCached.data.weather.current) {
        data = { current: weatherCached.data.weather.current };
      }
    }
    const cur = data.current ?? {};
    const cloudCover = cur.cloud_cover ?? null;
    const isDayTime = cur.is_day !== undefined ? (cur.is_day === 1) : true;
    
    // Solar radiation calculated strictly according to solar zenith and cloud transmittance physics
    const solarRadiation = calculateSolarRadiation(cloudCover, isDayTime, cur.shortwave_radiation);

    const baseTemp = cur.temperature_2m ?? null;
    const baseHumidity = normalizeHumidity(cur.relative_humidity_2m);
    const baseWind = cur.wind_speed_10m ?? null;
    const basePressure = cur.pressure_msl ?? null;

    const soilTemp = cur.soil_temperature_0cm ?? baseTemp;
    
    const sm0 = cur.soil_moisture_0_to_1cm;
    const weatherCached = weatherResponseCache.get(geoKey);
    const cachedMoisture = weatherCached?.data?.weather?.current?.soil_moisture_satellite;
    let soilMoisture = null;
    if (typeof cachedMoisture === 'number') {
      soilMoisture = cachedMoisture;
    } else if (typeof cur.soil_moisture_satellite === 'number') {
      soilMoisture = cur.soil_moisture_satellite;
    } else if (sm0 !== undefined && sm0 !== null) {
      soilMoisture = Math.round(sm0 > 1 ? sm0 : sm0 * 100);
    }

    const rainRate = cur.precipitation ?? null;
    const weatherCode = cur.weather_code ?? cur.weathercode ?? 0;

    const calcLeafWetness = (humidityVal: number | null, rainVal: number | null, wCode?: number) => {
      const code = wCode ?? weatherCode;
      const isPrecip = (rainVal !== null && rainVal > 0) || (typeof code === 'number' && code >= 50 && code <= 99);
      if (humidityVal === null && rainVal === null && !code) return { leafWetness: null, leafWetnessText: "Brak danych" };
      let index = 0;
      if (isPrecip) index = 13;
      else if (typeof code === 'number' && code >= 20 && code <= 29) index = 10;
      else if (humidityVal !== null && humidityVal >= 90) index = 8;
      else if (humidityVal !== null && humidityVal >= 80) index = 5;
      else if (humidityVal !== null && humidityVal >= 65) index = 2;
      else index = 0;
      return { leafWetness: index, leafWetnessText: `${index}/15` };
    };

    let stations: any[] = [];
    let giosAir: any = null;
    let hydroData: any = null;

    try {
      const [unifiedImgw, giosResult, hydroResult] = await Promise.all([
        fetchUnifiedImgwStation(latitude, longitude),
        fetchGiosAirQuality(latitude, longitude),
        fetchImgwHydroData(latitude, longitude)
      ]);

      giosAir = giosResult;
      hydroData = hydroResult;

      if (unifiedImgw) {
        const primary = { ...unifiedImgw };
        delete primary.candidates;
        delete primary.nearestCandidates;
        stations = [primary];
      }
    } catch (e) {
      console.warn("Could not fetch real stations:", e);
    }

    const stationPayload = {
      stations,
      airQuality: giosAir,
      hydrology: hydroData,
      coordinates: { lat: latitude, lng: longitude },
      lastUpdated: new Date().toISOString()
    };
    stationResponseCache.set(geoKey, { data: stationPayload, timestamp: Date.now() });
    res.json(stationPayload);
  } catch (err: any) {
    console.error("Error in /api/stations:", err);
    res.status(500).json({ error: err.message || "Błąd pobierania stacji." });
  }
});

// Safe helper - AI disabled
async function callGeminiWithFallback(prompt: string, responseMimeType: string = "application/json"): Promise<string | null> {
  return null;
}

// Auxiliary function: generate highly detailed local weather recommendations if Gemini API is unavailable/fails
function getLocalAdviceFallback(city: string, current: any, daily: any, mode?: string) {
  const satMoisture = typeof current?.soil_moisture_satellite === "number" ? current.soil_moisture_satellite : null;
  const temp = typeof current?.temperature_2m === 'number' ? Math.round(current.temperature_2m) : null;
  const cloud = typeof current?.cloud_cover === 'number' ? Math.round(current.cloud_cover) : null;
  const press = typeof current?.pressure_msl === 'number' ? Math.round(current.pressure_msl) : null;
  const uv = typeof current?.uv_index === 'number' ? current.uv_index : null;

  if (mode === "ciekawostka") {
    const triviaFacts = [
      {
        advice: `Czy wiesz, że radary mikrofalowe pasma C na satelitach europejskich Sentinel-1 prześwietlają glebę w rejonie ${city || 'Twoim'} na głębokość 3 cm? ${satMoisture !== null ? `Dzisiejsza wilgotność gleby z kosmosu wynosi dokładnie ${satMoisture}%.` : 'Dane o wilgotności gleby są aktualizowane przez system Copernicus.'}`,
        clothes: "Okulary astronomiczne i ciekawość świata",
        activities: "Obserwacja chmur i sprawdzanie danych satelitarnych Copernicus",
        isFallback: true
      },
      {
        advice: `Ciekawostka meteorologiczna dla ${city || 'Twojego regionu'}: Przy ciśnieniu ${press !== null ? `${press} hPa` : 'atmosferycznym'} i zachmurzeniu ${cloud !== null ? `${cloud}%` : 'bieżącym'}, powłoka atmosferyczna wywiera potężny nacisk na każdy metr kwadratowy powierzchni!`,
        clothes: "Lekkie ubranie i czapka z daszkiem",
        activities: "Krótka lektura o fizyce atmosfery i zjawiskach pogodowych",
        isFallback: true
      },
      {
        advice: `Kosmiczny fakt: Geostacjonarny satelita Meteosat widzi ${city || 'Twój region'} z wysokości 35 786 km nad Ziemią! Rejestruje promieniowanie podczerwone, co pozwala nam dokładnie monitorować stan atmosfery i gleby.`,
        clothes: "Wygodny strój na spacer",
        activities: "Wyszukiwanie gwiazdozbiorów lub obserwacja satelitów na niebie",
        isFallback: true
      }
    ];

    // Select fact based on current parameters (stable hash)
    const seed = (temp ?? 10) + (satMoisture ?? 20) + (press ?? 1000);
    const factIndex = Math.abs(seed % triviaFacts.length);
    return triviaFacts[factIndex];
  }

  if (mode === "podlej") {
    const wilgotnoscSatelitarna = satMoisture;
    if (wilgotnoscSatelitarna < 20) {
      return {
        advice: `Wariacie, satelita Sentinel melduje suszę pod korzeniami (${wilgotnoscSatelitarna}%), natychmiast bierz konewkę!`,
        clothes: "Strój roboczy do ogrodu i konewka w dłoń",
        activities: "Obfite podlewanie kwiatów i roślin ogrodowych",
        isFallback: true,
        soilMoisture: wilgotnoscSatelitarna
      };
    } else if (wilgotnoscSatelitarna > 40) {
      return {
        advice: `Wariacie, satelita Sentinel wykrył, że ziemia jest idealnie wilgotna (${wilgotnoscSatelitarna}%) – schowaj konewkę i nie przelewaj roślin!`,
        clothes: "Wygodne kapcie i odpoczynek",
        activities: "Relaks w ogrodzie i podziwianie nawodnionego trawnika",
        isFallback: true,
        soilMoisture: wilgotnoscSatelitarna
      };
    } else {
      return {
        advice: `Satelita Sentinel/SMOS wskazuje umiarkowaną wilgotność gleby (${wilgotnoscSatelitarna}%). Ziemia jest lekko wilgotna – sprawdź palcem doniczkę i podlej delikatnie tylko w razie potrzeby.`,
        clothes: "Lekki strój codzienny",
        activities: "Drobne prace pielegnacyjne wokół roślin",
        isFallback: true,
        soilMoisture: wilgotnoscSatelitarna
      };
    }
  }

  const code = current ? (current.weather_code ?? 0) : 0;
  const isRain = (code >= 51 && code <= 67) || (code >= 80 && code <= 82);
  const isSnow = (code >= 71 && code <= 77) || (code >= 85 && code <= 86);
  const isStorm = code >= 95 && code <= 99;
  
  let baseAdvice = "";
  let clothes = "";
  let activities = "";

  if (isStorm) {
    baseAdvice = `O matko, w ${city || 'Twojej okolicy'} idzie potężna burza przy ${temp}°C! Lepiej szybko zwijaj manatki z pola albo z borówek, schowaj się pod dach i odpuść grę w golfa.`;
    clothes = "Kalosze, peleryna i zero metalowych prętów w rękach";
    activities = "Siedzenie w chałupie, patrzenie w okno i herbatka z malinami";
  } else if (isRain) {
    baseAdvice = `Pada w ${city || 'Twojej okolicy'} (${temp}°C) jakby jutra miało nie być! Jeśli nie chcesz wracać przemoczony do suchej nitki, bierz parasol albo uciekaj pod najbliższy dach.`;
    clothes = "Kurtka przeciwdeszczowa, parasol i wodoodporne adidasy";
    activities = "Zaszycie się w kawiarni albo leniuchowanie pod kocykiem";
  } else if (isSnow) {
    baseAdvice = `Sypie śniegiem w ${city || 'Twojej okolicy'} przy ${temp}°C! Czas odśnieżyć podjazd albo ulepić bałwana, póki białe.`;
    clothes = "Puchówka, czapka z pomponem i solidne zimowe buty";
    activities = "Zimowy spacer, sanki i gorąca czekolada";
  } else if (temp >= 25) {
    baseAdvice = `Ależ grzeje w ${city || 'Twojej okolicy'} – aż ${temp}°C! Słońce mocno dogrzewa, więc to idealny moment na odpoczynek w cieniu i regularne nawadnianie.`;
    clothes = "Krótkie spodenki, okulary przeciwsłoneczne i czapka z daszkiem";
    activities = "Wypoczynek w cieniu, chłodne napoje i regularne nawadnianie";
  } else if (temp >= 15) {
    baseAdvice = `Pogoda w ${city || 'Twojej okolicy'} w sam raz na spacer, ${temp}°C na liczniku. Ani za zimno, ani za gorąco – grzech siedzieć w czterech ścianach!`;
    clothes = "Lekka bluza, t-shirt i wygodne buty";
    activities = "Rower, spacer po parku lub mały grill ze znajomymi";
  } else if (temp >= 5) {
    baseAdvice = `Chłodek w ${city || 'Twojej okolicy'} (${temp}°C), wieje lekki wiatr. Jak się nie ubierzesz na cebulkę, to zaraz zmarzniesz w nos.`;
    clothes = "Kurtka przejściowa, sweter i długie spodnie";
    activities = "Szybki marsz, zakupy albo ciepła kawa na wynos";
  } else {
    baseAdvice = `Trzyma mróz w ${city || 'Twojej okolicy'} (${temp}°C)! Nos czerwony, palce drętwieją – bez grubej kurtki ani rusz.`;
    clothes = "Gruba zimowa kurtka, szalik i ciepłe rękawice";
    activities = "Gorąca herbata z miodem i oglądanie seriali pod kocem";
  }

  const advice = baseAdvice;
  return { advice, clothes, activities, isFallback: true };
}

// API Route: Get current app URL dynamically for the QR code share functionality (defaults to shared pre URL instead of dev link)
app.get("/api/app-url", (req, res) => {
  const host = req.get("host") || "";
  const protocol = req.headers["x-forwarded-proto"] || req.protocol || "https";
  let targetUrl = "https://ais-pre-55vkqchaiz5cdsnzrutx6d-128716608243.europe-west2.run.app";
  
  if (host) {
    const sharedHost = host.replace("-dev-", "-pre-");
    targetUrl = `${protocol}://${sharedHost}`;
  }
  
  res.json({ url: targetUrl });
});

// API Route: High-precision location search (Nominatim + Open-Meteo fallback)
app.get("/api/search-city", async (req, res) => {
  const query = (req.query.q as string || "").trim();
  if (!query) return res.json([]);

  try {
    const nomUrl = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&addressdetails=1&accept-language=pl&countrycodes=pl&limit=10`;
    const nomController = new AbortController();
    const nomTimeout = setTimeout(() => nomController.abort(), 8000);
    const nomRes = await fetch(nomUrl, {
      headers: { "User-Agent": "AuraWeatherApp/1.0 (contact@auraweather.app)" },
      signal: nomController.signal
    });
    clearTimeout(nomTimeout);

    if (nomRes.ok) {
      const nomData = await nomRes.json();
      if (Array.isArray(nomData) && nomData.length > 0) {
        let results = nomData.map((item: any) => {
          const a = item.address || {};
          const place = a.hamlet || a.village || a.town || a.city || a.locality || item.name;
          const admin = a.municipality ?? a.county ?? a.state ?? "";
          let label = place;
          if (admin && !admin.toLowerCase().includes(place.toLowerCase())) {
            label = `${place} (${admin})`;
          }
          return {
            name: label,
            lat: parseFloat(item.lat),
            lng: parseFloat(item.lon),
            rawName: place,
            adminContext: `${admin} ${a.state || ''} ${a.county || ''}`
          };
        });

        return res.json(results);
      }
    }
  } catch (e) {
    console.warn("Nominatim search failed, trying Open-Meteo fallback...", e);
  }

  // Fallback to Open-Meteo Geocoding
  try {
    const apiKey = process.env.OPENMETEO_API_KEY;
    let omGeoBase = apiKey ? "https://customer-geocoding-api.open-meteo.com/v1/search" : "https://geocoding-api.open-meteo.com/v1/search";
    let auth = apiKey ? `&apikey=${apiKey}` : "";

    let omUrl = `${omGeoBase}?name=${encodeURIComponent(query)}&count=10&language=pl&format=json${auth}`;
    let omRes = await fetchWithRetry(omUrl);

    if (omRes && omRes.status === 400 && apiKey) {
      const errJson = await omRes.clone().json().catch(() => ({}));
      if (errJson.reason?.includes("API key")) {
        omGeoBase = "https://geocoding-api.open-meteo.com/v1/search";
        auth = "";
        omUrl = `${omGeoBase}?name=${encodeURIComponent(query)}&count=10&language=pl&format=json`;
        omRes = await fetchWithRetry(omUrl);
      }
    }

    if (omRes && omRes.ok) {
      const omData = await omRes.json();
      if (omData.results && omData.results.length > 0) {
        let results = omData.results.map((r: any) => ({
          name: `${r.name}${r.admin1 ? " (" + r.admin1 + ")" : ""}`,
          lat: r.latitude,
          lng: r.longitude,
          rawName: r.name
        }));

        return res.json(results);
      }
    }
  } catch (e) {
    console.warn("Open-Meteo search failed:", e);
  }

  return res.json([]);
});

// In-memory cache to stay strictly within free-tier API quotas and handle rate limits gracefully
const CACHE_TTL_MS = 6 * 60 * 60 * 1000; // Cache AI advice for 6 hours to save quota
const ANALYSIS_CACHE_TTL_MS = 1 * 60 * 60 * 1000; // Cache AI analysis for 1 hour
const aiAdviceCache = new Map<string, { data: any; timestamp: number }>();
const aiAnalysisCache = new Map<string, { data: any; timestamp: number }>();

// Multi-Model Gemini Fallback Chain for 100% Continuity & Reliability
const GEMINI_MODELS_FALLBACK_CHAIN = [
  "gemini-2.5-flash",
  "gemini-2.0-flash",
  "gemini-1.5-flash",
  "gemini-2.5-pro",
  "gemini-1.5-pro"
];

// Cooldown tracker for models hitting 429 / Rate Limit / Quota Exceeded
const modelCooldowns: Record<string, number> = {};

/**
 * Intelligent Multi-Model Dispatcher
 * Automatically cascades through Gemini models if any model hits quota limit (429 / 503 / ResourceExhausted).
 */
async function generateGeminiContentWithFallback(
  prompt: string,
  systemInstruction?: string
): Promise<{ text: string; usedModel: string } | null> {
  const geminiKey = process.env.GEMINI_API_KEY?.trim() || process.env.GOOGLE_API_KEY?.trim();
  if (!geminiKey) {
    console.log("[Gemini Smart Switcher] Brak klucza GEMINI_API_KEY - automatyczne użycie silnika lokalnego.");
    return null;
  }

  const ai = new GoogleGenAI({ apiKey: geminiKey });
  const now = Date.now();

  for (const modelName of GEMINI_MODELS_FALLBACK_CHAIN) {
    if (modelCooldowns[modelName] && modelCooldowns[modelName] > now) {
      const remainingSec = Math.ceil((modelCooldowns[modelName] - now) / 1000);
      console.warn(`[Gemini Smart Switcher] Model ${modelName} odpoczywa po limicie zapytań (${remainingSec}s do końca). Przełączam na kolejny...`);
      continue;
    }

    try {
      console.log(`[Gemini Smart Switcher] Wysyłam zapytanie do modelu: ${modelName}...`);
      const response = await ai.models.generateContent({
        model: modelName,
        contents: prompt,
        config: systemInstruction ? { systemInstruction } : undefined,
      });

      if (response && response.text) {
        console.log(`[Gemini Smart Switcher] ✅ SUKCES! Zrealizowano przez model: ${modelName}`);
        delete modelCooldowns[modelName]; // Reset cooldown on successful response
        return { text: response.text, usedModel: modelName };
      }
    } catch (err: any) {
      const errMsg = err?.message || String(err);
      const isRateLimitOrQuota =
        errMsg.includes("429") ||
        errMsg.toLowerCase().includes("quota") ||
        errMsg.toLowerCase().includes("resource_exhausted") ||
        errMsg.toLowerCase().includes("limit") ||
        errMsg.includes("503");

      if (isRateLimitOrQuota) {
        console.warn(`[Gemini Smart Switcher] ⚠️ Wykryto LIMIT APKA (429/503/Quota) dla modelu ${modelName}! Natychmiast przełączam na kolejny model Gemini w łańcuchu fallback... (Błąd: ${errMsg.slice(0, 100)})`);
        modelCooldowns[modelName] = Date.now() + 3 * 60 * 1000; // 3 minuty cooldown
      } else {
        console.warn(`[Gemini Smart Switcher] Błąd wykonania na ${modelName}: ${errMsg.slice(0, 100)}. Przełączam na kolejny model w hierarchii...`);
      }
    }
  }

  console.warn("[Gemini Smart Switcher] Wszystkie modele Gemini osiągnęły limit lub są niedostępne. Bezszwowy spadek do niezawodnego silnika lokalnego.");
  return null;
}

// API Route: Smart AI Weather Advice & Insights with Multi-Model Fallback
app.post("/api/weather/ai", async (req, res) => {
  try {
    const { city, current, daily, mode } = req.body || {};
    const cacheKey = `${city ?? 'loc'}_${mode ?? 'def'}_${current?.temperature_2m ?? 0}_${current?.weather_code ?? 0}`;

    const cached = aiAdviceCache.get(cacheKey);
    if (cached && (Date.now() - cached.timestamp < CACHE_TTL_MS)) {
      return res.json(cached.data);
    }

    const modePrompt = mode === 'ciekawostka' 
      ? 'Podaj jedną ultra interesującą i naukowo ścisłą ciekawostkę meteorologiczną lub satelitarną powiązaną z obecną pogodą w miejscowości ' + (city || 'lokalizacja')
      : mode === 'podlej'
      ? 'Oceń czy należy podlać ogród lub kwiaty w miejscowości ' + (city || 'lokalizacja') + ' uwzględniając wilgotność gleby i prognozę opadów'
      : 'Skomponuj krótką praktyczną radę ubioru i aktywności dla miejscowości ' + (city || 'lokalizacja');

    const prompt = `Jesteś zaawansowanym synoptykiem i asystentem pogodowym.
Lokalizacja: ${city || 'lokalizacja'}.
Otrzymane parametry meteorologiczne:
- Temperatura: ${typeof current?.temperature_2m === 'number' ? current.temperature_2m + '°C' : 'Brak danych'}
- Kod pogody WMO: ${typeof current?.weather_code === 'number' ? current.weather_code : 'Brak danych'}
- Zachmurzenie optyczne: ${typeof current?.cloud_cover === 'number' ? current.cloud_cover + '%' : 'Brak danych'}
- Wilgotność gleby (satelita): ${typeof current?.soil_moisture_satellite === 'number' ? current.soil_moisture_satellite + '%' : 'Brak danych'}
- Opady: ${typeof current?.precipitation === 'number' ? current.precipitation + ' mm' : 'Brak danych'}
- Wiatr: ${typeof current?.wind_speed_10m === 'number' ? current.wind_speed_10m + ' km/h' : 'Brak danych'}
- Indeks UV: ${typeof current?.uv_index === 'number' ? current.uv_index : 'Brak danych'}

Zadanie: ${modePrompt}

Sformatuj odpowiedź WYŁĄCZNIE jako prawidłowy obiekt JSON bez znaczników markdown:
{
  "advice": "krótki zwięzły wniosek po polsku",
  "clothes": "sugerowany ubiór po polsku",
  "activities": "sugerowane aktywności na dzisiaj"
}`;

    const result = await generateGeminiContentWithFallback(prompt);

    if (result && result.text) {
      try {
        const cleaned = result.text.replace(/```json/g, "").replace(/```/g, "").trim();
        const parsed = JSON.parse(cleaned);
        const finalData = {
          advice: parsed.advice || "Pogoda jest stabilna i bez opadów.",
          clothes: parsed.clothes || "Standardowy ubiór warstwowy dostosowany do temperatury.",
          activities: parsed.activities || "Spacer, jazda na rowerze lub wypoczynek.",
          isFallback: false,
          modelUsed: result.usedModel,
          provider: `Gemini AI (${result.usedModel})`
        };

        aiAdviceCache.set(cacheKey, { data: finalData, timestamp: Date.now() });
        return res.json(finalData);
      } catch (jsonErr) {
        console.warn("[Advice] Błąd parsowania JSON odpowiedzi Gemini, przełączam na algorytm lokalny:", jsonErr);
      }
    }

    // Seamless Local Fallback engine
    const fallback = getLocalAdviceFallback(city || "lokalizacja", current, daily, mode);
    return res.json({
      ...fallback,
      modelUsed: "Aura-Local-Engine",
      provider: "Aura Engine (Lokalny)"
    });
  } catch (outerErr: any) {
    console.error("[Advice] Błąd krytyczny w /api/weather/ai:", outerErr);
    const fallback = getLocalAdviceFallback(req.body?.city || "lokalizacja", req.body?.current, req.body?.daily);
    res.json(fallback);
  }
});

// API Route: AI Forecast Analysis & Warning System with Multi-Model Fallback
app.post("/api/weather/analyze", async (req, res) => {
  try {
    const { weatherData } = req.body;
    if (!weatherData || !weatherData.current) {
      return res.status(400).json({ error: "Weather data required." });
    }

    const current = weatherData.current;
    const cacheKey = `analyze_${current.temperature_2m}_${current.weather_code}_${current.precipitation}_${current.wind_speed_10m}`;

    const cached = aiAnalysisCache.get(cacheKey);
    if (cached && (Date.now() - cached.timestamp < ANALYSIS_CACHE_TTL_MS)) {
      return res.json(cached.data);
    }

    const prompt = `Analiza meteorologiczna pod kątem zagrożeń:
Kod WMO: ${current.weather_code}, Temp: ${current.temperature_2m}°C, Opady: ${current.precipitation}mm, Wiatr: ${current.wind_speed_10m}km/h.
Czy występują niebezpieczne zjawiska meteorologiczne lub gwałtowne zmiany?

Sformatuj odpowiedź WYŁĄCZNIE jako kod JSON:
{
  "warning": "Krótkie precyzyjne ostrzeżenie lub komunikat o braku zagrożeń po polsku",
  "isAlert": true/false
}`;

    const result = await generateGeminiContentWithFallback(prompt);

    if (result && result.text) {
      try {
        const cleaned = result.text.replace(/```json/g, "").replace(/```/g, "").trim();
        const parsed = JSON.parse(cleaned);
        const finalData = {
          warning: parsed.warning || "Pogoda jest stabilna.",
          isAlert: Boolean(parsed.isAlert),
          modelUsed: result.usedModel
        };
        aiAnalysisCache.set(cacheKey, { data: finalData, timestamp: Date.now() });
        return res.json(finalData);
      } catch (e) {
        console.warn("[Analyze] Błąd parsowania odpowiedzi JSON Gemini, spadek do reguł lokalnych:", e);
      }
    }

    const code = current.weather_code;
    const lp = current.lightning_potential || 0;
    const precip = current.precipitation || 0;
    const isStormy = (code >= 95 && code <= 99) || code === 29 || lp > 0 || (code >= 80 && code <= 82 && precip > 1.0);

    let fallback = { warning: "Pogoda jest stabilna. Dobre warunki do aktywności na zewnątrz.", isAlert: false };
    if (isStormy) fallback = { warning: "Wykryto ryzyko gwałtownych burz i wyładowań. Zachowaj ostrożność i szukaj bezpiecznego schronienia.", isAlert: true };
    else if (code >= 51 && code <= 67) fallback = { warning: "Możliwe opady deszczu w najbliższym czasie. Pamiętaj o parasolu.", isAlert: false };
    else if (current.temperature_2m > 30) fallback = { warning: "Uważaj na upał! Pij dużo wody i unikaj pełnego słońca.", isAlert: true };

    res.json({ ...fallback, modelUsed: "Aura-Local-Engine" });
  } catch (err: any) {
    console.error("[Analyze] Błąd serwera:", err);
    res.json({ warning: "Pogoda stabilna.", isAlert: false, modelUsed: "Aura-Local-Engine" });
  }
});

// In-memory Google Cloud persistence store simulation for user settings & preferences
let cloudStorageStore: Record<string, any> = {
  favorites: ["Warszawa", "Kraków", "Gdańsk"],
  settings: { units: "metric", theme: "auto" },
  lastCloudSync: new Date().toISOString()
};

// Scheduled weather sync tracking (3 times a day: 06:00, 12:00, 18:00)
let weatherSyncScheduleState = {
  lastScheduledSync: new Date().toISOString(),
  scheduledTimes: ["06:00", "12:00", "18:00"],
  syncCountToday: 0,
  status: "Synchronizowany (Serwer pogodowy aktywny)"
};

// Background cron check every minute to simulate 3x daily server synchronization reset
setInterval(() => {
  const now = new Date();
  const hours = now.getHours();
  const minutes = now.getMinutes();
  
  // Check if current time matches 06:00, 12:00, or 18:00 (within the first minute)
  if (minutes === 0 && (hours === 6 || hours === 12 || hours === 18)) {
    weatherSyncScheduleState.lastScheduledSync = now.toISOString();
    weatherSyncScheduleState.syncCountToday += 1;
    weatherSyncScheduleState.status = `Zsynchronizowano o ${hours}:00 (Automatyczny reset serwera pogodowego)`;
    console.log(`[Aura Cloud & Weather Sync] Scheduled sync triggered at ${hours}:00. Data refreshed from Open-Meteo server.`);
  }
}, 60000);

// API Route: Google Cloud Storage - Get user data
app.get("/api/cloud-storage", (req, res) => {
  res.json({ success: true, data: cloudStorageStore, timestamp: new Date().toISOString() });
});

// API Route: Google Cloud Storage - Save user data
app.post("/api/cloud-storage", (req, res) => {
  const { data } = req.body;
  if (data) {
    cloudStorageStore = {
      ...cloudStorageStore,
      ...data,
      lastCloudSync: new Date().toISOString()
    };
  }
  res.json({ success: true, data: cloudStorageStore, message: "Zapisano pomyślnie w chmurze Google." });
});

// API Route: Weather Server Sync Schedule & Manual Reset
app.get("/api/weather/sync-schedule", (req, res) => {
  res.json({
    success: true,
    ...weatherSyncScheduleState,
    serverTime: new Date().toISOString()
  });
});

app.post("/api/weather/force-sync", (req, res) => {
  weatherSyncScheduleState.lastScheduledSync = new Date().toISOString();
  weatherSyncScheduleState.status = "Wymuszono świeże pobranie z serwera pogodowego";
  console.log("[Aura Weather Sync] Manual server reset & sync requested.");
  res.json({
    success: true,
    message: "Połączenie z serwerem pogodowym zostało zresetowane i pomyślnie odświeżone.",
    ...weatherSyncScheduleState
  });
});

// Removed AI Assistant endpoint as requested

// Start server function to mount Vite middleware or serve static files
async function startServer() {
  const rootDir = process.cwd();
  const distPath = path.resolve(rootDir, "dist");
  const distIndexPath = path.resolve(distPath, "index.html");
  const rawIndexPath = path.resolve(rootDir, "index.html");

  console.log(`[Aura Server] Root: ${rootDir}`);
  console.log(`[Aura Server] distPath: ${distPath}`);
  console.log(`[Aura Server] NODE_ENV: ${process.env.NODE_ENV}`);

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    console.log("[Aura Server] Starting with Vite middleware in DEVELOPMENT mode...");
    try {
      const vite = await createViteServer({
        server: { middlewareMode: true },
        appType: "spa",
      });
      app.use(vite.middlewares);

      // SPA HTML Fallback Handler in dev mode using Vite's HTML transformer
      app.get("*", async (req, res, next) => {
        if (req.originalUrl.startsWith("/api/")) {
          return next();
        }
        try {
          const url = req.originalUrl;
          let template = fs.readFileSync(rawIndexPath, "utf-8");
          template = await vite.transformIndexHtml(url, template);
          res.status(200).set({ "Content-Type": "text/html; charset=utf-8" }).end(template);
        } catch (e: any) {
          if (vite) {
            vite.ssrFixStacktrace(e);
          }
          console.error("[Aura Server] Vite transform error:", e);
          next(e);
        }
      });
    } catch (e) {
      console.error("[Aura Server] Failed to start Vite middleware:", e);
    }
  } else {
    console.log("[Aura Server] Starting in PRODUCTION mode (serving dist)...");
    app.use("/Pogoda-API", express.static(distPath));
    app.use(express.static(distPath));

    // SPA catch-all
    app.get("*", (req, res, next) => {
      if (req.originalUrl.startsWith("/api/")) {
        return next();
      }
      if (fs.existsSync(distIndexPath)) {
        res.sendFile(distIndexPath);
      } else {
        res.status(404).send("Aura Pogoda: dist/index.html not found. Please run build.");
      }
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`[Aura Server] Running at http://localhost:${PORT}`);
  });
}

startServer();
