import { useEffect, useMemo, useState } from "react";
import {
  Activity, AlertTriangle, BarChart3, CheckCircle2, ClipboardCheck,
  Cloud, Gauge, Leaf, MapPin, Menu, Settings2, ShieldAlert,
  Target, TrendingDown, TrendingUp, X
} from "lucide-react";
import {
  Area, AreaChart, CartesianGrid, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis, Legend
} from "recharts";
import { MapContainer, TileLayer, CircleMarker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";

const API_BASE = "http://127.0.0.1:8000";

async function apiFetch(path) {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `API request failed: ${response.status}`);
  }
  return response.json();
}

async function apiPost(path, body) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `API request failed: ${response.status}`);
  }
  return response.json();
}

const STATION_META = {
  gomti_nagar: { color: "#c94b3c" },
  kendriya_vidyalaya: { color: "#d28a2d" },
  lalbagh: { color: "#3d8b67" },
  talkatora_dic: { color: "#6a73b7" },
};

const FALLBACK_STATIONS = [
  { id: "gomti_nagar", name: "Gomti Nagar", lat: 26.8487, lng: 81.0070, color: "#c94b3c" },
  { id: "kendriya_vidyalaya", name: "Kendriya Vidyalaya", lat: 26.8650, lng: 80.9660, color: "#d28a2d" },
  { id: "lalbagh", name: "Lalbagh", lat: 26.8467, lng: 80.9462, color: "#3d8b67" },
  { id: "talkatora_dic", name: "Talkatora DIC", lat: 26.8320, lng: 80.9070, color: "#6a73b7" },
];

const pages = [
  { id: "dashboard", label: "Dashboard", icon: Gauge },
  { id: "analytics", label: "Analytics", icon: BarChart3 },
  { id: "forecast", label: "Forecast", icon: TrendingUp },
  { id: "risk", label: "Risk & Anomalies", icon: ShieldAlert },
  { id: "inspection", label: "Inspection Priority", icon: ClipboardCheck },
  { id: "policy", label: "Policy Simulator", icon: Settings2 },
  { id: "verification", label: "Policy Verification", icon: CheckCircle2 },
];

function riskClass(label) {
  return label === "Critical" ? "critical" : label === "High" ? "high" : label === "Moderate" ? "moderate" : "low";
}

function Badge({ station }) {
  return <span className={`risk-badge ${riskClass(station.riskLabel || "Low")}`}>{station.riskLabel || "Low"}</span>;
}

function toNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function riskLabelFromScore(score) {
  if (score < 25) return "Low";
  if (score < 50) return "Moderate";
  if (score < 75) return "High";
  return "Critical";
}

function normalizeRiskRow(row, station) {
  const current = toNumber(row["Current PM2.5"] ?? row.current_pm25);
  const forecast = toNumber(row["24h Forecast PM2.5"] ?? row.forecast_pm25 ?? row.predicted_pm25_24h);
  const risk = toNumber(row["Risk Score"] ?? row.risk_score ?? row["Tomorrow Risk"]);
  const anomalies = toNumber(row["Anomaly"] ?? row.anomaly ?? row.anomalies);
  const label = String(row["Tomorrow Risk"] ?? row["Risk Category"] ?? row.risk_category ?? riskLabelFromScore(risk));

  return {
    ...station,
    current,
    forecast,
    risk,
    riskLabel: label,
    anomalies,
    reason: anomalies > 0
      ? "Elevated PM2.5 + unusual recent pattern"
      : forecast > current
        ? "Forecast increase with recent environmental conditions"
        : "Stable forecast relative to current conditions",
    model: row["Forecast Model"] ?? row.forecast_model ?? "Station-specific ML model",
    rank: row["Risk Rank"] ?? row["Rank"] ?? row.risk_rank,
    recentMean: toNumber(row["Recent 24h Mean PM2.5"] ?? row.recent_24h_mean_pm25),
  };
}

function mergeStationMetadata(apiStations) {
  return apiStations.map((s) => ({
    ...s,
    color: STATION_META[s.id]?.color || "#4d7c69",
  }));
}

function selectedStations(selected, stations) {
  return selected === "all" ? stations : stations.filter((s) => s.id === selected);
}

function formatTime(timestamp) {
  if (!timestamp) return "";
  const d = new Date(timestamp);
  if (Number.isNaN(d.getTime())) return String(timestamp).slice(11, 16) || String(timestamp);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
}

function formatDateTime(timestamp) {
  if (!timestamp) return "";
  const d = new Date(timestamp);
  if (Number.isNaN(d.getTime())) return String(timestamp);
  return d.toLocaleString();
}

function Loading({ text = "Loading environmental intelligence..." }) {
  return <div className="notice">⏳ {text}</div>;
}

function ErrorNotice({ error }) {
  if (!error) return null;
  return <div className="notice" style={{ borderColor: "#d92d20" }}>⚠️ {error}</div>;
}

function StationMap({ selected = "all", stations, height = 390, showPopup = true }) {
  const shown = selected === "all" ? stations : stations.filter((s) => s.id === selected);
  const center = [26.8467, 80.9462];

  return <div className="map-card" style={{ height }}>
    <MapContainer center={center} zoom={11} scrollWheelZoom={true} style={{ height: "100%", width: "100%" }}>
      <TileLayer attribution='&copy; OpenStreetMap contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      {shown.map((s) => (
        <CircleMarker
          key={s.id}
          center={[s.lat, s.lng]}
          radius={selected === s.id ? 13 : 10}
          pathOptions={{ color: s.color, fillColor: s.color, fillOpacity: .82, weight: 3 }}
        >
          {showPopup && <Popup>
            <div className="map-popup">
              <strong>{s.name}</strong>
              <span>Current PM2.5: {s.current != null ? `${s.current} µg/m³` : "Loading"}</span>
              <span>24h forecast: {s.forecast != null ? `${s.forecast} µg/m³` : "Loading"}</span>
              <span>Risk score: {s.risk != null ? `${s.risk}/100` : "Loading"}</span>
              <b>{s.anomalies != null ? s.anomalies : 0} unusual pattern signal{s.anomalies === 1 ? "" : "s"}</b>
            </div>
          </Popup>}
        </CircleMarker>
      ))}
    </MapContainer>
  </div>;
}

function App() {
  const [activePage, setActivePage] = useState("dashboard");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [stationId, setStationId] = useState("all");
  const [stations, setStations] = useState(FALLBACK_STATIONS);
  const [apiError, setApiError] = useState("");
  const [fontScale, setFontScale] = useState(100);
  const currentPage = pages.find((p) => p.id === activePage);
  const navigate = (page) => { setActivePage(page); setMobileOpen(false); };

  useEffect(() => {
    apiFetch("/api/stations")
      .then((data) => {
        setStations(mergeStationMetadata(data.stations));
        setApiError("");
      })
      .catch((error) => {
        console.error("Eco Predict API error:", error);
        setApiError("Backend connection unavailable. Showing station fallback metadata.");
      });
  }, []);

  return <div className="app-shell" style={{ "--font-scale": `${fontScale / 100}` }}>
    <aside className={`sidebar ${mobileOpen ? "open" : ""}`}>
      <div className="brand">
        <div className="brand-mark"><Leaf size={24} /></div>
        <div><h1>Eco Predict</h1><p>Environmental Intelligence</p></div>
        <button className="close-mobile" onClick={() => setMobileOpen(false)}><X size={20} /></button>
      </div>
      <div className="nav-label">COMMAND CENTER</div>
      <nav>{pages.map((page) => { const Icon = page.icon; return <button key={page.id}
        className={`nav-item ${activePage === page.id ? "active" : ""}`} onClick={() => navigate(page.id)}>
        <Icon size={18} /><span>{page.label}</span></button>; })}</nav>
      <div className="source-card">
        <div className="source-title">DATA SOURCE</div><strong>CPCB / UPPCB</strong>
        <span>4 monitoring stations · Lucknow</span><div className="online"><span /> Multi-station data</div>
      </div>
    </aside>
    {mobileOpen && <div className="overlay" onClick={() => setMobileOpen(false)} />}
    <main className="main-content">
      <header className="topbar">
        <button className="menu-btn" onClick={() => setMobileOpen(true)}><Menu size={22} /></button>
        <div><div className="eyebrow">ECO PREDICT / COMMAND CENTER</div><h2>{currentPage?.label}</h2></div>
        <div className="topbar-actions">
          <div className="font-size-control" aria-label="Font size controls">
            <span className="font-size-label">Font size:</span>
            <button className="font-size-btn" onClick={() => setFontScale(v => Math.max(80, v - 10))} disabled={fontScale <= 80}>−</button>
            <span className="font-size-value" aria-live="polite">{fontScale}%</span>
            <button className="font-size-btn" onClick={() => setFontScale(v => Math.min(130, v + 10))} disabled={fontScale >= 130}>+</button>
          </div>
          <label className="station-picker"><MapPin size={16} />
            <select value={stationId} onChange={e => setStationId(e.target.value)}>
              <option value="all">All Lucknow Stations</option>
              {stations.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
            </select>
          </label>
        </div>
      </header>
      {apiError && <div style={{ padding: "10px 28px 0" }}><ErrorNotice error={apiError} /></div>}
      <div className="page-content">
        {activePage === "dashboard" && <Dashboard navigate={navigate} selected={stationId} stations={stations} />}
        {activePage === "analytics" && <Analytics selected={stationId} stations={stations} />}
        {activePage === "forecast" && <Forecast selected={stationId} stations={stations} />}
        {activePage === "risk" && <Risk selected={stationId} stations={stations} />}
        {activePage === "inspection" && <Inspection selected={stationId} stations={stations} />}
        {activePage === "policy" && <Policy selected={stationId} stations={stations} />}
        {activePage === "verification" && <Verification selected={stationId} stations={stations} />}
      </div>
    </main>
  </div>;
}

function Dashboard({ navigate, selected, stations }) {
  const [dashboard, setDashboard] = useState(null);
  const [forecasts, setForecasts] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    const query = encodeURIComponent(selected);
    Promise.all([
      apiFetch(`/api/dashboard?station=${query}`),
      Promise.all(selectedStations(selected, stations).map(s => apiFetch(`/api/forecast?station=${encodeURIComponent(s.id)}`)))
    ])
      .then(([dashboardData, forecastData]) => {
        if (cancelled) return;
        setDashboard(dashboardData);
        setForecasts(forecastData);
        setError("");
      })
      .catch(err => !cancelled && setError(err.message || "Could not load dashboard data."));
    return () => { cancelled = true; };
  }, [selected, stations]);

  if (error) return <PageIntro title="Environmental Dashboard" text="Live backend connection could not be loaded."><ErrorNotice error={error} /></PageIntro>;
  if (!dashboard) return <PageIntro title="Environmental Dashboard" text="Loading actual station intelligence from FastAPI."><Loading /></PageIntro>;

  const riskRows = dashboard.risk_rows || [];
  const list = selectedStations(selected, stations).map(s => {
    const row = riskRows.find(r => String(r.Station || "").toLowerCase() === s.name.toLowerCase());
    return row ? normalizeRiskRow(row, s) : s;
  });
  const sorted = [...list].sort((a, b) => b.risk - a.risk);
  const highestCurrent = [...list].sort((a, b) => b.current - a.current)[0];
  const highestForecast = [...list].sort((a, b) => b.forecast - a.forecast)[0];
  const focus = list[0];
  const current = selected === "all" ? dashboard.average_current_pm25 : focus.current;
  const forecast = selected === "all" ? dashboard.average_forecast_pm25 : focus.forecast;
  const risk = selected === "all" ? dashboard.highest_risk_score : focus.risk;
  const anomalies = list.reduce((sum, s) => sum + toNumber(s.anomalies), 0);
  const mapStations = stations.map(s => {
    const row = riskRows.find(r => String(r.Station || "").toLowerCase() === s.name.toLowerCase());
    return row ? normalizeRiskRow(row, s) : s;
  });
  const trendForecast = forecasts[0];
  const trend = trendForecast?.series?.slice(-24).map(item => ({
    time: formatTime(item.timestamp),
    pm25: toNumber(item.pm25),
  })) || [];

  return <>
    <section className="hero"><div><div className="hero-tag"><Activity size={15} /> SYSTEM OPERATIONAL</div>
      <h3>From environmental data<br />to government action.</h3>
      <p>Monitor → Predict → Detect → Prioritize → Simulate → Verify</p></div>
      <div className="hero-meta"><span>MONITORING</span><strong>{stations.length} Stations</strong></div>
    </section>
    <section className="kpi-grid">
      <Kpi title={selected === "all" ? "Average Current PM2.5" : "Current PM2.5"} value={toNumber(current).toFixed(1)} unit="µg/m³" note={selected === "all" ? "Across monitored stations" : "Latest monitored reading"} icon={Cloud} />
      <Kpi title={selected === "all" ? "Average 24h Forecast" : "24h Forecast"} value={toNumber(forecast).toFixed(1)} unit="µg/m³" note={selected === "all" ? "Across station forecasts" : "Station-specific ML forecast"} icon={TrendingUp} />
      <Kpi title={selected === "all" ? "Highest Risk Score" : "Risk Score"} value={toNumber(risk).toFixed(1)} unit="/ 100" note={selected === "all" ? dashboard.highest_risk_station : "Environmental screening score"} icon={AlertTriangle} />
      <Kpi title="Unusual Patterns" value={anomalies} unit="detected" note={selected === "all" ? "Across monitored stations" : "Recent unusual patterns"} icon={Activity} />
    </section>
    <section className="content-grid">
      <Panel title="PM2.5 Environmental Trend" subtitle={selected === "all" ? "Recent actual observations from the selected station set" : "Recent actual observations from the selected station"}>
        <div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><AreaChart data={trend}>
          <defs><linearGradient id="pmFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopOpacity={.28} /><stop offset="100%" stopOpacity={.02} /></linearGradient></defs>
          <CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="time" /><YAxis /><Tooltip />
          <Area type="monotone" dataKey="pm25" strokeWidth={2.5} fill="url(#pmFill)" />
        </AreaChart></ResponsiveContainer></div>
      </Panel>
      <Panel title="Station Risk Overview" subtitle="Tomorrow-focused environmental screening">
        <div className="station-risk-list">{sorted.map(s => <div className="station-risk-row" key={s.id}>
          <div className="station-name"><span className="station-dot" style={{ background: s.color }} /><strong>{s.name}</strong></div>
          <div className="station-risk-value"><strong>{toNumber(s.risk).toFixed(1)}</strong><Badge station={s} /></div>
        </div>)}</div>
      </Panel>
    </section>
    <section className="content-grid single-map-grid"><Panel title="Anomaly & Risk Map" subtitle="Spatial view of environmental screening across monitored stations">
      <StationMap selected={selected} stations={mapStations} height={330} /></Panel></section>
    <section className="action-grid">
      <ActionCard icon={TrendingUp} title="View Forecast" text="Review the next 24-hour PM2.5 forecast." onClick={() => navigate("forecast")} />
      <ActionCard icon={ShieldAlert} title="Review Risks" text="Inspect unusual patterns and environmental risk." onClick={() => navigate("risk")} />
      <ActionCard icon={Settings2} title="Run Policy Scenario" text="Estimate impact of a proposed intervention." onClick={() => navigate("policy")} />
    </section>
    <div className="notice">Highest current station: <strong>{highestCurrent?.name || "—"}</strong> · Highest 24h forecast station: <strong>{highestForecast?.name || "—"}</strong></div>
  </>;
}

function Analytics({ selected, stations }) {
  const [data, setData] = useState(null);
  const [forecasts, setForecasts] = useState([]);
  const [error, setError] = useState("");
  const list = selectedStations(selected, stations);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      apiFetch("/api/analytics"),
      Promise.all(list.map(s => apiFetch(`/api/forecast?station=${encodeURIComponent(s.id)}`)))
    ])
      .then(([analyticsData, forecastData]) => {
        if (cancelled) return;
        setData(analyticsData);
        setForecasts(forecastData);
        setError("");
      })
      .catch(err => !cancelled && setError(err.message || "Could not load analytics."));
    return () => { cancelled = true; };
  }, [selected, stations]);

  if (error) return <PageIntro title="Environmental Analytics" text="Actual backend analytics could not be loaded."><ErrorNotice error={error} /></PageIntro>;
  if (!data) return <PageIntro title="Environmental Analytics" text="Loading actual model and station analytics."><Loading /></PageIntro>;

  const chartData = forecasts[0]?.series?.slice(-24).map((item, i) => {
    const row = { time: formatTime(item.timestamp) };
    forecasts.forEach((f, idx) => { row[list[idx].id] = toNumber(f.series?.[Math.max(0, f.series.length - 24 + i)]?.pm25); });
    return row;
  }) || [];

  return <PageIntro title="Environmental Analytics" text="Historical and model-performance information served by FastAPI.">
    <section className="content-grid">
      <Panel title="Recent PM2.5 Station Comparison" subtitle="Actual observations returned by the station forecasting endpoint">
        <div className="chart-wrap tall"><ResponsiveContainer width="100%" height="100%"><LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="time" /><YAxis /><Tooltip /><Legend />
          {list.map(s => <Line key={s.id} type="monotone" dataKey={s.id} name={s.name} stroke={s.color} strokeWidth={2.5} dot={{ r: 2 }} />)}
        </LineChart></ResponsiveContainer></div>
      </Panel>
      <Panel title="Forecast Model Metrics" subtitle="Station-specific evaluation artifacts">
        <Table headers={["Station", "Model", "MAE", "RMSE"]} rows={(data.forecast_metrics || []).map((r, i) => [
          <strong key={i}>{r.Station || r.station || "—"}</strong>,
          r["Best Model"] || r.best_model || r["Model"] || r.model || "—",
          r.MAE != null ? toNumber(r.MAE).toFixed(2) : "—",
          r.RMSE != null ? toNumber(r.RMSE).toFixed(2) : "—",
        ])} />
      </Panel>
    </section>
    <Panel title="Station Data Quality" subtitle="Quality/coverage information returned by the ML pipeline">
      <Table headers={["Station", "Rows", "Missing PM2.5", "Coverage"]} rows={(data.station_quality || []).map((r, i) => [
        <strong key={i}>{r.Station || r.station || "—"}</strong>,
        r.Rows ?? r.rows ?? r["Row Count"] ?? "—",
        r["Missing PM2.5"] ?? r.missing_pm25 ?? "—",
        r.Coverage ?? r.coverage ?? r["Coverage %"] ?? "—",
      ])} />
    </Panel>
  </PageIntro>;
}

function Forecast({ selected, stations }) {
  const list = selectedStations(selected, stations);
  const [forecasts, setForecasts] = useState([]);
  const [riskData, setRiskData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      Promise.all(list.map(s => apiFetch(`/api/forecast?station=${encodeURIComponent(s.id)}`))),
      apiFetch(`/api/risk?station=${encodeURIComponent(selected)}`)
    ])
      .then(([forecastData, risk]) => {
        if (cancelled) return;
        setForecasts(forecastData);
        setRiskData(risk);
        setError("");
      })
      .catch(err => !cancelled && setError(err.message || "Could not load forecast data."));
    return () => { cancelled = true; };
  }, [selected, stations]);

  if (error) return <PageIntro title="AI Environmental Forecast" text="Actual ML forecast could not be loaded."><ErrorNotice error={error} /></PageIntro>;
  if (!forecasts.length) return <PageIntro title="AI Environmental Forecast" text="24-hour PM2.5 forecast generated by the trained multi-station ML pipeline."><Loading /></PageIntro>;

  const chartData = forecasts[0].series.map((item, i) => {
    const row = { time: formatTime(item.timestamp), timestamp: item.timestamp };
    forecasts.forEach((f, idx) => { row[list[idx].id] = toNumber(f.series?.[i]?.pm25); });
    return row;
  });
  const riskRows = riskData?.stations || [];
  const tableRows = list.map(s => {
    const r = riskRows.find(row => String(row.Station || "").toLowerCase() === s.name.toLowerCase());
    const n = r ? normalizeRiskRow(r, s) : s;
    const priority = n.risk >= 75 ? "Urgent Review" : n.risk >= 50 ? "High Priority" : n.risk >= 30 ? "Review" : "Monitor";
    return [<strong key={s.id}>{s.name}</strong>, `${toNumber(n.current).toFixed(2)} µg/m³`, `${toNumber(n.forecast).toFixed(2)} µg/m³`, <><strong>{toNumber(n.risk).toFixed(1)}</strong> <Badge station={n} /></>, priority];
  });

  return <PageIntro title="AI Environmental Forecast" text="24-hour PM2.5 forecast generated by the trained multi-station ML pipeline.">
    <Panel title="24-Hour PM2.5 Forecast" subtitle="Actual station-specific model output from FastAPI">
      <div className="chart-wrap tall"><ResponsiveContainer width="100%" height="100%"><LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="time" /><YAxis /><Tooltip /><Legend />
        {list.map(s => <Line key={s.id} type="monotone" dataKey={s.id} name={s.name} stroke={s.color} strokeWidth={2.8} dot={{ r: 2 }} />)}
      </LineChart></ResponsiveContainer></div>
    </Panel>
    <Panel title="Tomorrow's Station Risk" subtitle="Forecast-driven screening from the backend risk artifact">
      <Table headers={["Station", "Current", "24h Forecast", "Risk", "Priority"]} rows={tableRows} />
    </Panel>
    <div className="notice">Forecast values are produced by the station-specific Day 5 ML models through the FastAPI backend. They are forecasts, not guaranteed future observations.</div>
  </PageIntro>;
}

function Risk({ selected, stations }) {
  const list = selectedStations(selected, stations);
  const [riskData, setRiskData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    apiFetch(`/api/risk?station=${encodeURIComponent(selected)}`)
      .then(data => { if (!cancelled) { setRiskData(data); setError(""); } })
      .catch(err => !cancelled && setError(err.message || "Could not load risk data."));
    return () => { cancelled = true; };
  }, [selected]);

  if (error) return <PageIntro title="Risk & Anomalies" text="Actual backend risk data could not be loaded."><ErrorNotice error={error} /></PageIntro>;
  if (!riskData) return <PageIntro title="Risk & Anomalies" text="Spatial and station-level screening intelligence from the ML pipeline."><Loading /></PageIntro>;

  const rows = riskData.stations || [];
  const normalized = list.map(s => {
    const r = rows.find(row => String(row.Station || "").toLowerCase() === s.name.toLowerCase());
    return r ? normalizeRiskRow(r, s) : s;
  });
  const highest = Math.max(...normalized.map(s => toNumber(s.risk)), 0);
  const anomalyTotal = normalized.reduce((n, s) => n + toNumber(s.anomalies), 0);
  const mapStations = stations.map(s => {
    const r = rows.find(row => String(row.Station || "").toLowerCase() === s.name.toLowerCase());
    return r ? normalizeRiskRow(r, s) : s;
  });

  return <PageIntro title="Risk & Anomalies" text="Spatial and station-level screening intelligence generated by the multi-station ML pipeline.">
    <div className="three-grid">
      <StatBox icon={ShieldAlert} title="Highest Risk" value={`${highest.toFixed(1)} / 100`} detail="Station screening" />
      <StatBox icon={AlertTriangle} title="Unusual Patterns" value={anomalyTotal} detail="Recent anomaly signals" />
      <StatBox icon={Target} title="Priority" value={highest >= 75 ? "Urgent" : highest >= 50 ? "High" : highest >= 30 ? "Review" : "Monitor"} detail="Inspection screening" />
    </div>
    <section className="content-grid risk-map-grid">
      <Panel title="Anomaly & Risk Map" subtitle="Risk markers are screening indicators">
        <StationMap selected={selected} stations={mapStations} height={420} />
      </Panel>
      <Panel title="Recent Risk Events" subtitle="Station-level screening signals">
        <div className="risk-event-list">{[...normalized].sort((a, b) => b.risk - a.risk).map(s => <div className="risk-event" key={s.id}>
          <div className="risk-event-top"><div className="station-name"><span className="station-dot" style={{ background: s.color }} /><strong>{s.name}</strong></div><Badge station={s} /></div>
          <p>{s.reason}</p><div className="risk-event-meta"><span>PM2.5: {toNumber(s.current).toFixed(2)}</span><span>24h: {toNumber(s.forecast).toFixed(2)}</span><span>Unusual: {toNumber(s.anomalies)}</span></div>
        </div>)}</div>
      </Panel>
    </section>
    <div className="notice">ℹ️ {riskData.signal_explanation} Risk values are prototype prioritization indicators, not official regulatory classifications.</div>
  </PageIntro>;
}

function Inspection({ selected, stations }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    apiFetch("/api/inspection-priority")
      .then(result => { if (!cancelled) { setData(result); setError(""); } })
      .catch(err => !cancelled && setError(err.message || "Could not load inspection priority."));
    return () => { cancelled = true; };
  }, []);

  if (error) return <PageIntro title="Inspection Priority" text="Actual backend inspection ranking could not be loaded."><ErrorNotice error={error} /></PageIntro>;
  if (!data) return <PageIntro title="Inspection Priority" text="Prioritize limited inspection resources using environmental risk signals."><Loading /></PageIntro>;

  let rows = data.rows || [];
  if (selected !== "all") {
    const station = stations.find(s => s.id === selected);
    rows = rows.filter(r => String(r.Station || "").toLowerCase() === String(station?.name || "").toLowerCase());
  }

  return <PageIntro title="Inspection Priority" text="Prioritize limited inspection resources using environmental risk signals.">
    <Panel title="Priority Queue" subtitle="Cross-station screening recommendations from the Day 5 artifact">
      <Table headers={["Rank", "Location", "Risk", "Reason", "Action"]} rows={rows.map((r, i) => {
        const score = toNumber(r["Inspection Priority Score"] ?? r.inspection_priority_score ?? r["Risk Score"]);
        const risk = toNumber(r["Risk Score"] ?? r.risk_score);
        const action = r["Inspection Level"] || (score >= 75 ? "Urgent Review" : score >= 55 ? "High Priority" : score >= 30 ? "Review" : "Monitor");
        return [r["Inspection Rank"] ?? i + 1, <strong key={i}>{r.Station || "—"}</strong>, <strong>{risk.toFixed(1)}</strong>, `Priority score: ${score.toFixed(1)}`, action];
      })} />
    </Panel>
    <div className="notice">🎯 {data.note}</div>
  </PageIntro>;
}

function Policy({ selected, stations }) {
  const [targetStation, setTargetStation] = useState(selected === "all" ? stations[0]?.id : selected);
  const [library, setLibrary] = useState(null);
  const [recommendations, setRecommendations] = useState(null);
  const [selectedPolicy, setSelectedPolicy] = useState("P01");
  const [bundlePolicies, setBundlePolicies] = useState(["P01", "P03"]);
  const [simulation, setSimulation] = useState(null);
  const [bundleResult, setBundleResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (selected !== "all") {
      setTargetStation(selected);
      return;
    }
    apiFetch("/api/risk?station=all")
      .then(data => {
        const rows = data.stations || [];
        const highest = [...rows].sort((a, b) => toNumber(b["Risk Score"]) - toNumber(a["Risk Score"]))[0];
        const match = stations.find(s => String(s.name).toLowerCase() === String(highest?.Station || "").toLowerCase());
        if (match) setTargetStation(match.id);
      })
      .catch(() => setTargetStation(stations[0]?.id));
  }, [selected, stations]);

  useEffect(() => {
    if (!targetStation) return;
    let cancelled = false;
    Promise.all([
      apiFetch("/api/policy/library"),
      apiFetch(`/api/policy/recommend?station=${encodeURIComponent(targetStation)}`)
    ])
      .then(([lib, rec]) => {
        if (cancelled) return;
        setLibrary(lib);
        setRecommendations(rec);
        setError("");
      })
      .catch(err => !cancelled && setError(err.message || "Could not load policy intelligence."));
    return () => { cancelled = true; };
  }, [targetStation]);

  const targetName = stations.find(s => s.id === targetStation)?.name || targetStation;
  const policies = library?.policies || [];

  const toggleBundle = (id) => {
    setBundlePolicies(current => current.includes(id) ? current.filter(x => x !== id) : [...current, id]);
  };

  const runSimulation = async () => {
    setBusy(true);
    try {
      const result = await apiPost("/api/policy/simulate", { station: targetStation, policy_id: selectedPolicy });
      setSimulation(result);
      setError("");
    } catch (err) { setError(err.message || "Policy simulation failed."); }
    finally { setBusy(false); }
  };

  const runBundle = async () => {
    if (!bundlePolicies.length) return setError("Select at least one policy for the bundle.");
    setBusy(true);
    try {
      const result = await apiPost("/api/policy/bundle", { station: targetStation, policy_ids: bundlePolicies });
      setBundleResult(result);
      setError("");
    } catch (err) { setError(err.message || "Policy bundle simulation failed."); }
    finally { setBusy(false); }
  };

  if (error && !library) return <PageIntro title="Policy What-If Simulator" text="Scenario estimates for proposed environmental interventions."><ErrorNotice error={error} /></PageIntro>;
  if (!library) return <PageIntro title="Policy What-If Simulator" text="Scenario estimates for proposed environmental interventions."><Loading text="Loading policy library and recommendation engine..." /></PageIntro>;

  return <PageIntro title="Policy What-If Simulator" text="Scenario estimates for proposed environmental interventions.">
    <ErrorNotice error={error} />
    <section className="content-grid">
      <Panel title="Scenario Controls" subtitle={`Using actual ML forecast baseline for ${targetName}`}>
        <div style={{ display: "grid", gap: 12 }}>
          <label><strong>Simulation station</strong><select value={targetStation} onChange={e => setTargetStation(e.target.value)} style={{ width: "100%", padding: 10, marginTop: 6 }}>
            {stations.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
          </select></label>
          <label><strong>Single policy</strong><select value={selectedPolicy} onChange={e => setSelectedPolicy(e.target.value)} style={{ width: "100%", padding: 10, marginTop: 6 }}>
            {policies.map(p => <option key={p.policy_id} value={p.policy_id}>{p.policy_id} — {p.policy}</option>)}
          </select></label>
          <button className="action-card" onClick={runSimulation} disabled={busy}><Settings2 size={20} /><div><strong>{busy ? "Running..." : "Simulate Policy"}</strong><span>Use the backend's actual 24h ML forecast as baseline.</span></div><span className="arrow">→</span></button>
        </div>
      </Panel>
      <Panel title="Single Policy Result" subtitle="Baseline vs estimated scenario outcome">
        {simulation ? <div className="three-grid compact">
          <StatBox icon={Cloud} title="Baseline" value={toNumber(simulation.baseline_pm25).toFixed(2)} detail="PM2.5 µg/m³" />
          <StatBox icon={TrendingDown} title="Projected" value={toNumber(simulation.projected_pm25).toFixed(2)} detail="PM2.5 µg/m³" />
          <StatBox icon={TrendingDown} title="Reduction" value={`${toNumber(simulation.estimated_reduction).toFixed(2)} µg/m³`} detail={`${simulation.assumed_reduction_pct}% scenario assumption`} />
        </div> : <div className="notice">Run a policy simulation to see the actual backend result.</div>}
      </Panel>
    </section>

    <section className="content-grid">
      <Panel title="Multi-Policy Bundle" subtitle="Combine transparent scenario assumptions">
        <div style={{ display: "grid", gap: 10 }}>
          {policies.map(p => <label key={p.policy_id} style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <input type="checkbox" checked={bundlePolicies.includes(p.policy_id)} onChange={() => toggleBundle(p.policy_id)} />
            <span><strong>{p.policy_id}</strong> — {p.policy} <small>({p.assumed_pm25_reduction_pct}% assumption · {p.budget_units} budget units)</small></span>
          </label>)}
          <button className="action-card" onClick={runBundle} disabled={busy}><Settings2 size={20} /><div><strong>{busy ? "Running..." : "Simulate Policy Bundle"}</strong><span>Evaluate the selected intervention package.</span></div><span className="arrow">→</span></button>
        </div>
      </Panel>
      <Panel title="Bundle Result" subtitle="Combined scenario estimate">
        {bundleResult ? <div className="three-grid compact">
          <StatBox icon={Cloud} title="Baseline" value={toNumber(bundleResult.baseline_pm25).toFixed(2)} detail="PM2.5 µg/m³" />
          <StatBox icon={TrendingDown} title="Projected" value={toNumber(bundleResult.projected_pm25).toFixed(2)} detail={`${bundleResult.combined_assumed_reduction_pct}% combined assumption`} />
          <StatBox icon={TrendingDown} title="Reduction" value={`${toNumber(bundleResult.estimated_reduction).toFixed(2)} µg/m³`} detail={`${bundleResult.total_budget_units} budget units`} />
        </div> : <div className="notice">Run a bundle simulation to see the actual backend result.</div>}
      </Panel>
    </section>

    <Panel title="Budget-Aware Recommendations" subtitle={`Transparent comparison for ${recommendations?.station || targetName}`}>
      <Table headers={["Rank", "Policy", "Projected", "Reduction", "Budget", "Impact / Unit"]} rows={(recommendations?.recommendations || []).map(r => [
        r.recommendation_rank,
        <strong key={r.policy_id}>{r.policy_id} — {r.policy}</strong>,
        `${toNumber(r.projected_pm25).toFixed(2)} µg/m³`,
        `${toNumber(r.estimated_reduction).toFixed(2)} µg/m³`,
        r.budget_units,
        toNumber(r.impact_per_budget_unit).toFixed(3),
      ])} />
    </Panel>
    <div className="notice">ℹ️ {recommendations?.note || library.estimate_note}</div>
  </PageIntro>;
}

function Verification() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    apiFetch("/api/policy/verification")
      .then(result => { if (!cancelled) { setData(result); setError(""); } })
      .catch(err => !cancelled && setError(err.message || "Could not load verification framework."));
    return () => { cancelled = true; };
  }, []);

  if (error) return <PageIntro title="Policy Effectiveness Verification" text="Compare expected outcomes with observed environmental changes after an intervention."><ErrorNotice error={error} /></PageIntro>;
  if (!data) return <PageIntro title="Policy Effectiveness Verification" text="Compare expected outcomes with observed environmental changes after an intervention."><Loading /></PageIntro>;

  const verification = data.verification || {};
  const required = verification.required_inputs || [];

  return <PageIntro title="Policy Effectiveness Verification" text="Compare expected outcomes with observed environmental changes after an intervention.">
    <Panel title="Verification Framework" subtitle="Actual backend verification status">
      <div className="three-grid compact">
        <StatBox icon={CheckCircle2} title="Status" value="Awaiting" detail="Verified post-intervention observations" />
        <StatBox icon={Target} title="Required Inputs" value={required.length} detail="Evidence fields needed" />
        <StatBox icon={ClipboardCheck} title="Fabrication" value="None" detail="Observed values are not invented" />
      </div>
      <div className="notice" style={{ marginTop: 16 }}>{verification.status || "Awaiting verified post-intervention observations"}</div>
      <Table headers={["Required input", "Purpose"]} rows={required.map((item, i) => [<strong key={i}>{item}</strong>, "Required for effectiveness verification"])} />
    </Panel>
    <div className="notice">✓ {data.note}</div>
  </PageIntro>;
}

function Kpi({ title, value, unit, note, icon: Icon }) { return <div className="kpi-card"><div className="kpi-head"><span>{title}</span><Icon size={19} /></div><div className="kpi-value">{value}</div><div className="kpi-unit">{unit}</div><div className="kpi-note">{note}</div></div>; }
function Panel({ title, subtitle, children }) { return <section className="panel"><div className="panel-head"><div><h3>{title}</h3><p>{subtitle}</p></div></div>{children}</section>; }
function PageIntro({ title, text, children }) { return <><section className="page-intro"><div className="hero-tag">ENVIRONMENTAL INTELLIGENCE</div><h3>{title}</h3><p>{text}</p></section>{children}</>; }
function ActionCard({ icon: Icon, title, text, onClick }) { return <button className="action-card" onClick={onClick}><Icon size={22} /><div><strong>{title}</strong><span>{text}</span></div><span className="arrow">→</span></button>; }
function StatBox({ icon: Icon, title, value, detail }) { return <div className="stat-box"><Icon size={20} /><span>{title}</span><strong>{value}</strong><small>{detail}</small></div>; }
function Table({ rows, headers }) { return <div className="table-wrap"><table><thead><tr>{headers.map(h => <th key={h}>{h}</th>)}</tr></thead><tbody>{rows.length ? rows.map((row, i) => <tr key={i}>{row.map((cell, j) => <td key={j}>{cell}</td>)}</tr>) : <tr><td colSpan={headers.length}>No data returned by backend.</td></tr>}</tbody></table></div>; }

export default App;
