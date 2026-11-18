import { useEffect, useMemo, useState } from 'react';
import { Activity, ArrowUpRight, Clock3, Compass, Gauge, MapPinned, RefreshCw, Route, Server, Wifi, WifiOff } from 'lucide-react';
import { getHealth, getTrips, predictTrip } from './lib/api.js';
import RouteMap from './components/RouteMap.jsx';

function Stat({ icon: Icon, label, value, suffix = '' }) {
  return <div className="stat-card"><div className="stat-icon"><Icon size={17}/></div><div><div className="stat-label">{label}</div><div className="stat-value">{value ?? '—'}<small>{value == null ? '' : suffix}</small></div></div></div>;
}

export default function App() {
  const [trips, setTrips] = useState([]);
  const [selectedTrip, setSelectedTrip] = useState('');
  const [result, setResult] = useState(null);
  const [serverStatus, setServerStatus] = useState('checking');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function loadTrips() {
    setError('');
    try {
      const [health, availableTrips] = await Promise.all([getHealth(), getTrips()]);
      setServerStatus(health.status === 'ok' ? 'online' : 'offline');
      setTrips(availableTrips);
      setSelectedTrip((current) => current || availableTrips[0] || '');
    } catch (err) {
      setServerStatus('offline');
      setError(err.response?.data?.detail || err.message || 'Could not connect to the PathSnap API.');
    }
  }

  useEffect(() => { loadTrips(); }, []);

  async function runPrediction() {
    if (!selectedTrip) return;
    setLoading(true); setError(''); setResult(null);
    try {
      const data = await predictTrip(selectedTrip);
      if (!Array.isArray(data.raw_gps) || !Array.isArray(data.predicted_route)) {
        throw new Error('Prediction response must include raw_gps and predicted_route coordinate arrays.');
      }
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Prediction failed.');
    } finally { setLoading(false); }
  }

  const summary = result?.summary || {};
  const hasMap = Boolean(result?.raw_gps?.length || result?.predicted_route?.length);
  const observations = useMemo(() => summary.observations ?? result?.raw_gps?.length, [summary, result]);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-mark"><Route size={22}/></div><div><strong>PathSnap</strong><span>MAP MATCHING SYSTEM</span></div></div>
        <div className="side-label">WORKSPACE</div>
        <div className="nav-item active"><MapPinned size={17}/> Evaluator dashboard</div>
        <div className="sidebar-bottom"><div className={`connection ${serverStatus}`}><span className="status-dot"/><div><b>{serverStatus === 'online' ? 'Server connected' : serverStatus === 'checking' ? 'Checking server' : 'Server unavailable'}</b><small>FastAPI · Laptop 1</small></div></div><div className="side-foot">PathSnap · Team project</div></div>
      </aside>

      <main className="main-content">
        <header className="topbar"><div className="breadcrumb">Workspace <span>/</span> Evaluator</div><div className="top-actions"><span className="live-label"><i/> LIVE WORKSPACE</span><button className="icon-button" onClick={loadTrips} title="Refresh connection"><RefreshCw size={16}/></button></div></header>

        <section className="page-heading"><div><div className="eyebrow">DISTRIBUTED ROUTE RECONSTRUCTION</div><h1>Evaluator dashboard</h1><p>Inspect noisy GPS observations and the route reconstructed by PathSnap.</p></div><div className="heading-chip"><Activity size={16}/> Prediction workspace</div></section>

        <section className="control-panel"><div className="control-copy"><div className="control-icon"><Compass size={20}/></div><div><h2>Select a trip</h2><p>Choose a permitted evaluation trip to run map matching.</p></div></div><div className="control-actions"><select value={selectedTrip} onChange={(e) => setSelectedTrip(e.target.value)} disabled={!trips.length}><option value="">Select trip</option>{trips.map((trip) => <option key={trip} value={trip}>Trip {trip}</option>)}</select><button className="primary-button" onClick={runPrediction} disabled={!selectedTrip || loading || serverStatus !== 'online'}>{loading ? <RefreshCw className="spin" size={16}/> : <Route size={16}/>} {loading ? 'Reconstructing…' : 'Run reconstruction'} <ArrowUpRight size={15}/></button></div></section>

        {error && <div className="error-banner"><WifiOff size={17}/><div><b>Unable to complete request</b><span>{error}</span></div></div>}

        <section className="stats-grid">
          <Stat icon={MapPinned} label="Selected trip" value={result?.trip_id || selectedTrip || '—'}/>
          <Stat icon={Activity} label="GPS observations" value={observations}/>
          <Stat icon={Route} label="Matched segments" value={summary.matched_segments}/>
          <Stat icon={Clock3} label="Processing time" value={summary.processing_ms} suffix=" ms"/>
        </section>

        <section className="map-section"><div className="section-heading"><div><h2>Trajectory visualization</h2><p>Raw GPS points compared with the reconstructed path.</p></div><div className="map-tag"><span/> INTERACTIVE MAP</div></div><div className="map-card">{hasMap ? <RouteMap result={result}/> : <div className="empty-map"><div className="empty-map-icon"><MapPinned size={30}/></div><h3>Map preview will appear here</h3><p>Connect to the PathSnap server and run a trip reconstruction to visualize its GPS observations and predicted route.</p><div className="empty-hint"><Server size={14}/> Waiting for prediction data</div></div>}</div></section>

        <section className="bottom-grid"><div className="info-card"><div className="info-icon"><Wifi size={18}/></div><div><h3>Distributed deployment</h3><p>The evaluator interface runs on Laptop 2 and communicates with the FastAPI model server on Laptop 1 over the local network.</p></div></div><div className="info-card"><div className="info-icon"><Gauge size={18}/></div><div><h3>Evaluation integrity</h3><p>Only prediction outputs and operational summaries are displayed. Evaluator reference labels remain server-side and are never sent to this UI.</p></div></div></section>
        <footer>PATHSNAP <span>·</span> GPS MAP MATCHING &amp; ROUTE RECONSTRUCTION</footer>
      </main>
    </div>
  );
}
