from flask import Flask, send_file, render_template, request, redirect, url_for
import pyart
import matplotlib.pyplot as plt
import tempfile
import os
from datetime import datetime, timedelta
import numpy as np
import json

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static/radar", exist_ok=True)

# Global state
radar_groups = {}
available_fields = []
radar_extent = [[-10, 100], [10, 120]]  # fallback extent
time_bounds = {}

# --- Default configs (Qt style) ---
default_configs = {
	"DBZ2":   dict(vmin=-20, vmax=70, cmap="turbo"),
	"VEL2":   dict(vmin=-30, vmax=30, cmap="seismic"),
	"WIDTH2": dict(vmin=0, vmax=5, cmap="turbo"),
	"ZDR2":   dict(vmin=-5, vmax=5, cmap="turbo"),
	"KDP2":   dict(vmin=-1, vmax=5, cmap="viridis"),
	"RHOHV2": dict(vmin=0.8, vmax=1.05, cmap="seismic"),
	"VELC2":  dict(vmin=-30, vmax=30, cmap="seismic"),
	"SQI2":   dict(vmin=0, vmax=1, cmap="viridis"),
	"PHIDP2": dict(vmin=-180, vmax=180, cmap="twilight_shifted"),
	"HCLASS2":dict(vmin=0, vmax=10, cmap="turbo"),
	"SNR16":  dict(vmin=-20, vmax=40, cmap="turbo"),
	"PMI16":  dict(vmin=0, vmax=1, cmap="viridis"),
	"LOG16":  dict(vmin=0, vmax=5, cmap="plasma"),
	"CSP16":  dict(vmin=0, vmax=1, cmap="turbo"),
}


def render_radar_png(filepath, field, cmap_override="", vmin_override=None, vmax_override=None, filters=None):
    if filters is None:
        filters = {}

    radar = pyart.io.read_sigmet( filepath, file_field_names=True, full_xhdr=True, time_ordered="full" )
    warnings = []

    # --- Field fallback ---
    if field not in radar.fields:
        if "DBZ2" in radar.fields:
            field = "DBZ2"
        else:
            field = list(radar.fields.keys())[0]

    # --- Get data array ---
    data = radar.fields[field]["data"].copy()

    # --- Apply filters ---
    if filters.get("useSQI") and "SQI2" in radar.fields:
        sqi_min = float(filters.get("sqi") or 0)
        sqi = radar.fields["SQI2"]["data"]
        data = np.ma.masked_where(sqi < sqi_min, data)
    elif filters.get("useSQI"):
        warnings.append("SQI2 field not available")

    if filters.get("usePMI") and "PMI16" in radar.fields:
        pmi_min = float(filters.get("pmi") or 0)
        pmi = radar.fields["PMI16"]["data"]
        data = np.ma.masked_where(pmi < pmi_min, data)
    elif filters.get("usePMI"):
        warnings.append("PMI16 field not available")

    if filters.get("useLOG"):
        log_min = float(filters.get("log") or 0)
        data = np.ma.masked_where(data < log_min, data)

    if filters.get("clipRange"):
        if vmin_override is not None and vmax_override is not None:
            data = np.clip(data, vmin_override, vmax_override)

    if filters.get("maskInvalid"):
        data = np.ma.masked_invalid(data)

    if filters.get("speckle"):
        from scipy.ndimage import median_filter
        data = median_filter(data, size=3)

    if filters.get("clutter"):
        data = np.ma.masked_where(data < 0, data)

    if filters.get("dealias") and field.startswith("VEL"):
        try:
            data = pyart.correct.dealias_region_based(radar, vel_field=field)
        except Exception as e:
            warnings.append(f"Dealiasing failed: {e}")

    # --- Replace radar field with filtered data ---
    radar.fields[field]["data"] = data

    # --- Plot ---
    display = pyart.graph.RadarDisplay(radar)
    fig, ax = plt.subplots(figsize=(6, 6))
    display.plot(field, 0, ax=ax,
                 colorbar_flag=False,
                 vmin=vmin_override,
                 vmax=vmax_override,
                 cmap=(cmap_override or "turbo"),
                 title_flag=False)
    ax.axis("off")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir="static/radar")
    plt.savefig(tmp.name, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)

    # --- Extent (safe fallback) ---
    try:
        lat = radar.latitude['data'][0]
        lon = radar.longitude['data'][0]
        rng = radar.range['data'][-1] / 1000.0
        d = rng / 111.0
        extent = [lat - d, lon - d, lat + d, lon + d]
    except Exception:
        warnings.append("Extent calculation failed, using fallback box")
        extent = [-10, 100, 10, 120]

    return tmp.name, extent, radar, warnings



def get_radar_timestamp_and_epoch(radar):
	try:
		start = radar.time['units']
		base = start.split("since")[-1].strip()
		base_dt = datetime.fromisoformat(base.replace("Z", ""))
		sweep_time = base_dt + timedelta(seconds=float(radar.time['data'][0]))
		return sweep_time.strftime("%Y-%m-%d %H:%M:%S"), sweep_time.timestamp()
	except Exception:
		return "N/A", None


def find_field_name(radar, candidates):
	"""Return the first matching field name from candidates list."""
	for c in candidates:
		if c in radar.fields:
			return c
	return None
	

@app.route("/", methods=["GET", "POST"])
def index():
	global radar_groups, available_fields, radar_extent, time_bounds

	if request.method == "POST":
		radar_groups = {}
		time_bounds = {}

		for f in request.files.getlist("radarfiles"):
			filepath = os.path.join(UPLOAD_FOLDER, f.filename)
			f.save(filepath)

			try:
				radar = pyart.io.read_sigmet( filepath, file_field_names=True, full_xhdr=True, time_ordered="full" )
			except Exception as e:
				print(f"⚠️ Skipping {f.filename}: {e}")
				continue

			if len(radar.sweep_start_ray_index['data']) == 0:
				print(f"⚠️ Skipping {f.filename}: no sweeps found")
				continue

			site = radar.metadata.get("instrument_name", "UNKNOWN")
			task = radar.metadata.get("task_name") or radar.metadata.get("sigmet_task_name", "UNKNOWN")
			key = f"{site} - {task}"

			ts_str, ts_epoch = get_radar_timestamp_and_epoch(radar)
			radar_groups.setdefault(key, []).append({"file": filepath, "time": ts_str, "epoch": ts_epoch})

		# sort & bounds
		for key in radar_groups:
			radar_groups[key].sort(key=lambda x: x["file"])
			epochs_list = [f["epoch"] for f in radar_groups[key] if f["epoch"] is not None]
			if epochs_list:
				time_bounds[key] = {"min": min(epochs_list), "max": max(epochs_list)}

		# set available fields + extent
		if radar_groups:
			first_file = list(radar_groups.values())[0][0]["file"]
			try:
				radar = pyart.io.read_sigmet( first_file, file_field_names=True, full_xhdr=True, time_ordered="full" )
				available_fields = list(radar.fields.keys())
				lat = radar.latitude['data'][0]
				lon = radar.longitude['data'][0]
				rng = radar.range['data'][-1] / 1000.0
				d = rng / 111.0
				radar_extent = [[lat - d, lon - d], [lat + d, lon + d]]
			except Exception as e:
				print(f"⚠️ Could not read first file fields: {e}")
				available_fields = []

		return redirect(url_for("index"))

	return render_template(
		"map.html",
		fields=available_fields,
		extent=radar_extent,
		groups=list(radar_groups.keys()),
		times={g: [f["time"] for f in radar_groups[g]] for g in radar_groups},
		epochs={g: [f["epoch"] for f in radar_groups[g]] for g in radar_groups},
		bounds=time_bounds
	)


@app.route("/radar")
def radar_overlay():
	group = request.args.get("group")
	idx = int(request.args.get("frame", 0))

	field = request.args.get("field")
	# --- SAFETY CHECK: fall back if invalid field ---
	if not field or field not in available_fields:
		if "DBZ2" in available_fields:
			field = "DBZ2"
		elif available_fields:
			field = available_fields[0]
		else:
			return "No radar fields available", 404

	cmap_override = request.args.get("cmap", "")
	vmin_override = float(request.args.get("vmin", "") or "nan")
	vmax_override = float(request.args.get("vmax", "") or "nan")

	vmin_override = None if np.isnan(vmin_override) else vmin_override
	vmax_override = None if np.isnan(vmax_override) else vmax_override

	# --- Parse filters safely ---
	filters_arg = request.args.get("filters")
	try:
		filters = json.loads(filters_arg) if filters_arg else {}
	except Exception:
		filters = {}

	if not radar_groups or group not in radar_groups:
		return "No radar files loaded", 404

	files = radar_groups[group]
	idx = idx % len(files)
	filepath = files[idx]["file"]
	timestamp_str = files[idx]["time"]

	png, extent, radar, warnings = render_radar_png(
		filepath, field, cmap_override, vmin_override, vmax_override, filters
	)

	resp = send_file(png, mimetype="image/png")
	resp.headers['X-Extent'] = f"{extent[0]},{extent[1]},{extent[2]},{extent[3]}"
	resp.headers['X-Frames'] = str(len(files))
	resp.headers['X-Timestamp'] = timestamp_str
	if warnings:
		resp.headers['X-Warnings'] = "; ".join(warnings)
	return resp




if __name__ == "__main__":
	app.run(host="0.0.0.0", port=8010, debug=True)
