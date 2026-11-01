import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_csv(path):
    data = {k: [] for k in ['t', 'x', 'y', 'yaw', 'steer_rad', 'x_ref', 'y_ref', 'cte', 'heading_err']}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in data: data[k].append(float(row[k]))
    return {k: np.array(v) for k, v in data.items()}

def compute_metrics(d):
    dsteer = np.diff(d['steer_rad']) / np.maximum(np.diff(d['t']), 1e-3)
    return {
        'rms_cte': float(np.sqrt(np.mean(d['cte'] ** 2))),
        'max_cte': float(np.max(np.abs(d['cte']))),
        'rms_heading': float(np.sqrt(np.mean(d['heading_err'] ** 2))),
        'rms_srate': float(np.sqrt(np.mean(dsteer ** 2))),
    }

def main():
    files = {
        'PID': 'evaluation/pid_figure8_telemetry.csv',
        'LQR': 'evaluation/lqr_figure8_telemetry.csv',
        'LTV-MPC': 'evaluation/mpc_figure8_telemetry.csv',
        'NMPC': 'evaluation/nmpc_figure8_telemetry.csv'
    }
    
    data = {}
    metrics = {}
    for name, path in files.items():
        if os.path.exists(path):
            data[name] = load_csv(path)
            metrics[name] = compute_metrics(data[name])

    if not data:
        print("No CSV files found.")
        return

    print('================================================================================')
    print(f"{'Metric':<25} | {'PID':<10} | {'LQR':<10} | {'LTV-MPC':<10} | {'NMPC':<10}")
    print('--------------------------------------------------------------------------------')
    m_keys = [('RMS CTE (m)', 'rms_cte'), ('Max CTE (m)', 'max_cte'), ('RMS Heading (rad)', 'rms_heading'), ('Steer Rate (rad/s)', 'rms_srate')]
    for label, key in m_keys:
        row = f"{label:<25} | "
        for name in ['PID', 'LQR', 'LTV-MPC', 'NMPC']:
            row += f"{metrics[name][key]:<10.4f} | " if name in metrics else f"{'N/A':<10} | "
        print(row)
    print('================================================================================')

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(data[list(data.keys())[0]]['x_ref'], data[list(data.keys())[0]]['y_ref'], 'k--', lw=2, label='Reference')
    
    colors = {'PID': 'b', 'LQR': 'g', 'LTV-MPC': 'r', 'NMPC': 'm'}
    for name, d in data.items():
        ax.plot(d['x'], d['y'], f'{colors[name]}-', lw=1.5, label=f"{name} (CTE={metrics[name]['rms_cte']:.3f}m)")
        
    ax.set_title('4-Way Trajectory Tracking Comparison (Figure-8)')
    ax.axis('equal')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('evaluation/4way_comparison.png', dpi=150)

if __name__ == '__main__': main()
