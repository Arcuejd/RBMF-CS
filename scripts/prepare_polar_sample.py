"""Download public polar sample inputs and record source manifests."""
import argparse
import concurrent.futures
import hashlib
import json
import re
import ssl
from datetime import datetime, timedelta, timezone
import numpy as np
import time
import urllib.request
import certifi
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'polar_sample'
SOURCES = {
    'oras_catalog': 'https://icdc.cen.uni-hamburg.de/thredds/catalog/ftpthredds/EASYInit/oras5/r1x1/catalog.xml',
    'ice_north': 'https://noaadata.apps.nsidc.org/NOAA/G02202_V6/north/',
    'ice_ancillary': 'https://noaadata.apps.nsidc.org/NOAA/G02202_V6/ancillary/',
    'itp_directory': 'https://scienceweb.whoi.edu/itp/data/',
    'argo_index': 'https://data-argo.ifremer.fr/ar_index_global_prof.txt.gz',
}


def download(url, relative):
    dest = OUT / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        last = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'RBMF-CS research data preparation'})
                with urllib.request.urlopen(req, timeout=60, context=ssl.create_default_context(cafile=certifi.where())) as response, dest.with_suffix(dest.suffix + '.part').open('wb') as f:
                    while chunk := response.read(1024 * 1024):
                        f.write(chunk)
                dest.with_suffix(dest.suffix + '.part').replace(dest)
                break
            except Exception as exc:
                last = exc
                if attempt < 2:
                    time.sleep(2)
        if not dest.exists():
            raise RuntimeError(f'{url}: {last}')
    digest = hashlib.sha256()
    with dest.open('rb') as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    info = {'url': url, 'path': str(dest), 'bytes': dest.stat().st_size, 'sha256': digest.hexdigest(),
            'saved_utc':datetime.fromtimestamp(dest.stat().st_mtime, timezone.utc).isoformat()}
    print(json.dumps(info), flush=True)
    return info


def probe():
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download, url, 'catalogs/' + name + ('.txt.gz' if name == 'argo_index' else '.txt')): name
                   for name, url in SOURCES.items()}
        for f in concurrent.futures.as_completed(futures):
            name = futures[f]
            try:
                results[name] = f.result()
            except Exception as exc:
                results[name] = {'url': SOURCES[name], 'error': str(exc)}
                print(name, str(exc), flush=True)
    (OUT / 'probe.json').write_text(json.dumps(results, indent=2), encoding='utf-8')


def batch(jobs, filename):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        fs = {pool.submit(download, u, p): u for u, p in jobs}
        for f in concurrent.futures.as_completed(fs):
            try:
                results.append(f.result())
            except Exception as exc:
                results.append({'url': fs[f], 'error': str(exc)})
                print(str(exc), flush=True)
    (OUT / filename).write_text(json.dumps(results, indent=2), encoding='utf-8')


def discover():
    base = 'https://icdc.cen.uni-hamburg.de/thredds/catalog/ftpthredds/EASYInit/oras5/r1x1/'
    jobs = [(base + v + '/catalog.xml', 'catalogs/oras_' + v + '.xml') for v in ['votemper', 'vosaline', 'LSM_r1x1']]
    jobs.append(('https://noaadata.apps.nsidc.org/NOAA/G02202_V6/north/monthly/', 'catalogs/ice_monthly.html'))
    jobs.append(('https://scienceweb.whoi.edu/itp/data/rawloc/', 'catalogs/itp_rawloc.html'))
    dirs = re.findall(r'href="(itpsys\d+/)"', (OUT / 'catalogs/itp_directory.txt').read_text())
    jobs.extend(('https://scienceweb.whoi.edu/itp/data/' + d, 'catalogs/' + d[:-1] + '.html') for d in dirs)
    batch(jobs, 'discovery.json')


def inventory():
    base = 'https://icdc.cen.uni-hamburg.de/thredds/catalog/ftpthredds/EASYInit/oras5/r1x1/'
    jobs = [(base + v + '/opa0/catalog.xml', 'catalogs/oras_' + v + '_opa0.xml') for v in ['votemper', 'vosaline']]
    for p in (OUT / 'catalogs').glob('itpsys*.html'):
        name = next(iter(re.findall(r'href="(itp\d+rawlocs\.dat)"', p.read_text())), None)
        if name:
            jobs.append(('https://scienceweb.whoi.edu/itp/data/' + p.stem + '/' + name, 'itp/tracks/' + name))
    for month in ['201401', '201404', '201407', '201410']:
        name = next(iter(re.findall(r'href="(sic_psn25_' + month + r'[^\"]+\.nc)"', (OUT / 'catalogs/ice_monthly.html').read_text())), None)
        if not name:
            raise ValueError('Sea ice month missing: ' + month)
        jobs.append(('https://noaadata.apps.nsidc.org/NOAA/G02202_V6/north/monthly/' + name, 'ice/' + name))
    jobs.append(('https://noaadata.apps.nsidc.org/NOAA/G02202_V6/ancillary/G02202-ancillary-psn25-v06r00.nc', 'ice/G02202-ancillary-psn25-v06r00.nc'))
    rows = json.loads((ROOT / 'outputs/argo_2014_candidates.json').read_text())
    jobs.extend(('https://data-argo.ifremer.fr/dac/' + r['file'], 'argo/profiles/' + r['file']) for r in rows)
    batch(jobs, 'inventory_downloads.json')


def oras():
    jobs = []
    for variable in ['votemper', 'vosaline']:
        text = (OUT / ('catalogs/oras_' + variable + '_opa0.xml')).read_text()
        for month in ['201401', '201404', '201407', '201410']:
            name = variable + '_ORAS5_1m_' + month + '_r1x1.nc'
            if name not in text:
                raise ValueError('ORAS5 month absent from catalog: ' + name)
            url = 'https://icdc.cen.uni-hamburg.de/thredds/fileServer/ftpthredds/EASYInit/oras5/r1x1/' + variable + '/opa0/' + name
            jobs.append((url, 'oras5/' + name))
    text = (OUT / 'catalogs/oras_LSM_r1x1.xml').read_text()
    jobs.extend(('https://icdc.cen.uni-hamburg.de/thredds/fileServer/' + p, 'oras5/' + Path(p).name)
                for p in re.findall(r'urlPath="([^\"]+\.nc)"', text))
    batch(jobs, 'oras_downloads.json')


def itp_selected():
    jobs, selected = [], []
    for p in sorted((OUT / 'itp/tracks').glob('*.dat')):
        try:
            a = np.loadtxt(p, comments='%')
            if a.ndim != 2 or a.shape[1] != 4:
                continue
            rows = a[(a[:, 0] == 2014) & (a[:, 3] >= 70) & (a[:, 3] <= 90)]
            months = { (datetime(2014, 1, 1) + timedelta(days=float(r[1])-1)).month for r in rows }
            if not months.intersection({1, 4, 7, 10}):
                continue
            system = re.search(r'itp(\d+)', p.name).group(1)
            catalog = (OUT / ('catalogs/itpsys' + system + '.html')).read_text()
            name = 'itp' + system + 'final.zip'
            record = {'system': int(system), 'track_months': sorted(months), 'level3_available': name in catalog}
            selected.append(record)
            if name in catalog:
                jobs.append(('https://scienceweb.whoi.edu/itp/data/itpsys' + system + '/' + name, 'itp/final/' + name))
        except Exception as exc:
            print(p.name, str(exc), flush=True)
    (OUT / 'itp_selected.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
    print('ITP_SELECTED', json.dumps(selected), flush=True)
    batch(jobs, 'itp_downloads.json')


def finish():
    # Copy only the fixed reports generated for this preparation task.
    reports = OUT / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    for source, name in [
        (ROOT/'docs/polar_sample_download_report.md', 'README_zh.md'),
        (ROOT/'outputs/polar_sample_audit/coverage.json', 'coverage.json'),
        (ROOT/'outputs/polar_sample_audit/profile_inventory.json', 'profile_inventory.json'),
    ]:
        (reports/name).write_bytes(source.read_bytes())
    records = {}
    for name in ['discovery.json', 'inventory_downloads.json', 'oras_downloads.json', 'itp_downloads.json']:
        for r in json.loads((OUT/name).read_text()):
            if 'error' in r:
                raise ValueError('Unresolved download error: '+r['url'])
            records[r['path']] = r
    manifest = {'sample_months':['201401', '201404', '201407', '201410'],
                'prepared_date_local':'2026-10-11', 'timezone':'Asia/Shanghai',
                'sources':{'oras5':'ICDC r1x1 opa0', 'ice':'NOAA NSIDC G02202 V6 north monthly',
                           'argo':'GDAC profile index and 420 candidate files', 'itp':'WHOI Level III final'},
                'files':list(records.values()), 'status':'downloads_and_coverage_audit_complete',
                'physical_alignment_status':'not_yet_performed'}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Reports and manifest saved to', OUT, flush=True)


def iap_baseline():
    batch([
        ('https://www.ocean.iap.ac.cn/ftp/cheng/IAPv4.2_IAP_Temperature_gridded_1month_netcdf/', 'catalogs/iap_v42.html'),
        ('http://www.ocean.iap.ac.cn/ftp/cheng/IAPv4.2_IAP_Temperature_gridded_1month_netcdf/', 'catalogs/iap_v42_http.html'),
    ], 'iap_baseline_discovery.json')


def iap_catalogs():
    batch([
        ('http://www.ocean.iap.ac.cn/ftp/cheng/', 'catalogs/iap_cheng_root.html'),
        ('http://www.ocean.iap.ac.cn/ftp/cheng/IAPv4.2_IAP_Temperature_gridded_1month_netcdf/Monthly/', 'catalogs/iap_absolute_monthly.html'),
    ], 'iap_catalogs.json')


def iap_climatology_catalogs():
    batch([
        ('http://www.ocean.iap.ac.cn/ftp/cheng/IAP_TS_climatology2024/', 'catalogs/iap_ts_climatology2024.html'),
        ('http://www.ocean.iap.ac.cn/ftp/cheng/IAPv4_temperature_climatology/', 'catalogs/iap_v4_climatology.html'),
        ('http://www.ocean.iap.ac.cn/ftp/cheng/IAPv4.2_IAP_Temperature_gridded_1month_netcdf/Monthly/IAPv4_Temp_monthly_1_6000m_year_2000_month_01.nc', 'iap_baseline/IAPv4_Temp_monthly_1_6000m_year_2000_month_01.nc'),
    ], 'iap_climatology_catalogs.json')


def iap_climatology_files():
    jobs = []
    for m in range(1, 13):
        name = f'IAP_temperature_2024v_climatology_1_6000m_month_{m:02d}.nc'
        jobs.append(('http://www.ocean.iap.ac.cn/ftp/cheng/IAP_TS_climatology2024/' + name, 'iap_baseline/' + name))
        name = f'IAPv4_Temp_monthly_1_6000m_year_2000_month_{m:02d}.nc'
        jobs.append(('http://www.ocean.iap.ac.cn/ftp/cheng/IAPv4.2_IAP_Temperature_gridded_1month_netcdf/Monthly/' + name, 'iap_baseline/' + name))
    batch(jobs, 'iap_climatology_files.json')


def iap_baseline_validation():
    jobs = []
    for m in range(1,13):
        name = f'IAPv4_Temp_monthly_1_6000m_year_2001_month_{m:02d}.nc'
        jobs.append(('http://www.ocean.iap.ac.cn/ftp/cheng/IAPv4.2_IAP_Temperature_gridded_1month_netcdf/Monthly/' + name, 'iap_baseline/' + name))
    batch(jobs, 'iap_baseline_validation.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--discover', action='store_true')
    parser.add_argument('--inventory', action='store_true')
    parser.add_argument('--oras', action='store_true')
    parser.add_argument('--itp-selected', action='store_true')
    parser.add_argument('--finish', action='store_true')
    parser.add_argument('--iap-baseline', action='store_true')
    parser.add_argument('--iap-catalogs', action='store_true')
    parser.add_argument('--iap-climatology-catalogs', action='store_true')
    parser.add_argument('--iap-climatology-files', action='store_true')
    parser.add_argument('--iap-baseline-validation', action='store_true')
    args = parser.parse_args()
    OUT = args.output.resolve()
    if args.probe:
        probe()
    if args.discover:
        discover()
    if args.inventory:
        inventory()
    if args.oras:
        oras()
    if args.itp_selected:
        itp_selected()
    if args.finish:
        finish()
    if args.iap_baseline:
        iap_baseline()
    if args.iap_catalogs:
        iap_catalogs()
    if args.iap_climatology_catalogs:
        iap_climatology_catalogs()
    if args.iap_climatology_files:
        iap_climatology_files()
    if args.iap_baseline_validation:
        iap_baseline_validation()
