"""
Live End-to-End Workflow & Integration Test for PrivSynth
"""
import urllib.request
import urllib.parse
import json

BASE_URL = 'http://127.0.0.1:8000'

def test_workflow():
    print('1. Checking Health...')
    with urllib.request.urlopen(f'{BASE_URL}/api/health') as resp:
        data = json.loads(resp.read().decode())
        assert resp.status == 200
        assert data['status'] == 'ok'
    print(f'   -> Health OK: {data}')

    print('2. Checking API Info...')
    with urllib.request.urlopen(f'{BASE_URL}/api/') as resp:
        data = json.loads(resp.read().decode())
        assert resp.status == 200
        assert 'endpoints' in data
    print(f'   -> Endpoints count: {len(data["endpoints"])}')

    print('3. Upload CSV...')
    def post_multipart(url, filename, content, params=None):
        boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
        body = bytearray()
        body.extend(f'--{boundary}\r\n'.encode())
        body.extend(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode())
        body.extend(b'Content-Type: text/csv\r\n\r\n')
        body.extend(content)
        body.extend(b'\r\n')
        body.extend(f'--{boundary}--\r\n'.encode())
        
        full_url = url
        if params:
            full_url += '?' + urllib.parse.urlencode(params)
        req = urllib.request.Request(full_url, data=bytes(body))
        req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())

    sample_csv = b'age,workclass,income\n39,State-gov,<=50K\n50,Self-emp,<=50K\n38,Private,<=50K\n53,Private,<=50K\n28,Private,<=50K\n37,Private,<=50K\n49,Private,<=50K\n52,Self-emp,>50K\n31,Private,>50K\n42,Private,>50K\n37,Private,<=50K\n30,State-gov,>50K\n23,Private,<=50K\n32,Private,<=50K\n40,Private,>50K\n34,Private,<=50K\n35,Federal-gov,<=50K\n43,Private,<=50K\n54,Private,>50K\n35,Private,<=50K\n'

    status, up_res = post_multipart(f'{BASE_URL}/api/upload', 'test.csv', sample_csv)
    print(f'   -> Upload status: {status}, rows: {up_res.get("rows")}, cols: {up_res.get("columns")}')
    assert status == 200
    assert up_res['rows'] == 20

    print('4. Generate (Gaussian Copula)...')
    status, gen_res = post_multipart(f'{BASE_URL}/api/generate', 'test.csv', sample_csv, {'generator': 'gaussian_copula'})
    print(f'   -> Generate status: {status}, result_id: {gen_res.get("result_id")}')
    assert status == 200
    result_id = gen_res['result_id']

    print('5. Retrieve Result Metadata...')
    with urllib.request.urlopen(f'{BASE_URL}/api/results/{result_id}') as resp:
        res_data = json.loads(resp.read().decode())
        assert resp.status == 200
        assert res_data['rows'] == 20
        assert 'fidelity' in res_data['metrics']
    print('   -> Result Metadata retrieved successfully!')

    print('6. Download Generated CSV...')
    with urllib.request.urlopen(f'{BASE_URL}/api/results/{result_id}/download') as resp:
        assert resp.status == 200
        downloaded_csv = resp.read().decode()
        lines = [l for l in downloaded_csv.strip().split('\n') if l]
        print(f'   -> Downloaded lines: {len(lines)}')
        assert len(lines) == 21  # 1 header + 20 rows

    print('7. Error Handling Verification...')
    # Bad generator
    s, err_gen = post_multipart(f'{BASE_URL}/api/generate', 'test.csv', sample_csv, {'generator': 'fake_gen'})
    assert s == 400
    print('   -> Invalid generator rejected: 400')
    # Non-CSV
    s, err_csv = post_multipart(f'{BASE_URL}/api/upload', 'data.pdf', b'%PDF-1.4')
    assert s == 400
    print('   -> Non-CSV upload rejected: 400')
    # Traversal ID
    try:
        urllib.request.urlopen(f'{BASE_URL}/api/results/../../etc/passwd')
    except urllib.error.HTTPError as e:
        assert e.code == 404
        print('   -> Path traversal safely blocked: 404')

    print('\n========================================')
    print('ALL LIVE API WORKFLOW TESTS PASSED!')
    print('========================================')

if __name__ == '__main__':
    test_workflow()

