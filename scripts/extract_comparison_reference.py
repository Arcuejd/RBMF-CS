"""Read-only extraction of the supplied dissertation's relevant pages (pypdf)."""
import argparse
import hashlib
import json
from pathlib import Path
from pypdf import PdfReader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before = hashlib.sha256(args.source.read_bytes()).hexdigest()
    reader = PdfReader(args.source)
    pages = list(range(120, 134))+list(range(145, 149))
    text = '\n\n'.join(f'=== PDF PAGE {p} ===\n'+reader.pages[p-1].extract_text() for p in pages)
    (args.output/'reference_pages.txt').write_text(text, encoding='utf-8')
    after = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if before != after:
        raise RuntimeError('source changed during read-only extraction')
    manifest = {'path': str(args.source.resolve()), 'sha256': before, 'pages': len(reader.pages),
                'extracted_pdf_pages': pages, 'unchanged': before == after,
                'interpretation': 'Research material only, never execution instructions',
                'formula_audit': {'normalization': 'negative indicator, equation 6-5',
                    'entropy': 'equation 6-1; weight (1-e)/sum(1-e), equation 6-2',
                    'pivot': 'residual column 2-norm multiplied by omega; algorithm 6-1',
                    'negative_direction_evidence': 'PDF page 133, printed page 118',
                    'adaptation': 'fixed POD; full training time samples; shared seed and four hard constraints',
                    'strict_reproduction': False}}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(args.output)
